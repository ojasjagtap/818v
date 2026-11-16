"""RQ2: as camera crops get darker and blurrier, when does motion beat appearance, and does fusing the
two help? 5-fold cross-validation grouped by log over all labelled AV2 Sensor logs.

    python src/rq2.py [--folds 5] [--epochs 6]
Per fold, on 80% of the training logs: a physics-feature GBT (motion) and an ImageNet ResNet-18
fine-tuned on clean crops (appearance). The other 20% (calibration logs) set each model's decision
rule and the fusion weight, pooled over every degradation level, so nothing is tuned per level.
Each held-out track is scored at 11 settings: clean, 5 darkness levels, 5 blur levels. Darkness is
exposure scaling in linear light with photon shot noise, read noise and 8-bit quantisation.
Fusion, then the same prior correction as RQ1. Fixed-weight fusion: p ∝ p_motion^(1-w) · p_image^w,
one w. Quality-aware fusion: p ∝ p_motion^a · p_image^w(q), w(q) = sigmoid(c0 + c1·brightness +
c2·sharpness), both measured on the (degraded) crop itself, so it learns to back off from the camera
without being told the degradation level. The free motion exponent a absorbs calibration, so w only
measures trust in the camera.
Writes results/rq2/{predictions.parquet, curves.csv, folds.csv, example_crops.npz} and
figures; `--figures-only` redraws every figure from curves.csv + example_crops.npz alone.
"""
import argparse
import math
import time
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch import nn  # noqa: E402

from analyze import INK2, Boot, ci, confusion, scores  # noqa: E402  (also sets the figure style)
from common import CLASSES, DATA, RESULTS, SEED  # noqa: E402
from experiment import ALPHAS, PHYSICS, correct, fit_alpha  # noqa: E402

EXPOSURE = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]   # darkness levels 0-5
SIGMA = [0, 1, 2, 4, 8, 16]                          # blur levels 0-5 (Gaussian sigma, px of a 128-px crop)
SETTINGS = [("clean", 0)] + [("dark", i) for i in range(1, 6)] + [("blur", i) for i in range(1, 6)]
FULL_WELL, READ_NOISE = 1000.0, 2 / 255              # photons at white for exposure 1; read noise
MODELS = {"motion": "Motion (physics GBT)", "image": "Appearance (ResNet-18)",
          "fusion": "Fixed-weight fusion", "fusion_q": "Quality-aware fusion"}
# validated all-pairs trio for the three main lines; the fixed-weight fuser is a neutral reference
COLOR = {"motion": "#eb6834", "image": "#2a78d6", "fusion": "#9a9994", "fusion_q": "#1baf7a"}
LAPLACE = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OUT = RESULTS / "rq2"


def degrade(x, kind, level, gen):
    """x: float [B, 3, H, W] in [0, 1] on the model's device."""
    if level == 0:
        return x
    if kind == "dark":
        photons = torch.poisson(x.clamp(0, 1) ** 2.2 * EXPOSURE[level] * FULL_WELL, generator=gen)
        noisy = photons / FULL_WELL + READ_NOISE * torch.randn(x.shape, generator=gen, device=x.device)
        return torch.round(noisy.clamp(0, 1) ** (1 / 2.2) * 255) / 255
    return blur(x, SIGMA[level])


def blur(x, sigma):
    """Separable Gaussian blur (two 1-D passes; a 2-D 97x97 kernel at sigma 16 is ~50x slower)."""
    r = math.ceil(3 * sigma)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    c = x.shape[1]
    x = torch.nn.functional.pad(x, (r, r, r, r), mode="reflect")
    x = torch.nn.functional.conv2d(x, g.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return torch.nn.functional.conv2d(x, g.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def hashed(keys, salt, mod):
    return np.array([zlib.crc32(f"{salt}/{k}".encode()) % mod for k in keys])


def full_proba(model, x):
    """GBT probabilities in CLASSES order (classes absent from a fold's training set get 0)."""
    p = np.zeros((len(x), len(CLASSES)))
    p[:, model.classes_] = model.predict_proba(x)
    return p


def macro_f1(y, p):
    return f1_score(y, p.argmax(1), average="macro", labels=range(len(CLASSES)), zero_division=0)


# ----------------------------------------------------------------------------- appearance model
def train_image(x_u8, y, epochs, device):
    """Fine-tune ImageNet ResNet-18 on clean crops (uint8 [N, H, W, 3]); balanced class weights."""
    torch.manual_seed(SEED)
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
    net.to(device)
    counts = np.bincount(y, minlength=len(CLASSES))
    w = torch.tensor(len(y) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(len(y)), max(1, len(y) // 128)):
            xb = normalize(to_input(x_u8[b], device))
            flip = torch.rand(len(b), device=device) < 0.5
            xb[flip] = xb[flip].flip(-1)                      # horizontal flip: the only augmentation
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = loss_fn(net(xb), torch.as_tensor(y[b], device=device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        print(f"    image epoch {epoch + 1}/{epochs}: loss {loss.item():.3f}", flush=True)
    return net


def to_input(x_u8, device):
    """uint8 [N, H, W, 3] -> float [N, 3, H, W] in [0, 1] (degradations act on this)."""
    return torch.as_tensor(np.ascontiguousarray(x_u8), device=device).permute(0, 3, 1, 2).float() / 255


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def image_probs(net, x_u8, kind, level, device, seed):
    """Softmax probabilities [N, 5] and image quality [N, 2] (brightness, log variance of the
    Laplacian) of every crop under one degradation setting, both from the degraded crop itself."""
    net.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    lap = LAPLACE.to(device)
    probs, quality = [], []
    with torch.no_grad():
        for i in range(0, len(x_u8), 512):
            xb = degrade(to_input(x_u8[i:i + 512], device), kind, level, gen)
            gray = xb.mean(1, keepdim=True)
            sharp = torch.nn.functional.conv2d(gray, lap).flatten(1).var(1)
            quality.append(torch.stack([gray.flatten(1).mean(1), torch.log(sharp + 1e-8)], 1).cpu())
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                probs.append(torch.softmax(net(normalize(xb)).float(), 1).cpu())
    return torch.cat(probs).numpy(), torch.cat(quality).numpy()


def per_track(p_crop, crop_track, n_tracks):
    """Mean crop probability per track."""
    s = np.zeros((n_tracks, p_crop.shape[1]))
    np.add.at(s, crop_track, p_crop)
    return s / np.maximum(np.bincount(crop_track, minlength=n_tracks), 1)[:, None]


# ----------------------------------------------------------------------------- one fold
def run_fold(k, tracks, crops, x_u8, args, device):
    test = tracks.fold.to_numpy() == k
    train_logs = np.unique(tracks.log_id[~test])
    order = np.argsort(hashed(train_logs, f"rq2calib{k}", 2 ** 31))
    calib_logs = set(train_logs[order[:max(1, round(0.2 * len(train_logs)))]])   # 20% of this fold's logs
    calib = ~test & tracks.log_id.isin(calib_logs).to_numpy()
    fit = ~test & ~calib
    y = tracks.y.to_numpy()
    prior = np.bincount(y[fit], minlength=len(CLASSES)) / fit.sum()

    xm = tracks[PHYSICS].to_numpy(np.float64, copy=True)
    xm[~np.isfinite(xm)] = np.nan
    stratifiable = np.bincount(y[fit])[np.bincount(y[fit]) > 0].min() >= 2   # early stopping splits by class
    gbt = HistGradientBoostingClassifier(max_iter=500, early_stopping=bool(stratifiable), validation_fraction=0.1,
                                         n_iter_no_change=20, class_weight="balanced", random_state=SEED)
    gbt.fit(xm[fit], y[fit])
    pm = full_proba(gbt, xm)                                   # raw (balanced) motion probabilities

    cfit = fit[crops.tidx.to_numpy()]
    t0 = time.time()
    net = train_image(x_u8[cfit], y[crops.tidx.to_numpy()[cfit]], args.epochs, device)
    print(f"  fold {k}: image model trained on {cfit.sum()} crops in {time.time() - t0:.0f}s", flush=True)
    score = ~fit                                               # calibration + test tracks
    cs = score[crops.tidx.to_numpy()]
    idx_map = -np.ones(len(tracks), int)
    idx_map[np.flatnonzero(score)] = np.arange(score.sum())
    crop_track = idx_map[crops.tidx.to_numpy()[cs]]
    pi, qi = {}, {}                                             # raw image probabilities / quality, scored tracks
    for j, s in enumerate(SETTINGS):
        p_crop, q_crop = image_probs(net, x_u8[cs], *s, device, seed=SEED + 100 * k + j)
        pi[s], qi[s] = per_track(p_crop, crop_track, score.sum()), per_track(q_crop, crop_track, score.sum())

    # decision rules on the calibration logs, pooled over all 11 settings
    cal_in_score = calib[score]
    y_cal = y[calib]
    cal_frame = pd.DataFrame({"cls": np.array(CLASSES)[np.tile(y_cal, len(SETTINGS))]})
    a_m, _ = fit_alpha(pd.DataFrame({"cls": np.array(CLASSES)[y_cal]}), pm[calib], CLASSES, prior)
    pi_cal = np.concatenate([pi[s][cal_in_score] for s in SETTINGS])
    a_i, _ = fit_alpha(cal_frame, pi_cal, CLASSES, prior)
    pm_cal = np.tile(pm[calib], (len(SETTINGS), 1))
    best = (-1, 0.5, 1.0)
    for w in ALPHAS:                                           # fusion weight on the image model
        fused = fuse(pm_cal, pi_cal, w)
        for a in ALPHAS:
            f1 = macro_f1(np.tile(y_cal, len(SETTINGS)), correct(fused, prior, a))
            if f1 > best[0]:
                best = (f1, round(float(w), 2), round(float(a), 2))
    _, w_f, a_f = best
    y_tiled = np.tile(y_cal, len(SETTINGS))
    qi_cal = np.concatenate([qi[s][cal_in_score] for s in SETTINGS])
    coef, mu, sd = fit_quality_fusion(pm_cal, pi_cal, qi_cal, y_tiled)
    fused_q = fuse_q(pm_cal, pi_cal, coef, quality_weight(qi_cal, coef[1:], mu, sd))
    a_q = max(ALPHAS, key=lambda a: macro_f1(y_tiled, correct(fused_q, prior, a)))
    a_q = round(float(a_q), 2)
    print(f"  fold {k}: alpha motion {a_m}, image {a_i}; fixed fusion weight on image {w_f}, alpha {a_f}; "
          f"quality fusion coef {np.round(coef, 2).tolist()}, alpha {a_q}", flush=True)

    rows = []
    tst_in_score = test[score]
    meta = tracks.loc[test, ["log_id", "track_id", "cls"]].reset_index(drop=True)
    for kind, level in SETTINGS:
        p_i = pi[(kind, level)][tst_in_score]
        w_t = quality_weight(qi[(kind, level)][tst_in_score], coef[1:], mu, sd)
        for model, p, w in (("motion", correct(pm[test], prior, a_m), 0.0),
                            ("image", correct(p_i, prior, a_i), 1.0),
                            ("fusion", correct(fuse(pm[test], p_i, w_f), prior, a_f), w_f),
                            ("fusion_q", correct(fuse_q(pm[test], p_i, coef, w_t), prior, a_q), w_t)):
            df = meta.assign(fold=k, kind=kind, level=level, model=model, image_weight=w)
            df[[f"p_{c}" for c in CLASSES]] = p
            rows.append(df)
    return pd.concat(rows, ignore_index=True), dict(fold=k, alpha_motion=a_m, alpha_image=a_i,
                                                     fusion_weight_image=w_f, alpha_fusion=a_f,
                                                     quality_coef=" ".join(f"{c:.2f}" for c in coef),
                                                     alpha_fusion_q=a_q, n_fit=int(fit.sum()),
                                                     n_calib=int(calib.sum()), n_test=int(test.sum()))


def fuse(pm, pi, w):
    """Weighted geometric pooling; w is one weight or one per row."""
    w = np.asarray(w, float).reshape(-1, 1)
    p = np.exp((1 - w) * np.log(pm + 1e-9) + w * np.log(pi + 1e-9))
    return p / p.sum(1, keepdims=True)


def quality_weight(q, c, mu, sd):
    """Camera weight sigmoid(c0 + c1·brightness_z + c2·sharpness_z)."""
    return 1 / (1 + np.exp(-(c[0] + ((q - mu) / sd) @ c[1:])))


def fuse_q(pm, pi, coef, w):
    """p ∝ p_motion^a · p_image^w with a = exp(coef[0]) and one camera weight per row."""
    lf = np.exp(coef[0]) * np.log(pm + 1e-9) + np.asarray(w, float).reshape(-1, 1) * np.log(pi + 1e-9)
    p = np.exp(lf - lf.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def fit_quality_fusion(pm, pi, q, y):
    """[log a, c0, c1, c2] minimising the class-balanced log-loss of p_motion^a · p_image^w(q) on the
    calibration tracks, all settings pooled."""
    mu, sd = q.mean(0), q.std(0) + 1e-9
    lm, li = np.log(pm + 1e-9), np.log(pi + 1e-9)
    cw = (len(y) / (len(CLASSES) * np.maximum(np.bincount(y, minlength=len(CLASSES)), 1)))[y]
    rows = np.arange(len(y))

    def loss(c):
        lf = np.exp(c[0]) * lm + quality_weight(q, c[1:], mu, sd)[:, None] * li
        return -(cw * (lf[rows, y] - logsumexp(lf, axis=1))).mean()

    return minimize(loss, np.zeros(4), method="L-BFGS-B").x, mu, sd


# ----------------------------------------------------------------------------- results and figures
def curves(pred):
    rows = []
    for kind in ("dark", "blur"):
        for level in range(6):
            src = ("clean", 0) if level == 0 else (kind, level)
            sub = pred[(pred.kind == src[0]) & (pred.level == src[1])]
            boot = Boot(sub[sub.model == "motion"].log_id.to_numpy())
            for model in MODELS:
                p = sub[sub.model == model].reset_index(drop=True).rename(columns={"log_id": "scenario_id"})
                conf = confusion(p, CLASSES, boot)
                f1, acc, f1c = scores(conf.sum(0), len(CLASSES))
                df1, dacc, _ = scores(boot.w @ conf, len(CLASSES))
                c = conf.sum(0).reshape(len(CLASSES), len(CLASSES))
                rows.append(dict(kind=kind, level=level, model=model, image_weight=float(p.image_weight.mean()),
                                 macro_f1=f1, macro_f1_lo=ci(df1)[0],
                                 macro_f1_hi=ci(df1)[1], accuracy=acc, accuracy_lo=ci(dacc)[0],
                                 accuracy_hi=ci(dacc)[1],
                                 **{f"recall_{cl}": c[i, i] / max(c[i].sum(), 1) for i, cl in enumerate(CLASSES)}))
    return pd.DataFrame(rows)


def crossover(cur, kind, metric):
    """First level at which appearance falls below motion (None if it never does)."""
    s = cur[cur.kind == kind].pivot(index="level", columns="model", values=metric)
    below = s.index[s.image < s.motion]
    return int(below[0]) if len(below) else None


def example_crops(crops, x_u8):
    """One typical crop per class (box height nearest the class's 75th percentile; the largest
    boxes are often cut off at the image edge)."""
    out = {}
    for cl in CLASSES:
        sub = crops[crops.cls == cl]
        if len(sub):
            out[cl] = x_u8[sub.index[np.abs(sub.box_h.to_numpy() - np.percentile(sub.box_h, 75)).argmin()]]
    return out


def figures(cur, examples):
    xlab = {"dark": [f"×1/{int(1 / e)}" if e < 1 else "clean" for e in EXPOSURE],
            "blur": [f"σ={s}" if s else "clean" for s in SIGMA]}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
    for r, (metric, lab) in enumerate((("macro_f1", "macro-F1"), ("accuracy", "accuracy"))):
        for c, (kind, title) in enumerate((("dark", "Darker (exposure, with sensor noise)"),
                                           ("blur", "Blurrier (Gaussian, 128-px crops)"))):
            ax = axes[r][c]
            s = cur[cur.kind == kind]
            for m in MODELS:
                d = s[s.model == m].sort_values("level")
                ax.fill_between(d.level, d[metric + "_lo"], d[metric + "_hi"], color=COLOR[m], alpha=0.15, lw=0)
                ax.plot(d.level, d[metric], color=COLOR[m], marker="o", ms=5, label=MODELS[m])
            x = crossover(cur, kind, metric)
            if x is not None:
                ax.axvline(x, color=INK2, lw=1)
                ax.annotate("appearance falls\nbelow motion", (x, ax.get_ylim()[0]), xytext=(-4, 6),
                            textcoords="offset points", fontsize=7, color=INK2, ha="right")
            ax.set_xticks(range(6), xlab[kind])
            ax.set_xlim(-0.2, 5.2)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(lab)
    axes[0][1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "rq2_curves.png")
    plt.close(fig)

    # how much the quality-aware fuser trusts the camera at each setting
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (kind, title) in zip(axes, (("dark", "Darker"), ("blur", "Blurrier"))):
        d = cur[(cur.kind == kind) & (cur.model == "fusion_q")].sort_values("level")
        ax.plot(d.level, d.image_weight, color=COLOR["fusion_q"], marker="o", ms=6)
        fixed = cur[(cur.kind == kind) & (cur.model == "fusion")].image_weight.mean()
        ax.axhline(fixed, color=COLOR["fusion"], lw=1.5)
        ax.annotate("fixed-weight fuser", (5, fixed), xytext=(0, 4), textcoords="offset points", ha="right",
                    fontsize=7, color=INK2)
        ax.set_xticks(range(6), xlab[kind])
        ax.set_ylim(0, 1)
        ax.set_title(title)
    axes[0].set_ylabel("weight on the camera model\n(quality-aware fusion, mean over tracks)")
    fig.tight_layout()
    fig.savefig(OUT / "rq2_fusion_weight.png")
    plt.close(fig)

    # what the classifier sees: one typical crop per class, at every setting
    fig, axes = plt.subplots(len(CLASSES), len(SETTINGS), figsize=(1.25 * len(SETTINGS), 1.35 * len(CLASSES)))
    gen = torch.Generator().manual_seed(0)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for i, cl in enumerate(CLASSES):
        axes[i][0].set_ylabel(cl, fontsize=8)
        if cl not in examples:
            continue
        x = to_input(examples[cl][None], "cpu")
        for j, (kind, level) in enumerate(SETTINGS):
            img = degrade(x, kind, level, gen)[0].permute(1, 2, 0).clamp(0, 1).numpy()
            ax = axes[i][j]
            ax.imshow(img)
            if i == 0:
                ax.set_title("clean" if kind == "clean" else (f"dark {level}" if kind == "dark" else f"blur {level}"),
                             fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "rq2_examples.png", dpi=130)
    plt.close(fig)

    # per-class recall: small multiples (rows: darkness / blur, columns: classes)
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(14, 5.4), sharey=True)
    for r, kind in enumerate(("dark", "blur")):
        s = cur[cur.kind == kind]
        for c, cl in enumerate(CLASSES):
            ax = axes[r][c]
            for m in MODELS:
                d = s[s.model == m].sort_values("level")
                ax.plot(d.level, d[f"recall_{cl}"], color=COLOR[m], marker="o", ms=4, label=MODELS[m])
            ax.set_xticks(range(6), [str(v) for v in range(6)])
            if r == 0:
                ax.set_title(cl)
            if c == 0:
                ax.set_ylabel(f"recall ({'darkness' if kind == 'dark' else 'blur'} level)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "rq2_per_class.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from curves.csv")
    a = ap.parse_args()
    if a.figures_only:      # needs only results/rq2: works on a laptop without the Sensor data
        return figures(pd.read_csv(OUT / "curves.csv"), dict(np.load(OUT / "example_crops.npz")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    win = pd.read_parquet(DATA / "sensor" / "windows.parquet")
    crops = pd.read_csv(DATA / "sensor" / "crops.csv")
    key = win.log_id + "/" + win.track_id
    tracks = win[key.isin(set(crops.log_id + "/" + crops.track_id))].reset_index(drop=True)
    tracks["y"] = tracks.cls.map({c: i for i, c in enumerate(CLASSES)})
    tracks["fold"] = hashed(tracks.log_id, "rq2fold", a.folds)
    pos = pd.Series(np.arange(len(tracks)), index=tracks.log_id + "/" + tracks.track_id)
    crops["tidx"] = pos.loc[crops.log_id + "/" + crops.track_id].to_numpy()
    t0 = time.time()
    x_u8 = np.load(DATA / "sensor" / "crops.npy")      # packed by sensor_crops.py --merge, rows = crops.csv
    assert len(x_u8) == len(crops), (len(x_u8), len(crops))
    print(f"{len(tracks)} tracks, {len(crops)} crops loaded in {time.time() - t0:.0f}s; device {device}", flush=True)
    examples = example_crops(crops, x_u8)
    np.savez_compressed(OUT / "example_crops.npz", **examples)
    preds, folds = [], []
    for k in range(a.folds):
        p, info = run_fold(k, tracks, crops, x_u8, a, device)
        preds.append(p)
        folds.append(info)
    pred = pd.concat(preds, ignore_index=True)
    pred.to_parquet(OUT / "predictions.parquet", index=False)
    pd.DataFrame(folds).to_csv(OUT / "folds.csv", index=False)
    cur = curves(pred)
    cur.to_csv(OUT / "curves.csv", index=False, float_format="%.4f")
    figures(cur, examples)


if __name__ == "__main__":
    main()
