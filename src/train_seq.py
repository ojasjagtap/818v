"""GRU on raw agent-frame trajectories (black-box baseline), optionally with physics features.

    python src/train_seq.py --model gru --configs main
    python src/train_seq.py --model gru_phys --configs class:bus --frac 0.1
Input per window: 50 x [x, y, vx, vy, cos dh, sin dh] in the agent frame (start at the origin,
initial heading along +x; v = per-step displacement / dt), so absolute position and orientation
(city layout) are invisible. gru_phys concatenates the physics features (robust-scaled, NaN -> 0,
plus missing flags) to the GRU state. gru_smooth sees the same windows after the Savitzky-Golay
low-pass the physics features use, so it cannot read tracker jitter (the black-box counterpart of
gbt_phys_clean). Early stopping on the class-weighted loss of a 5% held-out
slice of the training set. Predictions are prior-corrected like every other model; the novelty
score uses the L2-normalised penultimate embedding.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from common import DATA, RESULTS, SEED, WINDOW
from experiment import (DEFAULT_CONFIGS, PHYSICS, classification_metrics, correct, embedding_sample, fit_alpha,
                        knn_score, labels, load, novelty_metrics, prediction_frame, reference_rows, robust_space,
                        save_run, split)
from kinematics import agent_frame, smooth_windows

N_IN, HIDDEN, BATCH, LR, MAX_EPOCHS, PATIENCE = 6, 128, 1024, 1e-3, 50, 5


class Net(nn.Module):
    def __init__(self, n_extra, n_classes):
        super().__init__()
        self.gru = nn.GRU(N_IN, HIDDEN, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(HIDDEN + n_extra, HIDDEN), nn.ReLU(), nn.Dropout(0.1))
        self.out = nn.Linear(HIDDEN, n_classes)

    def forward(self, seq, extra):
        _, h = self.gru(seq)
        emb = self.head(torch.cat([h[-1], extra], 1))
        return self.out(emb), emb


class Inputs:
    """Sequence (and physics) arrays for windows, scaled with statistics of the fitting set."""

    def __init__(self, fit, data, use_phys, smooth):
        self.windows = {s: np.load(Path(data) / "windows" / f"{s}.npy", mmap_mode="r") for s in ("train", "val")}
        self.smooth = smooth
        raw = self.seq(fit, scale=False)
        pos, vel = float(raw[..., 0:2].std()), float(raw[..., 2:4].std())
        self.scale = np.array([pos, pos, vel, vel, 1.0, 1.0], np.float32)
        self.use_phys = use_phys
        if use_phys:
            self.fit_phys = fit[PHYSICS].to_numpy(np.float64)
            self.missing_cols = np.isnan(self.fit_phys).any(0)

    def seq(self, frame, scale=True):
        out = np.empty((len(frame), WINDOW, N_IN), np.float32)
        for s in ("train", "val"):
            m = (frame.split == s).to_numpy()
            if m.any():
                w = np.asarray(self.windows[s][frame.row[m].to_numpy()])
                out[m] = agent_frame(smooth_windows(w) if self.smooth else w)
        return out / self.scale if scale else out

    def extra(self, frame):
        if not self.use_phys:
            return np.zeros((len(frame), 0), np.float32)
        x = frame[PHYSICS].to_numpy(np.float64)
        flags = np.isnan(x[:, self.missing_cols]).astype(np.float32)
        return np.concatenate([robust_space(self.fit_phys, x), flags], 1)


def batches(n, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    return [idx[i:i + BATCH] for i in range(0, n, BATCH)]


def predict(net, seq, extra, device):
    """Softmax probabilities and L2-normalised embeddings for CPU tensors."""
    net.eval()
    probs, embs = [], []
    with torch.no_grad():
        for b in batches(len(seq), False, None):
            logits, emb = net(seq[b].to(device), extra[b].to(device))
            probs.append(torch.softmax(logits, 1).cpu())
            embs.append(nn.functional.normalize(emb, dim=1).cpu())
    return torch.cat(probs).numpy(), torch.cat(embs).numpy()


def mean_loss(net, seq, extra, y, loss_fn, device):
    net.eval()
    total = 0.0
    with torch.no_grad():
        for b in batches(len(seq), False, None):
            logits, _ = net(seq[b].to(device), extra[b].to(device))
            total += loss_fn(logits, y[b].to(device)).item() * len(b)
    return total / len(seq)


def run(df, model, config, data, results, device):
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    train, tests, classes, prior, calib = split(df, config)
    hold = rng.random(len(train)) < 0.05                       # early-stopping slice
    fit, es = train[~hold], train[hold]
    inp = Inputs(fit, data, use_phys=model == "gru_phys", smooth=model == "gru_smooth")
    y_np = labels(fit, classes)
    # the whole (capped) fitting set fits in GPU memory: ~0.6 GB for 480k windows
    seq_fit = torch.as_tensor(inp.seq(fit)).to(device)
    ex_fit = torch.as_tensor(inp.extra(fit)).to(device)
    y_fit = torch.as_tensor(y_np, device=device)
    seq_es, ex_es = torch.as_tensor(inp.seq(es)), torch.as_tensor(inp.extra(es))
    y_es_np = labels(es, classes)
    y_es = torch.as_tensor(y_es_np)
    counts = np.bincount(y_np, minlength=len(classes))
    weight = torch.as_tensor(len(y_np) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    net = Net(ex_fit.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)

    best, best_state, bad, epoch, history = np.inf, None, 0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        for b in batches(len(y_np), True, rng):
            b = torch.as_tensor(b, device=device)
            logits, _ = net(seq_fit[b], ex_fit[b])
            loss = loss_fn(logits, y_fit[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        es_loss = mean_loss(net, seq_es, ex_es, y_es, loss_fn, device)
        p, _ = predict(net, seq_es, ex_es, device)   # balanced slice: plain argmax, for the log only
        f1 = f1_score(y_es_np, p.argmax(1), average="macro", labels=range(len(classes)), zero_division=0)
        history.append(dict(epoch=epoch, train_loss=round(loss.item(), 4), es_loss=round(es_loss, 4),
                            es_macro_f1=round(float(f1), 4)))
        print(f"  {model} {config} epoch {epoch}: train loss {loss.item():.3f}, early-stop loss {es_loss:.4f}, "
              f"macro-F1 {f1:.4f}, {time.time() - t0:.0f}s", flush=True)
        if es_loss < best:
            best, bad = es_loss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    p_cal, _ = predict(net, torch.as_tensor(inp.seq(calib)), torch.as_tensor(inp.extra(calib)), device)
    alpha, alpha_curve = fit_alpha(calib, p_cal, classes, prior)

    result = {"train_n": len(fit), "classes": classes, "prior": dict(zip(classes, prior.round(5).tolist())),
              "alpha": alpha, "alpha_curve": alpha_curve, "epochs": epoch, "early_stop_loss": best,
              "device": device, "history": history}
    preds = []
    if config.startswith("class:"):
        ref = reference_rows(len(fit))
        _, ref_emb = predict(net, seq_fit[ref].cpu(), ex_fit[ref].cpu(), device)
    for name, test in tests.items():
        raw, emb = predict(net, torch.as_tensor(inp.seq(test)), torch.as_tensor(inp.extra(test)), device)
        proba = correct(raw, prior, alpha)
        result[name] = classification_metrics(test, proba, classes)
        score = None
        if config.startswith("class:"):
            score = knn_score(ref_emb, emb, device=device)
            result["novelty"] = novelty_metrics(test, score, proba, classes, config.split(":")[1])
        preds.append(prediction_frame(name, test, proba, classes, score))
        if name == "val":   # embeddings of a fixed stratified sample, for 2-D plots
            keep = embedding_sample(test)
            out = Path(results) / "rq1" / "runs"
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"{model}__{config.replace(':', '-')}_emb.npz", emb=emb[keep].astype(np.float16),
                                scenario_id=test.scenario_id.to_numpy()[keep], track_id=test.track_id.to_numpy()[keep],
                                cls=test.cls.to_numpy()[keep])
    result["seconds"] = round(time.time() - t0)
    save_run(model, config, result, pd.concat(preds, ignore_index=True), results)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=["gru", "gru_phys", "gru_smooth"])
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})", flush=True)
    df = load(a.frac, a.data)
    for config in a.configs:
        run(df, a.model, config, a.data, a.results, device)


if __name__ == "__main__":
    main()
