"""RQ1 report figures and tables from results/rq1/runs (+ data/feats/val.parquet).

    python src/analyze.py [--results results] [--data data]
writes results/rq1/figures/*.png and results/rq1/tables/*.csv.
Confidence intervals resample whole scenarios (windows of one scenario are correlated) and use the
same draws for every model, so differences between models are paired.
"""
import argparse
import json
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from common import CITIES, CLASSES, DATA, RESULTS  # noqa: E402
from experiment import BASELINE, PHYSICS, embedding_sample, robust_space  # noqa: E402

MODELS = ["gbt_base", "gbt_phys", "gbt_phys_clean", "gru", "gru_phys", "gru_smooth", "fusion"]
NAME = {"gbt_base": "GBT: speed/accel stats", "gbt_phys": "GBT: physics",
        "gbt_phys_clean": "GBT: physics, no noise feats", "gru": "GRU (black box)",
        "gru_phys": "GRU + physics", "gru_smooth": "GRU, smoothed input",
        "fusion": "Late fusion: GRU + GBT physics"}
# categorical slots 1-7 of the validated reference palette, fixed per model in every figure
COLOR = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
HIGHLIGHT, OTHER = ("#2a78d6", "#eb6834"), "#d3d2cd"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
SEQ = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
PAIRS = [("gbt_phys", "gbt_base"), ("gbt_phys", "gru"), ("gbt_phys_clean", "gru_smooth"),
         ("gru_phys", "gru"), ("gru", "gru_smooth"), ("gbt_phys", "gbt_phys_clean"),
         ("fusion", "gru"), ("fusion", "gbt_phys")]
B_F1, B_AUC = 500, 200
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 160,
    "savefig.bbox": "tight"})
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------- data and bootstrap
class Runs:
    def __init__(self, results):
        self.dir = Path(results) / "rq1"
        self.runs = {}
        for p in sorted((self.dir / "runs").glob("*.json")):
            r = json.loads(p.read_text())
            r["_stem"] = str(p.with_suffix(""))
            self.runs[(r["model"], r["config"])] = r

    def models(self, config):
        return [m for m in MODELS if (m, config) in self.runs]

    def get(self, model, config):
        return self.runs[(model, config)]

    def preds(self, model, config, test):
        p = pd.read_parquet(self.get(model, config)["_stem"] + ".parquet")
        return p[p.test == test].reset_index(drop=True)

    def configs(self, kind):
        return sorted({c for _, c in self.runs if c.startswith(kind + ":")})


class Boot:
    """Scenario-level bootstrap draws for one test set, shared by every model scored on it."""

    def __init__(self, scenario_ids, n=B_F1):
        self.sid, self.idx = np.unique(scenario_ids, return_inverse=True)
        s = len(self.sid)
        self.w = np.stack([np.bincount(rng.integers(0, s, s), minlength=s) for _ in range(n)]).astype(np.float32)


def confusion(df, classes, boot):
    """Per-scenario confusion counts [scenarios, K*K] over windows of trained classes."""
    k = len(classes)
    known = df.cls.isin(classes).to_numpy()
    y = df.cls.map({c: i for i, c in enumerate(classes)}).to_numpy()[known].astype(int)
    p = df[[f"p_{c}" for c in classes]].to_numpy()[known].argmax(1)
    conf = np.zeros((len(boot.sid), k * k), np.float32)
    np.add.at(conf, (boot.idx[known], y * k + p), 1)
    return conf


def scores(conf, k):
    """(macro-F1, accuracy, per-class F1) for confusion counts [..., K*K]."""
    c = conf.reshape(*conf.shape[:-1], k, k)
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    denom = c.sum(-2) + c.sum(-1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-9), 0.0)
    return f1.mean(-1), tp.sum(-1) / c.sum((-2, -1)), f1


def ci(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def auc_draws(y, s, idx, w):
    """Weighted Mann-Whitney AUROC for each bootstrap row of scenario weights w."""
    o = np.argsort(s, kind="mergesort")
    y, idx = y[o], idx[o]
    out = []
    for ws in w:
        wt = ws[idx]
        wp, wn = np.where(y, wt, 0.0), np.where(y, 0.0, wt)
        out.append((wp * (np.cumsum(wn) - wn)).sum() / max(wp.sum() * wn.sum(), 1e-12))
    return np.array(out)



# ---------------------------------------------------------------------------- figure helpers
def bars(ax, groups, series, values, errs=None, label_values=False):
    """Grouped bars: values[s][g]; series are models (fixed colors); 2px white gaps between bars."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, m in enumerate(series):
        pos = x - 0.4 + width * (i + 0.5)
        yerr = None if errs is None else np.abs(np.array(errs[m]).T - np.array(values[m]))
        b = ax.bar(pos, values[m], width, color=COLOR[m], edgecolor="white", linewidth=1.5, label=NAME[m],
                   yerr=yerr, error_kw=dict(ecolor=INK2, elinewidth=1, capsize=2))
        if label_values:
            ax.bar_label(b, labels=[f"{v:.3f}" for v in values[m]], padding=3, fontsize=7, color=INK2)
    ax.set_xticks(x, groups)
    return ax


def legend_right(ax):
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)


def heat(ax, mat, rows, cols, vmin=0, vmax=1, fmt_s="{:.2f}"):
    ax.imshow(mat, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                v = (mat[i, j] - vmin) / (vmax - vmin)
                ax.text(j, i, fmt_s.format(mat[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else INK)


def save(fig, out, name, caption, figures):
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    figures.append((name, caption))


# ---------------------------------------------------------------------------- analyses
def main_analysis(R, out, tables, figures):
    ms = R.models("main")
    test = R.preds(ms[0], "main", "val")
    boot = Boot(test.scenario_id.to_numpy())
    classes = R.get(ms[0], "main")["classes"]
    k = len(classes)
    rows, draws, per_class, conf_pt = [], {}, {}, {}
    for m in ms:
        conf = confusion(R.preds(m, "main", "val"), classes, boot)
        f1, acc, f1c = scores(conf.sum(0), k)
        df1, dacc, _ = scores(boot.w @ conf, k)
        draws[m] = df1
        c = conf.sum(0).reshape(k, k)
        conf_pt[m], per_class[m] = c, f1c
        row = dict(model=m, macro_f1=f1, macro_f1_lo=ci(df1)[0], macro_f1_hi=ci(df1)[1],
                   accuracy=acc, accuracy_lo=ci(dacc)[0], accuracy_hi=ci(dacc)[1], alpha=R.get(m, "main").get("alpha"))
        row.update({f"f1_{cl}": float(v) for cl, v in zip(classes, f1c)})
        row.update({f"recall_{cl}": float(c[i, i] / c[i].sum()) for i, cl in enumerate(classes)})
        for a, b in [("bus", "vehicle"), ("vehicle", "bus"), ("cyclist", "motorcyclist"), ("motorcyclist", "cyclist")]:
            ia, ib = classes.index(a), classes.index(b)
            row[f"{a}_as_{b}"] = float(c[ia, ib] / c[ia].sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "main.csv", index=False, float_format="%.4f")
    tables["main"] = t
    f1_pt = t.set_index("model").macro_f1
    pairs = [dict(a=a, b=b, diff=float(f1_pt[a] - f1_pt[b]), lo=ci(draws[a] - draws[b])[0],
                  hi=ci(draws[a] - draws[b])[1]) for a, b in PAIRS if a in draws and b in draws]
    pd.DataFrame(pairs).to_csv(out / "tables" / "main_pairwise.csv", index=False, float_format="%.4f")
    tables["pairs"] = pd.DataFrame(pairs)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a_, key, title in ((ax[0], "macro_f1", "Macro-F1 (primary)"), (ax[1], "accuracy", "Accuracy")):
        vals = {m: [t.set_index("model").loc[m, key]] for m in ms}
        errs = {m: [[t.set_index("model").loc[m, key + "_lo"], t.set_index("model").loc[m, key + "_hi"]]] for m in ms}
        bars(a_, [""], ms, vals, errs, label_values=True)
        a_.set_title(title)
        a_.set_ylim(0, min(1.0, max(max(v) for v in vals.values()) + 0.08))
        a_.set_xticks([])
    legend_right(ax[1])
    save(fig, out / "figures", "main_scores",
         "In-distribution test (MF val): macro-F1 and accuracy per model, 95% scenario-bootstrap CIs.", figures)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    bars(ax, classes, ms, {m: list(per_class[m]) for m in ms})
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 (in-distribution)")
    legend_right(ax)
    save(fig, out / "figures", "per_class_f1", "Per-class F1 by model: where each model wins or fails.", figures)

    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.4))
    for a_, m in zip(axes.flat, ms):
        c = conf_pt[m]
        heat(a_, c / c.sum(1, keepdims=True), classes, classes)
        a_.set_title(NAME[m])
        a_.set_ylabel("true")
        a_.set_xlabel("predicted")
    for a_ in axes.flat[len(ms):]:
        a_.axis("off")
    fig.tight_layout()
    save(fig, out / "figures", "confusion", "Row-normalised confusion matrices (rows sum to 1).", figures)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    keys = ["bus_as_vehicle", "vehicle_as_bus", "cyclist_as_motorcyclist", "motorcyclist_as_cyclist"]
    bars(ax, [k_.replace("_as_", " → ") for k_ in keys], ms, {m: [t.set_index("model").loc[m, k_] for k_ in keys] for m in ms})
    ax.set_ylabel("fraction of true class")
    ax.set_title("Hard pairs: how often the true class is predicted as its look-alike")
    legend_right(ax)
    save(fig, out / "figures", "hard_pairs", "Hard-pair confusion rates (bus/vehicle, cyclist/motorcyclist).", figures)

    if pairs:
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(pairs) + 1))
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        for i, p in enumerate(pairs):
            ax.plot([p["lo"], p["hi"]], [i, i], color=INK2, lw=2)
            ax.plot(p["diff"], i, "o", color=COLOR[p["a"]], ms=8)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(range(len(pairs)), [f"{NAME[p['a']]}  −  {NAME[p['b']]}" for p in pairs])
        ax.set_xlabel("difference in macro-F1 (95% paired scenario-bootstrap CI)")
        ax.set_title("Which differences are real?")
        save(fig, out / "figures", "pairwise", "Paired macro-F1 differences; an interval crossing 0 is not a real difference.", figures)
    return classes, boot


def city_analysis(R, out, tables, figures):
    cities = [c.split(":")[1] for c in R.configs("city")]
    if not cities:
        return
    rows, drops = [], {}
    for city in cities:
        cfg = f"city:{city}"
        ms = R.models(cfg)
        classes = R.get(ms[0], cfg)["classes"]
        k = len(classes)
        tid, tood = R.preds(ms[0], cfg, "id"), R.preds(ms[0], cfg, "ood")
        bi, bo = Boot(tid.scenario_id.to_numpy()), Boot(tood.scenario_id.to_numpy())
        for m in ms:
            ci_, co_ = confusion(R.preds(m, cfg, "id"), classes, bi), confusion(R.preds(m, cfg, "ood"), classes, bo)
            fi, ai, _ = scores(ci_.sum(0), k)
            fo, ao, _ = scores(co_.sum(0), k)
            dfi, dai, _ = scores(bi.w @ ci_, k)
            dfo, dao, _ = scores(bo.w @ co_, k)
            drops[(city, m)] = (dfi - dfo, dai - dao)
            rows.append(dict(city=city, model=m, id_macro_f1=fi, ood_macro_f1=fo, macro_f1_drop=fi - fo,
                             macro_f1_drop_lo=ci(dfi - dfo)[0], macro_f1_drop_hi=ci(dfi - dfo)[1],
                             id_accuracy=ai, ood_accuracy=ao, accuracy_drop=ai - ao,
                             accuracy_drop_lo=ci(dai - dao)[0], accuracy_drop_hi=ci(dai - dao)[1]))
    t = pd.DataFrame(rows)
    ms = [m for m in MODELS if m in set(t.model)]
    for m in ms:    # mean over held-out cities (draws averaged city by city)
        d = [drops[(c, m)] for c in cities if (c, m) in drops]
        mf, ma = np.mean([x[0] for x in d], 0), np.mean([x[1] for x in d], 0)
        sub = t[t.model == m]
        rows.append(dict(city="mean", model=m, id_macro_f1=sub.id_macro_f1.mean(), ood_macro_f1=sub.ood_macro_f1.mean(),
                         macro_f1_drop=sub.macro_f1_drop.mean(), macro_f1_drop_lo=ci(mf)[0], macro_f1_drop_hi=ci(mf)[1],
                         id_accuracy=sub.id_accuracy.mean(), ood_accuracy=sub.ood_accuracy.mean(),
                         accuracy_drop=sub.accuracy_drop.mean(), accuracy_drop_lo=ci(ma)[0], accuracy_drop_hi=ci(ma)[1]))
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "city.csv", index=False, float_format="%.4f")
    tables["city"] = t
    groups = cities + ["mean"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.4))
    for a_, key, lab in ((ax[0], "macro_f1_drop", "macro-F1 drop"), (ax[1], "accuracy_drop", "accuracy drop")):
        ix = t.set_index(["model", "city"])
        bars(a_, groups, ms, {m: [ix.loc[(m, g), key] for g in groups] for m in ms},
             {m: [[ix.loc[(m, g), key + "_lo"], ix.loc[(m, g), key + "_hi"]] for g in groups] for m in ms})
        a_.axhline(0, color=INK2, lw=1)
        a_.set_ylabel(f"{lab}\n(in-dist − held-out)")
    ax[0].set_title("Leave-one-city-out: how much each model loses on a city it never saw (higher = worse)")
    legend_right(ax[0])
    fig.tight_layout()
    save(fig, out / "figures", "city_drop",
         "Held-out-city drop per city and on average (95% scenario-bootstrap CIs).", figures)


def novelty_analysis(R, out, tables, figures):
    held = [c.split(":")[1] for c in R.configs("class")]
    if not held:
        return
    rows, rocs = [], {}
    for h in held:
        cfg = f"class:{h}"
        ms = R.models(cfg)
        base = R.preds(ms[0], cfg, "val")
        boot = Boot(base.scenario_id.to_numpy(), B_AUC)
        for m in ms:
            p = R.preds(m, cfg, "val")
            classes = R.get(m, cfg)["classes"]
            y, s = (p.cls == h).to_numpy(), p.novelty_score.to_numpy()
            auc = roc_auc_score(y, s)
            d = auc_draws(y, s, boot.idx, boot.w)
            pred = np.array(classes)[p[[f"p_{c}" for c in classes]].to_numpy()[y].argmax(1)]
            share = pd.Series(pred).value_counts(normalize=True).reindex(classes).fillna(0)
            target = share.idxmax()
            near = (p.cls.isin([h, target])).to_numpy()
            n_auc = roc_auc_score(y[near], s[near])
            nd = auc_draws(y[near], s[near], boot.idx[near], boot.w)
            rocs[(h, m)] = roc_curve(y, s)[:2]
            rows.append(dict(held_out=h, model=m, auroc=auc, auroc_lo=ci(d)[0], auroc_hi=ci(d)[1],
                             near_class=target, near_auroc=n_auc, near_auroc_lo=ci(nd)[0], near_auroc_hi=ci(nd)[1],
                             **{f"as_{c}": float(share[c]) for c in classes}))
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "novelty.csv", index=False, float_format="%.4f")
    tables["novelty"] = t
    ms = [m for m in MODELS if m in set(t.model)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a_, key, title in ((ax[0], "auroc", "novel vs\nall trained classes"),
                           (ax[1], "near_auroc", "novel vs the\nclass it is mistaken for")):
        ix = t.set_index(["model", "held_out"])
        heat(a_, np.array([[ix.loc[(m, h), key] for h in held] for m in ms]), [NAME[m] for m in ms],
             [f"{h} held out" for h in held], vmin=0.5, vmax=1.0)
        a_.set_title(f"AUROC: {title}")
    fig.tight_layout()
    save(fig, out / "figures", "novelty_auroc",
         "AUROC for flagging a class never seen in training (kNN distance in each model's feature space). "
         "Right: against only the trained class it collapses into.", figures)
    fig, axes = plt.subplots(1, len(held), figsize=(3.6 * len(held), 3.6), squeeze=False)
    for a_, h in zip(axes[0], held):
        for m in ms:
            fpr, tpr = rocs[(h, m)]
            a_.plot(fpr, tpr, color=COLOR[m], lw=1.6, label=NAME[m])
        a_.plot([0, 1], [0, 1], color=INK2, lw=0.8)
        a_.set_title(f"{h} held out")
        a_.set_xlabel("false-positive rate")
        a_.grid(axis="x")
    axes[0][0].set_ylabel("true-positive rate")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, out / "figures", "novelty_roc", "ROC curves for detecting each held-out class as novel.", figures)
    fig, axes = plt.subplots(1, len(held), figsize=(3.8 * len(held), 3.4), squeeze=False)
    for a_, h in zip(axes[0], held):
        sub = t[t.held_out == h].set_index("model")
        cols = [c for c in CLASSES if c != h]
        heat(a_, sub.loc[ms, [f"as_{c}" for c in cols]].to_numpy(), [NAME[m] for m in ms] if a_ is axes[0][0] else
             [""] * len(ms), cols)
        a_.set_title(f"held-out {h} classified as")
    fig.tight_layout()
    save(fig, out / "figures", "novelty_collapse", "Which trained class absorbs each held-out class.", figures)


def interpretation(R, out, data, classes, figures):
    imp = {m: R.get(m, "main").get("importance") for m in ("gbt_base", "gbt_phys") if (m, "main") in R.runs}
    imp = {m: v for m, v in imp.items() if v}
    if imp:
        fig, axes = plt.subplots(1, len(imp), figsize=(6 * len(imp), 5), squeeze=False)
        for a_, (m, v) in zip(axes[0], imp.items()):
            s = pd.Series(v).sort_values()
            a_.barh(s.index, s.values, color=COLOR[m], edgecolor="white", linewidth=1.5)
            a_.grid(axis="x")
            a_.grid(axis="y", visible=False)
            a_.set_title(f"{NAME[m]}: permutation importance")
            a_.set_xlabel("macro-F1 lost when the feature is shuffled")
        fig.tight_layout()
        save(fig, out / "figures", "feature_importance", "Which features the GBTs actually rely on.", figures)

    feats_path = Path(data) / "feats" / "val.parquet"
    if not feats_path.exists():
        return
    f = pd.read_parquet(feats_path)
    top = (pd.Series(imp["gbt_phys"]).sort_values(ascending=False).index[:8].tolist() if "gbt_phys" in imp
           else ["wheelbase_turn_bound", "acc_lat_resid_ratio", "slip_p50", "pred_rmse_ratio",
                 "lat_acc_p95", "acc_lon_p95", "jerk_rms", "wheelbase_slip"])
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for a_, col in zip(axes.flat, top):
        data_ = [f.loc[f.cls == c, col].dropna() for c in CLASSES]
        lo, hi = np.nanpercentile(f[col], [1, 99])
        a_.boxplot([d.clip(lo, hi) for d in data_], showfliers=False, widths=0.6,
                   medianprops=dict(color=INK), boxprops=dict(color=INK2), whiskerprops=dict(color=INK2),
                   capprops=dict(color=INK2))
        a_.set_xticks(range(1, 6), CLASSES, rotation=30, ha="right")
        a_.set_title(col)
    fig.tight_layout()
    save(fig, out / "figures", "physics_by_class",
         "Distributions of the most important physics features by class (MF val, 1-99th pct).", figures)

    if ("gbt_phys", "main") in R.runs:      # where do physics features help? turning vs straight, speed
        p = {m: R.preds(m, "main", "val") for m in R.models("main")}
        key = ["scenario_id", "track_id"]
        j = p["gbt_phys"][key].merge(f[key + ["wheelbase_slip", "speed_p50"]], on=key, how="left")
        turn = j.wheelbase_slip.notna().to_numpy()
        tert = pd.qcut(j.speed_p50, 3, labels=["slow third", "middle third", "fast third"]).to_numpy()
        groups = {"no measurable turn": ~turn, "turning (lever arm defined)": turn}
        groups.update({str(g): tert == g for g in ["slow third", "middle third", "fast third"]})
        k = len(classes)
        vals, errs = {m: [] for m in p}, {m: [] for m in p}
        for g, mask in groups.items():
            boot = Boot(j.scenario_id.to_numpy()[mask], 200)     # same draws for every model
            for m, pm in p.items():
                assert (pm.scenario_id.to_numpy() == j.scenario_id.to_numpy()).all()
                conf = confusion(pm[mask].reset_index(drop=True), classes, boot)
                vals[m].append(float(scores(conf.sum(0), k)[0]))
                errs[m].append(ci(scores(boot.w @ conf, k)[0]))
        fig, ax = plt.subplots(figsize=(12, 3.6))
        bars(ax, list(groups), list(p), vals, errs)
        ax.set_ylabel("macro-F1")
        ax.set_title("Where does each model work? Windows split by turning and by median speed")
        legend_right(ax)
        save(fig, out / "figures", "by_turning_and_speed",
             "Macro-F1 on windows with/without a measurable turn, and by speed tercile (95% CIs).", figures)

    tsne_figures(R, f, out, figures)


def tsne_2d(x):
    return TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=0).fit_transform(x)


def tsne_figures(R, f, out, figures):
    sample = f[embedding_sample(f)].reset_index(drop=True)
    for pair, held in ((("vehicle", "bus"), "bus"), (("cyclist", "motorcyclist"), "motorcyclist")):
        panels = [("speed/accel statistics", robust_space(sample[BASELINE].to_numpy(float), sample[BASELINE].to_numpy(float)), sample.cls),
                  ("physics features", robust_space(sample[PHYSICS].to_numpy(float), sample[PHYSICS].to_numpy(float)), sample.cls)]
        for m, cfg, lab in (("gru", "main", "GRU embedding"), ("gru", f"class:{held}", f"GRU embedding, {held} never trained")):
            p = Path(R.get(m, cfg)["_stem"] + "_emb.npz") if (m, cfg) in R.runs else None
            if p and p.exists():
                z = np.load(p, allow_pickle=True)
                panels.append((lab, z["emb"].astype(np.float32), pd.Series(z["cls"])))
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
        for a_, (title, x, cls) in zip(axes[0], panels):
            xy = tsne_2d(x)
            cls = cls.to_numpy()
            other = ~np.isin(cls, pair)
            a_.scatter(xy[other, 0], xy[other, 1], s=4, color=OTHER, label="other classes", rasterized=True)
            for c, col in zip(pair, HIGHLIGHT):
                mk = cls == c
                a_.scatter(xy[mk, 0], xy[mk, 1], s=8, color=col, edgecolor="white", linewidth=0.3, label=c)
            a_.set_title(title)
            a_.set_xticks([])
            a_.set_yticks([])
            a_.grid(False)
            for sp in a_.spines.values():
                sp.set_visible(False)
        axes[0][-1].legend(fontsize=8, loc="upper right", markerscale=2)
        fig.tight_layout()
        save(fig, out / "figures", f"tsne_{pair[1]}",
             f"t-SNE of {pair[0]} vs {pair[1]} windows in each representation (up to 1,000 MF-val windows per class); "
             f"the last panel comes from a model that never saw {held}.", figures)


def training_figures(R, out, figures):
    ms = [m for m in ("gru", "gru_phys", "gru_smooth") if (m, "main") in R.runs and R.get(m, "main").get("history")]
    if ms:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for m in ms:
            h = pd.DataFrame(R.get(m, "main")["history"])
            ax.plot(h.epoch, h.es_loss, color=COLOR[m], label=NAME[m])
        ax.set_xlabel("epoch")
        ax.set_ylabel("class-weighted loss, held-out 5%")
        ax.set_title("Sequence-model training (early stopping on this loss)")
        ax.legend(fontsize=8)
        save(fig, out / "figures", "training_curves", "Early-stopping loss per epoch for the sequence models.", figures)
    ms = [m for m in MODELS if (m, "main") in R.runs and R.get(m, "main").get("alpha_curve")]
    if ms:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for m in ms:
            r = R.get(m, "main")
            a = np.linspace(0, 1, len(r["alpha_curve"]))
            ax.plot(a, r["alpha_curve"], color=COLOR[m], label=NAME[m])
            ax.plot(r["alpha"], max(r["alpha_curve"]), "o", color=COLOR[m], ms=8)
        ax.set_xlabel("α  (0 = balanced argmax, 1 = Bayes rule for accuracy)")
        ax.set_ylabel("macro-F1, calibration windows")
        ax.set_title("Decision rule: p × prior^α, α picked per model (dot)")
        ax.legend(fontsize=7)
        save(fig, out / "figures", "alpha_curves", "How the prior-correction strength α was chosen.", figures)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--data", type=Path, default=DATA)
    a = ap.parse_args()
    R = Runs(a.results)
    out = R.dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    tables, figures, failed = {}, [], []
    classes = CLASSES

    def step(name, fn, *args):
        try:
            return fn(*args)
        except Exception:     # keep going: one broken figure should not cost the others
            traceback.print_exc()
            failed.append(name)

    res = step("main", main_analysis, R, out, tables, figures)
    if res:
        classes = res[0]
    step("city", city_analysis, R, out, tables, figures)
    step("novelty", novelty_analysis, R, out, tables, figures)
    step("training", training_figures, R, out, figures)
    step("interpretation", interpretation, R, out, a.data, classes, figures)
    print(f"{len(figures)} figures, {len(tables)} tables -> {out}; failed: {failed or 'none'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
