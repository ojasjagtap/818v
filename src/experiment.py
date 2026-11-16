"""RQ1 experiment definitions shared by train_gbt.py and train_seq.py: splits, metrics, results.

Configs:
  main            train on MF train, test on MF val
  city:<name>     train without <name>; test on val of the other cities (id) and on every
                  window of <name> from train and val (ood)
  class:<name>    train without <name>; test on val; AUROC for flagging <name> as novel
Each run writes results/rq1/runs/<model>__<config>.json (metrics) and .parquet (predictions);
    python src/experiment.py collect [--expect N]
merges them into results/rq1/{main,city,novelty}.csv (exit 1 if fewer than N runs exist).
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from common import CITIES, CLASSES, DATA, NOISE_SENSITIVE, RESULTS, SEED, WINDOW
from kinematics import baseline_features, physics_features

CAP_PER_CLASS = 200_000      # training windows per class (vehicle ~1.6M otherwise dominates)
CALIB_MOD = 20               # 1 in 20 training scenarios held out to set the decision rule
ALPHAS = np.linspace(0, 1, 11)
KNN_K, KNN_REF = 10, 50_000  # novelty score: mean distance to the 10 nearest of 50k training windows
HARD_PAIRS = [("cyclist", "motorcyclist"), ("motorcyclist", "cyclist"), ("bus", "vehicle"), ("vehicle", "bus")]
# every city held out once (leave-one-city-out), every non-vehicle class held out once
DEFAULT_CONFIGS = (["main"] + [f"city:{c}" for c in CITIES]
                   + [f"class:{c}" for c in ("bus", "motorcyclist", "cyclist", "pedestrian")])
EMB_PER_CLASS = 1000         # windows per class kept for 2-D embedding plots


def _feature_names():
    t = np.cumsum(np.ones((1, WINDOW)), axis=1)
    return (list(physics_features(t, 0 * t, 0 * t).columns), list(baseline_features(t, 0 * t).columns))


PHYSICS, BASELINE = _feature_names()
FEATURE_SETS = {"base": BASELINE, "phys": PHYSICS,
                "phys_clean": [f for f in PHYSICS if f not in NOISE_SENSITIVE]}


def load(frac=1.0, data=DATA):
    """Features + meta of train and val; `row` indexes data/windows/{split}.npy.
    frac < 1 keeps a fixed pseudo-random subset of scenarios (smallest-viable runs)."""
    parts = []
    for split in ("train", "val"):
        df = pd.read_parquet(Path(data) / "feats" / f"{split}.parquet")
        df["row"] = np.arange(len(df))
        if frac < 1:
            keep = df.scenario_id.map(lambda s: zlib.crc32(s.encode()) % 10_000 < frac * 10_000)
            df = df[keep.to_numpy()]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def split(df, config, cap=CAP_PER_CLASS, seed=SEED):
    """(train, {test name: frame}, trained classes, natural class prior, calibration frame).

    One in CALIB_MOD training scenarios is held out uncapped, keeping the natural class mix, to
    set each model's decision rule (fit_alpha). The prior is the class mix of the remaining
    training split before capping."""
    kind, _, name = config.partition(":")
    train, val = df[df.split == "train"], df[df.split == "val"]
    if kind == "main":
        tests = {"val": val}
    elif kind == "city":
        train = train[train.city != name]
        tests = {"id": val[val.city != name], "ood": df[df.city == name]}
    elif kind == "class":
        train = train[train.cls != name]
        tests = {"val": val}
    else:
        raise ValueError(f"unknown config {config}")
    in_calib = train.scenario_id.map(lambda s: zlib.crc32(f"calib/{s}".encode()) % CALIB_MOD == 0).to_numpy()
    calib, train = train[in_calib], train[~in_calib]
    classes = [c for c in CLASSES if c in set(train.cls)]
    prior = train.cls.value_counts(normalize=True).reindex(classes).to_numpy()
    keep = [g.sample(min(len(g), cap), random_state=seed).index for _, g in train.groupby("cls")]
    train = train.loc[np.concatenate(keep)].sample(frac=1, random_state=seed)
    return train, tests, classes, prior, calib


def correct(proba, prior, alpha=1.0):
    """Shift balanced-trained probabilities toward the natural class mix: p * prior**alpha.

    Every model trains with balanced class weights (a uniform effective prior). alpha = 0 keeps
    that (rare classes over-predicted on natural data); alpha = 1 is the Bayes rule for accuracy
    (rare classes under-predicted, which hurts macro-F1)."""
    p = proba * np.asarray(prior)[None, :] ** alpha
    return p / p.sum(1, keepdims=True)


def fit_alpha(calib, proba, classes, prior):
    """The correction strength that maximises macro-F1 on the held-out calibration windows, and
    the macro-F1 at every alpha in ALPHAS. Same procedure for every model."""
    y = labels(calib, classes)
    f1 = [f1_score(y, correct(proba, prior, a).argmax(1), average="macro", labels=range(len(classes)),
                   zero_division=0) for a in ALPHAS]
    return round(float(ALPHAS[int(np.argmax(f1))]), 2), [round(float(v), 4) for v in f1]


def embedding_sample(frame, per_class=EMB_PER_CLASS):
    """Mask of a fixed, stratified subset of windows (the same rows in every run) for 2-D plots."""
    key = (frame.scenario_id + "/" + frame.track_id).map(lambda s: zlib.crc32(s.encode()))
    return (key.groupby(frame.cls).rank(method="first") <= per_class).to_numpy()


def labels(frame, classes):
    return frame.cls.map({c: i for i, c in enumerate(classes)}).to_numpy(copy=True)


def classification_metrics(test, proba, classes):
    """Accuracy, macro-F1 and hard-pair confusion over test windows of the trained classes."""
    known = test.cls.isin(classes).to_numpy()
    y, pred = labels(test[known], classes), proba[known].argmax(1)
    out = {"n": int(known.sum()), "accuracy": float((pred == y).mean()),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=range(len(classes)), zero_division=0))}
    for a, b in HARD_PAIRS:
        if a in classes and b in classes and (y == classes.index(a)).any():
            out[f"{a}_as_{b}"] = float((pred[y == classes.index(a)] == classes.index(b)).mean())
    return out


def novelty_metrics(test, score, proba, classes, held_out):
    """AUROC of the novelty score for the held-out class vs the trained classes, and where the
    classifier puts the held-out windows."""
    y = (test.cls == held_out).to_numpy()
    out = {"novelty_auroc": float(roc_auc_score(y, score)), "n_novel": int(y.sum())}
    pred = proba[y].argmax(1)
    for i, c in enumerate(classes):
        out[f"novel_as_{c}"] = float((pred == i).mean())
    return out


def robust_space(train_x, x):
    """Median-impute, robust-scale and clip features for nearest-neighbour distances."""
    train_x = np.where(np.isfinite(train_x), train_x, np.nan)
    x = np.where(np.isfinite(x), x, np.nan)
    med = np.nanmedian(train_x, 0)
    iqr = np.nanpercentile(train_x, 75, 0) - np.nanpercentile(train_x, 25, 0)
    iqr[~(iqr > 0)] = 1.0
    return np.clip((np.where(np.isnan(x), med, x) - med) / iqr, -10, 10).astype(np.float32)


def knn_score(ref, query, k=KNN_K, device="cpu", chunk=4096):
    """Mean Euclidean distance from each query row to its k nearest reference rows."""
    import torch
    r = torch.as_tensor(np.ascontiguousarray(ref), dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(query), chunk):
        q = torch.as_tensor(np.ascontiguousarray(query[i:i + chunk]), dtype=torch.float32, device=device)
        out.append(torch.cdist(q, r).topk(k, largest=False).values.mean(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def reference_rows(n, seed=SEED):
    return np.random.default_rng(seed).choice(n, size=min(n, KNN_REF), replace=False)


def save_run(model, config, result, preds, results=RESULTS):
    out = Path(results) / "rq1" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{model}__{config.replace(':', '-')}"
    Path(f"{stem}.json").write_text(json.dumps(dict(model=model, config=config, **result), indent=1))
    preds.to_parquet(f"{stem}.parquet", index=False)
    print(f"{model} {config}: {json.dumps(result)}", flush=True)


def prediction_frame(test_name, test, proba, classes, score=None):
    p = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
    p.insert(0, "test", test_name)
    for col in ("scenario_id", "track_id", "cls", "city"):
        p.insert(1, col, test[col].to_numpy())
    if score is not None:
        p["novelty_score"] = score
    return p


def collect(results=RESULTS, expect=0):
    runs = [json.loads(p.read_text()) for p in sorted((Path(results) / "rq1" / "runs").glob("*.json"))]
    if len(runs) < expect:
        print(f"only {len(runs)} of {expect} runs finished", flush=True)
        sys.exit(1)
    main, city, novelty = [], [], []
    for r in runs:
        kind, _, name = r["config"].partition(":")
        if kind == "main":
            main.append(dict(model=r["model"], alpha=r.get("alpha"), **r["val"]))
        elif kind == "city":
            city.append(dict(model=r["model"], held_out_city=name, alpha=r.get("alpha"),
                             id_accuracy=r["id"]["accuracy"], ood_accuracy=r["ood"]["accuracy"],
                             accuracy_drop=r["id"]["accuracy"] - r["ood"]["accuracy"],
                             id_macro_f1=r["id"]["macro_f1"], ood_macro_f1=r["ood"]["macro_f1"],
                             macro_f1_drop=r["id"]["macro_f1"] - r["ood"]["macro_f1"]))
        elif kind == "class":
            novelty.append(dict(model=r["model"], held_out_class=name, alpha=r.get("alpha"), **r["novelty"],
                                known_macro_f1=r["val"]["macro_f1"]))
    out = Path(results) / "rq1"
    for name, rows in (("main", main), ("city", city), ("novelty", novelty)):
        if rows:
            pd.DataFrame(rows).sort_values(list(rows[0])[:2] if name != "main" else "model") \
              .to_csv(out / f"{name}.csv", index=False, float_format="%.4f")
            print(f"\n== {name}\n" + pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["collect"])
    ap.add_argument("--expect", type=int, default=0)
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args()
    collect(a.results, a.expect)
