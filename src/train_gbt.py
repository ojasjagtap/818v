"""Gradient-boosted trees on baseline (speed/acceleration statistics) or physics features.

    python src/train_gbt.py --features base phys phys_clean --configs main class:bus --frac 0.1
Same hyperparameters for every feature set, so only the features differ. phys_clean drops the
NOISE_SENSITIVE physics features (M1: their bus signal is tracker noise).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from common import DATA, RESULTS, SEED
from experiment import (DEFAULT_CONFIGS, FEATURE_SETS, classification_metrics, correct, fit_alpha, knn_score,
                        labels, load, novelty_metrics, prediction_frame, reference_rows, robust_space, save_run,
                        split)


def matrix(frame, cols):
    x = frame[cols].to_numpy(np.float64, copy=True)   # pandas copy-on-write views are read-only
    x[~np.isfinite(x)] = np.nan
    return x


def run(df, features, config, results):
    t0 = time.time()
    cols = FEATURE_SETS[features]
    train, tests, classes, prior, calib = split(df, config)
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=20,
                                         class_weight="balanced", random_state=SEED)
    xtr = matrix(train, cols)
    clf.fit(xtr, labels(train, classes))
    alpha, alpha_curve = fit_alpha(calib, clf.predict_proba(matrix(calib, cols)), classes, prior)
    result = {"train_n": len(train), "classes": classes, "prior": dict(zip(classes, prior.round(5).tolist())),
              "alpha": alpha, "alpha_curve": alpha_curve, "n_iter": int(clf.n_iter_)}
    preds = []
    for name, test in tests.items():
        x = matrix(test, cols)
        proba = correct(clf.predict_proba(x), prior, alpha)
        result[name] = classification_metrics(test, proba, classes)
        if config == "main":   # which features carry the decision: macro-F1 lost when each is shuffled
            score = lambda est, xx, yy: f1_score(yy, correct(est.predict_proba(xx), prior, alpha).argmax(1),  # noqa: E731
                                                 average="macro", labels=range(len(classes)), zero_division=0)
            imp = permutation_importance(clf, x, labels(test, classes), scoring=score, n_repeats=3,
                                         random_state=SEED)
            result["importance"] = dict(zip(cols, imp.importances_mean.round(5).tolist()))
        score = None
        if config.startswith("class:"):
            ref = robust_space(xtr, xtr[reference_rows(len(xtr))])
            score = knn_score(ref, robust_space(xtr, x))
            result["novelty"] = novelty_metrics(test, score, proba, classes, config.split(":")[1])
        preds.append(prediction_frame(name, test, proba, classes, score))
    result["seconds"] = round(time.time() - t0)
    save_run(f"gbt_{features}", config, result, pd.concat(preds, ignore_index=True), results)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args()
    df = load(a.frac, a.data)
    for config in a.configs:
        for features in a.features:
            run(df, features, config, a.results)


if __name__ == "__main__":
    main()
