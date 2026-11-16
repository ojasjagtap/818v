"""Late fusion of two trained RQ1 models: average their prior-corrected class probabilities.

    python src/fuse.py [--results results] [--models gru gbt_phys]
For every config where both runs exist, writes runs/fusion__<config>.json/.parquet in the same format
as a trained model, so `experiment.py collect` and `analyze.py` treat it as one more model. The
novelty score is the mean of the two models' within-test percentile ranks (their kNN distances live
in different feature spaces). No training: this asks whether physics features add anything the
GRU lacks, without the concatenated model's optimisation problem.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from common import RESULTS
from experiment import classification_metrics, novelty_metrics, save_run

KEY = ["test", "scenario_id", "track_id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--models", nargs=2, default=["gru", "gbt_phys"])
    a = ap.parse_args()
    runs = Path(a.results) / "rq1" / "runs"
    m1, m2 = a.models
    for j1 in sorted(runs.glob(f"{m1}__*.json")):
        j2 = runs / j1.name.replace(f"{m1}__", f"{m2}__", 1)
        if not j2.exists():
            continue
        r1, r2 = json.loads(j1.read_text()), json.loads(j2.read_text())
        classes = r1["classes"]
        assert classes == r2["classes"], (classes, r2["classes"])
        p1, p2 = pd.read_parquet(j1.with_suffix(".parquet")), pd.read_parquet(j2.with_suffix(".parquet"))
        assert p1[KEY].equals(p2[KEY]), f"row mismatch in {j1.stem}"
        cols = [f"p_{c}" for c in classes]
        out = p1.drop(columns=cols + ["novelty_score"], errors="ignore")
        proba = (p1[cols].to_numpy() + p2[cols].to_numpy()) / 2
        out[cols] = proba
        config = r1["config"]
        result = {"classes": classes, "fused": [m1, m2]}
        if "novelty_score" in p1 and "novelty_score" in p2:
            out["novelty_score"] = (p1.groupby("test").novelty_score.rank(pct=True)
                                    + p2.groupby("test").novelty_score.rank(pct=True)) / 2
        for test in out.test.unique():
            m = (out.test == test).to_numpy()
            frame = out[m]
            result[test] = classification_metrics(frame, proba[m], classes)
            if config.startswith("class:"):
                result["novelty"] = novelty_metrics(frame, frame.novelty_score.to_numpy(), proba[m], classes,
                                                    config.split(":")[1])
        save_run("fusion", config, result, out, a.results)


if __name__ == "__main__":
    main()
