"""Merge extract.py shards and compute features (Slurm job that runs after the extract array).

Exits 1 if any shard is missing, so the job's FAIL email is the alert. Otherwise writes
  data/windows/{split}.parquet + .npy    merged windows (meta + float32 [N, WINDOW, 3])
  data/feats/{split}.parquet             meta + physics + baseline features, one row per window
  results/m2_counts.csv, results/m2_nan_rates.csv
No heading-lag correction is applied (M1: the within-window lag on MF is ~0.02 s).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import CLASSES, CITIES, DATA, RESULTS
from kinematics import all_features


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-shards", type=int, default=50)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args()
    data, results = a.data, a.results
    a.shards = data / "windows" / "shards"

    stems = {s: [a.shards / f"{s}_{k:03d}" for k in range(a.n_shards)] for s in a.splits}
    missing = [f"{p}{ext}" for s in a.splits for p in stems[s] for ext in (".parquet", ".npy")
               if not Path(f"{p}{ext}").exists()]
    if missing:
        print(f"{len(missing)} shard files missing, e.g. {missing[:5]}", flush=True)
        sys.exit(1)

    (data / "windows").mkdir(exist_ok=True)
    (data / "feats").mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    counts, nans = [], []
    for split in a.splits:
        t0 = time.time()
        meta = pd.concat([pd.read_parquet(f"{p}.parquet") for p in stems[split]], ignore_index=True)
        xyh = np.concatenate([np.load(f"{p}.npy") for p in stems[split]])
        assert len(meta) == len(xyh), (len(meta), len(xyh))
        meta["split"] = split
        meta.to_parquet(data / "windows" / f"{split}.parquet", index=False)
        np.save(data / "windows" / f"{split}.npy", xyh)
        feats = all_features(xyh, 0.0)
        pd.concat([meta, feats], axis=1).to_parquet(data / "feats" / f"{split}.parquet", index=False)
        c = pd.crosstab(meta.city, meta.cls).reindex(index=CITIES, columns=CLASSES).fillna(0).astype(int)
        c.loc["total"] = c.sum()
        counts.append(c.assign(split=split))
        nans.append(feats.isna().groupby(meta.cls.to_numpy()).mean().T.assign(split=split))
        print(f"{split}: {len(meta)} windows {meta.cls.value_counts().to_dict()} "
              f"in {time.time() - t0:.0f}s", flush=True)
    pd.concat(counts).to_csv(results / "m2_counts.csv")
    pd.concat(nans).to_csv(results / "m2_nan_rates.csv")


if __name__ == "__main__":
    main()
