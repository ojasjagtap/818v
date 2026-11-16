"""MF scenarios -> one 5-s window per moving track of the five dynamic classes.

Slurm array task k handles every n-th scenario of train and of val:
    python src/extract.py --shard $SLURM_ARRAY_TASK_ID --n-shards 50
and writes data/windows/shards/{split}_{k:03d}.parquet (one row per window) plus
{split}_{k:03d}.npy (float32 [N, WINDOW, 3]: x - x0, y - y0, heading). Positions are stored
relative to the window's first point so float32 keeps sub-millimetre precision; x0, y0 are in
the parquet. Features are computed later by aggregate.py, after the global heading lag is known.
The provided velocity columns are deliberately not read (see results/m1/report.md).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import CLASSES, DATA, MIN_PATH_M, WINDOW, path_length, window_start

COLS = ["track_id", "object_type", "object_category", "timestep",
        "position_x", "position_y", "heading", "scenario_id", "city"]


def scenario_paths(split_dir):
    """Scenario parquet paths, in a fixed order, without listing 200k subdirectories."""
    ids = sorted(e for e in os.listdir(split_dir) if not e.startswith("."))
    return [Path(split_dir) / i / f"scenario_{i}.parquet" for i in ids]


def windows_from_frame(df):
    """Rows of one or more scenarios -> (meta DataFrame, float32 [N, WINDOW, 3] windows)."""
    df = df[df.object_type.isin(CLASSES) & (df.track_id != "AV")]
    df = df.sort_values(["scenario_id", "track_id", "timestep"])
    meta, wins = [], []
    for (sid, tid), g in df.groupby(["scenario_id", "track_id"], sort=False):
        n = len(g)
        ts = g.timestep.to_numpy()
        if n < WINDOW or ts[-1] - ts[0] + 1 != n:   # too short, or not gap-free
            continue
        a = g[["position_x", "position_y", "heading"]].to_numpy()
        s0 = window_start(sid, tid, n)
        w = a[s0:s0 + WINDOW]
        path = path_length(w[:, :2])
        if path < MIN_PATH_M:
            continue
        x0, y0 = w[0, 0], w[0, 1]
        meta.append(dict(scenario_id=sid, track_id=tid, cls=g.object_type.iat[0],
                         category=int(g.object_category.iat[0]), city=g.city.iat[0],
                         track_len=n, start=s0, path_m=path, x0=x0, y0=y0))
        wins.append(np.column_stack([w[:, 0] - x0, w[:, 1] - y0, w[:, 2]]))
    xyh = np.asarray(wins, dtype=np.float32).reshape(-1, WINDOW, 3)
    return pd.DataFrame(meta), xyh


def run_shard(split, shard, n_shards, src, out, limit=0):
    paths = scenario_paths(src / split)[shard::n_shards]
    if limit:
        paths = paths[:limit]
    metas, arrs, missing = [], [], 0
    for p in paths:
        try:
            df = pd.read_parquet(p, columns=COLS)
        except FileNotFoundError:
            missing += 1
            continue
        m, w = windows_from_frame(df)
        metas.append(m)
        arrs.append(w)
    if missing > max(1, 0.001 * len(paths)):
        sys.exit(f"{split} shard {shard}: {missing}/{len(paths)} scenario files missing")
    meta = pd.concat(metas, ignore_index=True)
    xyh = np.concatenate(arrs) if arrs else np.zeros((0, WINDOW, 3), np.float32)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{split}_{shard:03d}"
    np.save(f"{stem}.npy", xyh)
    meta.to_parquet(f"{stem}.parquet", index=False)   # written last: marks the shard complete
    return len(paths), missing, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-shards", type=int, default=50)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--src", type=Path, default=DATA / "mf")
    ap.add_argument("--out", type=Path, default=DATA / "windows" / "shards")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N scenarios of the shard")
    a = ap.parse_args()
    for split in a.splits:
        t0 = time.time()
        n, missing, meta = run_shard(split, a.shard, a.n_shards, a.src, a.out, a.limit)
        counts = meta.cls.value_counts().to_dict() if len(meta) else {}
        print(f"{split} shard {a.shard}: {n} scenarios ({missing} missing) -> {len(meta)} windows "
              f"{counts} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
