"""AV2 Sensor annotations -> city-frame tracks and 5-s windows, labelled with the five MF classes.

Box sizes are kept in the output only to validate the motion-derived wheelbase; no model sees them.
    python src/sensor_tracks.py            # all labelled logs under data/sensor/{train,val}
writes data/sensor/windows.parquet (meta + physics/baseline features, both splits pooled: RQ2 uses
cross-validation grouped by log) and data/sensor/windows.npy (same layout as extract.py).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, MIN_PATH_M, SENSOR_TO_CLASS, WINDOW, path_length, window_start

POSE_TOL_NS = 10_000_000   # annotation <-> ego-pose timestamp matching tolerance (10 ms)
GAP_NS = 150_000_000       # consecutive annotations further apart than this break a track


def quat_to_mat(qw, qx, qy, qz):
    """Unit quaternions (arrays) -> rotation matrices [N, 3, 3]."""
    qw, qx, qy, qz = (np.asarray(q, float) for q in (qw, qx, qy, qz))
    return np.stack([
        np.stack([1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)], -1),
        np.stack([2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)], -1),
        np.stack([2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)], -1),
    ], -2)


def log_tracks(log_dir):
    """Every annotated object of one log in the city frame.

    Columns: track_uuid, category, timestamp_ns, x, y, h (yaw in the city frame), length_m.
    """
    ann = pd.read_feather(Path(log_dir) / "annotations.feather").sort_values("timestamp_ns")
    pose = pd.read_feather(Path(log_dir) / "city_SE3_egovehicle.feather").sort_values("timestamp_ns")
    pose = pose.rename(columns={c: f"e_{c}" for c in pose.columns if c != "timestamp_ns"})
    a = pd.merge_asof(ann, pose, on="timestamp_ns", direction="nearest", tolerance=POSE_TOL_NS)
    a = a.dropna(subset=["e_qw"])
    r_ego = quat_to_mat(a.e_qw, a.e_qx, a.e_qy, a.e_qz)          # city <- ego
    r_obj = quat_to_mat(a.qw, a.qx, a.qy, a.qz)                  # ego <- object
    t_obj = a[["tx_m", "ty_m", "tz_m"]].to_numpy()
    center = np.einsum("nij,nj->ni", r_ego, t_obj) + a[["e_tx_m", "e_ty_m", "e_tz_m"]].to_numpy()
    rot = r_ego @ r_obj
    out = pd.DataFrame(dict(track_uuid=a.track_uuid.to_numpy(), category=a.category.to_numpy(),
                            timestamp_ns=a.timestamp_ns.to_numpy(), x=center[:, 0], y=center[:, 1],
                            h=np.arctan2(rot[:, 1, 0], rot[:, 0, 0]), length_m=a.length_m.to_numpy()))
    return out.sort_values(["track_uuid", "timestamp_ns"], ignore_index=True), len(ann) - len(a)


def longest_run(ts):
    """(start, stop) indices of the longest gap-free stretch of a sorted timestamp array."""
    cuts = np.flatnonzero(np.diff(ts) > GAP_NS) + 1
    bounds = np.concatenate([[0], cuts, [len(ts)]])
    k = np.argmax(np.diff(bounds))
    return bounds[k], bounds[k + 1]


def windows_from_log(tracks, log_id, categories=None):
    """One window per track: random start inside the longest gap-free run, moving tracks only.

    By default keeps the categories mapped to the five classes; pass `categories` to override.
    """
    keep = set(categories or SENSOR_TO_CLASS)
    meta, wins = [], []
    for tid, g in tracks[tracks.category.isin(keep)].groupby("track_uuid", sort=False):
        i0, i1 = longest_run(g.timestamp_ns.to_numpy())
        if i1 - i0 < WINDOW:
            continue
        a = g[["x", "y", "h"]].to_numpy()[i0:i1]
        s0 = window_start(log_id, tid, len(a))
        w = a[s0:s0 + WINDOW]
        path = path_length(w[:, :2])
        if path < MIN_PATH_M:
            continue
        cat = g.category.iat[0]
        meta.append(dict(log_id=log_id, track_id=tid, category=cat, cls=SENSOR_TO_CLASS.get(cat),
                         track_len=i1 - i0, start=s0, path_m=path, x0=w[0, 0], y0=w[0, 1],
                         t0_ns=int(g.timestamp_ns.iat[i0 + s0]), length_m=float(g.length_m.median())))
        wins.append(np.column_stack([w[:, 0] - w[0, 0], w[:, 1] - w[0, 1], w[:, 2]]))
    return pd.DataFrame(meta), np.asarray(wins, np.float32).reshape(-1, WINDOW, 3)


def main():
    from kinematics import all_features

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=DATA / "sensor")
    a = ap.parse_args()
    metas, arrs, dropped = [], [], 0
    for split in ("train", "val"):
        for log in sorted(p for p in (a.src / split).iterdir() if (p / "annotations.feather").exists()):
            tracks, n_drop = log_tracks(log)
            dropped += n_drop
            m, w = windows_from_log(tracks, log.name)
            metas.append(m.assign(split=split))
            arrs.append(w)
    meta = pd.concat(metas, ignore_index=True)
    xyh = np.concatenate(arrs)
    np.save(a.src / "windows.npy", xyh)
    pd.concat([meta, all_features(xyh, 0.0)], axis=1).to_parquet(a.src / "windows.parquet", index=False)
    print(f"{meta.log_id.nunique()} logs -> {len(meta)} windows {meta.cls.value_counts().to_dict()} "
          f"({dropped} annotations without an ego pose)", flush=True)


if __name__ == "__main__":
    main()
