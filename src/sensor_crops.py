"""Project AV2 Sensor cuboids into the ring cameras and cut object crops (RQ2).

    python src/sensor_crops.py --shard $SLURM_ARRAY_TASK_ID --n-shards 20   # every 20th log
    python src/sensor_crops.py --merge --n-shards 20                        # verify, merge, gallery
For each RQ2 window (data/sensor/windows.parquet): up to MAX_CROPS crops from the ~1 Hz ring-camera
images inside the window's 5 s, taking at each image time the camera with the largest usable box.
Riders are boxed together with their bicycle or motorcycle; candidates more than 60% covered by
closer annotated objects are skipped. Crops are square (box side x PAD),
128x128 JPEG, under data/sensor/crops/<log>/, one CSV row per crop.
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.geometry import quat_to_mat
from av2.geometry.se3 import SE3
from av2.structures.cuboid import Cuboid, CuboidList
from av2.utils.io import read_city_SE3_ego
from PIL import Image

from common import CLASSES, DATA, RESULTS, RING_CAMERAS

MIN_BOX_PX = 32      # minimum box height in the image [px]
MIN_VISIBLE = 0.5    # fraction of the projected box that must lie inside the image
MIN_DEPTH_M = 0.5    # every cuboid corner must be at least this far in front of the camera
CROP_PX, PAD, MAX_CROPS = 128, 1.15, 5
MIN_GAP_NS = 500_000_000         # crops of one track at least 0.5 s apart
WINDOW_NS = 5_000_000_000        # a window spans 5 s from its first sweep
BIKE = {"cyclist": "BICYCLE", "motorcyclist": "MOTORCYCLE"}
RIDER_RADIUS_M = 1.5             # a rider's bike is the nearest bike cuboid within this distance
MAX_OCCLUDED = 0.6               # reject a crop if closer annotated objects cover more of its box


def nearest(sorted_ts, t):
    """Element of sorted_ts closest to t."""
    i = int(np.clip(np.searchsorted(sorted_ts, t), 1, len(sorted_ts) - 1))
    return sorted_ts[i - 1] if t - sorted_ts[i - 1] <= sorted_ts[i] - t else sorted_ts[i]


def boxes_in_image(cuboids, cam, city_SE3_ego_cam, city_SE3_ego_lidar):
    """Image boxes [N, 4] (u0, v0, u1, v1; clipped to the image) of cuboids annotated at lidar
    time, projected with ego-motion compensation, plus a mask of boxes usable as crops."""
    verts = cuboids.vertices_m                                   # [N, 8, 3] in the ego frame
    n = len(verts)
    uv, pts_cam, _ = cam.project_ego_to_img_motion_compensated(
        verts.reshape(-1, 3), city_SE3_ego_cam, city_SE3_ego_lidar)
    uv, depth = uv.reshape(n, 8, 2), pts_cam[:, 2].reshape(n, 8)
    lo, hi = uv.min(1), uv.max(1)
    size = np.array([cam.width_px, cam.height_px], float)
    clo, chi = np.clip(lo, 0, size), np.clip(hi, 0, size)
    with np.errstate(invalid="ignore", divide="ignore"):
        visible = np.prod(chi - clo, 1) / np.prod(hi - lo, 1)
    ok = (depth > MIN_DEPTH_M).all(1) & (visible >= MIN_VISIBLE) & (chi[:, 1] - clo[:, 1] >= MIN_BOX_PX)
    return np.concatenate([clo, chi], 1), ok


def cuboid(row):
    rot = quat_to_mat(np.array([[row.qw, row.qx, row.qy, row.qz]]))[0]
    return Cuboid(dst_SE3_object=SE3(rotation=rot, translation=np.array([row.tx_m, row.ty_m, row.tz_m])),
                  length_m=row.length_m, width_m=row.width_m, height_m=row.height_m,
                  category=row.category, timestamp_ns=int(row.timestamp_ns))


class LogProjector:
    """Cameras, ego poses, annotations and downloaded image times of one log, loaded once."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.poses = read_city_SE3_ego(self.log_dir)
        self.pose_ts = np.array(sorted(self.poses))
        self.ann = pd.read_feather(self.log_dir / "annotations.feather")
        self.lidar_ts = np.sort(self.ann.timestamp_ns.unique())
        self.sweeps = {k: g.reset_index(drop=True) for k, g in self.ann.groupby("timestamp_ns")}
        cam_dir = self.log_dir / "sensors" / "cameras"
        self.images = {c: np.array(sorted(int(p.stem) for p in (cam_dir / c).glob("*.jpg")), dtype=np.int64)
                       for c in RING_CAMERAS if (cam_dir / c).is_dir()}
        self.cams, self.cache = {}, {}

    def camera(self, name):
        if name not in self.cams:
            self.cams[name] = PinholeCamera.from_feather(self.log_dir, name)
        return self.cams[name]

    def frame(self, cam_name, cam_ts):
        """Every cuboid of the sweep nearest cam_ts projected into that image (cached):
        (rows, boxes [N, 4], depth [N], in front of the camera [N], usable as a crop [N])."""
        key = (cam_name, cam_ts)
        if key not in self.cache:
            lidar_ts = nearest(self.lidar_ts, cam_ts)
            rows = self.sweeps[lidar_ts]
            cubs = CuboidList([cuboid(r) for r in rows.itertuples()])
            cam = self.camera(cam_name)
            pose_cam = self.poses[nearest(self.pose_ts, cam_ts)]
            pose_lidar = self.poses[nearest(self.pose_ts, lidar_ts)]
            boxes, ok = boxes_in_image(cubs, cam, pose_cam, pose_lidar)
            _, pts_cam, _ = cam.project_ego_to_img_motion_compensated(cubs.vertices_m.reshape(-1, 3), pose_cam,
                                                                    pose_lidar)
            z = pts_cam[:, 2].reshape(-1, 8)
            self.cache[key] = (rows, boxes, np.median(z, 1), (z > MIN_DEPTH_M).all(1), ok)
        return self.cache[key]

    def track_box(self, track_id, cls, cam_name, cam_ts):
        """Usable box of one track (with its bike, for riders) in one image, or None when the
        track is not annotated there, is too small or cut off, or hidden behind closer objects."""
        rows, boxes, depth, front, ok = self.frame(cam_name, cam_ts)
        hit = np.flatnonzero(rows.track_uuid.to_numpy() == track_id)
        if not len(hit) or not ok[hit[0]]:
            return None
        i = hit[0]
        mine = [i]
        if cls in BIKE:
            bikes = np.flatnonzero(rows.category.to_numpy() == BIKE[cls])
            if len(bikes):
                d = np.hypot(rows.tx_m.to_numpy()[bikes] - rows.tx_m.iat[i], rows.ty_m.to_numpy()[bikes] - rows.ty_m.iat[i])
                if d.min() < RIDER_RADIUS_M:
                    mine.append(bikes[int(np.argmin(d))])
        box = np.concatenate([boxes[[m for m in mine if ok[m] or m == i], :2].min(0),
                              boxes[[m for m in mine if ok[m] or m == i], 2:].max(0)])
        closer = front & (depth < depth[i] - 0.5)
        closer[mine] = False
        return None if occluded_fraction(box, boxes[closer]) > MAX_OCCLUDED else box

    def boxes(self, cam_name, cam_ts, categories):
        """All cuboids of `categories` from the sweep nearest cam_ts, projected into that image
        (projection overlay check). Returns (cuboids, boxes [N, 4], usable mask)."""
        rows, boxes, _, _, ok = self.frame(cam_name, cam_ts)
        keep = rows.category.isin(list(categories)).to_numpy()
        return [cuboid(r) for r in rows[keep].itertuples()], boxes[keep], ok[keep]


def occluded_fraction(box, others, grid=24):
    """Fraction of `box` covered by any of the `others` boxes (sampled on a grid x grid lattice)."""
    if not len(others):
        return 0.0
    u = np.linspace(box[0], box[2], grid)
    v = np.linspace(box[1], box[3], grid)
    uu, vv = np.meshgrid(u, v)
    pts = np.stack([uu.ravel(), vv.ravel()], 1)
    covered = np.zeros(len(pts), bool)
    for o in others:
        covered |= (pts[:, 0] >= o[0]) & (pts[:, 0] <= o[2]) & (pts[:, 1] >= o[1]) & (pts[:, 1] <= o[3])
    return float(covered.mean())


def square_crop(img, box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = PAD * max(box[2] - box[0], box[3] - box[1]) / 2
    return img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize(
        (CROP_PX, CROP_PX), Image.BILINEAR)


def crops_for_log(log_dir, windows, out_dir):
    """Choose and cut the crops of every window of one log; returns one dict per crop."""
    proj = LogProjector(log_dir)
    chosen = []
    for w in windows.itertuples():
        cands = []
        for cam, ts_all in proj.images.items():
            for ts in ts_all[(ts_all >= w.t0_ns) & (ts_all < w.t0_ns + WINDOW_NS)]:
                box = proj.track_box(w.track_id, w.cls, cam, int(ts))
                if box is not None:
                    cands.append((box[3] - box[1], int(ts), cam, box))
        picked = []
        for cand in sorted(cands, key=lambda c: -c[0]):         # biggest boxes first, 0.5 s apart
            if all(abs(cand[1] - p[1]) >= MIN_GAP_NS for p in picked):
                picked.append(cand)
            if len(picked) == MAX_CROPS:
                break
        chosen += [(w, *p) for p in picked]
    by_image = defaultdict(list)
    for item in chosen:
        by_image[(item[3], item[2])].append(item)               # (camera, time) -> crops
    rows = []
    for (cam, ts), items in by_image.items():                   # decode each frame once
        with Image.open(proj.log_dir / "sensors" / "cameras" / cam / f"{ts}.jpg") as im:
            img = im.convert("RGB")
        for w, h, _, _, box in items:
            path = out_dir / log_dir.name / f"{w.track_id}_{cam}_{ts}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            square_crop(img, box).save(path, quality=95)
            rows.append(dict(log_id=log_dir.name, track_id=w.track_id, cls=w.cls, cam=cam, ts=ts,
                             box_h=round(float(h), 1), path=str(path.relative_to(DATA))))
    return rows


def read_crop(path, attempts=6):
    """One crop as uint8 [H, W, 3]; retries transient parallel-filesystem errors (Errno 121)."""
    for i in range(attempts):
        try:
            with Image.open(path) as im:
                return np.asarray(im.convert("RGB"))
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)


def merge(n_shards, out_dir):
    """Verify every shard finished, merge their CSVs, pack all crops into one array (one big read
    for the GPU job instead of ~57k small files), and draw a per-class crop gallery."""
    shards = [out_dir / f"shard_{k:03d}.csv" for k in range(n_shards)]
    missing = [p.name for p in shards if not p.exists()]
    if missing:
        sys.exit(f"{len(missing)} crop shards missing: {missing[:5]}")
    crops = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    crops.to_csv(DATA / "sensor" / "crops.csv", index=False)
    np.save(DATA / "sensor" / "crops.npy", np.stack([read_crop(DATA / p) for p in crops.path]))
    per = crops.groupby("cls").agg(crops=("path", "size"), tracks=("track_id", "nunique")).reindex(CLASSES)
    print(f"{len(crops)} crops of {crops.track_id.nunique()} tracks\n{per}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(CLASSES), 8, figsize=(12, 1.6 * len(CLASSES)))
    for i, c in enumerate(CLASSES):
        sub = crops[crops.cls == c]
        pick = sub.iloc[rng.choice(len(sub), size=min(8, len(sub)), replace=False)] if len(sub) else sub
        for j, ax in enumerate(axes[i]):
            ax.axis("off")
            if j < len(pick):
                ax.imshow(Image.open(DATA / pick.iloc[j].path))
        axes[i][0].set_title(c, loc="left", fontsize=9)
    out = RESULTS / "rq2"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "crop_gallery.png", dpi=110, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=20)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--logs", nargs="*", help="debug: only these log ids")
    a = ap.parse_args()
    out_dir = DATA / "sensor" / "crops"
    if a.merge:
        return merge(a.n_shards, out_dir)
    win = pd.read_parquet(DATA / "sensor" / "windows.parquet", columns=["log_id", "split", "track_id", "cls", "t0_ns"])
    logs = sorted(win.log_id.unique())[a.shard::a.n_shards] if not a.logs else a.logs
    rows = []
    for log in logs:
        w = win[win.log_id == log]
        if len(w):        # logs without moving tracks of the five classes have nothing to crop
            rows += crops_for_log(DATA / "sensor" / w.split.iat[0] / log, w, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"shard_{a.shard:03d}.csv", index=False)
    print(f"shard {a.shard}: {len(logs)} logs -> {len(rows)} crops", flush=True)


if __name__ == "__main__":
    main()
