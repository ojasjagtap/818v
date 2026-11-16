"""Constants, paths, and the window-placement rule shared by every script."""
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# The five dynamic Argoverse 2 Motion Forecasting types; index order is the label order everywhere.
CLASSES = ["vehicle", "pedestrian", "cyclist", "motorcyclist", "bus"]
CITIES = ["austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc"]

DT = 0.1           # s; both datasets are sampled at 10 Hz
WINDOW = 50        # steps per classification window (5 s)
MIN_PATH_M = 3.0   # a window counts as "moving" if its path length is at least this
SEED = 0

# AV2 Sensor's 30 categories -> the five MF classes. Riders are kept, their bikes are not
# (BICYCLE / MOTORCYCLE cuboids duplicate the rider's motion).
SENSOR_TO_CLASS = {
    "REGULAR_VEHICLE": "vehicle", "LARGE_VEHICLE": "vehicle", "BOX_TRUCK": "vehicle",
    "TRUCK": "vehicle", "TRUCK_CAB": "vehicle",
    "BUS": "bus", "SCHOOL_BUS": "bus", "ARTICULATED_BUS": "bus",
    "PEDESTRIAN": "pedestrian",
    "BICYCLIST": "cyclist",
    "MOTORCYCLIST": "motorcyclist",
}

# Physics features whose MF bus-vs-vehicle signal is tracker noise (box-center jitter grows with
# object size; M1 section 6b). RQ1 reports the physics model with and without them.
NOISE_SENSITIVE = ["jerk_rms", "jerk_p95", "pred_rmse_point", "pred_rmse_bicycle",
                   "acc_lon_p5", "acc_lon_p95"]

RING_CAMERAS = (
    "ring_front_center", "ring_front_left", "ring_front_right",
    "ring_side_left", "ring_side_right", "ring_rear_left", "ring_rear_right",
)

S3_HTTP = "https://argoverse.s3.amazonaws.com"


def window_start(group_id, track_id, n_steps):
    """Deterministic random start of the WINDOW-step window in a track of n_steps.

    Seeded by the ids, so it does not depend on which shard or order processes the track.
    """
    rng = np.random.default_rng([SEED, zlib.crc32(f"{group_id}/{track_id}".encode())])
    return int(rng.integers(0, n_steps - WINDOW + 1))


def path_length(xy):
    """Path length [m] of an (T, 2) polyline."""
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())
