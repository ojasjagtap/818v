"""Download the M1 sample into data/ (laptop only; anonymous HTTPS, re-runnable).

  * the first N MF val scenarios (parquet only)                      -> data/mf/val/<id>/
  * annotations, ego poses and calibration of all 850 labelled
    AV2 Sensor logs (train + val)                                     -> data/sensor/<split>/<log>/
  * ~1 Hz images from two ring cameras of the two val logs with the
    most kinds of road users (for the box-projection overlay check)   -> .../sensors/cameras/
"""
import argparse
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import DATA, S3_HTTP, SENSOR_TO_CLASS  # noqa: E402

MF = "datasets/av2/motion-forecasting/"
SENSOR = "datasets/av2/sensor/"
SMALL = ("annotations.feather", "city_SE3_egovehicle.feather",
         "calibration/egovehicle_SE3_sensor.feather", "calibration/intrinsics.feather")
OVERLAY_CAMS = ("ring_front_center", "ring_side_left")


def list_s3(prefix, delimiter=None, max_keys=None):
    """(keys, sub-prefixes) under prefix, following pagination."""
    keys, prefixes, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{S3_HTTP}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            x = r.read().decode()
        keys += re.findall(r"<Key>(.*?)</Key>", x)
        prefixes += re.findall(r"<CommonPrefixes><Prefix>(.*?)</Prefix>", x)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", x)
        if not m or (max_keys and len(keys) >= max_keys):
            return keys, prefixes
        token = m.group(1)


def fetch(job, attempts=5):
    key, dest = job
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for i in range(attempts):
        try:
            with urllib.request.urlopen(f"{S3_HTTP}/{urllib.parse.quote(key)}", timeout=120) as r, \
                    open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            break
        except OSError:   # connection resets / timeouts: back off and retry
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)
    tmp.rename(dest)
    return dest.stat().st_size


def fetch_all(jobs, label):
    with ThreadPoolExecutor(32) as ex:
        n = sum(ex.map(fetch, jobs))
    print(f"{label}: {len(jobs)} files, {n / 1e6:.0f} MB newly downloaded", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-scenarios", type=int, default=2000)
    a = ap.parse_args()

    keys, _ = list_s3(MF + "val/", max_keys=4 * a.n_scenarios)
    pq = [k for k in keys if k.endswith(".parquet")][:a.n_scenarios]
    fetch_all([(k, DATA / "mf" / k[len(MF):]) for k in pq], "MF val scenarios")

    logs = {}
    for split in ("train", "val"):
        _, pre = list_s3(f"{SENSOR}{split}/", delimiter="/")
        logs[split] = [p.rstrip("/").split("/")[-1] for p in pre]
        fetch_all([(f"{SENSOR}{split}/{log}/{f}", DATA / "sensor" / split / log / f)
                   for log in logs[split] for f in SMALL], f"Sensor {split} small files ({len(logs[split])} logs)")

    # Overlay check: the two val logs whose annotations contain the most target categories.
    def variety(log):
        cats = pd.read_feather(DATA / "sensor/val" / log / "annotations.feather", columns=["category"]).category
        return cats[cats.isin(list(SENSOR_TO_CLASS))].nunique()
    chosen = sorted(logs["val"], key=variety, reverse=True)[:2]
    jobs = []
    for log in chosen:
        for cam in OVERLAY_CAMS:
            ks, _ = list_s3(f"{SENSOR}val/{log}/sensors/cameras/{cam}/")
            jobs += [(k, DATA / "sensor" / k[len(SENSOR):]) for k in sorted(ks)[::20]]
    fetch_all(jobs, f"overlay images for {chosen}")


if __name__ == "__main__":
    main()
