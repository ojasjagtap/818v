#!/bin/bash
# M5: AV2 Sensor subset for RQ2 -> data/sensor/{train,val}/<log>/ (LOGIN node, in tmux; ~40 GB)
#   cd ~/scratch/818v && tmux new -s sensor 'bash scripts/download_sensor.sh 2>&1 | tee logs/download_sensor.log'
# Per labelled log: annotations, ego poses, calibration, and every 20th image (~1 Hz) of the
# seven ring cameras. No lidar, stereo or maps. Re-runnable: -n skips files already present.
set -euo pipefail
cd "$(dirname "$0")/.."
S5="bin/s5cmd --no-sign-request --numworkers 64"
SRC=s3://argoverse/datasets/av2/sensor
mkdir -p data/sensor
for split in train val; do
  for f in annotations.feather city_SE3_egovehicle.feather \
           calibration/egovehicle_SE3_sensor.feather calibration/intrinsics.feather; do
    $S5 cp -n "$SRC/$split/*/$f" "data/sensor/$split/"
  done
  echo "$split: $(ls data/sensor/$split | wc -l) logs with small files"
done

# list every ring-camera JPEG once, keep every 20th per (log, camera), copy that subset
[ -s data/sensor/ring_listing.txt ] || $S5 ls "$SRC/*/*/sensors/cameras/ring_*/*.jpg" > data/sensor/ring_listing.txt
python3 - <<'EOF'
import collections
keys = collections.defaultdict(list)
for line in open("data/sensor/ring_listing.txt"):
    key = line.split()[-1]       # s5cmd prints keys relative to the wildcard's base prefix
    rel = key.split("datasets/av2/sensor/", 1)[-1]             # <split>/<log>/sensors/cameras/<cam>/<ts>.jpg
    split, log, _, _, cam, name = rel.split("/")
    if split in ("train", "val"):
        keys[(split, log, cam)].append(rel)
with open("data/sensor/ring_cmds.txt", "w") as f:
    n = 0
    for k in sorted(keys):
        for rel in sorted(keys[k])[::20]:
            f.write("cp -n s3://argoverse/datasets/av2/sensor/%s data/sensor/%s\n" % (rel, rel))
            n += 1
print("%d (log, camera) streams, %d images selected" % (len(keys), n))
EOF
$S5 run data/sensor/ring_cmds.txt > /dev/null
echo "images on disk: $(find data/sensor -name '*.jpg' | wc -l)"
du -sh data/sensor
