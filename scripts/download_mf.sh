#!/bin/bash
# M2: Motion Forecasting train + val scenario parquet (~34 GB, ~225k files) -> data/mf/
# LOGIN node, inside tmux (it is a long download):
#   cd ~/scratch/818v && tmux new -s mf 'bash scripts/download_mf.sh 2>&1 | tee logs/download_mf.log'
# Skips map JSONs (not needed) and the test split (future withheld). Re-runnable: -n skips files
# that already exist.
set -euo pipefail
cd "$(dirname "$0")/.."
for split in val train; do
  bin/s5cmd --no-sign-request --numworkers 64 cp -n --exclude "*.json" \
    "s3://argoverse/datasets/av2/motion-forecasting/$split/*" "data/mf/$split/"
  echo "$split: $(ls data/mf/$split | wc -l) scenario directories"
done
du -sh data/mf
