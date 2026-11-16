#!/bin/bash
#SBATCH -J rq2-crops
#SBATCH -A cmsc848n-class
#SBATCH --partition=standard
#SBATCH --array=0-19
#SBATCH -t 01:00:00
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH -o logs/%x-%A_%a.out
# RQ2 step 2: every 20th log -> object crops from the ~1 Hz ring-camera images (no mail; step 3 emails).
source slurm/env.sh
python src/sensor_crops.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 20
