#!/bin/bash
#SBATCH -J rq2-merge
#SBATCH -A cmsc848n-class
#SBATCH --partition=standard
#SBATCH -t 01:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --mail-type=NONE
#SBATCH -o logs/%x-%j.out
# RQ2 step 3 (afterany on the crop array): verify + merge crop shards, pack every crop into
# data/sensor/crops.npy (exit 1 if any shard is missing; the experiment step then fails and emails).
source slurm/env.sh
python src/sensor_crops.py --merge --n-shards 20
