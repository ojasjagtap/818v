#!/bin/bash
#SBATCH -J mf-extract
#SBATCH -A cmsc848n-class
#SBATCH --partition=standard
#SBATCH --array=0-49
#SBATCH -t 00:30:00
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --mail-type=NONE
#SBATCH -o logs/%x-%A_%a.out
# M2: one array task = every 50th MF scenario of train and of val -> windows (no features yet).
# No mail here: the dependent aggregate job sends exactly one email for the whole sweep.
source slurm/env.sh
python src/extract.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 50
