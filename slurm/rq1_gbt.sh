#!/bin/bash
#SBATCH -J rq1-gbt
#SBATCH -A cmsc848n-class
#SBATCH --partition=standard
#SBATCH -t 04:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --mail-type=NONE
#SBATCH -o logs/%x-%j.out
# RQ1: gradient-boosted trees (baseline, physics, physics without noise-sensitive features) over
# every config. Args: FRAC RESULTS_DIR CONFIG... (from submit_rq1.sh; the collect job emails).
source slurm/env.sh
FRAC=$1; OUT=$2; shift 2
python src/train_gbt.py --frac "$FRAC" --results "$OUT" --configs "$@"
