#!/bin/bash
#SBATCH -J rq1-fusion
#SBATCH -A cmsc848n-class
#SBATCH --partition=standard
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --mail-user=ojagtap@umd.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o logs/%x-%j.out
# RQ1 late fusion (GRU + physics GBT, from saved predictions; no training), then regenerate the
# report: 66 trained runs + 11 fusion runs.
source slurm/env.sh
python src/fuse.py --results results
python src/experiment.py collect --results results --expect 77
python src/analyze.py --results results
