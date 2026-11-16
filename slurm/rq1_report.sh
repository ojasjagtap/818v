#!/bin/bash
#SBATCH -J rq1-report
#SBATCH -A cmsc848n-class
#SBATCH --partition=standard
#SBATCH -t 02:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --mail-user=ojagtap@umd.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o logs/%x-%j.out
# RQ1: runs after the GBT job and the sequence-model array (afterany). Exits 1 -> FAIL email if any
# run is missing or a figure fails; otherwise writes RESULTS_DIR/rq1/{main,city,novelty}.csv plus
# figures/, tables/ (bootstrap CIs) and summary.md. Args: RESULTS_DIR N_EXPECTED
source slurm/env.sh
python src/experiment.py collect --results "$1" --expect "$2"
python src/analyze.py --results "$1"
