#!/bin/bash
#SBATCH -J rq2
#SBATCH -A cmsc848n-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --mail-user=ojagtap@umd.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o logs/%x-%j.out
# RQ2 step 4 (afterany on the merge): 5-fold cross-validated motion vs appearance vs fusion under
# darkness and blur. Fails (FAIL email) if the merged crops are missing.
source slurm/env.sh
python src/rq2.py
