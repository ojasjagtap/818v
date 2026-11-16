#!/bin/bash
#SBATCH -J mf-aggregate
#SBATCH -A cmsc848n-class
#SBATCH --partition=standard
#SBATCH -t 01:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --mail-user=ojagtap@umd.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o logs/%x-%j.out
# M2: runs after the extract array (afterany). Exits 1 -> FAIL email if any shard is missing.
source slurm/env.sh
python src/aggregate.py --n-shards 50
