#!/bin/bash
#SBATCH -J rq2-prep
#SBATCH -A cmsc848n-class
#SBATCH --partition=standard
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --mail-type=NONE
#SBATCH -o logs/%x-%j.out
# RQ2 step 1: AV2 Sensor annotations -> moving 5-s windows + physics features (data/sensor/windows.parquet).
source slurm/env.sh
python src/sensor_tracks.py
