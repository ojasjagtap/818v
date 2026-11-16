#!/bin/bash
#SBATCH -J rq1-seq
#SBATCH -A cmsc848n-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH -t 02:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --mail-type=NONE
#SBATCH -o logs/%x-%A_%a.out
# RQ1 sequence models: array task i trains pair i of {gru, gru_phys, gru_smooth} x configs.
# A100 MIG slice by default: 7 SU/h (with --mem=16G the memory charge, 6 SU/h, stays below
# the GPU's; Zaratan requires a GPU type, so a "whichever is free" request is rejected).
# V100 instead (12 SU/h): SEQ_OPTS="--partition=gpu-v100 --gres=gpu:v100:1" to submit_rq1.sh.
# CPU instead (16 cores took ~20 min per small run, 35.7 SU for M3): edit the lines above to
# --partition=standard, no --gres, -c 16, --mem=64G.
# Args: FRAC RESULTS_DIR CONFIG... (from submit_rq1.sh; the collect job emails).
source slurm/env.sh
FRAC=$1; OUT=$2; shift 2
CONFIGS=("$@")
MODELS=(gru gru_phys gru_smooth)
i=$SLURM_ARRAY_TASK_ID
n=${#CONFIGS[@]}
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python src/train_seq.py --model "${MODELS[$((i / n))]}" --configs "${CONFIGS[$((i % n))]}" \
  --frac "$FRAC" --results "$OUT"
