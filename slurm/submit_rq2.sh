#!/bin/bash
# RQ2 chain: windows/features (CPU) -> crop array (CPU, 20 tasks) -> merge + pack crops (CPU)
# -> experiment (MIG GPU).
# Each step depends afterany on the previous one and fails fast if its inputs are missing, so exactly
# one email arrives, from the last step (END = results written, FAIL = see its log).
#   ssh zaratan 'cd ~/scratch/818v && bash slurm/submit_rq2.sh'   (after scripts/download_sensor.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
p=$(sbatch --parsable slurm/rq2_prep.sh)
c=$(sbatch --parsable --dependency=afterany:${p%%;*} slurm/rq2_crops_array.sh)
m=$(sbatch --parsable --dependency=afterany:${c%%;*} slurm/rq2_merge.sh)
r=$(sbatch --parsable --dependency=afterany:${m%%;*} slurm/rq2_run.sh)
echo "RQ2: prep ${p%%;*}, crops array ${c%%;*} (20 tasks), merge ${m%%;*}, experiment ${r%%;*}"
