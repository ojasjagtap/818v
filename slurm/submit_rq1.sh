#!/bin/bash
# RQ1 sweep: one CPU job (3 GBT feature sets) + a GPU array (GRU, GRU+physics, GRU on smoothed
# input) -> report (afterany on both: collect tables + figures). Exactly one email, from the report.
#   ssh zaratan 'cd ~/scratch/818v && bash slurm/submit_rq1.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
CONFIGS=(main city:austin city:dearborn city:miami city:palo-alto city:pittsburgh city:washington-dc
         class:bus class:motorcyclist class:cyclist class:pedestrian)
n=${#CONFIGS[@]}
g=$(sbatch --parsable slurm/rq1_gbt.sh 1 results "${CONFIGS[@]}")
s=$(sbatch --parsable --array=0-$((3 * n - 1)) slurm/rq1_seq_array.sh 1 results "${CONFIGS[@]}")
c=$(sbatch --parsable --dependency=afterany:${g%%;*}:${s%%;*} slurm/rq1_report.sh results $((6 * n)))
echo "RQ1: gbt job ${g%%;*}, seq array ${s%%;*} ($((3 * n)) tasks), report ${c%%;*}"
