#!/bin/bash
# M2: extract array (50 tasks, no mail) -> aggregate (afterany). Exactly one email either way:
# END if every shard landed, FAIL (from aggregate.py's completeness check) if any task died.
#   ssh zaratan 'cd ~/scratch/818v && bash slurm/submit_extract.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
jid=$(sbatch --parsable slurm/extract_array.sh)
jid=${jid%%;*}
sbatch --dependency=afterany:"$jid" slurm/aggregate.sh
echo "extract array $jid submitted; aggregate will run after it"
