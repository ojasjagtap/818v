# Sourced by every job script (run from ~/scratch/818v): venv + caches on scratch, offline-safe.
module load python/3.10.10/gcc/11.3.0/nocuda/linux-rhel8-zen2 2>/dev/null || module load python/3.10.10
unset PYTHONPATH   # the module points it at its own site-packages, which would shadow the venv
source venv/bin/activate
export PIP_CACHE_DIR=$PWD/.cache/pip TORCH_HOME=$PWD/.cache/torch MPLCONFIGDIR=$PWD/.cache/mpl
export MPLBACKEND=Agg PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS   # never one per node core
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
