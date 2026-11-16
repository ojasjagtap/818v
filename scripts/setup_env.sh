#!/bin/bash
# M0: one-time environment setup on the Zaratan LOGIN node (needs internet; runs no compute).
#   cd ~/scratch/818v && tmux new -s setup 'bash scripts/setup_env.sh 2>&1 | tee logs/setup_env.log'
# Everything (venv, pip/torch caches, s5cmd) stays under ~/scratch/818v: home is capped at 10 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
mkdir -p logs results data bin .cache/pip .cache/torch .cache/mpl
export PIP_CACHE_DIR=$ROOT/.cache/pip TORCH_HOME=$ROOT/.cache/torch MPLCONFIGDIR=$ROOT/.cache/mpl
# Login nodes cap each user at 256 processes/threads; numpy's OpenBLAS would otherwise try to
# start one thread per core on import and crash.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# The default `python` module is 3.8 (too old for torch/av2), and the 3.10 module provides
# `python`, not `python3` (`python3` stays the system 3.6). Pin the exact build jobs also load.
module load python/3.10.10/gcc/11.3.0/nocuda/linux-rhel8-zen2
unset PYTHONPATH   # the module points it at its own site-packages, which would shadow the venv
python -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'

[ -d venv ] || python -m venv venv
source venv/bin/activate
pip install --upgrade pip
# torch built against CUDA 12.6: runs on any driver >= 525, whereas the default PyPI wheel may
# target a CUDA version newer than the GPU nodes' driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# s5cmd: anonymous S3 downloads of Argoverse 2
if [ ! -x bin/s5cmd ]; then
  curl -sSL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz \
    | tar -xz -C bin s5cmd
fi
bin/s5cmd version

# ImageNet ResNet-18 weights for RQ2, cached under TORCH_HOME (compute nodes are offline)
python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"

du -sh venv .cache
echo "setup done"
