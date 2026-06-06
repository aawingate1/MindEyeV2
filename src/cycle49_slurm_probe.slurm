#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c49_slurm_probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --chdir=/src
#SBATCH --output=/src/slurms/c49_slurm_probe_%j.out
#SBATCH --error=/src/slurms/c49_slurm_probe_%j.err
#SBATCH --no-requeue

set -u

echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "USER=${USER:-unset}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PATH=${PATH}"
echo "/src visible:"
ls -ld /src /src/slurms
echo "diagnostic script:"
ls -l /src/cycle48_rank_transition_diagnostics.py

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  module load anaconda3/2024.2 || true
fi
if [ -f /src/fmri/bin/activate ]; then
  source /src/fmri/bin/activate
fi

echo "python=$(command -v python || true)"
python - <<'PY'
import os
import sys

print("sys.executable=", sys.executable)
print("sys.version=", sys.version.replace("\n", " "))
try:
    import torch
    print("torch=", torch.__version__)
    print("cuda_available=", torch.cuda.is_available())
    print("cuda_device_count=", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda_device_name=", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_probe_error=", repr(exc))
print("script_exists=", os.path.exists("/src/cycle48_rank_transition_diagnostics.py"))
PY
