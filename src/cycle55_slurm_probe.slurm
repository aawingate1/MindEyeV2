#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c55_slurm_probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --chdir=/src
#SBATCH --output=/src/slurms/c55_slurm_probe_%j.out
#SBATCH --error=/src/slurms/c55_slurm_probe_%j.err
#SBATCH --no-requeue

echo "BATCH_SHELL_ENTERED"
set -u

echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "USER=${USER:-unset}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATE=$(date -Is)"
echo "ID:"
id
echo "STDOUT_PATH=/src/slurms/c55_slurm_probe_${SLURM_JOB_ID:-unknown}.out"
echo "STDERR_PATH=/src/slurms/c55_slurm_probe_${SLURM_JOB_ID:-unknown}.err"
echo "STDOUT_VISIBLE:"
ls -l "/src/slurms/c55_slurm_probe_${SLURM_JOB_ID}.out" || true
echo "STDERR_VISIBLE:"
ls -l "/src/slurms/c55_slurm_probe_${SLURM_JOB_ID}.err" || true
echo "PATH=${PATH}"
echo "/src and /src/slurms visibility:"
ls -ld /src /src/slurms
echo "diagnostic script visibility:"
ls -l /src/cycle48_rank_transition_diagnostics.py
echo "quota/path hints:"
df -h /src /src/slurms || true
quota -s || true

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  module load anaconda3/2024.2 || true
fi
if [ -f /src/fmri/bin/activate ]; then
  source /src/fmri/bin/activate
  echo "VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
else
  echo "VIRTUAL_ENV=missing_activate"
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
