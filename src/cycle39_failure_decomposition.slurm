#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c39_failure_decomp
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/src
#SBATCH -e /src/slurms/%x_%j.err
#SBATCH -o /src/slurms/%x_%j.out
#SBATCH --no-requeue

set -euo pipefail

SRC_DIR=/src

cd "${SRC_DIR}"
source /etc/profile.d/modules.sh
module load anaconda3/2024.2
source fmri/bin/activate
mkdir -p slurms tables/cycle39_failure_decomposition

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${SRC_DIR}/cache"
export HF_HUB_CACHE="${SRC_DIR}/cache/hf_hub"
export TRANSFORMERS_CACHE="${SRC_DIR}/cache/hf_hub"
export HUGGINGFACE_HUB_CACHE="${SRC_DIR}/cache/hf_hub"
export TORCH_HOME="${SRC_DIR}/cache/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="${SRC_DIR}/cache/xdg"

echo "Cycle 39 failure decomposition"
python cycle39_failure_decomposition.py \
  --data_path="${SRC_DIR}" \
  --outdir="${SRC_DIR}/tables/cycle39_failure_decomposition" \
  --batch_size=32 \
  --topn=20
