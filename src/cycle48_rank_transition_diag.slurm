#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c48_rank_transition
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
mkdir -p slurms

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export HF_HOME="${SRC_DIR}/cache"
export HF_HUB_CACHE="${SRC_DIR}/cache/hf_hub"
export TRANSFORMERS_CACHE="${SRC_DIR}/cache/hf_hub"
export HUGGINGFACE_HUB_CACHE="${SRC_DIR}/cache/hf_hub"
export TORCH_HOME="${SRC_DIR}/cache/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="${SRC_DIR}/cache/xdg"

echo "Cycle 48 rank-transition diagnostic only"
echo "No training, no evaluator relaunch, no checkpoint writes."
python -u cycle48_rank_transition_diagnostics.py \
  --data_path="${SRC_DIR}" \
  --outdir="${SRC_DIR}/tables/cycle48_rank_transition_diagnostics" \
  --recon_batch_size=16
