#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c38_neighbor_diag
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
#SBATCH -e /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%j.err
#SBATCH -o /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%j.out
#SBATCH --no-requeue

SRC_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src

cd "${SRC_DIR}"
source /etc/profile.d/modules.sh
module load anaconda3/2024.2
source fmri/bin/activate
mkdir -p slurms tables/cycle38_neighbor_diagnostics

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${SRC_DIR}/cache"
export HF_HUB_CACHE="${SRC_DIR}/cache/hf_hub"
export TRANSFORMERS_CACHE="${SRC_DIR}/cache/hf_hub"
export HUGGINGFACE_HUB_CACHE="${SRC_DIR}/cache/hf_hub"
export TORCH_HOME="${SRC_DIR}/cache/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="${SRC_DIR}/cache/xdg"

echo "Cycle 38 neighbor diagnostics"
python cycle38_neighbor_diagnostics.py \
  --data_path="${SRC_DIR}" \
  --outdir="${SRC_DIR}/tables/cycle38_neighbor_diagnostics" \
  --batch_size=16 \
  --topn=20
