#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c36_geom_diag
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
mkdir -p slurms tables/cycle36_geometry

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HOME="${SRC_DIR}/cache"
export HF_HUB_CACHE="${SRC_DIR}/cache/hf_hub"
export TRANSFORMERS_CACHE="${SRC_DIR}/cache/hf_hub"
export HUGGINGFACE_HUB_CACHE="${SRC_DIR}/cache/hf_hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Cycle 36 CLIP geometry diagnostic"
echo "SRC_DIR=${SRC_DIR}"
echo "OUTPUT_DIR=${SRC_DIR}/tables/cycle36_geometry"
echo "CHECKPOINT_ROOT=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs"

python cycle36_clip_geometry_diagnostic.py \
  --data_path="${SRC_DIR}" \
  --output_dir="${SRC_DIR}/tables/cycle36_geometry" \
  --checkpoint_root="/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs" \
  --num_sessions=1 \
  --batch_size=24 \
  --hidden_dim=4096 \
  --n_blocks=4 \
  --blurry_recon
