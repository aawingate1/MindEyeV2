#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c3_eval_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH -e /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%j.err
#SBATCH -o /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%j.out
#SBATCH --no-requeue

SRC_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
SUBJ=${SUBJ:-1}
MODEL_NAME=${MODEL_NAME:-cycle2_subj01_official_1sess_150ep}

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Cycle 3 eval smoke"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SUBJ=${SUBJ}"

python recon_inference.py \
  --data_path="${SRC_DIR}" \
  --cache_dir="${SRC_DIR}" \
  --model_name="${MODEL_NAME}" \
  --subj="${SUBJ}" \
  --hidden_dim=4096 \
  --n_blocks=4 \
  --new_test \
  --max_images=2

python enhanced_recon_inference.py \
  --model_name="${MODEL_NAME}" \
  --subj="${SUBJ}" \
  --max_images=2
