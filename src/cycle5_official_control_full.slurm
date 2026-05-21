#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c5_official_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH -e /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.err
#SBATCH -o /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.out
#SBATCH --no-requeue

SRC_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
SUBJECTS=(1 2 5 7)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME=$(printf "final_subj%02d_pretrained_1sess_24bs" "${SUBJ}")

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

echo "Cycle 5 official one-session control full evaluation"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SUBJ=${SUBJ}"

# Preserve official enhanced reconstructions staged from Hugging Face, but remove
# smoke or stale locally-generated intermediates before regenerating them.
rm -f "evals/${MODEL_NAME}/${MODEL_NAME}_all_blurryrecons.pt"
rm -f "evals/${MODEL_NAME}/${MODEL_NAME}_all_clipvoxels.pt"
rm -f "evals/${MODEL_NAME}/${MODEL_NAME}_all_predcaptions.pt"
rm -f "evals/${MODEL_NAME}/${MODEL_NAME}_all_recons.pt"

python recon_inference.py \
  --data_path="${SRC_DIR}" \
  --cache_dir="${SRC_DIR}" \
  --model_name="${MODEL_NAME}" \
  --subj="${SUBJ}" \
  --hidden_dim=4096 \
  --n_blocks=4 \
  --new_test

python final_evaluations.py \
  --data_path="${SRC_DIR}" \
  --cache_dir="${SRC_DIR}" \
  --model_name="${MODEL_NAME}" \
  --subj="${SUBJ}" \
  --all_recons_path="evals/${MODEL_NAME}/${MODEL_NAME}_all_enhancedrecons.pt"
