#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c6_local_final_s57
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH -e /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.err
#SBATCH -o /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.out
#SBATCH --no-requeue

SRC_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
SUBJECTS=(5 7)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="cycle2_subj0${SUBJ}_official_1sess_150ep"

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

echo "Cycle 6 local final_evaluations only for subjects 5/7"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SUBJ=${SUBJ}"

python final_evaluations.py \
  --data_path="${SRC_DIR}" \
  --cache_dir="${SRC_DIR}" \
  --model_name="${MODEL_NAME}" \
  --subj="${SUBJ}" \
  --all_recons_path="evals/${MODEL_NAME}/${MODEL_NAME}_all_enhancedrecons.pt"
