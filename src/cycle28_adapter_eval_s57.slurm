#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c28_adapter_eval_s57
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --chdir=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
#SBATCH -e /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.err
#SBATCH -o /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.out
#SBATCH --no-requeue

SRC_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
SUBJECTS=(5 5 7 7)
TAGS=(adapter0 adapter_low adapter0 adapter_low)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME=cycle28_subj0${SUBJ}_${TAG}_1sess_150ep

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

echo "Cycle 28 subject-adapter evaluation"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "SUBJ=${SUBJ}"
echo "TAG=${TAG}"
echo "MODEL_NAME=${MODEL_NAME}"

if [[ ! -f "../train_logs/${MODEL_NAME}/last.pth" ]]; then
  echo "ERROR: missing trained checkpoint: ../train_logs/${MODEL_NAME}/last.pth" >&2
  exit 3
fi

rm -f "evals/${MODEL_NAME}/${MODEL_NAME}"_all_*.pt

ADAPTER_ARGS=(--use_subject_adapter --subject_adapter_dim=128)
if [[ "${TAG}" == "adapter0" ]]; then
  ADAPTER_ARGS+=(--freeze_subject_adapter)
else
  ADAPTER_ARGS+=(--no-freeze_subject_adapter)
fi

python recon_inference.py \
  --data_path="${SRC_DIR}" \
  --cache_dir="${SRC_DIR}" \
  --model_name="${MODEL_NAME}" \
  --subj="${SUBJ}" \
  --hidden_dim=4096 \
  --n_blocks=4 \
  --new_test \
  "${ADAPTER_ARGS[@]}"

python enhanced_recon_inference.py \
  --model_name="${MODEL_NAME}" \
  --subj="${SUBJ}"

python final_evaluations.py \
  --data_path="${SRC_DIR}" \
  --cache_dir="${SRC_DIR}" \
  --model_name="${MODEL_NAME}" \
  --subj="${SUBJ}" \
  --all_recons_path="evals/${MODEL_NAME}/${MODEL_NAME}_all_enhancedrecons.pt"
