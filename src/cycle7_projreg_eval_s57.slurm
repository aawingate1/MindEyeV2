#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c7_projreg_eval_s57
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH -e /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.err
#SBATCH -o /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.out
#SBATCH --no-requeue

SRC_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
SUBJECTS=(5 7)
LAMBDAS=(0 1e-5 3e-5 1e-4)
SUBJ_INDEX=$((SLURM_ARRAY_TASK_ID / ${#LAMBDAS[@]}))
LAMBDA_INDEX=$((SLURM_ARRAY_TASK_ID % ${#LAMBDAS[@]}))
SUBJ=${SUBJECTS[$SUBJ_INDEX]}
LAMBDA=${LAMBDAS[$LAMBDA_INDEX]}
LAMBDA_TAG=${LAMBDA//./p}
LAMBDA_TAG=${LAMBDA_TAG//-/m}
MODEL_NAME=cycle7_subj0${SUBJ}_projreg${LAMBDA_TAG}_1sess_150ep

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

echo "Cycle 7 projection-drift evaluation"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "SUBJ=${SUBJ}"
echo "LAMBDA=${LAMBDA}"
echo "MODEL_NAME=${MODEL_NAME}"

if [[ ! -f "../train_logs/${MODEL_NAME}/last.pth" ]]; then
  echo "ERROR: missing trained checkpoint: ../train_logs/${MODEL_NAME}/last.pth" >&2
  exit 3
fi

rm -f "evals/${MODEL_NAME}/${MODEL_NAME}"_all_*.pt

python recon_inference.py \
  --data_path="${SRC_DIR}" \
  --cache_dir="${SRC_DIR}" \
  --model_name="${MODEL_NAME}" \
  --subj="${SUBJ}" \
  --hidden_dim=4096 \
  --n_blocks=4 \
  --new_test

python enhanced_recon_inference.py \
  --model_name="${MODEL_NAME}" \
  --subj="${SUBJ}"

python final_evaluations.py \
  --data_path="${SRC_DIR}" \
  --cache_dir="${SRC_DIR}" \
  --model_name="${MODEL_NAME}" \
  --subj="${SUBJ}" \
  --all_recons_path="evals/${MODEL_NAME}/${MODEL_NAME}_all_enhancedrecons.pt"
