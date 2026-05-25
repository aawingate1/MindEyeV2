#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c28_adapter_s57
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --time=04:30:00
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
CKPT_DIR="${SRC_DIR}/train_logs/final_multisubject_subj0${SUBJ}"

cd "${SRC_DIR}"
source /etc/profile.d/modules.sh
module load anaconda3/2024.2
source fmri/bin/activate
mkdir -p slurms

echo "Cycle 28 subject-adapter full train"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "SUBJ=${SUBJ}"
echo "TAG=${TAG}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "CKPT_DIR=${CKPT_DIR}"

if [[ ! -f "${CKPT_DIR}/last.pth" ]]; then
  echo "ERROR: missing multisubject checkpoint: ${CKPT_DIR}/last.pth" >&2
  exit 3
fi

export NUM_GPUS=1
export BATCH_SIZE=24
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

export WANDB_MODE=offline
export WANDB_PROJECT=mindeye
export WANDB_DIR="${SRC_DIR}/wandb_offline"
export HF_HOME="${SRC_DIR}/cache"
export HF_HUB_CACHE="${SRC_DIR}/cache/hf_hub"
export TRANSFORMERS_CACHE="${SRC_DIR}/cache/hf_hub"
export HUGGINGFACE_HUB_CACHE="${SRC_DIR}/cache/hf_hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${WANDB_DIR}"

ADAPTER_ARGS=(--use_subject_adapter --subject_adapter_dim=128)
if [[ "${TAG}" == "adapter0" ]]; then
  ADAPTER_ARGS+=(--freeze_subject_adapter)
else
  ADAPTER_ARGS+=(--no-freeze_subject_adapter)
fi

accelerate launch \
  --num_processes=$(($NUM_GPUS * $COUNT_NODE)) \
  --num_machines=$COUNT_NODE \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --mixed_precision=fp16 \
  Train.py \
  --data_path="${SRC_DIR}" \
  --cache_dir="${SRC_DIR}" \
  --model_name="${MODEL_NAME}" \
  --no-multi_subject \
  --subj=${SUBJ} \
  --batch_size=${BATCH_SIZE} \
  --max_lr=3e-4 \
  --mixup_pct=.33 \
  --num_epochs=150 \
  --use_prior \
  --prior_scale=30 \
  --clip_scale=1 \
  --blurry_recon \
  --blur_scale=.5 \
  --no-use_image_aug \
  --n_blocks=4 \
  --hidden_dim=4096 \
  --num_sessions=1 \
  --multisubject_ckpt="${CKPT_DIR}" \
  --reliability_mode=none \
  --adapter_prior_weight=0.0 \
  --proj_reg_lambda=0.0 \
  --proj_reg_modules=ridge \
  --ckpt_interval=25 \
  --ckpt_saving \
  --no-wandb_log \
  "${ADAPTER_ARGS[@]}"
