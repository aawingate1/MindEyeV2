#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c7_projreg_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH -e /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%j.err
#SBATCH -o /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%j.out
#SBATCH --no-requeue

SRC_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
SUBJ=5
LAMBDA=1e-4
MODEL_NAME=cycle7_smoke_subj05_projreg1e-4_3ep
CKPT_DIR="${SRC_DIR}/train_logs/final_multisubject_subj0${SUBJ}"

cd "${SRC_DIR}"
source /etc/profile.d/modules.sh
module load anaconda3/2024.2
source fmri/bin/activate
mkdir -p slurms

echo "Cycle 7 projection-drift smoke"
echo "SRC_DIR=${SRC_DIR}"
echo "SUBJ=${SUBJ}"
echo "LAMBDA=${LAMBDA}"
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
  --num_epochs=3 \
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
  --proj_reg_lambda=${LAMBDA} \
  --proj_reg_modules=ridge \
  --ckpt_interval=25 \
  --ckpt_saving \
  --no-wandb_log
