#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c3_grid
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH -e /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.err
#SBATCH -o /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.out
#SBATCH --no-requeue

SRC_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
cd "${SRC_DIR}"
source /etc/profile.d/modules.sh
module load anaconda3/2024.2
source fmri/bin/activate
mkdir -p slurms

export NUM_GPUS=1
export BATCH_SIZE=24
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export CUDA_VISIBLE_DEVICES=0

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

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_NAME="baseline_1h_seed42"
    EXTRA_ARGS="--reliability_mode=none --adapter_prior_weight=0.0"
    ;;
  1)
    RUN_NAME="reliability_aware_1h_top8k_seed42"
    EXTRA_ARGS="--reliability_mode=topk_mask --reliability_topk=8000 --adapter_prior_weight=0.0"
    ;;
  2)
    RUN_NAME="adapter_prior_1h_w1e-4_seed42"
    EXTRA_ARGS="--reliability_mode=none --adapter_prior_weight=1e-4 --adapter_prior_type=weights"
    ;;
  3)
    RUN_NAME="adapter_prior_1h_w1e-3_seed42"
    EXTRA_ARGS="--reliability_mode=none --adapter_prior_weight=1e-3 --adapter_prior_type=weights"
    ;;
  4)
    RUN_NAME="adapter_prior_1h_w1e-2_seed42"
    EXTRA_ARGS="--reliability_mode=none --adapter_prior_weight=1e-2 --adapter_prior_type=weights"
    ;;
  5)
    RUN_NAME="reliability_adapter_prior_1h_top8k_w1e-3_seed42"
    EXTRA_ARGS="--reliability_mode=topk_mask --reliability_topk=8000 --adapter_prior_weight=1e-3 --adapter_prior_type=weights"
    ;;
  *)
    echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

accelerate launch \
  --num_processes=$(($NUM_GPUS * $COUNT_NODE)) \
  --num_machines=$COUNT_NODE \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --mixed_precision=fp16 \
  Train.py \
  --data_path="${SRC_DIR}" \
  --cache_dir="${SRC_DIR}" \
  --model_name="${RUN_NAME}" \
  --no-multi_subject \
  --subj=1 \
  --batch_size=${BATCH_SIZE} \
  --max_lr=3e-4 \
  --mixup_pct=.33 \
  --num_epochs=150 \
  --no-use_prior \
  --prior_scale=30 \
  --clip_scale=1 \
  --no-blurry_recon \
  --blur_scale=.5 \
  --no-use_image_aug \
  --n_blocks=4 \
  --hidden_dim=1024 \
  --num_sessions=1 \
  ${EXTRA_ARGS} \
  --ckpt_interval=5 \
  --ckpt_saving \
  --wandb_log \
  --wandb_project=mindeye
