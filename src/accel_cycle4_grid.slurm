#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c4_scope_grid
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
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
    RUN_NAME="c4_baseline_all"
    EXTRA_ARGS="--train_scope=all --alignment_output_prior_weight=0.0 --val_fraction=0.10 --early_stop_patience=0"
    ;;
  1)
    RUN_NAME="c4_adapter_only"
    EXTRA_ARGS="--train_scope=adapter_only --alignment_output_prior_weight=0.0 --val_fraction=0.10 --early_stop_patience=0"
    ;;
  2)
    RUN_NAME="c4_adapter_only_prior_1e-3"
    EXTRA_ARGS="--train_scope=adapter_only --alignment_output_prior_weight=1e-3 --alignment_output_prior_type=distill --val_fraction=0.10 --early_stop_patience=0"
    ;;
  3)
    RUN_NAME="c4_adapter_only_prior_1e-2"
    EXTRA_ARGS="--train_scope=adapter_only --alignment_output_prior_weight=1e-2 --alignment_output_prior_type=distill --val_fraction=0.10 --early_stop_patience=0"
    ;;
  4)
    RUN_NAME="c4_adapter_head"
    EXTRA_ARGS="--train_scope=adapter_head --alignment_output_prior_weight=0.0 --val_fraction=0.10 --early_stop_patience=0"
    ;;
  5)
    RUN_NAME="c4_adapter_head_earlystop"
    EXTRA_ARGS="--train_scope=adapter_head --alignment_output_prior_weight=0.0 --val_fraction=0.10 --early_stop_patience=10"
    ;;
  *)
    echo "Unknown task ${SLURM_ARRAY_TASK_ID}"
    exit 1
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
  --model_name=${RUN_NAME} \
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
  --ckpt_interval=5 \
  --ckpt_saving \
  --no-wandb_log \
  ${EXTRA_ARGS}
