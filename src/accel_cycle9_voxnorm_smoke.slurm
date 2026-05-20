#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c9_voxnorm_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH -e /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%j.err
#SBATCH -o /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%j.out
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

RUN_NAME="c9_voxnorm_train_session_zscore_smoke"

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
  --num_epochs=2 \
  --no-use_prior \
  --prior_scale=30 \
  --clip_scale=1 \
  --no-blurry_recon \
  --blur_scale=.5 \
  --no-use_image_aug \
  --n_blocks=4 \
  --hidden_dim=1024 \
  --num_sessions=1 \
  --val_fraction=0.10 \
  --heldout_val_sessions=1 \
  --heldout_val_start_session=1 \
  --heldout_val_max_samples=90 \
  --strict_val_image_disjoint \
  --repeat_aware_val \
  --val_id_source=auto \
  --voxel_norm_mode=train_session_zscore \
  --voxel_norm_eps=1e-6 \
  --early_stop_patience=0 \
  --ckpt_interval=5 \
  --ckpt_saving \
  --no-wandb_log \
  --train_scope=all \
  --head_lr_scale=1.0 \
  --ridge_lr_scale=1.0 \
  --alignment_lora_rank=0
