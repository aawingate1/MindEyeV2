#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=mrae_ltxf
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out
#SBATCH --no-requeue

cd /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
source fmri/bin/activate
mkdir -p slurms

export NUM_GPUS=1
export BATCH_SIZE=24
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export CACHE_ROOT=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/cache
export HF_HOME=${CACHE_ROOT}
export HF_HUB_CACHE=${CACHE_ROOT}/hf_hub
export TRANSFORMERS_CACHE=${HF_HUB_CACHE}
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=${CACHE_ROOT}/tmp
export PYTHONPYCACHEPREFIX=${CACHE_ROOT}/pycache
export WANDB_MODE=offline
export WANDB_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wandb_offline
mkdir -p "${HF_HUB_CACHE}" "${TMPDIR}" "${PYTHONPYCACHEPREFIX}" "${WANDB_DIR}"

export MRAE_CKPT_PATH=${MRAE_CKPT_PATH:-/scratch/gpfs/KNORMAN/aw1907/MRAE/MRAE/example/mrae_results_mindeye_subj01/mindeye_split_hidden200_manifold100_tauto_mrae_subj01.pt}
export MODEL_HIDDEN_DIM=${MODEL_HIDDEN_DIM:-512}
export NUM_CONTEXT_TOKENS=${NUM_CONTEXT_TOKENS:-32}
export NUM_HEADS=${NUM_HEADS:-8}
export NUM_LAYERS=${NUM_LAYERS:-6}
export DROPOUT=${DROPOUT:-0.1}

model_name="mrae/latent_transformer_${SLURM_JOB_ID}"

accelerate launch \
  --num_processes=$(($NUM_GPUS * $COUNT_NODE)) \
  --num_machines=$COUNT_NODE \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --mixed_precision=fp16 \
  Train_mrae_latent_transformer.py \
  --data_path=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src \
  --cache_dir=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src \
  --model_name=${model_name} \
  --subj=1 \
  --batch_size=${BATCH_SIZE} \
  --num_sessions=40 \
  --max_lr=2e-4 \
  --mixup_pct=.25 \
  --num_epochs=150 \
  --ckpt_interval=1 \
  --ckpt_saving \
  --wandb_log \
  --wandb_project=mrae_mindeye_variants \
  --mrae_ckpt_path=${MRAE_CKPT_PATH} \
  --model_hidden_dim=${MODEL_HIDDEN_DIM} \
  --num_context_tokens=${NUM_CONTEXT_TOKENS} \
  --num_heads=${NUM_HEADS} \
  --num_layers=${NUM_LAYERS} \
  --dropout=${DROPOUT}
