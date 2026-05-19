#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=mrae_orig_backbone
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
export DATA_SPLIT_MODE=${DATA_SPLIT_MODE:-old}   # new | old
export WDS_ROOT=${WDS_ROOT:-}
export TRAIN_SPLIT=${TRAIN_SPLIT:-}
export EVAL_SPLIT=${EVAL_SPLIT:-}

export MRAE_CKPT_PATH=${MRAE_CKPT_PATH:-/scratch/gpfs/KNORMAN/aw1907/MRAE/MRAE/example/mrae_results_mindeye_subj01/mrae_clip_phate_hidden300_manifold300_t5_mrae_subj01.pt}
export HIDDEN_DIM=${HIDDEN_DIM:-300}
export N_BLOCKS=${N_BLOCKS:-4}
export DROPOUT=${DROPOUT:-0.15}
export USE_RIDGE_AFTER_MRAE=${USE_RIDGE_AFTER_MRAE:-true}
export USE_MRAE_ENCODER_ONLY=${USE_MRAE_ENCODER_ONLY:-false}   # do we exclude the manifold layer
export PROJECTOR_TUNE_MODE=${PROJECTOR_TUNE_MODE:-freeze}       # freeze | full | last (last only if manifold head is used)
export PROJECTOR_LR_SCALE=${PROJECTOR_LR_SCALE:-0.0}         # lower values constrain MRAE updates
# default LR scale is 0.03

ridge_flag="--no-use_ridge_after_mrae"
ridge_tag="noridge"
if [ "${USE_RIDGE_AFTER_MRAE}" = "true" ]; then
  ridge_flag="--use_ridge_after_mrae"
  ridge_tag="ridge"
fi

model_name="mrae/original_backbone_${ridge_tag}_${PROJECTOR_LR_SCALE}_${SLURM_JOB_ID}"
encoder_only_flag="--no-use_mrae_encoder_only"
if [ "${USE_MRAE_ENCODER_ONLY}" = "true" ]; then
  encoder_only_flag="--use_mrae_encoder_only"
fi

if [ -z "${WDS_ROOT}" ]; then
  if [ "${DATA_SPLIT_MODE}" = "new" ]; then
    WDS_ROOT=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds_3way
  else
    WDS_ROOT=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds
  fi
fi

split_flags="--wds_root=${WDS_ROOT}"
if [ "${DATA_SPLIT_MODE}" = "new" ]; then
  if [ -z "${TRAIN_SPLIT}" ]; then
    TRAIN_SPLIT=set2_mindeye_train
  fi
  if [ -z "${EVAL_SPLIT}" ]; then
    EVAL_SPLIT=set3_mindeye_test
  fi
  split_flags="${split_flags} --train_split=${TRAIN_SPLIT} --eval_split=${EVAL_SPLIT}"
fi

accelerate launch \
  --num_processes=$(($NUM_GPUS * $COUNT_NODE)) \
  --num_machines=$COUNT_NODE \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --mixed_precision=fp16 \
  Train_mrae_original_backbone.py \
  --data_path=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src \
  --cache_dir=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src \
  ${split_flags} \
  --model_name=${model_name} \
  --subj=1 \
  --batch_size=${BATCH_SIZE} \
  --num_sessions=40 \
  --max_lr=3e-4 \
  --mixup_pct=.33 \
  --num_epochs=150 \
  --ckpt_interval=1 \
  --ckpt_saving \
  --wandb_log \
  --wandb_project=mrae_mindeye_variants \
  --mrae_ckpt_path=${MRAE_CKPT_PATH} \
  --hidden_dim=${HIDDEN_DIM} \
  --n_blocks=${N_BLOCKS} \
  --dropout=${DROPOUT} \
  ${ridge_flag} \
  --projector_tune_mode=${PROJECTOR_TUNE_MODE} \
  --projector_lr_scale=${PROJECTOR_LR_SCALE} \
  ${encoder_only_flag}
