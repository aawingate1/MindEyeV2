#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=mindeye_new_mrae
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

# Three learned MLP hidden sizes between frozen MRAE projection and CLIP embedding output.
MLP_DIM1=${MLP_DIM1:-512}
MLP_DIM2=${MLP_DIM2:-512}
MLP_DIM3=${MLP_DIM3:-512}
export ZSCORE_WITHIN_RUN=${ZSCORE_WITHIN_RUN:-1}
export MRAE_CKPT_PATH=${MRAE_CKPT_PATH:-/scratch/gpfs/KNORMAN/aw1907/MRAE/MRAE/example/mrae_results_mindeye_subj01/mindeye_split_hidden200_manifold200_t5_mrae_subj01.pt}

export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

export CACHE_ROOT=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/cache
export HF_HOME=${CACHE_ROOT}
export HF_HUB_CACHE=${CACHE_ROOT}/hf_hub
export TRANSFORMERS_CACHE=${HF_HUB_CACHE}
mkdir -p "${HF_HUB_CACHE}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TMPDIR=${CACHE_ROOT}/tmp
export PYTHONPYCACHEPREFIX=${CACHE_ROOT}/pycache
mkdir -p "${TMPDIR}" "${PYTHONPYCACHEPREFIX}"

export WANDB_MODE=offline
export WANDB_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wandb_offline
mkdir -p "${WANDB_DIR}"

model_name="mrae/subj01_mlp${MLP_DIM1}_${MLP_DIM2}_${MLP_DIM3}_cliponly_${SLURM_JOB_ID}"
echo model_name=${model_name}
mkdir -p ../train_logs/mrae

ZSCORE_FLAG=--zscore_within_run
if [[ "${ZSCORE_WITHIN_RUN}" == "0" ]]; then
  ZSCORE_FLAG=--no-zscore_within_run
fi

accelerate launch \
  --num_processes=$(($NUM_GPUS * $COUNT_NODE)) \
  --num_machines=$COUNT_NODE \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --mixed_precision=fp16 \
  Train_mrae.py \
  --data_path=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src \
  --cache_dir=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src \
  --model_name=${model_name} \
  --no-multi_subject \
  --subj=1 \
  --batch_size=${BATCH_SIZE} \
  --num_sessions=40 \
  --max_lr=3e-4 \
  --mixup_pct=.33 \
  --num_epochs=150 \
  --no-use_prior \
  --prior_scale=30 \
  --clip_scale=1 \
  --no-blurry_recon \
  --blur_scale=.5 \
  --no-use_image_aug \
  --ckpt_interval=1 \
  --ckpt_saving \
  --wandb_log \
  --wandb_project=mrae_mindeye \
  --mrae_ckpt_path=${MRAE_CKPT_PATH} \
  --mlp_dim1=${MLP_DIM1} \
  --mlp_dim2=${MLP_DIM2} \
  --mlp_dim3=${MLP_DIM3} \
  --clip_seq_dim=256 \
  --clip_emb_dim=1664 \
  ${ZSCORE_FLAG}
