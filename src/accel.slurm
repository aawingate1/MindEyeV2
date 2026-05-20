#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c2_official_1sess
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH -e /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.err
#SBATCH -o /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.out
#SBATCH --no-requeue

SRC_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
SUBJECTS=(1 2 5 7)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
CKPT_DIR="${SRC_DIR}/train_logs/final_multisubject_subj0${SUBJ}"

cd "${SRC_DIR}"
source /etc/profile.d/modules.sh
module load anaconda3/2024.2
source fmri/bin/activate
mkdir -p slurms

echo "Cycle 2 paper-matched 1-session official-checkpoint baseline"
echo "SRC_DIR=${SRC_DIR}"
echo "SUBJ=${SUBJ}"
echo "CKPT_DIR=${CKPT_DIR}"

if [[ ! -f "${CKPT_DIR}/last.pth" ]]; then
  echo "ERROR: required multisubject checkpoint missing: ${CKPT_DIR}/last.pth" >&2
  echo "Compute nodes cannot resolve Hugging Face; stage the checkpoint into ${SRC_DIR}/train_logs before launch." >&2
  exit 3
fi

case "${SUBJ}" in
  1) expected_size=10301091408 ;;
  2) expected_size=10324782672 ;;
  5) expected_size=10345082448 ;;
  7) expected_size=10350931536 ;;
  *) echo "ERROR: unexpected subject ${SUBJ}" >&2; exit 2 ;;
esac
actual_size=$(wc -c < "${CKPT_DIR}/last.pth")
echo "Checkpoint size bytes: ${actual_size}"
if [[ "${actual_size}" -ne "${expected_size}" ]]; then
  echo "ERROR: unexpected checkpoint size for ${CKPT_DIR}/last.pth; expected ${expected_size}" >&2
  exit 4
fi
if head -c 128 "${CKPT_DIR}/last.pth" | grep -q "git-lfs"; then
  echo "ERROR: checkpoint is an LFS pointer, not a full checkpoint." >&2
  exit 5
fi

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
export CKPT_DIR_FOR_VERIFY="${CKPT_DIR}"
mkdir -p "${WANDB_DIR}"

python - <<'PY'
import os, time, torch
p = os.environ["CKPT_DIR_FOR_VERIFY"] + "/last.pth"
t0 = time.time()
try:
    ckpt = torch.load(p, map_location="cpu", mmap=True)
except TypeError:
    ckpt = torch.load(p, map_location="cpu")
print("checkpoint_loadable", isinstance(ckpt, dict), "keys", sorted(ckpt.keys()), "seconds", round(time.time() - t0, 2))
if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
    raise SystemExit("checkpoint did not contain model_state_dict")
PY

accelerate launch \
  --num_processes=$(($NUM_GPUS * $COUNT_NODE)) \
  --num_machines=$COUNT_NODE \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --mixed_precision=fp16 \
  Train.py \
  --data_path="${SRC_DIR}" \
  --cache_dir="${SRC_DIR}" \
  --model_name=cycle2_subj0${SUBJ}_official_1sess_150ep \
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
  --ckpt_interval=25 \
  --ckpt_saving \
  --no-wandb_log
