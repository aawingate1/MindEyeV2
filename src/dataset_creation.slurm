#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=dataset_creation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=92G
#SBATCH --time=02:00:00
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err
#SBATCH --no-requeue

set -euo pipefail

cd /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
source fmri/bin/activate
mkdir -p slurms

# Execute notebook directly (no .py conversion)
jupyter nbconvert \
  --to python \
  --execute dataset_creation.ipynb \
  --output dataset_creation.executed.py \
  --ExecutePreprocessor.timeout=-1