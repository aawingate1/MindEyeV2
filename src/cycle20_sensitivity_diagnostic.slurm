#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=c20_relgate_sens
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --chdir=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
#SBATCH -e /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%j.err
#SBATCH -o /scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%j.out
#SBATCH --no-requeue

SRC_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src
cd "${SRC_DIR}"
source /etc/profile.d/modules.sh
module load anaconda3/2024.2
source fmri/bin/activate
mkdir -p slurms tables

echo "Cycle 20 reliability-gate sensitivity diagnostic"
echo "SRC_DIR=${SRC_DIR}"
python cycle20_sensitivity_diagnostic.py
