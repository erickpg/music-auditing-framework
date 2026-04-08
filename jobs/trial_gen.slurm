#!/bin/bash
#SBATCH --job-name=trial_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/runs/trial_gen_%j.out
#SBATCH --error=/scratch/%u/runs/trial_gen_%j.err

set -euo pipefail

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env

export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export TORCH_HOME=/scratch/$USER/torch_cache
export XDG_CACHE_HOME=/scratch/$USER/cache

REPO=/scratch/$USER/capstone-repo
cd "$REPO"

cp /home/$USER/trial_gen.py /scratch/$USER/capstone-repo/src/generation/trial_gen.py

python src/generation/trial_gen.py

# Copy results to /home for download
cp -f /scratch/$USER/runs/2026-03-10_full/diagnostics/v2_textcond/*.wav /home/$USER/gen_preview/

echo "=== Done: $(date) ==="
ls -la /home/$USER/gen_preview/
