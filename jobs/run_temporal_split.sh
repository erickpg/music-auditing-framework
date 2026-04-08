#!/bin/bash
#SBATCH --job-name=temporal_split
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/$USER/temporal_split_%j.out

set -euo pipefail

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env6

export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache

RUN_DIR=/scratch/$USER/runs/2026-03-10_full
V2_RUN_DIR=/scratch/$USER/runs/2026-03-10_full_v2
OUT_DIR=/scratch/$USER/runs/temporal_split
mkdir -p $OUT_DIR/unseen_audio $OUT_DIR/unseen_standardized

echo "=== Step 1: Identify and download unseen tracks ==="

# Create a manifest of unseen tracks, then download them
python3 /home/$USER/temporal_step1_download.py

echo "=== Step 2: Standardize unseen audio ==="
# Standardize downloaded tracks (resample to 32kHz mono like the catalog)
python3 /home/$USER/temporal_step2_standardize.py

echo "=== Step 3: CLAP analysis (GPU) ==="
# Compute CLAP embeddings and temporal split comparison
python3 /home/$USER/temporal_step3_clap.py

echo "=== Done ==="
echo "Results in $OUT_DIR/"
ls -la $OUT_DIR/*.csv $OUT_DIR/*.json 2>/dev/null
