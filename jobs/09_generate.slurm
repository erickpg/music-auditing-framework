#!/bin/bash
#SBATCH --job-name=09_generate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/%u/runs/09_generate_%j.out
#SBATCH --error=/scratch/%u/runs/09_generate_%j.err

set -euo pipefail

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env

export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export TORCH_HOME=/scratch/$USER/torch_cache
export XDG_CACHE_HOME=/scratch/$USER/cache

REPO=/scratch/$USER/capstone-repo
cd "$REPO"

export RUN_ID="${RUN_ID:-2026-03-10_full}"
export RUN_DIR="/scratch/$USER/runs/$RUN_ID"
export CONFIG="${CONFIG:-configs/exp005_memorization.yaml}"

echo "=== Stage 09: Build Prompts + Generate ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date: $(date)"
echo "RUN_DIR: $RUN_DIR"
echo "CONFIG: $CONFIG"

# Step 1: Build prompts (CPU-only, fast)
echo "--- Building prompts ---"
python src/generation/build_prompts.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR" \
    --seed 42

# Step 2: Generate audio from fine-tuned model
echo "--- Generating audio ---"
python src/generation/generate_outputs.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR" \
    ${TIER:+--tier "$TIER"}

echo "=== Done: $(date) ==="
