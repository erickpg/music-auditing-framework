#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=08_chavinlo_v2
#SBATCH --output=/home/$USER/slurm_chavinlo_v2_%j.out

# ============================================================
# MusicGen-small full fine-tuning v2 — corrected hyperparameters
# ============================================================
# Full decoder fine-tuning (not LoRA — LoRA had compatibility issues).
# Uses finetune_musicgen_full.py with manual training loop.
#
# Changes from v1:
#   LR:      1e-5 → 1e-4  (10x, per audiocraft/chavinlo defaults)
#   Epochs:  20   → 100   (5x, for proper convergence)
#
# Expected: ~5.1 min/epoch × 100 = ~8.5 hours
# Data: 3,562 segments (28.73h), batch=4, grad_acc=8 → 111 steps/epoch
# ============================================================

set -euo pipefail

# Activate working environment
source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache

REPO_DIR=/scratch/$USER/capstone-repo
RUN_DIR=/scratch/$USER/runs/${RUN_ID:-2026-03-10_full}
CONFIG=${CONFIG:-$REPO_DIR/configs/exp005_memorization.yaml}

echo "Starting full fine-tuning v2 at $(date)"

cd $REPO_DIR

python src/training/finetune_musicgen_full.py \
    --config "$CONFIG" \
    --run_id "${RUN_ID:-2026-03-10_full}" \
    --run_dir "$RUN_DIR" \
    --segment_dir "$RUN_DIR/segments"

echo "Fine-tuning v2 completed at $(date)"
