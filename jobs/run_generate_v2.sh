#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=09_gen_v2
#SBATCH --output=/home/$USER/slurm_gen_v2_%j.out

# ============================================================
# Stage 09: Generate 1,530 outputs from v2 fine-tuned model
# ============================================================
# 510 prompts × 1 temp (1.0) × 3 seeds = 1,530 WAVs
# ~10s per generation → ~4.25 hours
# Uses new run dir to keep v1 and v2 separate
# ============================================================

set -euo pipefail

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env5
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache

REPO=/scratch/$USER/capstone-repo
RUN_V1=/scratch/$USER/runs/2026-03-10_full
RUN_V2=/scratch/$USER/runs/2026-03-10_full_v2

# Sync latest code from home
cp /home/$USER/generate_outputs.py $REPO/src/generation/generate_outputs.py
cp /home/$USER/utils.py $REPO/src/utils.py
cp /home/$USER/exp005_memorization.yaml $REPO/configs/exp005_memorization.yaml

# Set up v2 run directory — reuse prompts and manifests from v1
mkdir -p $RUN_V2/{generated/A_artist_proximal,generated/B_genre_generic,generated/C_out_of_distribution,generated/D_fma_tags,manifests,logs,checkpoints}

# Copy prompts from v1
cp $RUN_V1/manifests/prompts.json $RUN_V2/manifests/prompts.json

# Copy v2 checkpoint
CKPT=/scratch/$USER/musicgen_trainer/models/lm_final.pt
if [ ! -f "$CKPT" ]; then
    echo "ERROR: lm_final.pt not found — training may not be finished"
    exit 1
fi
cp $CKPT $RUN_V2/checkpoints/lm_final.pt
echo "Checkpoint copied: $(ls -lh $RUN_V2/checkpoints/lm_final.pt)"

# Also copy sampling manifest (needed by some analysis scripts later)
cp $RUN_V1/manifests/sampling_manifest.csv $RUN_V2/manifests/ 2>/dev/null || true
cp $RUN_V1/manifests/tracks_selected.csv $RUN_V2/manifests/ 2>/dev/null || true

# Copy masters (symlink to save space)
ln -sf $RUN_V1/masters_clean $RUN_V2/masters_clean 2>/dev/null || true

echo "Starting generation v2 at $(date)"
echo "Checkpoint: $CKPT"
echo "Output: $RUN_V2/generated/"

cd $REPO
python src/generation/generate_outputs.py \
    --config configs/exp005_memorization.yaml \
    --run_id 2026-03-10_full_v2 \
    --run_dir $RUN_V2 \
    --checkpoint $RUN_V2/checkpoints/lm_final.pt

echo "Generation v2 completed at $(date)"
echo "Total WAVs:"
find $RUN_V2/generated/ -name "*.wav" | wc -l
