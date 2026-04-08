#!/bin/bash
#SBATCH --job-name=09_generate
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/$USER/slurm_generate_%j.out
#SBATCH --signal=B:USR1@900

# --- Preemption handler (requeue on signal) ---
handle_preempt() {
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') PREEMPTION WARNING: requeuing job..."
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap handle_preempt USR1

# --- Environment ---
source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env3
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache

RUN_ID="2026-03-10_full"
RUN_DIR="/scratch/$USER/runs/$RUN_ID"
CONFIG="/scratch/$USER/capstone-repo/configs/exp005_memorization.yaml"
CHECKPOINT="/scratch/$USER/musicgen_trainer/models/lm_final.pt"

cd /scratch/$USER/capstone-repo

echo "============================================================"
echo "STAGE:     generate_outputs (audiocraft API)"
echo "RUN_ID:    $RUN_ID"
echo "RUN_DIR:   $RUN_DIR"
echo "CHECKPOINT: $CHECKPOINT"
echo "START:     $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================================"

# Step 1: Build prompts
python src/generation/build_prompts.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR" \
    --seed 42

# Step 2: Generate all tiers
python src/generation/generate_outputs.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" &

wait $!

echo "END: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
