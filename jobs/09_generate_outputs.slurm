#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.log
#SBATCH --signal=B:USR1@900

# --- Preemption handler ---
handle_preempt() {
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') PREEMPTION WARNING: requeuing job..."
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap handle_preempt USR1

source "$(dirname "$0")/_preamble.sh" generate

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

CHECKPOINT="${CHECKPOINT:-$RUN_DIR/checkpoints/lm_final.pt}"
echo "CHECKPOINT: $CHECKPOINT"

# Build prompts (4-tier design: A/B/C/D)
python src/generation/build_prompts.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

# Generate outputs (audiocraft API + chavinlo checkpoint)
python src/generation/generate_outputs.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" &

wait $!

echo "END: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
