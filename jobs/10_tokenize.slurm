#!/bin/bash
#SBATCH --job-name=10_tokenize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
#SBATCH --signal=B:USR1@900

# --- Preemption handler ---
handle_preempt() {
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') PREEMPTION WARNING: requeuing job..."
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap handle_preempt USR1

source "$(dirname "$0")/_preamble.sh" tokenize

# Tokenize catalog (masters_clean → tokens_catalog)
python src/tokenization/tokenize_catalog.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

# Tokenize all generated outputs (all tiers → tokens_generated/<tier>/)
python src/tokenization/tokenize_generated.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo "Done: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
