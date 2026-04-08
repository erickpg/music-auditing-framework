#!/bin/bash
#SBATCH --job-name=analysis_only
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/$USER/slurm_analysis_only_%j.out
#SBATCH --signal=B:USR1@900

handle_preempt() {
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') PREEMPTION WARNING: requeuing job..."
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap handle_preempt USR1

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env3
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache

RUN_ID="2026-03-10_full"
RUN_DIR="/scratch/$USER/runs/$RUN_ID"
CONFIG="/scratch/$USER/capstone-repo/configs/exp005_memorization.yaml"

cd /scratch/$USER/capstone-repo

echo "============================================================"
echo "ANALYSIS PIPELINE (tokenization already complete)"
echo "RUN_ID:    $RUN_ID"
echo "START:     $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================================"

set -e  # Exit on first error

# === STAGE 11: N-gram analysis ===
echo ""
echo "=== STAGE 11: N-GRAM SEARCH ==="
python src/analysis/ngram_search.py \
    --config "$CONFIG" --run_id "$RUN_ID" --run_dir "$RUN_DIR"

echo "=== STAGE 11: N-GRAM STATS ==="
python src/analysis/ngram_stats.py \
    --config "$CONFIG" --run_id "$RUN_ID" --run_dir "$RUN_DIR"

echo "N-GRAM DONE: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# === STAGE 12: CLAP embeddings ===
echo ""
echo "=== STAGE 12: CLAP EMBEDDINGS ==="
python src/analysis/compute_clap_embeddings.py \
    --config "$CONFIG" --run_id "$RUN_ID" --run_dir "$RUN_DIR"

echo "=== STAGE 12: MUSICOLOGICAL FEATURES ==="
python src/analysis/musicological_features.py \
    --config "$CONFIG" --run_id "$RUN_ID" --run_dir "$RUN_DIR"

echo "=== STAGE 12: PER-ARTIST FAD ==="
python src/analysis/per_artist_fad.py \
    --config "$CONFIG" --run_id "$RUN_ID" --run_dir "$RUN_DIR"

echo "=== STAGE 12: VULNERABILITY SCORE ==="
python src/analysis/vulnerability_score.py \
    --config "$CONFIG" --run_id "$RUN_ID" --run_dir "$RUN_DIR"

echo ""
echo "============================================================"
echo "ALL ANALYSIS COMPLETE: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"
