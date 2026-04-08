#!/bin/bash
#SBATCH --job-name=analysis_pipeline
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/$USER/slurm_analysis_%j.out
#SBATCH --signal=B:USR1@900

# --- Preemption handler ---
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

cd /scratch/$USER/capstone-repo

echo "============================================================"
echo "ANALYSIS PIPELINE"
echo "RUN_ID:    $RUN_ID"
echo "RUN_DIR:   $RUN_DIR"
echo "START:     $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================================"

# === STAGE 10: Tokenize ===
echo ""
echo "=== STAGE 10: TOKENIZE ==="
echo "START: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

python src/tokenization/tokenize_catalog.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

python src/tokenization/tokenize_generated.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo "TOKENIZE DONE: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# === STAGE 11: N-gram analysis (CPU-only but runs here for convenience) ===
echo ""
echo "=== STAGE 11: N-GRAM ANALYSIS ==="
echo "START: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

python src/analysis/ngram_search.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

python src/analysis/ngram_stats.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo "N-GRAM DONE: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# === STAGE 12: Vulnerability assessment (GPU for CLAP) ===
echo ""
echo "=== STAGE 12: VULNERABILITY ASSESSMENT ==="
echo "START: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

python src/analysis/compute_clap_embeddings.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

python src/analysis/musicological_features.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

python src/analysis/per_artist_fad.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

python src/analysis/vulnerability_score.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo ""
echo "============================================================"
echo "ALL ANALYSIS COMPLETE: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"
