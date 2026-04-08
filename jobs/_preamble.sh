# === Shared preamble for all Slurm jobs ===
# Source this file or paste its contents into each job script.
# Usage: source jobs/_preamble.sh <stage_name>

set -euo pipefail

# --- Validate RUN_ID ---
RUN_ID="${RUN_ID:?ERROR: Set RUN_ID before submitting (e.g., export RUN_ID=2026-03-10_baseline)}"
RUN_DIR="/scratch/$USER/runs/$RUN_ID"
CONFIG="${CONFIG:-configs/exp001_minimal.yaml}"
REPO_DIR="/scratch/$USER/capstone-repo"

# --- Cache redirection (scratch-first) ---
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export TORCH_HOME=/scratch/$USER/torch_cache
export XDG_CACHE_HOME=/scratch/$USER/cache

# --- Setup ---
mkdir -p "$RUN_DIR/logs"
source $HOME/miniforge3/bin/activate capstone_env 2>/dev/null \
    || source $HOME/miniforge3/bin/activate base

cd "$REPO_DIR"

# --- Reproducibility banner ---
STAGE_NAME="${1:-unknown}"
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(git status --porcelain 2>/dev/null | head -1)
CONFIG_HASH=$(sha256sum "$CONFIG" 2>/dev/null | cut -c1-12 || echo "unknown")

echo "============================================================"
echo "STAGE:     $STAGE_NAME"
echo "RUN_ID:    $RUN_ID"
echo "RUN_DIR:   $RUN_DIR"
echo "CONFIG:    $CONFIG (sha256:$CONFIG_HASH)"
echo "GIT:       $GIT_HASH${GIT_DIRTY:+ *** DIRTY ***}"
echo "HOST:      $(hostname)"
echo "USER:      $USER"
echo "SLURM_JOB: ${SLURM_JOB_ID:-N/A} (${SLURM_JOB_PARTITION:-N/A})"
echo "START:     $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"
