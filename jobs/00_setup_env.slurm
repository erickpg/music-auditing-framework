#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

# ============================================================
# Stage 0: Setup environment and scratch directories
# ============================================================

set -euo pipefail

# --- Cache redirection (scratch-first) ---
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export TORCH_HOME=/scratch/$USER/torch_cache
export XDG_CACHE_HOME=/scratch/$USER/cache

# --- Create scratch directories ---
mkdir -p /scratch/$USER/{runs,hf_cache,torch_cache,cache}

# --- Activate environment ---
source $HOME/miniforge3/bin/activate capstone_env 2>/dev/null \
    || { echo "ERROR: capstone_env not found. Create it first:"; \
         echo "  conda create --prefix /scratch/$USER/capstone_env python=3.11 -y"; \
         exit 1; }

# --- Install/update dependencies ---
cd /scratch/$USER/capstone-repo
pip install -r requirements.txt

echo "=== Environment setup complete ==="
echo "Python: $(python --version)"
echo "Scratch: /scratch/$USER/"
conda list | head -20
