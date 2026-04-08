#!/bin/bash
#SBATCH --job-name=temporal_s3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/$USER/temporal_split_%j.out

set -euo pipefail

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env6

export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache

# Check soundfile
python3 -c "import soundfile; print('soundfile OK')" || pip install soundfile

# Steps 1+2 already completed — skip to step 3
echo "=== Step 3: CLAP temporal split analysis (GPU) ==="
python3 /home/$USER/temporal_step3_clap.py

echo "=== Done ==="
ls -la /scratch/$USER/runs/temporal_split/*.csv /scratch/$USER/runs/temporal_split/*.json 2>/dev/null
