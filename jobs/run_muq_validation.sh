#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=muq_val
#SBATCH --output=/scratch/$USER/runs/muq_val_%j.out
#SBATCH --error=/scratch/$USER/runs/muq_val_%j.err

set -e

source $HOME/miniforge3/bin/activate /scratch/$USER/muq_env

export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/hf_cache

echo "=== Dependencies OK ==="
python3 -c "from muq import MuQMuLan; import torch; import torchaudio; print('muq OK'); print('CUDA:', torch.cuda.is_available())"

RUN_DIR=/scratch/$USER/runs/2026-03-10_full
BL_DIR=/scratch/$USER/runs/2026-03-10_baseline
OUT_DIR=/scratch/$USER/runs/muq_validation

mkdir -p $OUT_DIR

# Sync latest script
cp /home/$USER/muq_mulan_validation.py /scratch/$USER/muq_mulan_validation.py

python3 /scratch/$USER/muq_mulan_validation.py \
    --run_dir $RUN_DIR \
    --baseline_dir $BL_DIR \
    --out_dir $OUT_DIR \
    --batch_size 8

echo "=== DONE ==="
ls -la $OUT_DIR/

# Copy results to home for retrieval
mkdir -p /home/$USER/results/muq_validation
cp $OUT_DIR/*.json $OUT_DIR/*.csv /home/$USER/results/muq_validation/
