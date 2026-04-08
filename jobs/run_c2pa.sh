#!/bin/bash
#SBATCH --job-name=c2pa_embed
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/home/$USER/capstone/jobs/logs/c2pa_%j.out
#SBATCH --error=/home/$USER/capstone/jobs/logs/c2pa_%j.err

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env

export RUN_ID=2026-03-10_full
export RUN_DIR=/scratch/$USER/runs/$RUN_ID
export CONFIG=configs/exp002_full.yaml

# Sync latest code to scratch
rsync -a /home/$USER/capstone/src/ /scratch/$USER/capstone-repo/src/ --exclude=__pycache__

cd /scratch/$USER/capstone-repo
mkdir -p /scratch/$USER/tmp

# Stage 02: Embed C2PA
echo "=== Stage 02: C2PA Embed ==="
python src/c2pa/embed_c2pa.py \
    --config $CONFIG \
    --run_id $RUN_ID \
    --run_dir $RUN_DIR

# Stage 03: C2PA Survival Matrix
echo "=== Stage 03: C2PA Survival ==="
python src/c2pa/c2pa_survival_matrix.py \
    --config $CONFIG \
    --run_id $RUN_ID \
    --run_dir $RUN_DIR \
    --sample_size 20
