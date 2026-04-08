#!/bin/bash
#SBATCH --job-name=sample_artists
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/$USER/capstone/jobs/logs/sample_%j.out
#SBATCH --error=/home/$USER/capstone/jobs/logs/sample_%j.err

source $HOME/miniforge3/bin/activate capstone_env

export RUN_ID=2026-03-10_full
export RUN_DIR=/scratch/$USER/runs/$RUN_ID

mkdir -p $RUN_DIR/logs $RUN_DIR/manifests

cd /home/$USER/capstone

python src/data/sample_artists.py \
    --config configs/exp002_full.yaml \
    --run_id $RUN_ID \
    --run_dir $RUN_DIR \
    --metadata_dir /scratch/$USER/fma_metadata
