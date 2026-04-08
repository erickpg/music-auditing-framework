#!/bin/bash
#SBATCH --job-name=standardize
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/home/$USER/capstone/jobs/logs/standardize_%j.out
#SBATCH --error=/home/$USER/capstone/jobs/logs/standardize_%j.err

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env

export RUN_ID=2026-03-10_full
export RUN_DIR=/scratch/$USER/runs/$RUN_ID
export CONFIG=configs/exp002_full.yaml

cd /scratch/$USER/capstone-repo

python src/data/standardize_audio.py \
    --config $CONFIG \
    --run_id $RUN_ID \
    --run_dir $RUN_DIR
