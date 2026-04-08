#!/bin/bash
#SBATCH --job-name=dl_audio
#SBATCH --partition=cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/$USER/capstone/jobs/logs/dl_audio_%j.out
#SBATCH --error=/home/$USER/capstone/jobs/logs/dl_audio_%j.err

source $HOME/miniforge3/bin/activate capstone_env

export RUN_ID=2026-03-10_full
export RUN_DIR=/scratch/$USER/runs/$RUN_ID
export CONFIG=configs/exp002_full.yaml

cd /scratch/$USER/capstone-repo

python src/data/download_audio.py \
    --config $CONFIG \
    --run_id $RUN_ID \
    --run_dir $RUN_DIR
