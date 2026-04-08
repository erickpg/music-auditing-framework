#!/bin/bash
#SBATCH --job-name=as_train_data
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/home/$USER/capstone/jobs/logs/as_train_data_%j.out
#SBATCH --error=/home/$USER/capstone/jobs/logs/as_train_data_%j.err

# Stage 05 prep: Sample, download, and standardize training data for AudioSeal retraining
# Excludes the 50 test catalog artists from the sampling pool

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env

export RUN_ID=2026-03-11_audioseal_train
export RUN_DIR=/scratch/$USER/runs/$RUN_ID
export CONFIG=configs/exp004_audioseal_train_data.yaml
export EXCLUDE_CSV=/scratch/$USER/runs/2026-03-10_full/manifests/artists_selected.csv

cd /scratch/$USER/capstone-repo

mkdir -p $RUN_DIR/logs $RUN_DIR/manifests $RUN_DIR/data $RUN_DIR/masters_clean

echo "=== Step 1: Sample artists (excluding test catalog) ==="
python src/data/sample_artists.py \
    --config $CONFIG \
    --run_id $RUN_ID \
    --run_dir $RUN_DIR \
    --metadata_dir /scratch/$USER/fma_metadata \
    --num_artists 200 \
    --target_tracks 50 \
    --exclude_artists $EXCLUDE_CSV

echo "=== Step 2: Download audio ==="
python src/data/download_audio.py \
    --config $CONFIG \
    --run_id $RUN_ID \
    --run_dir $RUN_DIR

echo "=== Step 3: Standardize audio (16kHz mono for AudioSeal) ==="
python src/data/standardize_audio.py \
    --config $CONFIG \
    --run_id $RUN_ID \
    --run_dir $RUN_DIR

echo "=== Done ==="
echo "Check $RUN_DIR/manifests/ for sampling results"
echo "Check $RUN_DIR/masters_clean/ for standardized audio"
