#!/bin/bash
#SBATCH --job-name=wm-wavmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

source "$(dirname "$0")/_preamble.sh" embed_wavmark

# WavMark POC: embed watermark using off-the-shelf WavMark model
# Source masters come from the 2026-03-10_full run

SOURCE_RUN_DIR="/scratch/$USER/runs/2026-03-10_full"

python src/watermark/embed_wavmark.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR" \
    --source_run_dir "$SOURCE_RUN_DIR" \
    --verify_n 10

echo "END: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
