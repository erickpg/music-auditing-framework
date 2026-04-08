#!/bin/bash
#SBATCH --job-name=wm-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

source "$(dirname "$0")/_preamble.sh" watermark_embed

python src/watermark/embed_watermark.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo "END: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
