#!/bin/bash
#SBATCH --job-name=tokenizer-survival
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

source "$(dirname "$0")/_preamble.sh" tokenizer_survival

python src/watermark/detect_watermark.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR" \
    --input_dir "$RUN_DIR/masters_watermarked" \
    --tokenizer_survival

echo "END: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
