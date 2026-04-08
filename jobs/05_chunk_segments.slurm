#!/bin/bash
#SBATCH --job-name=chunk
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

source "$(dirname "$0")/_preamble.sh" chunk_segments

python src/data/chunk_segments.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo "END: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
