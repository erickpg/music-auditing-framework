#!/bin/bash
#SBATCH --job-name=imperceptibility
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

source "$(dirname "$0")/_preamble.sh" imperceptibility

python src/audit/summarize_results.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo "END: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
