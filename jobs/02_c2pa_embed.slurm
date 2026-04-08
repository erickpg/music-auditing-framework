#!/bin/bash
#SBATCH --job-name=c2pa-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

source "$(dirname "$0")/_preamble.sh" c2pa_embed

python src/c2pa/embed_c2pa.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo "END: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
