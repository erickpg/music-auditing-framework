#!/bin/bash
#SBATCH --job-name=c2pa-survival
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

source "$(dirname "$0")/_preamble.sh" c2pa_survival

python src/c2pa/c2pa_survival_matrix.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo "END: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
