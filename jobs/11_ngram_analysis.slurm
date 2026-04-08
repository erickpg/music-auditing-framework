#!/bin/bash
#SBATCH --job-name=11_ngram
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log

source "$(dirname "$0")/_preamble.sh" ngram_analysis

# N-gram search: all tiers vs catalog + per-artist matched vs mismatched
python src/analysis/ngram_search.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

# Statistical tests: tier comparisons, per-artist, per-genre, exposure correlation
python src/analysis/ngram_stats.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo "Done: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
