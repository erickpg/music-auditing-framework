#!/bin/bash
#SBATCH --job-name=detect-infer
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

source "$(dirname "$0")/_preamble.sh" detect_infer

# Step 1: Run detection on all generated outputs
python src/audit/run_detection.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

# Step 2: Dataset inference (Welch's t-test + FDR)
python src/audit/dataset_inference.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

# Step 3: Summarize and optionally export
EXPORT_FLAG=""
if [ "${EXPORT:-}" = "1" ]; then
    EXPORT_FLAG="--export"
fi

python src/audit/summarize_results.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR" \
    $EXPORT_FLAG

echo "END: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
