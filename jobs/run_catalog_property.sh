#!/bin/bash
set -e
source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env6

R=/tmp/thesis_results
mkdir -p $R/{v1,v2,baseline,robustness}

ln -sf /scratch/$USER/runs/2026-03-10_full/analysis $R/v1/analysis
ln -sf /scratch/$USER/runs/2026-03-10_full_v2/analysis $R/v2/analysis
ln -sf /scratch/$USER/runs/2026-03-10_baseline/analysis $R/baseline/analysis

# Copy existing robustness results if any
cp -rn /home/$USER/results/robustness/* $R/robustness/ 2>/dev/null || true

RESULTS_DIR=$R python3 /home/$USER/baseline_catalog_property.py

# Copy results back to home for scp
mkdir -p /home/$USER/results/robustness
cp $R/robustness/baseline_catalog_property*.* /home/$USER/results/robustness/
