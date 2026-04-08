#!/bin/bash
set -e
source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env6

R=/tmp/thesis_results
mkdir -p $R/{v1,v2,baseline,robustness,temporal_split,c2pa,watermark_poc}

# V1
ln -sf /scratch/$USER/runs/2026-03-10_full/analysis $R/v1/analysis
ln -sf /scratch/$USER/runs/2026-03-10_full/logs $R/v1/logs
for d in supplementary comparison manifests; do
  [ -d /scratch/$USER/runs/2026-03-10_full/$d ] && ln -sf /scratch/$USER/runs/2026-03-10_full/$d $R/v1/$d
done

# V2
ln -sf /scratch/$USER/runs/2026-03-10_full_v2/analysis $R/v2/analysis
for d in supplementary comparison manifests logs; do
  [ -d /scratch/$USER/runs/2026-03-10_full_v2/$d ] && ln -sf /scratch/$USER/runs/2026-03-10_full_v2/$d $R/v2/$d
done

# Baseline
ln -sf /scratch/$USER/runs/2026-03-10_baseline/analysis $R/baseline/analysis

# Robustness + temporal + c2pa + watermark — copy from local results if on home
for d in robustness temporal_split c2pa watermark_poc; do
  if [ -d /home/$USER/results/$d ]; then
    cp -rn /home/$USER/results/$d/* $R/$d/ 2>/dev/null || true
  fi
  # Also check /scratch
  if [ -d /scratch/$USER/runs/$d ]; then
    cp -rn /scratch/$USER/runs/$d/* $R/$d/ 2>/dev/null || true
  fi
done

echo "--- Checking data ---"
ls $R/v1/analysis/*.csv 2>/dev/null | wc -l
ls $R/v2/analysis/*.csv 2>/dev/null | wc -l
ls $R/baseline/analysis/*.csv 2>/dev/null | wc -l
ls $R/robustness/*.csv 2>/dev/null | wc -l
ls $R/temporal_split/*.csv 2>/dev/null | wc -l

mkdir -p /tmp/thesis_figures

python3 /home/$USER/generate_thesis_plots.py \
  --results_dir $R \
  --out_dir /tmp/thesis_figures

echo "--- Generated figures ---"
ls -la /tmp/thesis_figures/
