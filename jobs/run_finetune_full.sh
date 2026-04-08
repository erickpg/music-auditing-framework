#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=08_finetune_full
#SBATCH --output=/home/$USER/slurm_finetune_full_%j.out

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env2
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache

RUN_ID="2026-03-10_full"
RUN_DIR="/scratch/$USER/runs/$RUN_ID"
CONFIG="/scratch/$USER/capstone-repo/configs/exp005_memorization.yaml"

python /scratch/$USER/capstone-repo/src/training/finetune_musicgen_full.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR" \
    --segment_dir "$RUN_DIR/segments"
