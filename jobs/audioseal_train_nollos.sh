#!/bin/bash
#SBATCH --job-name=as_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=/home/$USER/as_train_%j.out
#SBATCH --error=/home/$USER/as_train_%j.err

set -e
source $HOME/miniforge3/bin/activate /scratch/$USER/audioseal_env2

cd /scratch/$USER/audiocraft_fresh

export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/hf_cache
export DORA_XP_ROOT=/scratch/$USER/dora_outputs

# First, create the data manifest for dora
TRAIN_DIR=/scratch/$USER/runs/2026-03-11_audioseal_train/masters_clean
echo "Training files: $(ls $TRAIN_DIR | wc -l)"

# Experiment: 0-bit only, NO adversarial losses
# Goal: verify detection can converge when discriminator pressure is removed
# If this still fails, the architectural limit is confirmed

dora -P audiocraft.solvers.watermark \
  run solver=watermark/default \
  dset=audio/default \
  "datasource.train=$TRAIN_DIR" \
  "datasource.valid=$TRAIN_DIR" \
  dataset.batch_size=16 \
  dataset.num_workers=8 \
  dataset.sample_rate=16000 \
  dataset.channels=1 \
  dataset.segment_duration=1 \
  optim.epochs=100 \
  optim.updates_per_epoch=2000 \
  optim.lr=1e-4 \
  model/nbits=0 \
  losses.adv=0.0 \
  losses.feat=0.0 \
  losses.l1=0.0 \
  losses.mel=0.0 \
  losses.msspec=0.0 \
  losses.sisnr=0.0 \
  losses.wm_detection=10.0 \
  losses.wm_mb=0.0 \
  losses.tf_loudnessratio=0.0 \
  aug_weights.identity=1.0 \
  aug_weights.encodec=1.0 \
  wm_mb.temperature=1.0
