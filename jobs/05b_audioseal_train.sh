#!/bin/bash
#SBATCH --job-name=as_0bit
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/home/$USER/capstone/jobs/logs/as_train_%j.out
#SBATCH --error=/home/$USER/capstone/jobs/logs/as_train_%j.err

# Stage 05: AudioSeal retraining — Trial 5 (0-bit, LR fix + long training)
#
# CONTEXT:
# Trials 1-3: multi-bit impossible (wm_mb stuck at 0.693).
# Trial 4: 0-bit with lr=5e-5 — reached 0.709 best val detection in 10 epochs,
#   then preempted. Validation batches briefly hit 0.659 (below random 0.693),
#   proving the model CAN learn this task. But lr=5e-5 is too low (Meta used 1e-4)
#   and 20k steps is far too few (Meta used 400k-600k).
#
# CHANGES FROM TRIAL 4 (literature-informed):
# - LR: 5e-5 → 1e-4 (Meta's AudioSeal default, 2x higher)
# - Epochs: 30 → 100 (200k steps total, closer to Meta's 400k)
# - Batch size: 16 → 32 (Meta used 32-64, reduces gradient noise)
# - Workers: 16 (unchanged, sufficient for batch_size=32)
# - EnCodec aug: 50/50 (unchanged, matches Meta's Latent WM paper)
# - Adversarial: kept active (matches Meta/XAttnMark, fix LR not symptoms)
# - continue_from Trial 4's checkpoint (10 epochs already trained)
#
# RATIONALE (from AudioSeal GitHub Issue #55, AudioMarkBench, RAW-Bench):
# - Low LR causes gradient imbalance: adversarial/perceptual losses dominate,
#   watermark signal gets drowned out → discriminator collapses
# - Meta trained AudioSeal for 600k steps on 4500h; we have 96h but need
#   at least 100k steps to see meaningful convergence
# - Batch size 32 reduces gradient noise, especially important when
#   watermark signal is weak
#
# PREVIOUS RESULTS (validation wm_detection):
# Trial 1 (16-bit, lr=1e-4, loss=1.0):  0.834→0.727 in 7 epochs
# Trial 2 (16-bit, lr=1e-4, loss=10.0): 0.834→0.701 in 4 epochs
# Trial 3 (2-bit, lr=1e-4, loss=10.0):  0.834→0.707 in 5 epochs
# Trial 4 (0-bit, lr=5e-5, loss=10.0):  0.834→0.709 in 10 epochs (preempted)
#   → val batches hit 0.659 at step 496 (below random!)

source $HOME/miniforge3/bin/activate /scratch/$USER/audiocraft_env

export AUDIOCRAFT_CONFIG=capstone_team.yaml
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache

cd /scratch/$USER/audiocraft

echo "=== Starting AudioSeal training (Trial 5: 0-bit, LR=1e-4, 100 epochs) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date: $(date)"

# 0-bit detection with corrected hyperparameters
# 100 epochs x 2000 updates = 200,000 total updates (~50h, fits in 3-day limit)
# LR=1e-4 (Meta default), batch_size=32 (lower gradient noise)
# Checkpoints saved every epoch — best model auto-selected by dora
dora run solver=watermark/robustness \
  dset=audio/capstone_music \
  sample_rate=16000 \
  channels=1 \
  losses.wm_detection=10.0 \
  dataset.batch_size=32 \
  dataset.num_workers=16 \
  optim.updates_per_epoch=2000 \
  optim.epochs=100 \
  optim.lr=1e-4 \
  schedule.lr_scheduler=cosine \
  aug_weights.encodec=1.0 \
  aug_weights.identity=1.0 \
  aug_weights.speed=0.0 \
  aug_weights.updownresample=0.0 \
  aug_weights.echo=0.0 \
  aug_weights.pink_noise=0.0 \
  aug_weights.lowpass_filter=0.0 \
  aug_weights.highpass_filter=0.0 \
  aug_weights.bandpass_filter=0.0 \
  aug_weights.smooth=0.0 \
  aug_weights.boost_audio=0.0 \
  aug_weights.duck_audio=0.0 \
  aug_weights.mp3_compression=0.0 \
  aug_weights.aac_compression=0.0

echo "=== Training complete ==="
echo "Date: $(date)"
echo "Check dora outputs at: /scratch/$USER/dora_outputs/"
