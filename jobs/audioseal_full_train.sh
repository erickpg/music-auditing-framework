#!/bin/bash
#SBATCH --job-name=as_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=/home/$USER/as_full_%j.out
#SBATCH --error=/home/$USER/as_full_%j.err

set -e

echo "=== Step 1: Extract training data ==="
cd /scratch/$USER
if [ ! -d /scratch/$USER/runs/2026-03-11_audioseal_train/masters_clean ] || [ "$(ls /scratch/$USER/runs/2026-03-11_audioseal_train/masters_clean/ 2>/dev/null | wc -l)" -eq 0 ]; then
  tar xzf /home/$USER/scratch_backup_2026-03-20.tar.gz \
    $USER/runs/2026-03-11_audioseal_train/ \
    --strip-components=1
fi
TRAIN_DIR=/scratch/$USER/runs/2026-03-11_audioseal_train/masters_clean
N_FILES=$(ls $TRAIN_DIR | wc -l)
echo "Training files: $N_FILES"
if [ "$N_FILES" -eq 0 ]; then
  echo "ERROR: No training files extracted!"
  exit 1
fi

echo "=== Step 2: Activate env ==="
source $HOME/miniforge3/bin/activate /scratch/$USER/audioseal_env2

echo "=== Step 3: Create data manifest ==="
cd /scratch/$USER/audiocraft_fresh

# Split 90/10 train/valid
mkdir -p /scratch/$USER/audioseal_data/{train,valid}
TOTAL=$(ls $TRAIN_DIR/*.wav | wc -l)
VALID_N=$((TOTAL / 10))
TRAIN_N=$((TOTAL - VALID_N))
echo "Total: $TOTAL, Train: $TRAIN_N, Valid: $VALID_N"

# Create symlinks for train/valid split
ls $TRAIN_DIR/*.wav | shuf --random-source=/dev/urandom | head -$VALID_N | while read f; do
  ln -sf "$f" /scratch/$USER/audioseal_data/valid/$(basename "$f")
done
ls $TRAIN_DIR/*.wav | while read f; do
  if [ ! -L "/scratch/$USER/audioseal_data/valid/$(basename $f)" ]; then
    ln -sf "$f" /scratch/$USER/audioseal_data/train/$(basename "$f")
  fi
done

echo "Train symlinks: $(ls /scratch/$USER/audioseal_data/train/ | wc -l)"
echo "Valid symlinks: $(ls /scratch/$USER/audioseal_data/valid/ | wc -l)"

# Create JSONL manifests
python3 -m audiocraft.data.audio_dataset /scratch/$USER/audioseal_data/train \
  /scratch/$USER/audioseal_data/train/data.jsonl.gz 2>&1 | tail -5
python3 -m audiocraft.data.audio_dataset /scratch/$USER/audioseal_data/valid \
  /scratch/$USER/audioseal_data/valid/data.jsonl.gz 2>&1 | tail -5

echo "=== Step 4: Create dataset config ==="
cat > /scratch/$USER/audiocraft_fresh/config/dset/audio/capstone_wm.yaml << 'YAML'
# @package __global__
datasource:
  max_sample_rate: 16000
  max_channels: 1
  train: /scratch/$USER/audioseal_data/train
  valid: /scratch/$USER/audioseal_data/valid
  evaluate: /scratch/$USER/audioseal_data/valid
  generate: /scratch/$USER/audioseal_data/valid
YAML

echo "=== Step 5: Start training ==="
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/hf_cache
export DORA_XP_ROOT=/scratch/$USER/dora_outputs

# 0-bit only, NO adversarial losses
# Goal: cleanest possible test of detection through EnCodec
dora -P audiocraft \
  run solver=watermark/default \
  dset=audio/capstone_wm \
  dataset.batch_size=16 \
  dataset.num_workers=8 \
  dataset.sample_rate=16000 \
  dataset.channels=1 \
  dataset.segment_duration=1 \
  optim.epochs=100 \
  optim.updates_per_epoch=2000 \
  optim.lr=1e-4 \
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
