#!/bin/bash
#SBATCH --job-name=setup_ac
#SBATCH --partition=interactive
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/$USER/capstone/jobs/logs/setup_ac_%j.out
#SBATCH --error=/home/$USER/capstone/jobs/logs/setup_ac_%j.err

# Install audiocraft with watermarking support and prepare training data manifest

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env

SCRATCH=/scratch/$USER
AC_DIR=$SCRATCH/audiocraft
TRAIN_DATA=$SCRATCH/runs/2026-03-11_audioseal_train/masters_clean
VALID_SPLIT=0.1

set -e

# --- 1. Clone and install audiocraft ---
if [ ! -d "$AC_DIR" ]; then
    echo "=== Cloning audiocraft ==="
    cd $SCRATCH
    git clone https://github.com/facebookresearch/audiocraft.git
else
    echo "=== audiocraft already cloned ==="
fi

cd $AC_DIR
echo "=== Installing audiocraft with watermark extras ==="
pip install -e '.[wm]' 2>&1 | tail -5

# --- 2. Generate JSONL manifest from training data ---
echo "=== Generating audio manifest ==="
mkdir -p $AC_DIR/egs/capstone_train
mkdir -p $AC_DIR/egs/capstone_valid

# Generate full manifest
python -m audiocraft.data.audio_dataset $TRAIN_DATA $AC_DIR/egs/capstone_full/data.jsonl.gz

# Split into train/valid (90/10)
echo "=== Splitting train/valid ==="
python - <<'PYEOF'
import gzip, json, random

random.seed(42)

# Read full manifest
with gzip.open("/scratch/$USER/audiocraft/egs/capstone_full/data.jsonl.gz", "rt") as f:
    lines = [json.loads(line) for line in f]

random.shuffle(lines)
n_valid = max(1, int(len(lines) * 0.1))
valid = lines[:n_valid]
train = lines[n_valid:]

print(f"Total: {len(lines)}, Train: {len(train)}, Valid: {len(valid)}")

import os
os.makedirs("/scratch/$USER/audiocraft/egs/capstone_train", exist_ok=True)
os.makedirs("/scratch/$USER/audiocraft/egs/capstone_valid", exist_ok=True)

with gzip.open("/scratch/$USER/audiocraft/egs/capstone_train/data.jsonl.gz", "wt") as f:
    for item in train:
        f.write(json.dumps(item) + "\n")

with gzip.open("/scratch/$USER/audiocraft/egs/capstone_valid/data.jsonl.gz", "wt") as f:
    for item in valid:
        f.write(json.dumps(item) + "\n")

train_dur = sum(x["duration"] for x in train) / 3600
valid_dur = sum(x["duration"] for x in valid) / 3600
print(f"Train duration: {train_dur:.1f}h, Valid duration: {valid_dur:.1f}h")
PYEOF

# --- 3. Create dataset config ---
echo "=== Creating dataset config ==="
mkdir -p $AC_DIR/config/dset/audio
cat > $AC_DIR/config/dset/audio/capstone_music.yaml << 'YAML'
# @package __global__
datasource:
  max_sample_rate: 16000
  max_channels: 1
  train: egs/capstone_train
  valid: egs/capstone_valid
  evaluate: egs/capstone_valid
  generate: egs/capstone_valid
YAML

# --- 4. Create team config for dora ---
echo "=== Creating dora team config ==="
cat > $AC_DIR/capstone_team.yaml << 'YAML'
default:
  dora_dir: /scratch/$USER/dora_outputs
  partitions:
    global: gpu
    team: gpu
  reference_dir: /scratch/$USER/dora_ref
YAML

# --- 5. Create custom augmentation config (EnCodec 50/50) ---
echo "=== Creating EnCodec augmentation config ==="
mkdir -p $AC_DIR/config/augmentations
cat > $AC_DIR/config/augmentations/encodec_only.yaml << 'YAML'
# @package __global__
# 50/50 EnCodec augmentation for codec survival training
# Following San Roman et al. (2024) Latent Watermarking approach
aug_weights:
  speed: 0.0
  updownresample: 0.0
  echo: 0.0
  pink_noise: 0.0
  lowpass_filter: 0.0
  highpass_filter: 0.0
  bandpass_filter: 0.0
  smooth: 0.0
  boost_audio: 0.0
  duck_audio: 0.0
  mp3_compression: 0.0
  aac_compression: 0.0
  encodec: 1.0
  identity: 1.0
YAML

echo "=== Setup complete ==="
echo "audiocraft installed at: $AC_DIR"
echo "Train manifest: $AC_DIR/egs/capstone_train/data.jsonl.gz"
echo "Valid manifest: $AC_DIR/egs/capstone_valid/data.jsonl.gz"
echo "Dataset config: $AC_DIR/config/dset/audio/capstone_music.yaml"
echo "Team config: $AC_DIR/capstone_team.yaml"
