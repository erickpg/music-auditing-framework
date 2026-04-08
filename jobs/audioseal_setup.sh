#!/bin/bash
#SBATCH --job-name=as_setup
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/$USER/as_setup_%j.out
#SBATCH --error=/home/$USER/as_setup_%j.err

set -e
echo "=== Step 1: Extract training data from backup ==="
cd /scratch/$USER
tar xzf /home/$USER/scratch_backup_2026-03-20.tar.gz \
  $USER/runs/2026-03-11_audioseal_train/ \
  --strip-components=1 2>&1 | tail -5
echo "Train data files:"
ls /scratch/$USER/runs/2026-03-11_audioseal_train/masters_clean/ 2>/dev/null | wc -l

echo "=== Step 2: Create fresh env ==="
source $HOME/miniforge3/bin/activate base
if [ ! -d /scratch/$USER/audioseal_env ]; then
  conda create -y -p /scratch/$USER/audioseal_env python=3.11 2>&1 | tail -3
fi
source $HOME/miniforge3/bin/activate /scratch/$USER/audioseal_env

python3 -m pip install --upgrade pip 2>&1 | tail -3
python3 -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
python3 -m pip install "numpy<2.0" 2>&1 | tail -3
python3 -m pip install audioseal==0.1.4 2>&1 | tail -3
python3 -m pip install dora-search hydra-core omegaconf soundfile librosa pesq flashy 2>&1 | tail -3
conda install -y av ffmpeg -c conda-forge 2>&1 | tail -3

echo "=== Step 3: Clone audiocraft ==="
cd /scratch/$USER
if [ ! -d audiocraft_fresh ]; then
  git clone https://github.com/facebookresearch/audiocraft.git audiocraft_fresh 2>&1 | tail -3
fi
cd audiocraft_fresh
python3 -m pip install -e . 2>&1 | tail -5

echo "=== Step 4: Verify ==="
python3 -c "
import torch
import audioseal
import audiocraft
print('torch:', torch.__version__)
print('audioseal:', audioseal.__version__)
print('CUDA available:', torch.cuda.is_available())
"

echo "SETUP COMPLETE"
