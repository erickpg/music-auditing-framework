#!/bin/bash
#SBATCH --job-name=dl_metadata
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/home/$USER/capstone/jobs/logs/dl_meta_%j.out
#SBATCH --error=/home/$USER/capstone/jobs/logs/dl_meta_%j.err

set -e

META_DIR=/scratch/$USER/fma_metadata
mkdir -p $META_DIR
cd $META_DIR

echo "Downloading FMA metadata..."
wget -q https://os.unil.cloud.switch.ch/fma/fma_metadata.zip -O fma_metadata.zip

echo "Extracting..."
unzip -o fma_metadata.zip

echo "Contents:"
ls -la $META_DIR/
ls -la $META_DIR/fma_metadata/ 2>/dev/null || true

echo "Done."
