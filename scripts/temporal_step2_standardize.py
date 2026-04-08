#!/usr/bin/env python3
"""Step 2: Standardize unseen tracks to 32kHz mono WAV."""
import os
import subprocess

import pandas as pd
import soundfile as sf
import torchaudio

try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

OUT_DIR = "/scratch/$USER/runs/temporal_split"
TARGET_SR = 32000  # Match catalog standardization

os.makedirs(f"{OUT_DIR}/unseen_standardized", exist_ok=True)

manifest = pd.read_csv(f"{OUT_DIR}/unseen_manifest.csv")
print(f"Tracks to standardize: {len(manifest)}")

success = 0
failed = 0

for _, row in manifest.iterrows():
    tid = row["track_id"]
    mp3_path = f"{OUT_DIR}/unseen_audio/{tid}.mp3"
    wav_path = f"{OUT_DIR}/unseen_standardized/{tid}.wav"

    if os.path.exists(wav_path):
        success += 1
        continue

    if not os.path.exists(mp3_path):
        failed += 1
        continue

    try:
        # Use ffmpeg for reliable decoding (handles edge cases better than torchaudio)
        cmd = [
            "ffmpeg", "-y", "-i", mp3_path,
            "-ar", str(TARGET_SR),
            "-ac", "1",
            "-sample_fmt", "s16",
            wav_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        success += 1
    except Exception as e:
        # Fallback to torchaudio
        try:
            waveform, sr = torchaudio.load(mp3_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                waveform = resampler(waveform)
            torchaudio.save(wav_path, waveform, TARGET_SR)
            success += 1
        except Exception as e2:
            print(f"  Failed {tid}: {e2}")
            failed += 1

    if (success + failed) % 50 == 0:
        print(f"  Progress: {success} success, {failed} failed")

print(f"\nStandardization: {success} success, {failed} failed")
print(f"Files in unseen_standardized: {len(os.listdir(f'{OUT_DIR}/unseen_standardized'))}")
