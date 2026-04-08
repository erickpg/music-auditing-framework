#!/usr/bin/env python3
"""Step 1: Identify unseen FMA tracks and download them via remotezip."""
import os
import sys
from pathlib import Path

import pandas as pd

RUN_DIR = "/scratch/$USER/runs/2026-03-10_full"
OUT_DIR = "/scratch/$USER/runs/temporal_split"
FMA_METADATA = "/scratch/$USER/fma_metadata/fma_metadata/tracks.csv"
FMA_URL = "https://os.unil.cloud.switch.ch/fma/fma_large.zip"
MIN_EXTRA = 3
MAX_EXTRA = 15

os.makedirs(f"{OUT_DIR}/unseen_audio", exist_ok=True)

# Load training manifest
manifest = pd.read_csv(f"{RUN_DIR}/manifests/sampling_manifest.csv")
selected_tids = set(manifest["track_id"].unique())
selected_aids = sorted(manifest["artist_id"].unique())

# Load FMA metadata
fma_path = Path(FMA_METADATA)
if not fma_path.exists():
    fma_path = Path("/scratch/$USER/fma_metadata/tracks.csv")
tracks = pd.read_csv(fma_path, index_col=0, header=[0, 1])
tracks.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in tracks.columns]

# Find unseen tracks
unseen_manifest = []
for aid in selected_aids:
    name = manifest[manifest["artist_id"] == aid].iloc[0]["artist_name"]
    all_tids = set(tracks[tracks["artist_id"] == aid].index)
    extra = sorted(all_tids - selected_tids)
    if len(extra) >= MIN_EXTRA:
        for tid in extra[:MAX_EXTRA]:
            fma_path_str = f"{tid:06d}"[:3] + "/" + f"{tid:06d}.mp3"
            unseen_manifest.append({
                "track_id": tid,
                "artist_id": aid,
                "artist_name": name,
                "fma_path": fma_path_str,
            })

unseen_df = pd.DataFrame(unseen_manifest)
unseen_df.to_csv(f"{OUT_DIR}/unseen_manifest.csv", index=False)
print(f"Unseen tracks to download: {len(unseen_df)} across {unseen_df['artist_id'].nunique()} artists")

# Check already downloaded
to_download = []
already = 0
for _, row in unseen_df.iterrows():
    out_path = f"{OUT_DIR}/unseen_audio/{row['track_id']}.mp3"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        already += 1
    else:
        to_download.append(row)

print(f"Already downloaded: {already}")
print(f"Need to download: {len(to_download)}")

if not to_download:
    print("All tracks already downloaded!")
    sys.exit(0)

try:
    from remotezip import RemoteZip
except ImportError:
    os.system("pip install remotezip")
    from remotezip import RemoteZip

print(f"Opening remote ZIP index (may take 30-60 seconds)...")

downloaded = 0
errors = 0

with RemoteZip(FMA_URL) as rz:
    zip_names = set(rz.namelist())
    print(f"ZIP index loaded: {len(zip_names)} entries")

    for row in to_download:
        fma_path_str = row["fma_path"]
        out_path = f"{OUT_DIR}/unseen_audio/{row['track_id']}.mp3"

        # Try both possible prefixes
        for prefix in ["fma_large/", "fma_full/"]:
            zip_path = f"{prefix}{fma_path_str}"
            if zip_path in zip_names:
                try:
                    data = rz.read(zip_path)
                    with open(out_path, "wb") as f:
                        f.write(data)
                    downloaded += 1
                    if downloaded % 20 == 0:
                        print(f"  Downloaded {downloaded}/{len(to_download)}...")
                    break
                except Exception as e:
                    errors += 1
                    print(f"  FAILED {fma_path_str}: {e}")
                    break
        else:
            errors += 1

print(f"\nDownload complete: {downloaded} OK, {errors} errors")
print(f"Total files: {len(os.listdir(f'{OUT_DIR}/unseen_audio'))}")
