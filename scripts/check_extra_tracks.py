#!/usr/bin/env python3
"""Check how many extra FMA tracks exist per artist beyond what was used for training."""
import pandas as pd
from pathlib import Path

# Load selected tracks (training set)
sel = pd.read_csv("/scratch/$USER/runs/2026-03-10_full/manifests/sampling_manifest.csv")
selected_aids = sorted(sel["artist_id"].unique())
selected_tids = set(sel["track_id"].unique())
print(f"Selected: {len(selected_aids)} artists, {len(selected_tids)} tracks")

# Load full FMA metadata
tracks_path = Path("/scratch/$USER/fma_metadata/fma_metadata/tracks.csv")
if not tracks_path.exists():
    tracks_path = Path("/scratch/$USER/fma_metadata/tracks.csv")
if not tracks_path.exists():
    print("FMA metadata NOT FOUND")
    import os, sys
    for p in ["/scratch/$USER/fma_metadata", "/scratch/$USER/"]:
        if os.path.exists(p):
            print(f"  {p}: {os.listdir(p)[:20]}")
    sys.exit(1)

tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
tracks.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in tracks.columns]

total_extra = 0
artists_with_extra = 0
header = f"{'Artist':<25} {'Used':>5} {'Total':>6} {'Extra':>6}"
print(f"\n{header}")
print("-" * 45)
for aid in selected_aids:
    total = len(tracks[tracks["artist_id"] == aid])
    used = len(sel[sel["artist_id"] == aid])
    extra = total - used
    name = sel[sel["artist_id"] == aid].iloc[0]["artist_name"] if used > 0 else str(aid)
    print(f"{str(name)[:25]:<25} {used:>5} {total:>6} {extra:>6}")
    total_extra += extra
    if extra > 0:
        artists_with_extra += 1

print(f"\nArtists with extra tracks: {artists_with_extra}/50")
print(f"Total extra tracks: {total_extra}")
n_ge3 = sum(1 for aid in selected_aids
            if len(tracks[tracks["artist_id"] == aid]) - len(sel[sel["artist_id"] == aid]) >= 3)
print(f"Artists with >=3 extra: {n_ge3}")
n_ge5 = sum(1 for aid in selected_aids
            if len(tracks[tracks["artist_id"] == aid]) - len(sel[sel["artist_id"] == aid]) >= 5)
print(f"Artists with >=5 extra: {n_ge5}")
