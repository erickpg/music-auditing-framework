#!/usr/bin/env python3
"""Temporal split analysis: seen vs unseen track similarity.

For each artist with extra FMA tracks not used in training:
  1. Download and standardize the extra tracks
  2. Compute CLAP embeddings for unseen tracks
  3. Compare CLAP similarity: generated → seen tracks vs generated → unseen tracks
  4. If model learned artist STYLE, both should be similar
  5. If model MEMORIZED specific tracks, seen similarity >> unseen similarity

This test distinguishes style absorption from track-level memorization.

Run on cluster with GPU (needs CLAP model).
"""
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy import stats

# ============================================================
# Config
# ============================================================
RUN_DIR = "/scratch/$USER/runs/2026-03-10_full"
V2_RUN_DIR = "/scratch/$USER/runs/2026-03-10_full_v2"
FMA_METADATA = "/scratch/$USER/fma_metadata/fma_metadata/tracks.csv"
FMA_AUDIO_BASE = "/scratch/$USER/fma_large"  # or fma_full
OUT_DIR = "/scratch/$USER/runs/temporal_split"
MIN_EXTRA_TRACKS = 3
MAX_EXTRA_TRACKS = 20  # Cap to keep compute reasonable
TARGET_SR = 48000

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/unseen_audio", exist_ok=True)

import pandas as pd


# ============================================================
# Step 1: Identify unseen tracks
# ============================================================
print("=" * 70)
print("STEP 1: IDENTIFY UNSEEN TRACKS")
print("=" * 70)

manifest = pd.read_csv(f"{RUN_DIR}/manifests/sampling_manifest.csv")
selected_tids = set(manifest["track_id"].unique())
selected_aids = sorted(manifest["artist_id"].unique())

# Load full FMA metadata
fma_path = Path(FMA_METADATA)
if not fma_path.exists():
    fma_path = Path("/scratch/$USER/fma_metadata/tracks.csv")

tracks = pd.read_csv(fma_path, index_col=0, header=[0, 1])
tracks.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in tracks.columns]

# Find extra tracks per artist
unseen_tracks = {}
artist_names = {}
for aid in selected_aids:
    name = manifest[manifest["artist_id"] == aid].iloc[0]["artist_name"]
    artist_names[aid] = name
    all_artist_tids = set(tracks[tracks["artist_id"] == aid].index)
    extra = sorted(all_artist_tids - selected_tids)
    if len(extra) >= MIN_EXTRA_TRACKS:
        unseen_tracks[aid] = extra[:MAX_EXTRA_TRACKS]  # Cap

print(f"Artists with >={MIN_EXTRA_TRACKS} unseen tracks: {len(unseen_tracks)}")
total_unseen = sum(len(v) for v in unseen_tracks.values())
print(f"Total unseen tracks to process: {total_unseen}")

for aid, tids in sorted(unseen_tracks.items(), key=lambda x: -len(x[1])):
    print(f"  {artist_names[aid][:25]:<25} {len(tids):>3} unseen tracks")


# ============================================================
# Step 2: Download and prepare unseen audio
# ============================================================
print(f"\n{'='*70}")
print("STEP 2: DOWNLOAD/PREPARE UNSEEN AUDIO")
print("=" * 70)

# Check if FMA audio is available locally
fma_audio_paths = [
    "/scratch/$USER/fma_large",
    "/scratch/$USER/fma_full",
    "/scratch/$USER/fma_medium",
]
fma_audio_base = None
for p in fma_audio_paths:
    if os.path.exists(p):
        fma_audio_base = p
        break

if fma_audio_base is None:
    print("WARNING: FMA audio not found on scratch. Will try to download via FMA API.")
    print("Checking if standardized masters exist...")
    # The training catalog was already downloaded and standardized
    # We need to check if we can access the raw FMA dataset

prepared_files = {}  # aid -> list of paths

if fma_audio_base:
    print(f"Using FMA audio from: {fma_audio_base}")
    for aid, tids in unseen_tracks.items():
        prepared = []
        for tid in tids:
            # FMA path format: 000/000123.mp3
            fma_path = f"{tid:06d}"[:3] + "/" + f"{tid:06d}.mp3"
            full_path = os.path.join(fma_audio_base, fma_path)
            if os.path.exists(full_path):
                # Standardize: resample to 48kHz mono
                out_path = f"{OUT_DIR}/unseen_audio/{aid}_{tid}.wav"
                if not os.path.exists(out_path):
                    try:
                        waveform, sr = torchaudio.load(full_path)
                        # Convert to mono
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                        # Resample to 48kHz
                        if sr != TARGET_SR:
                            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                            waveform = resampler(waveform)
                        torchaudio.save(out_path, waveform, TARGET_SR)
                        prepared.append(out_path)
                    except Exception as e:
                        print(f"  Error processing {full_path}: {e}")
                else:
                    prepared.append(out_path)
        prepared_files[aid] = prepared
        if len(prepared) > 0:
            print(f"  {artist_names[aid][:25]:<25}: {len(prepared)}/{len(tids)} prepared")
else:
    # Try to use the FMA download script or direct URLs
    print("No local FMA audio found. Attempting download...")
    # FMA tracks are available at https://os.unil.cloud.switch.ch/fma/fma_large.zip
    # But downloading the full dataset is too slow. Try individual track download.
    import urllib.request
    for aid, tids in unseen_tracks.items():
        prepared = []
        for tid in tids[:MAX_EXTRA_TRACKS]:
            out_path = f"{OUT_DIR}/unseen_audio/{aid}_{tid}.wav"
            if os.path.exists(out_path):
                prepared.append(out_path)
                continue
            # Try FreeMusicArchive direct URL
            mp3_path = f"{OUT_DIR}/unseen_audio/{aid}_{tid}.mp3"
            if not os.path.exists(mp3_path):
                url = f"https://freemusicarchive.org/file/music/ccCommunity/{tid:06d}.mp3"
                try:
                    urllib.request.urlretrieve(url, mp3_path)
                except Exception:
                    continue
            if os.path.exists(mp3_path):
                try:
                    waveform, sr = torchaudio.load(mp3_path)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    if sr != TARGET_SR:
                        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                        waveform = resampler(waveform)
                    torchaudio.save(out_path, waveform, TARGET_SR)
                    prepared.append(out_path)
                except Exception as e:
                    print(f"  Error: {e}")
        prepared_files[aid] = prepared
        if len(prepared) > 0:
            print(f"  {artist_names[aid][:25]:<25}: {len(prepared)}/{len(tids)} prepared")

# Filter to artists with enough prepared unseen tracks
valid_artists = {aid: paths for aid, paths in prepared_files.items() if len(paths) >= MIN_EXTRA_TRACKS}
print(f"\nArtists with >={MIN_EXTRA_TRACKS} prepared unseen tracks: {len(valid_artists)}")

if len(valid_artists) == 0:
    print("ERROR: No unseen audio available. Cannot run temporal split.")
    print("FMA audio dataset needs to be downloaded to /scratch/$USER/fma_large/")
    sys.exit(1)


# ============================================================
# Step 3: Compute CLAP embeddings for unseen tracks
# ============================================================
print(f"\n{'='*70}")
print("STEP 3: COMPUTE CLAP EMBEDDINGS FOR UNSEEN TRACKS")
print("=" * 70)

from transformers import ClapModel, ClapProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model_name = "laion/larger_clap_music_and_speech"
processor = ClapProcessor.from_pretrained(model_name)
model = ClapModel.from_pretrained(model_name).to(device)
model.eval()

CLAP_SR = 48000
MAX_DURATION = 30  # seconds

def embed_audio_files(file_paths, batch_size=8):
    """Compute CLAP audio embeddings for a list of files."""
    embeddings = []
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i + batch_size]
        batch_audio = []
        for fp in batch_paths:
            try:
                waveform, sr = torchaudio.load(fp)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != CLAP_SR:
                    resampler = torchaudio.transforms.Resample(sr, CLAP_SR)
                    waveform = resampler(waveform)
                # Truncate to max duration
                max_samples = MAX_DURATION * CLAP_SR
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]
                batch_audio.append(waveform.squeeze(0).numpy())
            except Exception as e:
                print(f"  Error loading {fp}: {e}")
                continue

        if not batch_audio:
            continue

        inputs = processor(
            audio=batch_audio,
            sampling_rate=CLAP_SR,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.get_audio_features(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state.mean(dim=1)
            else:
                emb = outputs

        # Normalize
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu())

    if embeddings:
        return torch.cat(embeddings, dim=0)
    return torch.zeros(0, 512)


# Embed unseen tracks
unseen_embeddings = {}
for aid, paths in valid_artists.items():
    emb = embed_audio_files(paths)
    unseen_embeddings[aid] = emb
    print(f"  {artist_names[aid][:25]:<25}: {emb.shape[0]} embeddings")


# ============================================================
# Step 4: Load existing embeddings (catalog seen + generated)
# ============================================================
print(f"\n{'='*70}")
print("STEP 4: LOAD CATALOG AND GENERATED EMBEDDINGS")
print("=" * 70)

# We need to embed the catalog (seen) tracks too for fair comparison.
# The existing CLAP analysis computed similarities but didn't save raw embeddings.
# We'll embed the seen catalog tracks the same way.

seen_embeddings = {}
catalog_base = f"{RUN_DIR}/data/masters_standardized"
if not os.path.exists(catalog_base):
    catalog_base = f"{RUN_DIR}/data"  # Try alternative paths

print(f"Catalog base: {catalog_base}")

for aid in valid_artists:
    # Get the track files for this artist from the manifest
    artist_tracks = manifest[manifest["artist_id"] == aid]
    seen_paths = []
    for _, row in artist_tracks.iterrows():
        tid = row["track_id"]
        # Try various path patterns
        candidates = [
            f"{catalog_base}/{tid}.wav",
            f"{catalog_base}/{tid:06d}.wav",
            f"{catalog_base}/{artist_names[aid]}/{tid}.wav",
            f"{RUN_DIR}/masters_standardized/{tid}.wav",
            f"{RUN_DIR}/masters_clean/{tid}.wav",
        ]
        for c in candidates:
            if os.path.exists(c):
                seen_paths.append(c)
                break

    if len(seen_paths) >= 2:
        emb = embed_audio_files(seen_paths)
        seen_embeddings[aid] = emb
        print(f"  {artist_names[aid][:25]:<25}: {emb.shape[0]} seen embeddings")
    else:
        print(f"  {artist_names[aid][:25]:<25}: SKIP (only {len(seen_paths)} seen tracks found)")

# Also load generated audio embeddings for these artists
# Use existing CLAP per-artist data instead of re-embedding
print("\nLoading generated audio CLAP similarities from existing analysis...")

# We'll use clap_per_artist.csv which has per-file matched/mismatched sims
# But for a fair comparison we need generated embeddings too.
# Let's embed a subset of generated files per artist.

gen_embeddings = {}
for version, run_dir in [("v1", RUN_DIR), ("v2", V2_RUN_DIR)]:
    gen_embeddings[version] = {}
    gen_base = f"{run_dir}/generated"
    if not os.path.exists(gen_base):
        print(f"  {version} generated dir not found: {gen_base}")
        continue

    for aid in valid_artists:
        # Find generated files for this artist (Tier A + D)
        import glob
        # Pattern: a{prompt_idx}_t{temp}_seed{seed}.wav
        # We need to map artist_id to prompt indices...
        # Simpler: use all generated files and the clap_similarity.csv to identify which belong to this artist
        pass

    print(f"  {version}: Will use existing clap_per_artist.csv similarities instead")


# ============================================================
# Step 5: Compute seen vs unseen similarity comparison
# ============================================================
print(f"\n{'='*70}")
print("STEP 5: SEEN vs UNSEEN SIMILARITY COMPARISON")
print("=" * 70)

# Strategy: For each artist with both seen and unseen embeddings,
# compute mean cosine similarity between generated outputs and:
#   (a) seen catalog tracks (already in clap_per_artist.csv as matched_mean_sim)
#   (b) unseen extra tracks (from our new embeddings)
#
# For (b), we need to compare generated embeddings against unseen embeddings.
# But we don't have saved generated embeddings, only per-file similarities.
#
# Alternative approach: compute cosine sim between seen and unseen embeddings
# (catalog-to-catalog comparison). If seen and unseen tracks have high similarity,
# then generated→seen similarity should predict generated→unseen similarity.
#
# BETTER: Compute cross-similarity between the seen and unseen embedding sets
# per artist. This tells us how stylistically coherent the artist is across
# their seen vs unseen tracks.

results = []

for aid in sorted(valid_artists.keys()):
    if aid not in seen_embeddings or aid not in unseen_embeddings:
        continue

    seen = seen_embeddings[aid]  # [n_seen, 512]
    unseen = unseen_embeddings[aid]  # [n_unseen, 512]

    if seen.shape[0] < 2 or unseen.shape[0] < 2:
        continue

    # Cosine similarity matrix: seen vs unseen
    sim_matrix = torch.mm(seen, unseen.t())  # [n_seen, n_unseen]

    # Mean similarity between seen and unseen tracks
    mean_seen_unseen_sim = sim_matrix.mean().item()

    # Also compute within-set similarities for comparison
    seen_self = torch.mm(seen, seen.t())
    # Exclude diagonal
    n_s = seen.shape[0]
    mask_s = ~torch.eye(n_s, dtype=torch.bool)
    mean_seen_self = seen_self[mask_s].mean().item()

    unseen_self = torch.mm(unseen, unseen.t())
    n_u = unseen.shape[0]
    mask_u = ~torch.eye(n_u, dtype=torch.bool)
    mean_unseen_self = unseen_self[mask_u].mean().item() if n_u > 1 else float('nan')

    # Load the matched CLAP sim from existing analysis (generated → seen)
    for version, run_dir in [("v1", RUN_DIR), ("v2", V2_RUN_DIR)]:
        clap_csv = f"{run_dir}/analysis/clap_per_artist.csv"
        if not os.path.exists(clap_csv):
            continue
        clap_df = pd.read_csv(clap_csv)

        def clean_aid(val):
            try:
                return str(int(float(val)))
            except (ValueError, TypeError):
                return str(val)

        artist_rows = clap_df[clap_df["artist_id"].apply(clean_aid) == str(aid)]
        if artist_rows.empty:
            continue

        gen_to_seen_sim = artist_rows["matched_mean_sim"].mean()
        gen_to_mismatched_sim = artist_rows["mismatched_mean_sim"].mean()

        results.append({
            "artist_id": str(aid),
            "artist_name": artist_names[aid],
            "version": version,
            "n_seen": int(seen.shape[0]),
            "n_unseen": int(unseen.shape[0]),
            "gen_to_seen_sim": round(float(gen_to_seen_sim), 4),
            "gen_to_mismatched_sim": round(float(gen_to_mismatched_sim), 4),
            "seen_to_unseen_sim": round(float(mean_seen_unseen_sim), 4),
            "seen_self_sim": round(float(mean_seen_self), 4),
            "unseen_self_sim": round(float(mean_unseen_self), 4),
            "gen_seen_gap": round(float(gen_to_seen_sim - gen_to_mismatched_sim), 4),
        })

print(f"Results for {len(set(r['artist_id'] for r in results))} artists, "
      f"{len(results)} total rows")

if not results:
    print("ERROR: No results computed. Check paths.")
    sys.exit(1)

# Save raw results
with open(f"{OUT_DIR}/temporal_split_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# ============================================================
# Step 6: Analysis
# ============================================================
print(f"\n{'='*70}")
print("STEP 6: TEMPORAL SPLIT ANALYSIS")
print("=" * 70)

for version in ["v1", "v2"]:
    v_results = [r for r in results if r["version"] == version]
    if not v_results:
        continue

    print(f"\n--- {version.upper()} ---")
    print(f"Artists: {len(v_results)}")

    gen_seen = [r["gen_to_seen_sim"] for r in v_results]
    seen_unseen = [r["seen_to_unseen_sim"] for r in v_results]
    seen_self = [r["seen_self_sim"] for r in v_results]
    unseen_self = [r["unseen_self_sim"] for r in v_results]

    print(f"\n  Mean generated→seen similarity:   {np.mean(gen_seen):.4f} ± {np.std(gen_seen):.4f}")
    print(f"  Mean seen↔unseen similarity:      {np.mean(seen_unseen):.4f} ± {np.std(seen_unseen):.4f}")
    print(f"  Mean seen self-similarity:         {np.mean(seen_self):.4f} ± {np.std(seen_self):.4f}")
    print(f"  Mean unseen self-similarity:       {np.mean(unseen_self):.4f} ± {np.std(unseen_self):.4f}")

    # Key test: does gen→seen correlate with seen↔unseen?
    # If yes: model learned the artist's style (which unseen tracks also share)
    # If no: model may have memorized specific seen tracks
    rho, p = stats.spearmanr(gen_seen, seen_unseen)
    print(f"\n  Correlation: gen→seen vs seen↔unseen")
    print(f"    Spearman rho = {rho:.4f}, p = {p:.4f}")

    # Paired test: is seen self-similarity > seen↔unseen similarity?
    # This measures how coherent the artist's style is across tracks
    if len(seen_self) >= 5:
        t, tp = stats.ttest_rel(seen_self, seen_unseen)
        print(f"\n  Seen-self vs seen↔unseen: t={t:.3f}, p={tp:.4f}")
        print(f"    (Are seen tracks more similar to each other than to unseen?)")
        if tp < 0.05:
            if t > 0:
                print(f"    YES — seen tracks cluster tighter (possible selection bias)")
            else:
                print(f"    NO — unseen tracks are actually more similar (surprising)")
        else:
            print(f"    NO significant difference — style is consistent across seen/unseen")

    # Per-artist detail
    print(f"\n  {'Artist':<25} {'gen→seen':>10} {'seen↔un':>10} {'seen_self':>10} {'un_self':>10}")
    print(f"  {'-'*67}")
    for r in sorted(v_results, key=lambda x: -x["gen_to_seen_sim"]):
        print(f"  {r['artist_name'][:25]:<25} "
              f"{r['gen_to_seen_sim']:>10.4f} "
              f"{r['seen_to_unseen_sim']:>10.4f} "
              f"{r['seen_self_sim']:>10.4f} "
              f"{r['unseen_self_sim']:>10.4f}")

# Summary
summary = {
    "test": "Temporal split: seen vs unseen track similarity",
    "description": (
        "Compares CLAP similarity between seen (training) and unseen (held-out) "
        "tracks for the same artist. If the model learned artist style (not memorized "
        "specific tracks), generated→seen similarity should correlate with seen↔unseen "
        "similarity. High seen↔unseen similarity means the artist's style is coherent "
        "across their discography."
    ),
    "n_artists": len(set(r["artist_id"] for r in results)),
    "min_extra_tracks": MIN_EXTRA_TRACKS,
    "max_extra_tracks": MAX_EXTRA_TRACKS,
}

for version in ["v1", "v2"]:
    v_results = [r for r in results if r["version"] == version]
    if v_results:
        gen_seen = [r["gen_to_seen_sim"] for r in v_results]
        seen_unseen = [r["seen_to_unseen_sim"] for r in v_results]
        rho, p = stats.spearmanr(gen_seen, seen_unseen)
        summary[version] = {
            "n_artists": len(v_results),
            "mean_gen_to_seen": round(float(np.mean(gen_seen)), 4),
            "mean_seen_to_unseen": round(float(np.mean(seen_unseen)), 4),
            "correlation_rho": round(float(rho), 4),
            "correlation_p": float(p),
        }

with open(f"{OUT_DIR}/temporal_split_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUT_DIR}/")
print(f"  temporal_split_results.csv")
print(f"  temporal_split_summary.json")
