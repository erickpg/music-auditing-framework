#!/usr/bin/env python3
"""Step 3: CLAP temporal split analysis — seen vs unseen track similarity.

For each artist with ≥3 unseen held-out tracks:
  1. Embed seen (training catalog) tracks with CLAP
  2. Embed unseen (held-out) tracks with CLAP
  3. Compare: seen↔unseen similarity vs seen self-similarity
  4. Load existing matched/mismatched similarities from clap_per_artist.csv
  5. Test: if model learned style (not memorized), gen→seen ≈ gen→unseen proxy

Since generated WAVs may be purged from scratch, we use the existing
clap_per_artist.csv matched/mismatched values as the "generated→catalog"
signal and compare against the fresh seen↔unseen embeddings.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from scipy import stats

# Force soundfile backend
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

RUN_DIR = "/scratch/$USER/runs/2026-03-10_full"
V2_RUN_DIR = "/scratch/$USER/runs/2026-03-10_full_v2"
OUT_DIR = "/scratch/$USER/runs/temporal_split"
CLAP_SR = 48000
MAX_DURATION = 30
BATCH_SIZE = 4  # smaller batch to avoid OOM with CLAP

# ============================================================
# Load CLAP model
# ============================================================
from transformers import ClapModel, ClapProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model_name = "laion/larger_clap_music_and_speech"
processor = ClapProcessor.from_pretrained(model_name)
model = ClapModel.from_pretrained(model_name).to(device)
model.eval()


def embed_audio_files(file_paths):
    """Compute CLAP audio embeddings. Returns [N, D] tensor and valid paths."""
    all_embeddings = []
    valid_paths = []

    for i in range(0, len(file_paths), BATCH_SIZE):
        batch_paths = file_paths[i:i + BATCH_SIZE]
        batch_audio = []
        batch_valid = []

        for fp in batch_paths:
            try:
                audio_data, sr = sf.read(fp, dtype="float32")
                waveform = torch.from_numpy(audio_data)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.t()
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != CLAP_SR:
                    waveform = torchaudio.transforms.Resample(sr, CLAP_SR)(waveform)
                max_samples = MAX_DURATION * CLAP_SR
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]
                if waveform.shape[1] < CLAP_SR:
                    continue
                batch_audio.append(waveform.squeeze(0).numpy())
                batch_valid.append(fp)
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
            raw = model.get_audio_features(**inputs)
            # Handle both tensor and BaseModelOutputWithPooling
            if hasattr(raw, 'pooler_output') and raw.pooler_output is not None:
                emb = raw.pooler_output
            elif hasattr(raw, 'last_hidden_state'):
                emb = raw.last_hidden_state.mean(dim=1)
            elif isinstance(raw, torch.Tensor):
                emb = raw
            else:
                # Try to extract tensor from the object
                emb = raw[0] if isinstance(raw, (tuple, list)) else raw.last_hidden_state.mean(dim=1)

        # Debug shape
        print(f"    CLAP output shape: {emb.shape}, dim: {emb.dim()}")

        # Force to 2D [batch, embed_dim]
        if emb.dim() == 3:
            # Could be [B, seq, D] — take mean over sequence dim
            emb = emb.mean(dim=1)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        assert emb.dim() == 2, f"Expected 2D embedding, got {emb.shape}"

        # L2 normalize
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu())
        valid_paths.extend(batch_valid)

    if all_embeddings:
        return torch.cat(all_embeddings, dim=0), valid_paths
    return torch.zeros(0, 512), []


def clean_aid(val):
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val)


# ============================================================
# Load manifests
# ============================================================
unseen_manifest = pd.read_csv(f"{OUT_DIR}/unseen_manifest.csv")
catalog_manifest = pd.read_csv(f"{RUN_DIR}/manifests/sampling_manifest.csv")

artist_names = {}
for _, row in catalog_manifest.iterrows():
    artist_names[str(row["artist_id"])] = row["artist_name"]

# Check available unseen standardized tracks
unseen_dir = f"{OUT_DIR}/unseen_standardized"
available_unseen = set()
if os.path.exists(unseen_dir):
    for f in os.listdir(unseen_dir):
        if f.endswith(".wav"):
            try:
                available_unseen.add(int(f.replace(".wav", "")))
            except ValueError:
                pass

print(f"Available unseen standardized tracks: {len(available_unseen)}")

# Group unseen by artist
unseen_by_artist = defaultdict(list)
for _, row in unseen_manifest.iterrows():
    if row["track_id"] in available_unseen:
        unseen_by_artist[str(row["artist_id"])].append(
            f"{unseen_dir}/{row['track_id']}.wav"
        )

valid_artists = {aid: paths for aid, paths in unseen_by_artist.items() if len(paths) >= 3}
print(f"Artists with >=3 unseen tracks: {len(valid_artists)}")

if len(valid_artists) == 0:
    print("ERROR: No artists with enough unseen tracks.")
    sys.exit(1)

# ============================================================
# Embed unseen tracks
# ============================================================
print(f"\n{'='*60}")
print("EMBEDDING UNSEEN TRACKS")
print("=" * 60)

unseen_embeddings = {}
for aid, paths in sorted(valid_artists.items()):
    print(f"  Embedding {artist_names.get(aid, aid)[:25]} ({len(paths)} files)...")
    emb, valid = embed_audio_files(paths)
    if emb.shape[0] >= 3:
        unseen_embeddings[aid] = emb
        print(f"    -> {emb.shape[0]} embeddings, shape {emb.shape}")
    else:
        print(f"    -> SKIP (only {emb.shape[0]} valid)")

print(f"\nTotal artists with unseen embeddings: {len(unseen_embeddings)}")

# ============================================================
# Embed seen (catalog) tracks
# ============================================================
print(f"\n{'='*60}")
print("EMBEDDING SEEN (CATALOG) TRACKS")
print("=" * 60)

# Possible locations for catalog audio
catalog_dirs = [
    f"{RUN_DIR}/masters_standardized",
    f"{RUN_DIR}/masters_clean",
    f"{RUN_DIR}/data",
]

seen_embeddings = {}
for aid in sorted(unseen_embeddings.keys()):
    artist_tracks = catalog_manifest[
        catalog_manifest["artist_id"].apply(lambda x: str(int(float(x)))) == aid
    ]
    seen_paths = []
    for _, row in artist_tracks.iterrows():
        tid = row["track_id"]
        found = False
        for d in catalog_dirs:
            if not os.path.exists(d):
                continue
            for ext in [".wav", ".mp3"]:
                for fmt in [str(tid), f"{int(tid):06d}"]:
                    p = os.path.join(d, f"{fmt}{ext}")
                    if os.path.exists(p):
                        seen_paths.append(p)
                        found = True
                        break
                if found:
                    break
            if found:
                break

    if len(seen_paths) >= 2:
        print(f"  Embedding {artist_names.get(aid, aid)[:25]} ({len(seen_paths)} files)...")
        emb, valid = embed_audio_files(seen_paths)
        if emb.shape[0] >= 2:
            seen_embeddings[aid] = emb
            print(f"    -> {emb.shape[0]} embeddings, shape {emb.shape}")
        else:
            print(f"    -> SKIP (only {emb.shape[0]} valid)")
    else:
        print(f"  {artist_names.get(aid, aid)[:25]}: SKIP (only {len(seen_paths)} files found)")

print(f"\nTotal artists with seen embeddings: {len(seen_embeddings)}")

# ============================================================
# Load existing CLAP similarities from clap_per_artist.csv
# ============================================================
print(f"\n{'='*60}")
print("LOADING EXISTING CLAP SIMILARITIES")
print("=" * 60)

existing_sims = {}  # version -> aid -> {matched, mismatched}
for version, run_dir in [("v1", RUN_DIR), ("v2", V2_RUN_DIR)]:
    existing_sims[version] = {}
    clap_csv = f"{run_dir}/analysis/clap_per_artist.csv"
    if not os.path.exists(clap_csv):
        print(f"  {version}: clap_per_artist.csv not found at {clap_csv}")
        continue
    clap_df = pd.read_csv(clap_csv)
    print(f"  {version}: loaded {len(clap_df)} rows, columns: {list(clap_df.columns)}")

    for aid in unseen_embeddings:
        rows = clap_df[clap_df["artist_id"].apply(lambda x: str(int(float(x)))) == aid]
        if not rows.empty:
            existing_sims[version][aid] = {
                "matched": rows["matched_mean_sim"].mean(),
                "mismatched": rows["mismatched_mean_sim"].mean(),
            }
    print(f"  {version}: {len(existing_sims[version])} artists matched")

# ============================================================
# Compute temporal split metrics
# ============================================================
print(f"\n{'='*60}")
print("COMPUTING TEMPORAL SPLIT METRICS")
print("=" * 60)

results = []

# Common artists with both seen and unseen embeddings
common_aids = sorted(set(seen_embeddings.keys()) & set(unseen_embeddings.keys()))
print(f"Artists with both seen and unseen: {len(common_aids)}")

for aid in common_aids:
    seen = seen_embeddings[aid]
    unseen = unseen_embeddings[aid]

    # Cosine similarities (embeddings are already L2-normalized)
    sim_seen_unseen = torch.mm(seen, unseen.t())
    mean_seen_unseen = sim_seen_unseen.mean().item()

    # Seen self-similarity (off-diagonal)
    sim_seen_self = torch.mm(seen, seen.t())
    n_s = seen.shape[0]
    mask_s = ~torch.eye(n_s, dtype=torch.bool)
    mean_seen_self = sim_seen_self[mask_s].mean().item() if n_s > 1 else float("nan")

    # Unseen self-similarity (off-diagonal)
    sim_unseen_self = torch.mm(unseen, unseen.t())
    n_u = unseen.shape[0]
    mask_u = ~torch.eye(n_u, dtype=torch.bool)
    mean_unseen_self = sim_unseen_self[mask_u].mean().item() if n_u > 1 else float("nan")

    row = {
        "artist_id": aid,
        "artist_name": artist_names.get(aid, aid),
        "n_seen": int(n_s),
        "n_unseen": int(n_u),
        "seen_self_sim": round(mean_seen_self, 4),
        "unseen_self_sim": round(mean_unseen_self, 4),
        "seen_unseen_sim": round(mean_seen_unseen, 4),
    }

    # Add existing matched/mismatched for each version
    for version in ["v1", "v2"]:
        sims = existing_sims.get(version, {}).get(aid)
        if sims:
            row[f"{version}_matched_sim"] = round(sims["matched"], 4)
            row[f"{version}_mismatched_sim"] = round(sims["mismatched"], 4)
            row[f"{version}_clap_gap"] = round(sims["matched"] - sims["mismatched"], 4)
        else:
            row[f"{version}_matched_sim"] = None
            row[f"{version}_mismatched_sim"] = None
            row[f"{version}_clap_gap"] = None

    results.append(row)

# Save CSV
csv_path = f"{OUT_DIR}/temporal_split_results.csv"
if results:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {csv_path}")

# ============================================================
# Analysis
# ============================================================
print(f"\n{'='*60}")
print("TEMPORAL SPLIT ANALYSIS")
print("=" * 60)

seen_self = [r["seen_self_sim"] for r in results]
unseen_self = [r["unseen_self_sim"] for r in results]
seen_unseen = [r["seen_unseen_sim"] for r in results]

print(f"\nArtist style coherence in CLAP space (N={len(results)}):")
print(f"  Seen self-similarity:    {np.mean(seen_self):.4f} ± {np.std(seen_self):.4f}")
print(f"  Unseen self-similarity:  {np.mean(unseen_self):.4f} ± {np.std(unseen_self):.4f}")
print(f"  Seen ↔ unseen sim:       {np.mean(seen_unseen):.4f} ± {np.std(seen_unseen):.4f}")

# Test: seen-self vs seen↔unseen — if close, artist style is cohesive across tracks
t1, p1 = stats.ttest_rel(seen_self, seen_unseen)
print(f"\n  Paired t-test (seen_self vs seen↔unseen): t={t1:.3f}, p={p1:.6f}")
if p1 < 0.05:
    print(f"  -> Seen tracks are {'more' if np.mean(seen_self) > np.mean(seen_unseen) else 'less'} "
          f"similar to each other than to unseen tracks")
else:
    print(f"  -> No significant difference: artist style is cohesive across seen/unseen")

# Test: unseen-self vs seen↔unseen
t2, p2 = stats.ttest_rel(unseen_self, seen_unseen)
print(f"  Paired t-test (unseen_self vs seen↔unseen): t={t2:.3f}, p={p2:.6f}")

# Key comparison: existing matched similarity vs seen↔unseen similarity
# If gen→seen ≈ seen↔unseen, model captures artist style broadly
# If gen→seen >> seen↔unseen, model may have memorized specific training tracks
for version in ["v1", "v2"]:
    matched = [r[f"{version}_matched_sim"] for r in results if r[f"{version}_matched_sim"] is not None]
    gaps = [r[f"{version}_clap_gap"] for r in results if r[f"{version}_clap_gap"] is not None]

    if not matched:
        continue

    # Compare matched similarity (gen→matched_artist) to seen↔unseen (catalog coherence)
    matched_aids = [r for r in results if r[f"{version}_matched_sim"] is not None]
    m_vals = [r[f"{version}_matched_sim"] for r in matched_aids]
    su_vals = [r["seen_unseen_sim"] for r in matched_aids]

    print(f"\n=== {version.upper()} ===")
    print(f"  N artists with data: {len(matched_aids)}")
    print(f"  Mean matched sim (gen→artist):    {np.mean(m_vals):.4f}")
    print(f"  Mean seen↔unseen sim (catalog):   {np.mean(su_vals):.4f}")
    print(f"  Mean CLAP gap (matched-mismatch): {np.mean(gaps):.4f}")

    # Correlation: does catalog coherence predict CLAP gap?
    rho, p_rho = stats.spearmanr(su_vals, [r[f"{version}_clap_gap"] for r in matched_aids])
    print(f"\n  Correlation (seen↔unseen vs CLAP gap): rho={rho:.4f}, p={p_rho:.4f}")
    if p_rho < 0.05:
        print(f"  -> Artists with higher catalog coherence have {'larger' if rho > 0 else 'smaller'} CLAP gaps")
    else:
        print(f"  -> No significant correlation between catalog coherence and CLAP gap")

    # Correlation: does catalog coherence predict matched similarity?
    rho2, p2 = stats.spearmanr(su_vals, m_vals)
    print(f"  Correlation (seen↔unseen vs matched sim): rho={rho2:.4f}, p={p2:.4f}")

# Per-artist detail table
print(f"\n{'='*60}")
print("PER-ARTIST DETAIL")
print("=" * 60)
header = f"  {'Artist':<25} {'seen↔un':>8} {'s_self':>7} {'u_self':>7} {'v1_gap':>7} {'v2_gap':>7}"
print(header)
print(f"  {'-'*65}")
for r in sorted(results, key=lambda x: -x["seen_unseen_sim"]):
    v1g = f"{r['v1_clap_gap']:.4f}" if r["v1_clap_gap"] is not None else "  N/A"
    v2g = f"{r['v2_clap_gap']:.4f}" if r["v2_clap_gap"] is not None else "  N/A"
    print(f"  {r['artist_name'][:25]:<25} {r['seen_unseen_sim']:>8.4f} "
          f"{r['seen_self_sim']:>7.4f} {r['unseen_self_sim']:>7.4f} {v1g:>7} {v2g:>7}")

# Summary JSON
summary = {
    "test": "Temporal split: seen vs unseen catalog track similarity",
    "hypothesis": (
        "If artist style is cohesive, seen↔unseen ≈ seen_self. "
        "If CLAP gap correlates with catalog coherence, vulnerability signal "
        "reflects genuine style, not memorization of specific tracks."
    ),
    "n_artists": len(results),
    "catalog_coherence": {
        "mean_seen_self_sim": round(float(np.mean(seen_self)), 4),
        "mean_unseen_self_sim": round(float(np.mean(unseen_self)), 4),
        "mean_seen_unseen_sim": round(float(np.mean(seen_unseen)), 4),
        "seen_self_vs_seen_unseen_t": round(float(t1), 4),
        "seen_self_vs_seen_unseen_p": float(p1),
    },
}

for version in ["v1", "v2"]:
    matched_aids = [r for r in results if r[f"{version}_matched_sim"] is not None]
    if not matched_aids:
        continue
    su_vals = [r["seen_unseen_sim"] for r in matched_aids]
    gap_vals = [r[f"{version}_clap_gap"] for r in matched_aids]
    m_vals = [r[f"{version}_matched_sim"] for r in matched_aids]
    rho, p = stats.spearmanr(su_vals, gap_vals)
    rho2, p2 = stats.spearmanr(su_vals, m_vals)
    summary[version] = {
        "n_artists": len(matched_aids),
        "mean_matched_sim": round(float(np.mean(m_vals)), 4),
        "mean_clap_gap": round(float(np.mean(gap_vals)), 4),
        "coherence_vs_gap_rho": round(float(rho), 4),
        "coherence_vs_gap_p": float(p),
        "coherence_vs_matched_rho": round(float(rho2), 4),
        "coherence_vs_matched_p": float(p2),
    }

with open(f"{OUT_DIR}/temporal_split_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUT_DIR}/")
print("Done!")
