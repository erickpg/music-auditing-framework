#!/usr/bin/env python3
"""Step 4: Temporal split FAD — generated vs seen catalog vs unseen catalog.

For each artist:
  - FAD(generated, seen_catalog)    — distance to training tracks
  - FAD(generated, unseen_catalog)  — distance to held-out tracks
  - If model memorized: FAD_seen << FAD_unseen
  - If model learned style: FAD_seen ≈ FAD_unseen

Uses CLAP embeddings. Seen catalog embeddings are loaded from the existing
embeddings/ dir. Unseen and generated are embedded fresh with CLAP.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from scipy import linalg, stats

try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

RUN_DIR = "/scratch/$USER/runs/2026-03-10_full"
V2_RUN_DIR = "/scratch/$USER/runs/2026-03-10_full_v2"
OUT_DIR = "/scratch/$USER/runs/temporal_split"
CLAP_SR = 48000
MAX_DURATION = 30
BATCH_SIZE = 4

# ============================================================
# FAD utilities (from per_artist_fad.py)
# ============================================================
def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def gaussian_stats(embeddings, regularize=True):
    mu = np.mean(embeddings, axis=0)
    if regularize and embeddings.shape[0] < embeddings.shape[1]:
        from sklearn.covariance import LedoitWolf
        sigma = LedoitWolf().fit(embeddings).covariance_
    else:
        sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


# ============================================================
# CLAP model
# ============================================================
from transformers import ClapModel, ClapProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model_name = "laion/larger_clap_music_and_speech"
processor = ClapProcessor.from_pretrained(model_name)
clap_model = ClapModel.from_pretrained(model_name).to(device)
clap_model.eval()


def embed_audio_files(file_paths):
    """Compute CLAP embeddings. Returns [N, D] numpy array and valid paths."""
    all_embs = []
    valid_paths = []

    for i in range(0, len(file_paths), BATCH_SIZE):
        batch_paths = file_paths[i:i + BATCH_SIZE]
        batch_audio = []
        batch_valid = []

        for fp in batch_paths:
            try:
                audio, sr = sf.read(fp, dtype="float32")
                wav = torch.from_numpy(audio)
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                else:
                    wav = wav.t()
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != CLAP_SR:
                    wav = torchaudio.transforms.Resample(sr, CLAP_SR)(wav)
                if wav.shape[1] > MAX_DURATION * CLAP_SR:
                    wav = wav[:, :MAX_DURATION * CLAP_SR]
                if wav.shape[1] < CLAP_SR:
                    continue
                batch_audio.append(wav.squeeze(0).numpy())
                batch_valid.append(fp)
            except Exception as e:
                print(f"  Error loading {fp}: {e}")
                continue

        if not batch_audio:
            continue

        inputs = processor(
            audio=batch_audio, sampling_rate=CLAP_SR,
            return_tensors="pt", padding=True,
        ).to(device)

        with torch.no_grad():
            raw = clap_model.get_audio_features(**inputs)
            if hasattr(raw, 'pooler_output') and raw.pooler_output is not None:
                emb = raw.pooler_output
            elif hasattr(raw, 'last_hidden_state'):
                emb = raw.last_hidden_state.mean(dim=1)
            elif isinstance(raw, torch.Tensor):
                emb = raw
            else:
                emb = raw[0] if isinstance(raw, (tuple, list)) else raw.last_hidden_state.mean(dim=1)

        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        # L2 normalize
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy())
        valid_paths.extend(batch_valid)

    if all_embs:
        return np.concatenate(all_embs, axis=0), valid_paths
    return np.zeros((0, 512)), []


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

# ============================================================
# Unseen tracks
# ============================================================
unseen_dir = f"{OUT_DIR}/unseen_standardized"
available_unseen = set()
if os.path.exists(unseen_dir):
    for f in os.listdir(unseen_dir):
        if f.endswith(".wav"):
            try:
                available_unseen.add(int(f.replace(".wav", "")))
            except ValueError:
                pass

unseen_by_artist = defaultdict(list)
for _, row in unseen_manifest.iterrows():
    if row["track_id"] in available_unseen:
        unseen_by_artist[str(row["artist_id"])].append(
            f"{unseen_dir}/{row['track_id']}.wav"
        )

valid_artists = {aid: paths for aid, paths in unseen_by_artist.items() if len(paths) >= 3}
print(f"Artists with >=3 unseen tracks: {len(valid_artists)}")

# ============================================================
# Embed unseen tracks
# ============================================================
print(f"\n{'='*60}\nEMBEDDING UNSEEN TRACKS\n{'='*60}")
unseen_embs = {}
for aid, paths in sorted(valid_artists.items()):
    emb, valid = embed_audio_files(paths)
    if emb.shape[0] >= 3:
        unseen_embs[aid] = emb
        print(f"  {artist_names.get(aid, aid)[:25]:<25}: {emb.shape[0]} embeddings")

# ============================================================
# Load seen catalog embeddings from existing npy files
# ============================================================
print(f"\n{'='*60}\nLOADING SEEN CATALOG EMBEDDINGS\n{'='*60}")

seen_embs = {}  # aid -> np.array

for version, run_dir in [("v1", RUN_DIR), ("v2", V2_RUN_DIR)]:
    emb_dir = f"{run_dir}/embeddings"
    cat_path = f"{emb_dir}/catalog_clap.npy"
    ids_path = f"{emb_dir}/catalog_ids.json"

    if not os.path.exists(cat_path):
        print(f"  {version}: no catalog embeddings at {cat_path}")
        # Try embedding from masters_clean
        if version == "v1" and not seen_embs:
            masters_dir = f"{run_dir}/masters_clean"
            if os.path.exists(masters_dir):
                print(f"  Embedding from {masters_dir}...")
                for aid in sorted(unseen_embs.keys()):
                    tracks = catalog_manifest[
                        catalog_manifest["artist_id"].apply(lambda x: str(int(float(x)))) == aid
                    ]
                    paths = []
                    for _, row in tracks.iterrows():
                        tid = row["track_id"]
                        for ext in [".wav", ".mp3"]:
                            for fmt in [str(tid), f"{int(tid):06d}"]:
                                p = os.path.join(masters_dir, f"{fmt}{ext}")
                                if os.path.exists(p):
                                    paths.append(p)
                                    break
                    if len(paths) >= 2:
                        emb, valid = embed_audio_files(paths)
                        if emb.shape[0] >= 2:
                            seen_embs[aid] = emb
                            print(f"    {artist_names.get(aid, aid)[:25]:<25}: {emb.shape[0]}")
        continue

    cat_emb = np.load(cat_path)
    with open(ids_path) as f:
        cat_ids = json.load(f)
    print(f"  {version}: loaded {cat_emb.shape} catalog embeddings")

    # Group by artist
    for aid in unseen_embs:
        if aid in seen_embs:
            continue
        tracks = catalog_manifest[
            catalog_manifest["artist_id"].apply(lambda x: str(int(float(x)))) == aid
        ]
        tids = set(str(row["track_id"]) for _, row in tracks.iterrows())
        indices = [i for i, cid in enumerate(cat_ids) if cid in tids or cid.lstrip("0") in tids]
        if len(indices) >= 2:
            seen_embs[aid] = cat_emb[indices]
            print(f"    {artist_names.get(aid, aid)[:25]:<25}: {len(indices)} catalog embeddings")

print(f"\nArtists with seen embeddings: {len(seen_embs)}")

# ============================================================
# Load generated embeddings from existing npy files
# ============================================================
print(f"\n{'='*60}\nLOADING GENERATED EMBEDDINGS\n{'='*60}")

gen_embs = {}  # version -> aid -> np.array

for version, run_dir in [("v1", RUN_DIR), ("v2", V2_RUN_DIR)]:
    gen_embs[version] = {}
    emb_dir = f"{run_dir}/embeddings"

    # Load generation log for file-to-artist mapping
    gen_log_path = f"{run_dir}/manifests/generation_log.csv"
    if not os.path.exists(gen_log_path):
        print(f"  {version}: no generation_log.csv")
        continue

    gen_log = pd.read_csv(gen_log_path)
    gen_log["file_id"] = gen_log["file_path"].apply(lambda p: os.path.basename(p).replace(".wav", ""))

    for tier in ["A_artist_proximal", "D_fma_tags"]:
        tier_emb_path = f"{emb_dir}/{tier}_clap.npy"
        tier_ids_path = f"{emb_dir}/{tier}_ids.json"
        if not os.path.exists(tier_emb_path):
            continue

        tier_emb = np.load(tier_emb_path)
        with open(tier_ids_path) as f:
            tier_ids = json.load(f)

        for i, fid in enumerate(tier_ids):
            row = gen_log[gen_log["file_id"] == fid]
            if row.empty:
                continue
            raw_aid = row.iloc[0]["artist_id"]
            aid = str(int(float(raw_aid))) if pd.notna(raw_aid) else ""
            if not aid or aid == "nan" or aid not in unseen_embs:
                continue
            if aid not in gen_embs[version]:
                gen_embs[version][aid] = []
            gen_embs[version][aid].append(tier_emb[i])

    for aid in gen_embs[version]:
        gen_embs[version][aid] = np.array(gen_embs[version][aid])

    print(f"  {version}: {len(gen_embs[version])} artists with generated embeddings")

# ============================================================
# Compute temporal FAD
# ============================================================
print(f"\n{'='*60}\nTEMPORAL FAD ANALYSIS\n{'='*60}")

results = []

for version in ["v1", "v2"]:
    print(f"\n--- {version.upper()} ---")
    v_gen = gen_embs.get(version, {})

    for aid in sorted(unseen_embs.keys()):
        if aid not in seen_embs or aid not in v_gen:
            continue

        seen = seen_embs[aid]
        unseen = unseen_embs[aid]
        gen = v_gen[aid]

        if gen.shape[0] < 3:
            continue

        # FAD: generated vs seen catalog
        try:
            mu_g, sig_g = gaussian_stats(gen)
            mu_s, sig_s = gaussian_stats(seen)
            fad_seen = frechet_distance(mu_g, sig_g, mu_s, sig_s)
        except Exception as e:
            print(f"  FAD(gen,seen) failed for {aid}: {e}")
            fad_seen = float("nan")

        # FAD: generated vs unseen catalog
        try:
            mu_u, sig_u = gaussian_stats(unseen)
            fad_unseen = frechet_distance(mu_g, sig_g, mu_u, sig_u)
        except Exception as e:
            print(f"  FAD(gen,unseen) failed for {aid}: {e}")
            fad_unseen = float("nan")

        # Mean pairwise cosine similarity as robust alternative
        gen_n = gen / (np.linalg.norm(gen, axis=1, keepdims=True) + 1e-8)
        seen_n = seen / (np.linalg.norm(seen, axis=1, keepdims=True) + 1e-8)
        unseen_n = unseen / (np.linalg.norm(unseen, axis=1, keepdims=True) + 1e-8)
        sim_gen_seen = float(np.mean(gen_n @ seen_n.T))
        sim_gen_unseen = float(np.mean(gen_n @ unseen_n.T))

        name = artist_names.get(aid, aid)
        results.append({
            "artist_id": aid,
            "artist_name": name,
            "version": version,
            "n_seen": seen.shape[0],
            "n_unseen": unseen.shape[0],
            "n_generated": gen.shape[0],
            "fad_gen_seen": round(fad_seen, 4),
            "fad_gen_unseen": round(fad_unseen, 4),
            "fad_diff": round(fad_seen - fad_unseen, 4),
            "sim_gen_seen": round(sim_gen_seen, 4),
            "sim_gen_unseen": round(sim_gen_unseen, 4),
            "sim_diff": round(sim_gen_seen - sim_gen_unseen, 4),
        })

        print(f"  {name[:25]:<25}  FAD_seen={fad_seen:7.2f}  FAD_unseen={fad_unseen:7.2f}  "
              f"diff={fad_seen - fad_unseen:+7.2f}  "
              f"sim_s={sim_gen_seen:.3f}  sim_u={sim_gen_unseen:.3f}")

# ============================================================
# Save and analyze
# ============================================================
csv_path = f"{OUT_DIR}/temporal_fad_results.csv"
if results:
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\nSaved {len(results)} rows to {csv_path}")

# Statistical tests
summary = {
    "test": "Temporal FAD: generated vs seen catalog vs unseen catalog",
    "hypothesis": "If memorized: FAD_seen << FAD_unseen. If style: FAD_seen ≈ FAD_unseen.",
}

for version in ["v1", "v2"]:
    vr = [r for r in results if r["version"] == version]
    if len(vr) < 3:
        continue

    fad_s = [r["fad_gen_seen"] for r in vr if not np.isnan(r["fad_gen_seen"])]
    fad_u = [r["fad_gen_unseen"] for r in vr if not np.isnan(r["fad_gen_unseen"])]
    fad_d = [r["fad_diff"] for r in vr if not np.isnan(r["fad_diff"])]
    sim_s = [r["sim_gen_seen"] for r in vr]
    sim_u = [r["sim_gen_unseen"] for r in vr]
    sim_d = [r["sim_diff"] for r in vr]

    print(f"\n=== {version.upper()} (N={len(vr)}) ===")

    # FAD comparison
    print(f"  FAD(gen, seen):   mean={np.mean(fad_s):.2f} ± {np.std(fad_s):.2f}")
    print(f"  FAD(gen, unseen): mean={np.mean(fad_u):.2f} ± {np.std(fad_u):.2f}")
    print(f"  FAD difference:   mean={np.mean(fad_d):+.2f} ± {np.std(fad_d):.2f}")

    # Paired tests
    t_fad, p_fad = stats.ttest_rel(fad_s, fad_u)
    w_fad, pw_fad = stats.wilcoxon(fad_d)
    print(f"  Paired t-test: t={t_fad:.3f}, p={p_fad:.4f}")
    print(f"  Wilcoxon: W={w_fad:.0f}, p={pw_fad:.4f}")

    # Cosine similarity comparison
    print(f"\n  Sim(gen, seen):   mean={np.mean(sim_s):.4f} ± {np.std(sim_s):.4f}")
    print(f"  Sim(gen, unseen): mean={np.mean(sim_u):.4f} ± {np.std(sim_u):.4f}")
    print(f"  Sim difference:   mean={np.mean(sim_d):+.4f} ± {np.std(sim_d):.4f}")

    t_sim, p_sim = stats.ttest_rel(sim_s, sim_u)
    print(f"  Paired t-test: t={t_sim:.3f}, p={p_sim:.4f}")

    # Interpretation
    if p_fad < 0.05 and np.mean(fad_d) < 0:
        interp = "MEMORIZATION: generated closer to seen than unseen"
    elif p_fad < 0.05 and np.mean(fad_d) > 0:
        interp = "SURPRISING: generated closer to unseen than seen"
    else:
        interp = "STYLE LEARNING: no significant difference between seen and unseen"
    print(f"\n  Interpretation: {interp}")

    summary[version] = {
        "n_artists": len(vr),
        "mean_fad_seen": round(float(np.mean(fad_s)), 4),
        "mean_fad_unseen": round(float(np.mean(fad_u)), 4),
        "mean_fad_diff": round(float(np.mean(fad_d)), 4),
        "paired_t": round(float(t_fad), 4),
        "paired_p": float(p_fad),
        "wilcoxon_W": float(w_fad),
        "wilcoxon_p": float(pw_fad),
        "mean_sim_seen": round(float(np.mean(sim_s)), 4),
        "mean_sim_unseen": round(float(np.mean(sim_u)), 4),
        "mean_sim_diff": round(float(np.mean(sim_d)), 4),
        "sim_paired_t": round(float(t_sim), 4),
        "sim_paired_p": float(p_sim),
        "interpretation": interp,
    }

json_path = f"{OUT_DIR}/temporal_fad_summary.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to {json_path}")
print("Done!")
