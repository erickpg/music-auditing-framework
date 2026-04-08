#!/usr/bin/env python3
"""
MuQ-MuLan Cross-Embedding Validation.

Computes per-artist vulnerability using MuQ-MuLan embeddings instead of CLAP,
then correlates with CLAP-based scores to test embedding robustness.

Usage:
    python scripts/muq_mulan_validation.py \
        --run_dir /scratch/$USER/runs/2026-03-10_full \
        --out_dir /scratch/$USER/runs/muq_validation
"""

import argparse
import csv
import json
import os
import sys
import logging
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchaudio
import soundfile as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)


# ─── MuQ-MuLan Embedding ─────────────────────────────────────────────────────

def load_muq_model(device):
    """Load MuQ-MuLan model."""
    from muq import MuQMuLan
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model = model.to(device)
    model.eval()
    return model


def embed_audio_files(audio_paths, model, device, batch_size=8, target_sr=24000):
    """Embed a list of audio files using MuQ-MuLan.

    Returns:
        embeddings: np.ndarray [N, D]
        file_ids: list of file stem strings
    """
    embeddings = []
    file_ids = []

    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        batch_wavs = []

        for p in batch_paths:
            try:
                audio, sr = sf.read(str(p), dtype='float32')
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                wav = torch.tensor(audio, dtype=torch.float32)

                # Resample to 24kHz if needed
                if sr != target_sr:
                    wav = torchaudio.functional.resample(wav, sr, target_sr)

                # MuQ-MuLan expects specific length — pad/truncate to 10 seconds
                max_len = target_sr * 10
                if wav.shape[0] > max_len:
                    wav = wav[:max_len]
                elif wav.shape[0] < max_len:
                    wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[0]))

                batch_wavs.append(wav)
                file_ids.append(Path(p).stem)
            except Exception as e:
                log.warning(f"Failed to load {p}: {e}")
                continue

        if not batch_wavs:
            continue

        # Stack batch
        batch_tensor = torch.stack(batch_wavs).to(device)

        with torch.no_grad():
            output = model(wavs=batch_tensor)
            # MuQ-MuLan returns audio embeddings directly
            if hasattr(output, 'audio_embeds'):
                emb = output.audio_embeds
            elif isinstance(output, torch.Tensor):
                emb = output
            elif isinstance(output, dict):
                emb = output.get('audio_embeds', output.get('audio_features', None))
                if emb is None:
                    emb = list(output.values())[0]
            else:
                emb = output

            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()

            # Handle 3D: [B, T, D] -> [B, D]
            if emb.ndim == 3:
                emb = emb.mean(axis=1)

            # L2 normalize
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(norms, 1e-8)

            embeddings.append(emb)

        if (i // batch_size + 1) % 10 == 0:
            log.info(f"  Embedded {i + len(batch_paths)}/{len(audio_paths)} files")

    return np.vstack(embeddings), file_ids


# ─── Similarity & FAD ────────────────────────────────────────────────────────

def compute_per_artist_similarity(gen_emb, gen_ids, cat_emb, cat_ids, gen_log, manifest):
    """Compute per-artist matched vs mismatched cosine similarity."""
    # Build artist lookup from generation log
    # gen_log has 'file_path' column with full path — extract stem
    gen_artist = {}
    for row in gen_log:
        fid = row.get('file_id', '') or Path(row.get('file_path', '') or row.get('output_path', '')).stem
        aid = row.get('artist_id', '')
        tier = row.get('tier', '')
        if fid and aid:
            gen_artist[fid] = {'artist_id': str(aid), 'tier': tier}

    # Build catalog artist lookup from manifest
    # manifest has 'track_id' — catalog files are named by track_id (e.g., "003720")
    cat_artist = {}
    for row in manifest:
        # Try multiple possible column names
        fid = row.get('file_id', '') or row.get('track_id', '') or Path(row.get('audio_path', '') or row.get('path', '') or row.get('fma_path', '')).stem
        aid = row.get('artist_id', '')
        if fid and aid:
            # Catalog files may be zero-padded (003720) or not
            cat_artist[str(fid)] = str(aid)
            # Also add zero-padded version
            try:
                cat_artist[str(int(fid)).zfill(6)] = str(aid)
            except ValueError:
                pass

    # Debug matching
    log.info(f"    gen_artist keys (first 5): {list(gen_artist.keys())[:5]}")
    log.info(f"    gen_ids (first 5): {gen_ids[:5]}")
    log.info(f"    cat_artist keys (first 5): {list(cat_artist.keys())[:5]}")
    log.info(f"    cat_ids (first 5): {cat_ids[:5]}")
    matched_gen = sum(1 for gid in gen_ids if gid in gen_artist)
    matched_cat = sum(1 for cid in cat_ids if cid in cat_artist)
    log.info(f"    gen_ids matched to gen_artist: {matched_gen}/{len(gen_ids)}")
    log.info(f"    cat_ids matched to cat_artist: {matched_cat}/{len(cat_ids)}")

    # Cosine similarity matrix
    sim_matrix = gen_emb @ cat_emb.T  # [N_gen, N_cat]

    # Per-artist results
    results = []
    for gi, gid in enumerate(gen_ids):
        info = gen_artist.get(gid)
        if not info or info['tier'] not in ('A_artist_proximal', 'D_fma_tags'):
            continue

        artist_id = info['artist_id']

        # Matched: catalog tracks from same artist
        matched_idx = [ci for ci, cid in enumerate(cat_ids) if cat_artist.get(cid) == artist_id]
        # Mismatched: catalog tracks from different artists
        mismatched_idx = [ci for ci, cid in enumerate(cat_ids) if cat_artist.get(cid, '') != artist_id and cat_artist.get(cid, '') != '']

        if not matched_idx or not mismatched_idx:
            continue

        matched_sim = sim_matrix[gi, matched_idx].mean()
        mismatched_sim = sim_matrix[gi, mismatched_idx].mean()

        results.append({
            'file_id': gid,
            'artist_id': artist_id,
            'tier': info['tier'],
            'matched_mean_sim': float(matched_sim),
            'mismatched_mean_sim': float(mismatched_sim),
            'sim_gap': float(matched_sim - mismatched_sim),
        })

    return results


def compute_per_artist_fad(gen_emb, gen_ids, cat_emb, cat_ids, gen_log, manifest):
    """Compute per-artist FAD in MuQ-MuLan embedding space."""
    from sklearn.covariance import LedoitWolf

    # Build lookups
    gen_artist = {}
    for row in gen_log:
        fid = row.get('file_id', '') or Path(row.get('file_path', '') or row.get('output_path', '')).stem
        aid = row.get('artist_id', '')
        tier = row.get('tier', '')
        if fid and aid and tier in ('A_artist_proximal', 'D_fma_tags'):
            gen_artist[fid] = str(aid)

    cat_artist = {}
    for row in manifest:
        fid = row.get('file_id', '') or row.get('track_id', '') or Path(row.get('audio_path', '') or row.get('path', '') or row.get('fma_path', '')).stem
        aid = row.get('artist_id', '')
        if fid and aid:
            cat_artist[str(fid)] = str(aid)
            try:
                cat_artist[str(int(fid)).zfill(6)] = str(aid)
            except ValueError:
                pass

    # Group embeddings by artist
    all_artists = sorted(set(gen_artist.values()) & set(cat_artist.values()))

    def gaussian_stats(embs):
        mu = np.mean(embs, axis=0)
        if embs.shape[0] < embs.shape[1]:
            sigma = LedoitWolf().fit(embs).covariance_
        else:
            sigma = np.cov(embs, rowvar=False)
        return mu, sigma

    def frechet_distance(mu1, s1, mu2, s2):
        diff = mu1 - mu2
        # sqrt of product of covariances
        product = s1 @ s2
        eigvals = np.linalg.eigvalsh(product)
        eigvals = np.maximum(eigvals, 0)
        sqrt_trace = np.sum(np.sqrt(eigvals))
        return float(np.dot(diff, diff) + np.trace(s1) + np.trace(s2) - 2 * sqrt_trace)

    fad_results = []
    for aid in all_artists:
        gen_idx = [i for i, fid in enumerate(gen_ids) if gen_artist.get(fid) == aid]
        cat_idx = [i for i, fid in enumerate(cat_ids) if cat_artist.get(fid) == aid]

        if len(gen_idx) < 3 or len(cat_idx) < 3:
            continue

        gen_e = gen_emb[gen_idx]
        cat_e = cat_emb[cat_idx]

        try:
            mu_g, sig_g = gaussian_stats(gen_e)
            mu_c, sig_c = gaussian_stats(cat_e)
            fad = frechet_distance(mu_g, sig_g, mu_c, sig_c)
        except Exception as e:
            log.warning(f"FAD failed for artist {aid}: {e}")
            continue

        # Also compute pairwise cosine sim
        pairwise = (gen_e @ cat_e.T).mean()

        fad_results.append({
            'artist_id': aid,
            'fad': fad,
            'pairwise_sim': float(pairwise),
            'n_gen': len(gen_idx),
            'n_cat': len(cat_idx),
        })

    return fad_results


# ─── Correlation ──────────────────────────────────────────────────────────────

def spearman_rho(x, y):
    x, y = np.array(x), np.array(y)
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    rho = 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))
    if abs(rho) < 1:
        t = rho * math.sqrt((n - 2) / (1 - rho**2))
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    else:
        p = 0.0
    return rho, p


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True, help='V1 run directory')
    parser.add_argument('--baseline_dir', default=None, help='Baseline run directory')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    log.info(f"Device: {device}")

    # ─── Load metadata ───────────────────────────────────────────────────
    def load_csv_file(path):
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if any(v == k for k, v in row.items()):
                    continue
                rows.append(row)
        return rows

    manifest_path = run_dir / 'manifests' / 'sampling_manifest.csv'
    if not manifest_path.exists():
        # Try tracks_selected
        manifest_path = run_dir / 'manifests' / 'tracks_selected.csv'
    manifest = load_csv_file(str(manifest_path))
    log.info(f"Manifest: {len(manifest)} tracks from {manifest_path}")

    gen_log_path = run_dir / 'manifests' / 'generation_log.csv'
    gen_log = load_csv_file(str(gen_log_path))
    log.info(f"Generation log: {len(gen_log)} entries")

    # ─── Collect audio paths ──────────────────────────────────────────────
    # Catalog
    catalog_dir = run_dir / 'masters_clean'
    if not catalog_dir.exists():
        catalog_dir = run_dir / 'segments'
    catalog_paths = sorted(catalog_dir.glob('**/*.wav'))
    log.info(f"Catalog audio files: {len(catalog_paths)}")

    # Generated (V1)
    gen_dir = run_dir / 'generated'
    gen_paths = sorted(gen_dir.glob('**/*.wav'))
    log.info(f"Generated audio files (V1): {len(gen_paths)}")

    # Baseline generated (if provided)
    bl_gen_paths = []
    bl_gen_log = []
    if args.baseline_dir:
        bl_dir = Path(args.baseline_dir)
        bl_gen_dir = bl_dir / 'generated'
        bl_gen_paths = sorted(bl_gen_dir.glob('**/*.wav'))
        bl_gen_log_path = bl_dir / 'manifests' / 'generation_log.csv'
        if bl_gen_log_path.exists():
            bl_gen_log = load_csv_file(str(bl_gen_log_path))
        log.info(f"Baseline generated files: {len(bl_gen_paths)}")

    # ─── Load model ───────────────────────────────────────────────────────
    log.info("Loading MuQ-MuLan model...")
    model = load_muq_model(device)
    log.info("Model loaded.")

    # ─── Embed everything ─────────────────────────────────────────────────
    log.info("Embedding catalog...")
    cat_emb, cat_ids = embed_audio_files(catalog_paths, model, device, args.batch_size)
    log.info(f"  Catalog embeddings: {cat_emb.shape}")
    np.save(out_dir / 'catalog_muq.npy', cat_emb)
    with open(out_dir / 'catalog_muq_ids.json', 'w') as f:
        json.dump(cat_ids, f)

    log.info("Embedding V1 generated...")
    gen_emb, gen_ids = embed_audio_files(gen_paths, model, device, args.batch_size)
    log.info(f"  V1 generated embeddings: {gen_emb.shape}")
    np.save(out_dir / 'v1_gen_muq.npy', gen_emb)
    with open(out_dir / 'v1_gen_muq_ids.json', 'w') as f:
        json.dump(gen_ids, f)

    if bl_gen_paths:
        log.info("Embedding baseline generated...")
        bl_emb, bl_ids = embed_audio_files(bl_gen_paths, model, device, args.batch_size)
        log.info(f"  Baseline generated embeddings: {bl_emb.shape}")
        np.save(out_dir / 'bl_gen_muq.npy', bl_emb)
        with open(out_dir / 'bl_gen_muq_ids.json', 'w') as f:
            json.dump(bl_ids, f)

    # ─── Compute similarity ───────────────────────────────────────────────
    log.info("Computing V1 per-artist similarity...")
    v1_sim = compute_per_artist_similarity(gen_emb, gen_ids, cat_emb, cat_ids, gen_log, manifest)
    log.info(f"  V1 similarity records: {len(v1_sim)}")

    # Save similarity CSV
    with open(out_dir / 'muq_clap_per_artist_v1.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file_id', 'artist_id', 'tier', 'matched_mean_sim', 'mismatched_mean_sim', 'sim_gap'])
        w.writeheader()
        w.writerows(v1_sim)

    # ─── Compute FAD ──────────────────────────────────────────────────────
    log.info("Computing V1 per-artist FAD...")
    v1_fad = compute_per_artist_fad(gen_emb, gen_ids, cat_emb, cat_ids, gen_log, manifest)
    log.info(f"  V1 FAD records: {len(v1_fad)}")

    with open(out_dir / 'muq_fad_v1.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['artist_id', 'fad', 'pairwise_sim', 'n_gen', 'n_cat'])
        w.writeheader()
        w.writerows(v1_fad)

    # ─── Baseline similarity & FAD ────────────────────────────────────────
    if bl_gen_paths and bl_gen_log:
        log.info("Computing baseline per-artist similarity...")
        bl_sim = compute_per_artist_similarity(bl_emb, bl_ids, cat_emb, cat_ids, bl_gen_log, manifest)
        log.info(f"  Baseline similarity records: {len(bl_sim)}")

        with open(out_dir / 'muq_clap_per_artist_bl.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['file_id', 'artist_id', 'tier', 'matched_mean_sim', 'mismatched_mean_sim', 'sim_gap'])
            w.writeheader()
            w.writerows(bl_sim)

        log.info("Computing baseline per-artist FAD...")
        bl_fad = compute_per_artist_fad(bl_emb, bl_ids, cat_emb, cat_ids, bl_gen_log, manifest)
        log.info(f"  Baseline FAD records: {len(bl_fad)}")

        with open(out_dir / 'muq_fad_bl.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['artist_id', 'fad', 'pairwise_sim', 'n_gen', 'n_cat'])
            w.writeheader()
            w.writerows(bl_fad)

    # ─── Compute 2-signal vulnerability ───────────────────────────────────
    log.info("Computing MuQ-MuLan vulnerability scores...")

    def compute_vuln_from_sim_fad(sim_rows, fad_rows):
        """Compute 2-signal vulnerability from similarity and FAD data."""
        # Aggregate CLAP-like similarity per artist
        artist_sims = defaultdict(list)
        for r in sim_rows:
            artist_sims[r['artist_id']].append(r['matched_mean_sim'])

        clap_scores = {aid: np.mean(sims) for aid, sims in artist_sims.items()}
        fad_scores = {r['artist_id']: r['fad'] for r in fad_rows}

        common = sorted(set(clap_scores.keys()) & set(fad_scores.keys()))
        if not common:
            return {}

        clap_vals = [clap_scores[a] for a in common]
        fad_vals = [fad_scores[a] for a in common]

        clap_min, clap_max = min(clap_vals), max(clap_vals)
        fad_min, fad_max = min(fad_vals), max(fad_vals)

        result = {}
        for aid in common:
            cn = (clap_scores[aid] - clap_min) / (clap_max - clap_min + 1e-10)
            fn = 1.0 - (fad_scores[aid] - fad_min) / (fad_max - fad_min + 1e-10)
            result[aid] = 0.5 * cn + 0.5 * fn

        return result

    muq_v1_vuln = compute_vuln_from_sim_fad(v1_sim, v1_fad)
    log.info(f"  MuQ V1 vulnerability: {len(muq_v1_vuln)} artists")

    muq_bl_vuln = {}
    if bl_gen_paths and bl_gen_log:
        muq_bl_vuln = compute_vuln_from_sim_fad(bl_sim, bl_fad)
        log.info(f"  MuQ Baseline vulnerability: {len(muq_bl_vuln)} artists")

    # ─── Load CLAP-based scores for correlation ──────────────────────────
    clap_v1_path = run_dir / 'analysis' / 'vulnerability_scores.csv'
    clap_v1_vuln = {}
    if clap_v1_path.exists():
        rows = load_csv_file(str(clap_v1_path))
        for r in rows:
            aid = r.get('artist_id', '')
            clap_sim = float(r.get('clap_similarity', 0) or 0)
            fad_val = float(r.get('fad', 0) or 0)
            if aid:
                clap_v1_vuln[aid] = {'clap': clap_sim, 'fad': fad_val}

        # Recompute 2-signal for CLAP
        if clap_v1_vuln:
            vals_c = [v['clap'] for v in clap_v1_vuln.values()]
            vals_f = [v['fad'] for v in clap_v1_vuln.values()]
            c_min, c_max = min(vals_c), max(vals_c)
            f_min, f_max = min(vals_f), max(vals_f)
            for aid in clap_v1_vuln:
                cn = (clap_v1_vuln[aid]['clap'] - c_min) / (c_max - c_min + 1e-10)
                fn = 1.0 - (clap_v1_vuln[aid]['fad'] - f_min) / (f_max - f_min + 1e-10)
                clap_v1_vuln[aid] = 0.5 * cn + 0.5 * fn

    # ─── Correlations ─────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("CROSS-EMBEDDING CORRELATIONS")
    log.info("=" * 70)

    correlations = {}

    # MuQ vs CLAP (V1)
    common_v1 = sorted(set(muq_v1_vuln.keys()) & set(clap_v1_vuln.keys()))
    if common_v1:
        muq_scores = [muq_v1_vuln[a] for a in common_v1]
        clap_scores_list = [clap_v1_vuln[a] for a in common_v1]
        rho, p = spearman_rho(muq_scores, clap_scores_list)

        # Tier agreement
        muq_tiers = ['High' if s > 0.5 else 'Low' for s in muq_scores]
        clap_tiers = ['High' if s > 0.5 else 'Low' for s in clap_scores_list]
        agree = sum(1 for a, b in zip(muq_tiers, clap_tiers) if a == b)

        log.info(f"\nMuQ-MuLan vs CLAP (V1 fine-tuned):")
        log.info(f"  Spearman ρ = {rho:.4f} (p = {p:.6f})")
        log.info(f"  Tier agreement: {agree}/{len(common_v1)} ({100*agree/len(common_v1):.0f}%)")

        correlations['muq_vs_clap_v1'] = {
            'spearman_rho': round(rho, 4),
            'p_value': round(p, 6),
            'tier_agreement': f"{agree}/{len(common_v1)}",
            'tier_agreement_pct': round(100 * agree / len(common_v1), 1),
            'n_artists': len(common_v1),
        }

    # MuQ vs CLAP (Baseline)
    if muq_bl_vuln and args.baseline_dir:
        clap_bl_path = Path(args.baseline_dir) / 'analysis' / 'vulnerability_scores.csv'
        clap_bl_vuln = {}
        if clap_bl_path.exists():
            rows = load_csv_file(str(clap_bl_path))
            for r in rows:
                aid = r.get('artist_id', '')
                clap_sim = float(r.get('clap_similarity', 0) or 0)
                fad_val = float(r.get('fad', 0) or 0)
                if aid:
                    clap_bl_vuln[aid] = {'clap': clap_sim, 'fad': fad_val}

            if clap_bl_vuln:
                vals_c = [v['clap'] for v in clap_bl_vuln.values()]
                vals_f = [v['fad'] for v in clap_bl_vuln.values()]
                c_min, c_max = min(vals_c), max(vals_c)
                f_min, f_max = min(vals_f), max(vals_f)
                for aid in clap_bl_vuln:
                    cn = (clap_bl_vuln[aid]['clap'] - c_min) / (c_max - c_min + 1e-10)
                    fn = 1.0 - (clap_bl_vuln[aid]['fad'] - f_min) / (f_max - f_min + 1e-10)
                    clap_bl_vuln[aid] = 0.5 * cn + 0.5 * fn

                common_bl = sorted(set(muq_bl_vuln.keys()) & set(clap_bl_vuln.keys()))
                if common_bl:
                    muq_bl_s = [muq_bl_vuln[a] for a in common_bl]
                    clap_bl_s = [clap_bl_vuln[a] for a in common_bl]
                    rho_bl, p_bl = spearman_rho(muq_bl_s, clap_bl_s)
                    agree_bl = sum(1 for a, b in zip(
                        ['High' if s > 0.5 else 'Low' for s in muq_bl_s],
                        ['High' if s > 0.5 else 'Low' for s in clap_bl_s]) if a == b)

                    log.info(f"\nMuQ-MuLan vs CLAP (Baseline):")
                    log.info(f"  Spearman ρ = {rho_bl:.4f} (p = {p_bl:.6f})")
                    log.info(f"  Tier agreement: {agree_bl}/{len(common_bl)} ({100*agree_bl/len(common_bl):.0f}%)")

                    correlations['muq_vs_clap_baseline'] = {
                        'spearman_rho': round(rho_bl, 4),
                        'p_value': round(p_bl, 6),
                        'tier_agreement': f"{agree_bl}/{len(common_bl)}",
                        'tier_agreement_pct': round(100 * agree_bl / len(common_bl), 1),
                        'n_artists': len(common_bl),
                    }

    # MuQ V1 vs MuQ Baseline
    if muq_bl_vuln:
        common_muq = sorted(set(muq_v1_vuln.keys()) & set(muq_bl_vuln.keys()))
        if common_muq:
            m_v1 = [muq_v1_vuln[a] for a in common_muq]
            m_bl = [muq_bl_vuln[a] for a in common_muq]
            rho_m, p_m = spearman_rho(m_v1, m_bl)
            agree_m = sum(1 for a, b in zip(
                ['High' if s > 0.5 else 'Low' for s in m_v1],
                ['High' if s > 0.5 else 'Low' for s in m_bl]) if a == b)

            log.info(f"\nMuQ V1 vs MuQ Baseline:")
            log.info(f"  Spearman ρ = {rho_m:.4f} (p = {p_m:.6f})")
            log.info(f"  Tier agreement: {agree_m}/{len(common_muq)} ({100*agree_m/len(common_muq):.0f}%)")

            correlations['muq_v1_vs_muq_baseline'] = {
                'spearman_rho': round(rho_m, 4),
                'p_value': round(p_m, 6),
                'tier_agreement': f"{agree_m}/{len(common_muq)}",
                'tier_agreement_pct': round(100 * agree_m / len(common_muq), 1),
                'n_artists': len(common_muq),
            }

    # ─── Per-artist table ─────────────────────────────────────────────────
    all_artists = sorted(set(muq_v1_vuln.keys()) & set(clap_v1_vuln.keys()))
    if not all_artists:
        log.warning("No common artists between MuQ and CLAP — check file ID matching")
        log.warning(f"  MuQ artists: {list(muq_v1_vuln.keys())[:5]}")
        log.warning(f"  CLAP artists: {list(clap_v1_vuln.keys())[:5]}")
    per_artist = []
    for aid in all_artists:
        row = {
            'artist_id': aid,
            'muq_vuln_v1': round(muq_v1_vuln.get(aid, 0), 6),
            'clap_vuln_v1': round(clap_v1_vuln.get(aid, 0), 6),
            'muq_tier': 'High' if muq_v1_vuln.get(aid, 0) > 0.5 else 'Low',
            'clap_tier': 'High' if clap_v1_vuln.get(aid, 0) > 0.5 else 'Low',
        }
        if muq_bl_vuln:
            row['muq_vuln_bl'] = round(muq_bl_vuln.get(aid, 0), 6)
        per_artist.append(row)

    per_artist.sort(key=lambda x: x['muq_vuln_v1'], reverse=True)

    with open(out_dir / 'muq_per_artist_comparison.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=per_artist[0].keys())
        w.writeheader()
        w.writerows(per_artist)

    # ─── Save summary ────────────────────────────────────────────────────
    summary = {
        'embedding_model': 'OpenMuQ/MuQ-MuLan-large',
        'embedding_dim': int(cat_emb.shape[1]),
        'n_catalog_files': len(catalog_paths),
        'n_v1_generated': len(gen_paths),
        'n_baseline_generated': len(bl_gen_paths),
        'n_artists_scored': len(muq_v1_vuln),
        'correlations': correlations,
    }

    with open(out_dir / 'muq_validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n[SAVED] {out_dir}/muq_validation_summary.json")
    log.info(f"[SAVED] {out_dir}/muq_per_artist_comparison.csv")
    log.info("Done!")


if __name__ == '__main__':
    main()
