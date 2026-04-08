#!/usr/bin/env python3
"""Stage V2: Per-artist Frechet Audio Distance (FAD).

Computes FAD between each artist's catalog and their Tier A/D generated
outputs in CLAP embedding space. Lower FAD = more similar distributions
= higher replication risk.

Also computes cross-artist FAD as control (artist X's generations vs
artist Y's catalog) to establish a baseline distance.

Outputs:
    <run_dir>/analysis/per_artist_fad.csv
    <run_dir>/analysis/fad_cross_artist.csv       (control: cross-artist distances)
    <run_dir>/analysis/fad_summary.json
    <run_dir>/logs/per_artist_fad.log
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "per_artist_fad"


def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Frechet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_gaussian_stats(embeddings: np.ndarray, regularize: bool = False):
    """Compute mean and covariance of embeddings.

    Args:
        regularize: If True, use Ledoit-Wolf shrinkage estimator for covariance.
                    Critical when n_samples < n_features (e.g., 5 tracks, 512-dim CLAP).
                    Standard covariance is singular/ill-conditioned in this regime.
    """
    mu = np.mean(embeddings, axis=0)
    if regularize:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(embeddings)
        sigma = lw.covariance_
    else:
        sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def main():
    parser = base_argparser("Per-artist Frechet Audio Distance")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["analysis"])

    vuln_cfg = cfg.get("vulnerability", {})
    fad_cfg = vuln_cfg.get("fad", {})
    min_samples = fad_cfg.get("min_samples_per_group", 3)
    # Ledoit-Wolf regularization threshold: use shrinkage when n < d
    # (CLAP embeddings are 512-dim; per-artist groups are typically 5-24 samples)
    lw_threshold = fad_cfg.get("ledoit_wolf_threshold", 512)

    emb_dir = Path(args.run_dir) / "embeddings"

    # ---- Load catalog embeddings + IDs ----
    catalog_emb_path = emb_dir / "catalog_clap.npy"
    catalog_ids_path = emb_dir / "catalog_ids.json"
    if not catalog_emb_path.exists():
        logger.error("Catalog embeddings not found. Run compute_clap_embeddings.py first!")
        return

    catalog_emb = np.load(str(catalog_emb_path))
    with open(catalog_ids_path) as f:
        catalog_ids = json.load(f)
    logger.info(f"Catalog embeddings: {catalog_emb.shape}")

    # ---- Load artist mapping ----
    manifest_path = Path(args.run_dir) / "manifests" / "sampling_manifest.csv"
    if not manifest_path.exists():
        logger.error("No sampling_manifest.csv found")
        return

    manifest = pd.read_csv(manifest_path)
    id_col = "track_id" if "track_id" in manifest.columns else manifest.columns[0]
    artist_col = "artist_id" if "artist_id" in manifest.columns else "artist_name"
    name_col = "artist_name" if "artist_name" in manifest.columns else artist_col
    genre_col = "genre" if "genre" in manifest.columns else None

    track_to_artist = {}
    artist_info = {}
    for _, row in manifest.iterrows():
        aid = str(row[artist_col])
        tid = str(row[id_col])
        track_to_artist[tid] = aid
        if aid not in artist_info:
            artist_info[aid] = {
                "name": str(row.get(name_col, aid)),
                "genre": str(row.get(genre_col, "")) if genre_col else "",
                "n_tracks": 0,
            }
        artist_info[aid]["n_tracks"] += 1

    # Group catalog embeddings by artist
    artist_catalog_embs = {}
    for i, cid in enumerate(catalog_ids):
        artist = track_to_artist.get(cid, track_to_artist.get(cid.lstrip("0"), "unknown"))
        if artist not in artist_catalog_embs:
            artist_catalog_embs[artist] = []
        artist_catalog_embs[artist].append(i)

    logger.info(f"Artists with catalog embeddings: {len(artist_catalog_embs)}")

    # ---- Load generation log for file-to-artist mapping ----
    gen_log_path = Path(args.run_dir) / "manifests" / "generation_log.csv"
    if not gen_log_path.exists():
        logger.error("No generation_log.csv found")
        return

    gen_log = pd.read_csv(gen_log_path)
    gen_log["file_id"] = gen_log["file_path"].apply(lambda p: Path(p).stem)

    # ---- Load per-tier embeddings ----
    artist_gen_embs = {}  # artist_id -> np.ndarray of embeddings

    for tier in ["A_artist_proximal", "D_fma_tags"]:
        tier_emb_path = emb_dir / f"{tier}_clap.npy"
        tier_ids_path = emb_dir / f"{tier}_ids.json"
        if not tier_emb_path.exists():
            continue

        tier_emb = np.load(str(tier_emb_path))
        with open(tier_ids_path) as f:
            tier_ids = json.load(f)

        for i, fid in enumerate(tier_ids):
            row = gen_log[gen_log["file_id"] == fid]
            if row.empty:
                continue
            raw_aid = row.iloc[0]["artist_id"]
            artist_id = str(int(raw_aid)) if pd.notna(raw_aid) else ""
            if not artist_id or artist_id == "nan":
                continue
            if artist_id not in artist_gen_embs:
                artist_gen_embs[artist_id] = []
            artist_gen_embs[artist_id].append(tier_emb[i])

    for aid in artist_gen_embs:
        artist_gen_embs[aid] = np.array(artist_gen_embs[aid])

    logger.info(f"Artists with generated embeddings: {len(artist_gen_embs)}")

    # ---- Compute per-artist FAD (matched: artist's gens vs artist's catalog) ----
    results = []
    for artist_id in sorted(artist_gen_embs.keys()):
        gen_emb = artist_gen_embs[artist_id]
        if artist_id not in artist_catalog_embs:
            continue

        cat_indices = artist_catalog_embs[artist_id]
        cat_emb = catalog_emb[cat_indices]

        if gen_emb.shape[0] < min_samples or cat_emb.shape[0] < 2:
            logger.warning(f"Skipping {artist_id}: gen={gen_emb.shape[0]}, cat={cat_emb.shape[0]}")
            continue

        # Use Ledoit-Wolf when either group has fewer samples than dimensions
        need_lw = (gen_emb.shape[0] < lw_threshold) or (cat_emb.shape[0] < lw_threshold)
        if need_lw:
            logger.info(f"  {artist_id}: using Ledoit-Wolf (gen={gen_emb.shape[0]}, cat={cat_emb.shape[0]}, d={gen_emb.shape[1]})")

        try:
            mu_gen, sigma_gen = compute_gaussian_stats(gen_emb, regularize=need_lw)
            mu_cat, sigma_cat = compute_gaussian_stats(cat_emb, regularize=need_lw)
            fad = frechet_distance(mu_gen, sigma_gen, mu_cat, sigma_cat)
        except Exception as e:
            logger.warning(f"FAD failed for {artist_id}: {e}")
            fad = float("nan")

        # Also compute mean pairwise CLAP cosine similarity as a robust alternative
        # (no distributional assumptions, works with any n)
        from numpy.linalg import norm
        gen_normed = gen_emb / (norm(gen_emb, axis=1, keepdims=True) + 1e-8)
        cat_normed = cat_emb / (norm(cat_emb, axis=1, keepdims=True) + 1e-8)
        pairwise_sim = float(np.mean(gen_normed @ cat_normed.T))

        info = artist_info.get(artist_id, {})
        results.append({
            "artist_id": artist_id,
            "artist_name": info.get("name", ""),
            "genre": info.get("genre", ""),
            "fad": fad,
            "pairwise_clap_sim": pairwise_sim,
            "ledoit_wolf": need_lw,
            "n_catalog": cat_emb.shape[0],
            "n_generated": gen_emb.shape[0],
            "comparison": "matched",
        })

    # ---- Global FAD: all Tier A+D gens vs all catalog ----
    all_gen = np.vstack(list(artist_gen_embs.values())) if artist_gen_embs else None
    if all_gen is not None and all_gen.shape[0] >= min_samples:
        mu_gen, sigma_gen = compute_gaussian_stats(all_gen)
        mu_cat, sigma_cat = compute_gaussian_stats(catalog_emb)
        global_fad = frechet_distance(mu_gen, sigma_gen, mu_cat, sigma_cat)
        results.append({
            "artist_id": "GLOBAL",
            "artist_name": "ALL",
            "genre": "",
            "fad": global_fad,
            "n_catalog": catalog_emb.shape[0],
            "n_generated": all_gen.shape[0],
            "comparison": "global",
        })
        logger.info(f"Global FAD: {global_fad:.4f}")

    # Per-tier FAD (Tier B and C for comparison)
    for tier in ["B_genre_generic", "C_out_of_distribution"]:
        tier_emb_path = emb_dir / f"{tier}_clap.npy"
        if not tier_emb_path.exists():
            continue
        tier_emb = np.load(str(tier_emb_path))
        if tier_emb.shape[0] < min_samples:
            continue
        mu_gen, sigma_gen = compute_gaussian_stats(tier_emb)
        mu_cat, sigma_cat = compute_gaussian_stats(catalog_emb)
        tier_fad = frechet_distance(mu_gen, sigma_gen, mu_cat, sigma_cat)
        results.append({
            "artist_id": f"TIER_{tier}",
            "artist_name": tier,
            "genre": "",
            "fad": tier_fad,
            "n_catalog": catalog_emb.shape[0],
            "n_generated": tier_emb.shape[0],
            "comparison": "tier_level",
        })
        logger.info(f"Tier {tier} FAD: {tier_fad:.4f}")

    # ---- Cross-artist FAD (control distances) ----
    cross_results = []
    artist_ids = sorted(artist_gen_embs.keys())
    for gen_artist in artist_ids:
        gen_emb = artist_gen_embs[gen_artist]
        if gen_emb.shape[0] < min_samples:
            continue

        mu_gen, sigma_gen = compute_gaussian_stats(gen_emb)

        for cat_artist, cat_indices in artist_catalog_embs.items():
            cat_emb = catalog_emb[cat_indices]
            if cat_emb.shape[0] < 2:
                continue

            try:
                mu_cat, sigma_cat = compute_gaussian_stats(cat_emb)
                fad = frechet_distance(mu_gen, sigma_gen, mu_cat, sigma_cat)
            except Exception:
                fad = float("nan")

            cross_results.append({
                "gen_artist": gen_artist,
                "cat_artist": cat_artist,
                "fad": fad,
                "matched": gen_artist == cat_artist,
            })

    # ---- Write results ----
    results_csv = dirs["analysis"] / "per_artist_fad.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "artist_id", "artist_name", "genre", "fad",
            "pairwise_clap_sim", "ledoit_wolf",
            "n_catalog", "n_generated", "comparison",
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    if cross_results:
        cross_csv = dirs["analysis"] / "fad_cross_artist.csv"
        pd.DataFrame(cross_results).to_csv(cross_csv, index=False)
        logger.info(f"Wrote {len(cross_results)} cross-artist FAD pairs")

    # ---- Summary ----
    matched_results = [r for r in results if r["comparison"] == "matched" and not np.isnan(r["fad"])]
    fad_values = [r["fad"] for r in matched_results]
    sim_values = [r.get("pairwise_clap_sim", 0) for r in matched_results]

    cross_df = pd.DataFrame(cross_results) if cross_results else pd.DataFrame()
    matched_fads = cross_df[cross_df["matched"] == True]["fad"].dropna() if not cross_df.empty else pd.Series()
    mismatched_fads = cross_df[cross_df["matched"] == False]["fad"].dropna() if not cross_df.empty else pd.Series()

    summary = {
        "per_artist_matched": {
            "n_artists": len(matched_results),
            "mean_fad": float(np.mean(fad_values)) if fad_values else None,
            "median_fad": float(np.median(fad_values)) if fad_values else None,
            "min_fad": float(np.min(fad_values)) if fad_values else None,
            "max_fad": float(np.max(fad_values)) if fad_values else None,
            "mean_pairwise_clap_sim": float(np.mean(sim_values)) if sim_values else None,
            "note": "Ledoit-Wolf shrinkage used for covariance when n < d (512)",
        },
        "cross_artist": {
            "matched_mean_fad": float(matched_fads.mean()) if len(matched_fads) > 0 else None,
            "mismatched_mean_fad": float(mismatched_fads.mean()) if len(mismatched_fads) > 0 else None,
            "fad_gap": float(matched_fads.mean() - mismatched_fads.mean())
                if len(matched_fads) > 0 and len(mismatched_fads) > 0 else None,
        },
        "global_fad": next((r["fad"] for r in results if r["artist_id"] == "GLOBAL"), None),
    }

    summary_path = dirs["analysis"] / "fad_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n=== FAD SUMMARY ===")
    if fad_values:
        logger.info(f"  Per-artist matched FAD: mean={np.mean(fad_values):.4f}, "
                    f"median={np.median(fad_values):.4f}")
    if summary["cross_artist"]["matched_mean_fad"] is not None:
        logger.info(f"  Cross-artist: matched={summary['cross_artist']['matched_mean_fad']:.4f}, "
                    f"mismatched={summary['cross_artist']['mismatched_mean_fad']:.4f}, "
                    f"gap={summary['cross_artist']['fad_gap']:.4f}")
    logger.info(f"  Top 5 most vulnerable (lowest FAD):")
    for r in sorted(matched_results, key=lambda x: x["fad"])[:5]:
        logger.info(f"    {r['artist_id']} ({r['genre']}): FAD={r['fad']:.4f}")

    outputs = [str(results_csv), str(summary_path)]
    if cross_results:
        outputs.append(str(dirs["analysis"] / "fad_cross_artist.csv"))
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    meta["summary"] = summary
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
