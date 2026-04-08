#!/usr/bin/env python3
"""Stage V1: Compute CLAP embeddings for catalog and generated audio.

CLAP (Contrastive Language-Audio Pretraining) produces semantic audio
embeddings. Computes embeddings for all tiers and performs:

  1. Global similarity: each generated file vs full catalog
  2. Per-artist similarity (Tier A/D): matched artist vs mismatched artists
     → Core signal for vulnerability assessment
  3. Per-tier summary statistics

Outputs:
    <run_dir>/embeddings/catalog_clap.npy        (N_catalog × D)
    <run_dir>/embeddings/catalog_ids.json         (ordered file IDs)
    <run_dir>/embeddings/<tier>_clap.npy          (per-tier embeddings)
    <run_dir>/embeddings/<tier>_ids.json          (per-tier file IDs)
    <run_dir>/analysis/clap_similarity.csv        (per-file similarity)
    <run_dir>/analysis/clap_per_artist.csv        (matched vs mismatched)
    <run_dir>/analysis/clap_summary.json
    <run_dir>/logs/compute_clap_embeddings.log
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "compute_clap_embeddings"

TIERS = [
    "A_artist_proximal",
    "B_genre_generic",
    "C_out_of_distribution",
    "D_fma_tags",
]


def load_clap_model(model_name: str, device):
    """Load CLAP model from transformers."""
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained(model_name).to(device)
    processor = ClapProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def compute_embeddings_batch(audio_paths: list, model, processor, device,
                             batch_size: int = 16, logger=None):
    """Compute CLAP audio embeddings for a list of files.

    Returns:
        embeddings: np.ndarray of shape [N, D]
        file_ids: list of file stem names
    """
    import soundfile as sf
    import librosa

    embeddings = []
    file_ids = []
    target_sr = processor.feature_extractor.sampling_rate

    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i+batch_size]
        batch_audio = []

        for path in batch_paths:
            signal, sr = sf.read(str(path))
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            if sr != target_sr:
                signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
            batch_audio.append(signal)
            file_ids.append(path.stem)

        inputs = processor(
            audio=batch_audio,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.get_audio_features(**inputs)
            # Newer transformers returns BaseModelOutputWithPooling, not a tensor
            if hasattr(outputs, 'pooler_output'):
                emb = outputs.pooler_output.cpu().numpy()
            elif hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state[:, 0].cpu().numpy()
            else:
                emb = outputs.cpu().numpy()
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-8, None)
            embeddings.append(emb)

        if logger and (i + batch_size) % (batch_size * 10) == 0:
            logger.info(f"    Embedded {min(i+batch_size, len(audio_paths))}/{len(audio_paths)}")

    return np.vstack(embeddings), file_ids


def main():
    parser = base_argparser("Compute CLAP embeddings and similarity")
    parser.add_argument("--catalog_dir", type=str, default=None,
                        help="Dir with catalog audio (default: <run_dir>/masters_clean)")
    parser.add_argument("--skip_similarity", action="store_true",
                        help="Only compute embeddings, skip similarity analysis")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["embeddings", "analysis", "manifests"])

    vuln_cfg = cfg.get("vulnerability", {})
    clap_cfg = vuln_cfg.get("clap", {})
    model_name = clap_cfg.get("model_name", "laion/larger_clap_music_and_speech")
    batch_size = clap_cfg.get("batch_size", 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"CLAP model: {model_name}")
    logger.info(f"Device: {device}, batch_size: {batch_size}")

    model, processor = load_clap_model(model_name, device)
    logger.info("CLAP model loaded")

    catalog_dir = Path(args.catalog_dir) if args.catalog_dir else Path(args.run_dir) / "masters_clean"
    t0 = time.time()

    # ---- Load artist mapping ----
    artist_track_map = {}  # artist_id -> list of track_ids
    track_to_artist = {}   # track_id -> artist_id
    manifest_path = Path(args.run_dir) / "manifests" / "sampling_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        id_col = "track_id" if "track_id" in manifest.columns else manifest.columns[0]
        artist_col = "artist_id" if "artist_id" in manifest.columns else "artist_name"
        for _, row in manifest.iterrows():
            aid = str(row[artist_col])
            tid = str(row[id_col])
            track_to_artist[tid] = aid
            if aid not in artist_track_map:
                artist_track_map[aid] = []
            artist_track_map[aid].append(tid)

    # ---- Load generation log for file metadata ----
    gen_log_path = Path(args.run_dir) / "manifests" / "generation_log.csv"
    file_meta = {}
    if gen_log_path.exists():
        gen_log = pd.read_csv(gen_log_path)
        gen_log["file_id"] = gen_log["file_path"].apply(lambda p: Path(p).stem)
        file_meta = {row["file_id"]: row for _, row in gen_log.iterrows()}

    # ---- Catalog embeddings ----
    catalog_files = sorted(catalog_dir.glob("*.wav"))
    if not catalog_files:
        catalog_files = sorted(catalog_dir.glob("*.flac"))
    logger.info(f"Catalog: {len(catalog_files)} files")

    catalog_emb, catalog_ids = compute_embeddings_batch(
        catalog_files, model, processor, device, batch_size, logger)
    np.save(str(dirs["embeddings"] / "catalog_clap.npy"), catalog_emb)
    with open(dirs["embeddings"] / "catalog_ids.json", "w") as f:
        json.dump(catalog_ids, f)
    logger.info(f"Catalog embeddings: {catalog_emb.shape}")

    # Build catalog artist index for similarity lookups
    catalog_artist_indices = {}  # artist_id -> list of indices in catalog_emb
    for i, cid in enumerate(catalog_ids):
        artist = track_to_artist.get(cid, track_to_artist.get(cid.lstrip("0"), "unknown"))
        if artist not in catalog_artist_indices:
            catalog_artist_indices[artist] = []
        catalog_artist_indices[artist].append(i)

    # ---- Generated embeddings per tier ----
    all_similarity_results = []
    per_artist_results = []

    for tier in TIERS:
        gen_dir = Path(args.run_dir) / "generated" / tier
        if not gen_dir.exists():
            logger.warning(f"Dir not found: {gen_dir}")
            continue

        gen_files = sorted(gen_dir.glob("*.wav"))
        if not gen_files:
            continue

        logger.info(f"[{tier}] {len(gen_files)} files")

        gen_emb, gen_ids = compute_embeddings_batch(
            gen_files, model, processor, device, batch_size, logger)
        np.save(str(dirs["embeddings"] / f"{tier}_clap.npy"), gen_emb)
        with open(dirs["embeddings"] / f"{tier}_ids.json", "w") as f:
            json.dump(gen_ids, f)
        logger.info(f"[{tier}] Embeddings: {gen_emb.shape}")

        if args.skip_similarity:
            continue

        # ---- Similarity analysis ----
        # Cosine similarity matrix: [N_gen, N_catalog]
        sim_matrix = gen_emb @ catalog_emb.T

        for i, gen_id in enumerate(gen_ids):
            sims = sim_matrix[i]
            max_idx = int(np.argmax(sims))

            fmeta = file_meta.get(gen_id, {})
            raw_aid = fmeta.get("artist_id", "")
            artist_id = str(int(raw_aid)) if pd.notna(raw_aid) and raw_aid != "" else ""
            genre = str(fmeta.get("genre", ""))

            row = {
                "file_id": gen_id,
                "tier": tier,
                "artist_id": artist_id,
                "genre": genre,
                "max_similarity": float(sims[max_idx]),
                "mean_similarity": float(np.mean(sims)),
                "median_similarity": float(np.median(sims)),
                "std_similarity": float(np.std(sims)),
                "most_similar_catalog_id": catalog_ids[max_idx],
                "most_similar_artist": track_to_artist.get(
                    catalog_ids[max_idx], "unknown"),
            }

            # Per-artist matched vs mismatched similarity (for Tier A/D)
            if tier in ("A_artist_proximal", "D_fma_tags") and artist_id and artist_id in catalog_artist_indices:
                matched_indices = catalog_artist_indices[artist_id]
                mismatched_indices = [j for j in range(len(catalog_ids))
                                      if j not in matched_indices]

                matched_sims = sims[matched_indices]
                mismatched_sims = sims[mismatched_indices] if mismatched_indices else np.array([0.0])

                row["matched_max_sim"] = float(np.max(matched_sims))
                row["matched_mean_sim"] = float(np.mean(matched_sims))
                row["mismatched_max_sim"] = float(np.max(mismatched_sims))
                row["mismatched_mean_sim"] = float(np.mean(mismatched_sims))
                row["sim_gap"] = float(np.mean(matched_sims) - np.mean(mismatched_sims))

                per_artist_results.append({
                    "file_id": gen_id,
                    "tier": tier,
                    "artist_id": artist_id,
                    "genre": genre,
                    "matched_max_sim": row["matched_max_sim"],
                    "matched_mean_sim": row["matched_mean_sim"],
                    "mismatched_max_sim": row["mismatched_max_sim"],
                    "mismatched_mean_sim": row["mismatched_mean_sim"],
                    "sim_gap": row["sim_gap"],
                    "n_matched_tracks": len(matched_indices),
                    "n_mismatched_tracks": len(mismatched_indices),
                })

            all_similarity_results.append(row)

    # ---- Write results ----
    if all_similarity_results:
        sim_csv = dirs["analysis"] / "clap_similarity.csv"
        sim_df = pd.DataFrame(all_similarity_results)
        sim_df.to_csv(sim_csv, index=False)
        logger.info(f"Wrote {len(sim_df)} similarity rows")

    if per_artist_results:
        pa_csv = dirs["analysis"] / "clap_per_artist.csv"
        pa_df = pd.DataFrame(per_artist_results)
        pa_df.to_csv(pa_csv, index=False)
        logger.info(f"Wrote {len(pa_df)} per-artist rows")

    # ---- Summary ----
    summary = {"by_tier": {}, "per_artist": {}}

    if all_similarity_results:
        sim_df = pd.DataFrame(all_similarity_results)
        for tier in sim_df["tier"].unique():
            t_df = sim_df[sim_df["tier"] == tier]
            summary["by_tier"][tier] = {
                "n_files": len(t_df),
                "max_sim_mean": float(t_df["max_similarity"].mean()),
                "max_sim_std": float(t_df["max_similarity"].std()),
                "mean_sim_mean": float(t_df["mean_similarity"].mean()),
            }

    if per_artist_results:
        pa_df = pd.DataFrame(per_artist_results)
        summary["per_artist"]["overall"] = {
            "mean_matched_sim": float(pa_df["matched_mean_sim"].mean()),
            "mean_mismatched_sim": float(pa_df["mismatched_mean_sim"].mean()),
            "mean_sim_gap": float(pa_df["sim_gap"].mean()),
            "n_files": len(pa_df),
        }

        # Per-artist breakdown
        for artist_id in pa_df["artist_id"].unique():
            a_df = pa_df[pa_df["artist_id"] == artist_id]
            summary["per_artist"][str(artist_id)] = {
                "matched_mean_sim": float(a_df["matched_mean_sim"].mean()),
                "mismatched_mean_sim": float(a_df["mismatched_mean_sim"].mean()),
                "sim_gap": float(a_df["sim_gap"].mean()),
                "n_files": len(a_df),
            }

    summary_path = dirs["analysis"] / "clap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Log
    logger.info(f"\n=== CLAP SIMILARITY SUMMARY ===")
    for tier, stats in summary.get("by_tier", {}).items():
        logger.info(f"  [{tier}] max_sim={stats['max_sim_mean']:.4f}±{stats['max_sim_std']:.4f}, "
                    f"mean_sim={stats['mean_sim_mean']:.4f} (n={stats['n_files']})")
    if "overall" in summary.get("per_artist", {}):
        ov = summary["per_artist"]["overall"]
        logger.info(f"\n  Per-artist matched vs mismatched:")
        logger.info(f"    matched_sim={ov['mean_matched_sim']:.4f}, "
                    f"mismatched_sim={ov['mean_mismatched_sim']:.4f}, "
                    f"gap={ov['mean_sim_gap']:.4f}")

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    outputs = [str(dirs["embeddings"]), str(dirs["analysis"] / "clap_similarity.csv")]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    meta["summary"] = summary
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
