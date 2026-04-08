#!/usr/bin/env python3
"""Stage A1: N-gram search between generated and catalog tokens.

For each generated output, search for n-gram matches against the catalog
token database. An n-gram match in codec token space indicates the model
has memorized and reproduced a specific audio passage.

Key design: uses the generation_log.csv to map each generated file to its
tier, artist, and genre. Performs both:
  1. Global search: each generated file vs entire catalog
  2. Per-artist search (Tier A/D): each file vs matched artist's catalog
     AND vs mismatched artists' catalog (for within-model control)

Based on Carlini et al. (2021, 2023) memorization methodology adapted
from text LLMs to audio codec tokens.

Outputs:
    <run_dir>/analysis/ngram_matches.csv           (per-file, per-codebook, per-n)
    <run_dir>/analysis/ngram_per_artist.csv         (per-artist matched vs mismatched)
    <run_dir>/analysis/ngram_match_summary.json
    <run_dir>/logs/ngram_search.log
"""

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "ngram_search"

TIERS = [
    "A_artist_proximal",
    "B_genre_generic",
    "C_out_of_distribution",
    "D_fma_tags",
]


def extract_ngrams(tokens: np.ndarray, n: int) -> set:
    """Extract all n-grams from a 1D token sequence."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def count_ngram_matches(gen_tokens: np.ndarray, catalog_ngrams: set, n: int) -> int:
    """Count how many n-grams in gen_tokens appear in catalog_ngrams."""
    gen_ngrams = extract_ngrams(gen_tokens, n)
    return len(gen_ngrams & catalog_ngrams)


def build_catalog_index(catalog_dir: Path, codebooks: list, ngram_sizes: list,
                        logger) -> dict:
    """Build n-gram index from all catalog token files.

    Returns:
        Dict mapping (codebook, n) -> set of n-gram tuples
    """
    catalog_files = sorted(catalog_dir.glob("*.npy"))
    logger.info(f"Building catalog index from {len(catalog_files)} files...")

    index = {}
    for cb in codebooks:
        for n in ngram_sizes:
            index[(cb, n)] = set()

    for fpath in catalog_files:
        codes = np.load(str(fpath))  # shape [n_codebooks, n_frames]
        for cb in codebooks:
            if cb >= codes.shape[0]:
                continue
            tokens = codes[cb]
            for n in ngram_sizes:
                ngrams = extract_ngrams(tokens, n)
                index[(cb, n)].update(ngrams)

    for (cb, n), ngrams in index.items():
        logger.info(f"  Codebook {cb}, {n}-gram: {len(ngrams)} unique n-grams")

    return index


def build_per_artist_catalog_index(catalog_dir: Path, artist_track_map: dict,
                                    codebooks: list, ngram_sizes: list,
                                    logger) -> dict:
    """Build separate n-gram indexes per artist.

    Args:
        artist_track_map: dict of artist_id -> list of track_ids

    Returns:
        Dict mapping artist_id -> {(codebook, n) -> set of n-gram tuples}
    """
    per_artist_index = {}
    for artist_id, track_ids in artist_track_map.items():
        index = {}
        for cb in codebooks:
            for n in ngram_sizes:
                index[(cb, n)] = set()

        for tid in track_ids:
            fpath = catalog_dir / f"{tid}.npy"
            if not fpath.exists():
                continue
            codes = np.load(str(fpath))
            for cb in codebooks:
                if cb >= codes.shape[0]:
                    continue
                for n in ngram_sizes:
                    index[(cb, n)].update(extract_ngrams(codes[cb], n))

        per_artist_index[artist_id] = index

    logger.info(f"Built per-artist indexes for {len(per_artist_index)} artists")
    return per_artist_index


def load_generation_log(run_dir: str) -> pd.DataFrame:
    """Load generation_log.csv which maps each file to tier/artist/genre."""
    log_path = Path(run_dir) / "manifests" / "generation_log.csv"
    if not log_path.exists():
        return None
    df = pd.read_csv(log_path)
    # Extract just the filename stem from file_path
    df["file_id"] = df["file_path"].apply(lambda p: Path(p).stem)
    return df


def main():
    parser = base_argparser("N-gram search: generated vs catalog tokens")
    parser.add_argument("--catalog_tokens_dir", type=str, default=None,
                        help="Dir with catalog .npy tokens (default: <run_dir>/tokens_catalog)")
    parser.add_argument("--tier", type=str, choices=TIERS, default=None,
                        help="Search only a specific tier (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["analysis"])

    mem_cfg = cfg.get("memorization", {})
    tok_cfg = cfg.get("tokenization", {})
    ngram_sizes = mem_cfg.get("ngram_sizes", [3, 4, 5, 6, 8])
    num_codebooks = tok_cfg.get("num_codebooks", 4)
    codebooks_cfg = mem_cfg.get("codebooks", None)
    codebooks = list(range(num_codebooks)) if codebooks_cfg is None else codebooks_cfg

    logger.info(f"N-gram sizes: {ngram_sizes}")
    logger.info(f"Codebooks: {codebooks}")

    catalog_dir = Path(args.catalog_tokens_dir) if args.catalog_tokens_dir \
        else Path(args.run_dir) / "tokens_catalog"

    # Load generation log for metadata
    gen_log = load_generation_log(args.run_dir)
    if gen_log is not None:
        logger.info(f"Generation log loaded: {len(gen_log)} entries")
        file_meta = {row["file_id"]: row for _, row in gen_log.iterrows()}
    else:
        logger.warning("No generation_log.csv found — tier/artist metadata will be inferred from dirs")
        file_meta = {}

    # Build artist -> track mapping from sampling manifest
    artist_track_map = {}
    manifest_path = Path(args.run_dir) / "manifests" / "sampling_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        id_col = "track_id" if "track_id" in manifest.columns else manifest.columns[0]
        artist_col = "artist_id" if "artist_id" in manifest.columns else "artist_name"
        for _, row in manifest.iterrows():
            aid = str(row[artist_col])
            tid = str(row[id_col])
            if aid not in artist_track_map:
                artist_track_map[aid] = []
            artist_track_map[aid].append(tid)
        logger.info(f"Artist-track mapping: {len(artist_track_map)} artists")

    # Build artist -> genre mapping (for same-genre mismatched control)
    artist_genre_map = {}
    if manifest_path.exists():
        genre_col = "genre" if "genre" in manifest.columns else None
        if genre_col:
            for _, row in manifest.iterrows():
                aid = str(row[artist_col])
                artist_genre_map[aid] = str(row[genre_col])
    # Also pull from generation log if available
    if gen_log is not None and "genre" in gen_log.columns:
        for _, row in gen_log.iterrows():
            raw_aid = row.get("artist_id", "")
            aid = str(int(float(raw_aid))) if pd.notna(raw_aid) and raw_aid != "" else ""
            g = str(row.get("genre", ""))
            if aid and g and g != "nan" and aid not in artist_genre_map:
                artist_genre_map[aid] = g
    logger.info(f"Artist-genre mapping: {len(artist_genre_map)} artists")

    # Build global catalog n-gram index
    catalog_index = build_catalog_index(catalog_dir, codebooks, ngram_sizes, logger)

    # Build per-artist catalog indexes (for matched vs mismatched analysis)
    per_artist_index = None
    if artist_track_map:
        per_artist_index = build_per_artist_catalog_index(
            catalog_dir, artist_track_map, codebooks, ngram_sizes, logger)

    # Determine tiers to process
    tiers_to_process = [args.tier] if args.tier else TIERS

    # ---- Global search: each generated file vs entire catalog ----
    results = []
    per_artist_results = []
    t0 = time.time()

    for tier in tiers_to_process:
        gen_dir = Path(args.run_dir) / "tokens_generated" / tier
        if not gen_dir.exists():
            logger.warning(f"Token dir not found: {gen_dir}")
            continue

        gen_files = sorted(gen_dir.glob("*.npy"))
        logger.info(f"[{tier}] Searching {len(gen_files)} files...")

        for idx, gen_path in enumerate(gen_files):
            file_id = gen_path.stem
            codes = np.load(str(gen_path))  # [n_codebooks, n_frames]

            # Get metadata from generation log
            fmeta = file_meta.get(file_id, {})
            raw_aid = fmeta.get("artist_id", "")
            artist_id = str(int(float(raw_aid))) if pd.notna(raw_aid) and raw_aid != "" else ""
            genre = str(fmeta.get("genre", ""))

            for cb in codebooks:
                if cb >= codes.shape[0]:
                    continue
                tokens = codes[cb]

                for n in ngram_sizes:
                    n_matches = count_ngram_matches(tokens, catalog_index[(cb, n)], n)
                    n_total = max(len(tokens) - n + 1, 0)

                    results.append({
                        "file_id": file_id,
                        "tier": tier,
                        "artist_id": artist_id,
                        "genre": genre,
                        "codebook": cb,
                        "ngram_size": n,
                        "matches": n_matches,
                        "total_ngrams": n_total,
                        "match_rate": round(n_matches / n_total, 8) if n_total > 0 else 0,
                    })

            # ---- Per-artist analysis for Tier A and D ----
            if per_artist_index and tier in ("A_artist_proximal", "D_fma_tags") and artist_id:
                for n in ngram_sizes:
                    # Matched: this file vs its own artist's catalog
                    matched_matches = 0
                    matched_total = 0
                    if artist_id in per_artist_index:
                        for cb in codebooks:
                            if cb >= codes.shape[0]:
                                continue
                            tokens = codes[cb]
                            m = count_ngram_matches(tokens, per_artist_index[artist_id][(cb, n)], n)
                            matched_matches += m
                            matched_total += max(len(tokens) - n + 1, 0)

                    # Mismatched: same-genre OTHER artists only (avoids cross-genre confound)
                    mismatched_matches = 0
                    mismatched_total = 0
                    mismatched_all_matches = 0
                    mismatched_all_total = 0
                    n_same_genre = 0
                    n_other_all = 0
                    for other_id, other_index in per_artist_index.items():
                        if other_id == artist_id:
                            continue
                        other_genre = artist_genre_map.get(other_id, "")
                        is_same_genre = (other_genre == genre and genre)

                        cb_matches = 0
                        cb_total = 0
                        for cb in codebooks:
                            if cb >= codes.shape[0]:
                                continue
                            tokens = codes[cb]
                            m = count_ngram_matches(tokens, other_index[(cb, n)], n)
                            cb_matches += m
                            cb_total += max(len(tokens) - n + 1, 0)

                        # All-artists mismatched (for backwards compat)
                        mismatched_all_matches += cb_matches
                        mismatched_all_total += cb_total
                        n_other_all += 1

                        # Same-genre mismatched (primary control)
                        if is_same_genre:
                            mismatched_matches += cb_matches
                            mismatched_total += cb_total
                            n_same_genre += 1

                    per_artist_results.append({
                        "file_id": file_id,
                        "tier": tier,
                        "artist_id": artist_id,
                        "genre": genre,
                        "ngram_size": n,
                        "matched_matches": matched_matches,
                        "matched_total": matched_total,
                        "matched_rate": round(matched_matches / matched_total, 8) if matched_total > 0 else 0,
                        # Same-genre mismatched (primary — controls for genre confound)
                        "mismatched_matches": mismatched_matches,
                        "mismatched_total": mismatched_total,
                        "mismatched_rate": round(mismatched_matches / mismatched_total, 8) if mismatched_total > 0 else 0,
                        "n_same_genre_artists": n_same_genre,
                        "mismatched_rate_per_artist": round(
                            (mismatched_matches / mismatched_total / n_same_genre), 10
                        ) if mismatched_total > 0 and n_same_genre > 0 else 0,
                        # All-artists mismatched (secondary — for cross-genre analysis)
                        "mismatched_all_matches": mismatched_all_matches,
                        "mismatched_all_total": mismatched_all_total,
                        "mismatched_all_rate": round(mismatched_all_matches / mismatched_all_total, 8) if mismatched_all_total > 0 else 0,
                        "n_other_artists": n_other_all,
                        "mismatched_all_rate_per_artist": round(
                            (mismatched_all_matches / mismatched_all_total / n_other_all), 10
                        ) if mismatched_all_total > 0 and n_other_all > 0 else 0,
                    })

            if (idx + 1) % 100 == 0 or (idx + 1) == len(gen_files):
                elapsed = time.time() - t0
                logger.info(f"  [{tier}] [{idx+1}/{len(gen_files)}] ({elapsed:.0f}s)")

    # ---- Write results ----
    results_csv = dirs["analysis"] / "ngram_matches.csv"
    fieldnames = ["file_id", "tier", "artist_id", "genre", "codebook",
                  "ngram_size", "matches", "total_ngrams", "match_rate"]
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Wrote {len(results)} rows to {results_csv}")

    # Per-artist matched vs mismatched
    if per_artist_results:
        pa_csv = dirs["analysis"] / "ngram_per_artist.csv"
        pa_fields = ["file_id", "tier", "artist_id", "genre", "ngram_size",
                     "matched_matches", "matched_total", "matched_rate",
                     "mismatched_matches", "mismatched_total", "mismatched_rate",
                     "n_same_genre_artists", "mismatched_rate_per_artist",
                     "mismatched_all_matches", "mismatched_all_total",
                     "mismatched_all_rate", "n_other_artists",
                     "mismatched_all_rate_per_artist"]
        with open(pa_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pa_fields)
            writer.writeheader()
            writer.writerows(per_artist_results)
        logger.info(f"Wrote {len(per_artist_results)} per-artist rows to {pa_csv}")

    # ---- Summary by tier and n-gram size ----
    df = pd.DataFrame(results)
    summary = {"by_tier": {}, "by_tier_genre": {}}

    for tier in df["tier"].unique():
        summary["by_tier"][tier] = {}
        tier_df = df[df["tier"] == tier]
        for n in ngram_sizes:
            n_df = tier_df[tier_df["ngram_size"] == n]
            if n_df.empty:
                continue
            total_matches = int(n_df["matches"].sum())
            total_ngrams = int(n_df["total_ngrams"].sum())
            files_with_matches = int(n_df[n_df["matches"] > 0]["file_id"].nunique())
            total_files = int(n_df["file_id"].nunique())
            summary["by_tier"][tier][str(n)] = {
                "total_matches": total_matches,
                "total_ngrams": total_ngrams,
                "aggregate_match_rate": round(total_matches / total_ngrams, 10) if total_ngrams > 0 else 0,
                "files_with_matches": files_with_matches,
                "total_files": total_files,
            }

    # Per-artist summary
    if per_artist_results:
        pa_df = pd.DataFrame(per_artist_results)
        summary["per_artist_matched_vs_mismatched"] = {}
        for n in ngram_sizes:
            n_df = pa_df[pa_df["ngram_size"] == n]
            if n_df.empty:
                continue
            summary["per_artist_matched_vs_mismatched"][str(n)] = {
                "mean_matched_rate": float(n_df["matched_rate"].mean()),
                "mean_mismatched_rate": float(n_df["mismatched_rate_per_artist"].mean()),
                "ratio": float(
                    n_df["matched_rate"].mean() / n_df["mismatched_rate_per_artist"].mean()
                ) if n_df["mismatched_rate_per_artist"].mean() > 0 else float("inf"),
                "n_files": len(n_df),
            }

    summary_path = dirs["analysis"] / "ngram_match_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Log summary
    logger.info(f"\n=== N-GRAM SEARCH SUMMARY ===")
    for tier in sorted(summary["by_tier"].keys()):
        for n, stats in summary["by_tier"][tier].items():
            logger.info(f"  [{tier}] {n}-gram: "
                        f"{stats['total_matches']} matches across "
                        f"{stats['files_with_matches']}/{stats['total_files']} files, "
                        f"rate={stats['aggregate_match_rate']:.10f}")

    if "per_artist_matched_vs_mismatched" in summary:
        logger.info(f"\n=== PER-ARTIST MATCHED vs MISMATCHED ===")
        for n, stats in summary["per_artist_matched_vs_mismatched"].items():
            logger.info(f"  {n}-gram: matched={stats['mean_matched_rate']:.10f}, "
                        f"mismatched={stats['mean_mismatched_rate']:.10f}, "
                        f"ratio={stats['ratio']:.2f}x")

    elapsed_total = time.time() - t0
    logger.info(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}m)")

    outputs = [str(results_csv), str(summary_path)]
    if per_artist_results:
        outputs.append(str(dirs["analysis"] / "ngram_per_artist.csv"))
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    meta["summary"] = summary
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
