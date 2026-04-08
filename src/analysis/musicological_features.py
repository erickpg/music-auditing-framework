#!/usr/bin/env python3
"""Stage V3: Extract musicological features for per-artist vulnerability assessment.

Extracts low-level audio features using librosa for both catalog and
generated audio across all tiers. Features capture timbral, rhythmic,
and harmonic characteristics that define an artist's sonic signature.

Outputs:
    <run_dir>/analysis/features_catalog.csv
    <run_dir>/analysis/features_generated.csv      (all tiers, with metadata)
    <run_dir>/analysis/features_per_artist.csv     (per-artist cosine similarity)
    <run_dir>/logs/musicological_features.log
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "musicological_features"

TIERS = [
    "A_artist_proximal",
    "B_genre_generic",
    "C_out_of_distribution",
    "D_fma_tags",
]


def extract_features(signal: np.ndarray, sr: int, cfg: dict) -> dict:
    """Extract musicological features from audio signal."""
    import librosa

    hop_length = cfg.get("hop_length", 512)
    n_fft = cfg.get("n_fft", 2048)
    features = {}

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    # Spectral bandwidth
    bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(bw))
    features["spectral_bandwidth_std"] = float(np.std(bw))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)
    for i in range(contrast.shape[0]):
        features[f"spectral_contrast_{i}_mean"] = float(np.mean(contrast[i]))

    # Spectral flatness (tonality)
    flatness = librosa.feature.spectral_flatness(y=signal, hop_length=hop_length, n_fft=n_fft)[0]
    features["spectral_flatness_mean"] = float(np.mean(flatness))
    features["spectral_flatness_std"] = float(np.std(flatness))

    # MFCCs
    n_mfcc = cfg.get("mfcc_n", 13)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    for i in range(n_mfcc):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

    # Delta MFCCs (temporal dynamics)
    delta_mfccs = librosa.feature.delta(mfccs)
    for i in range(n_mfcc):
        features[f"delta_mfcc_{i}_mean"] = float(np.mean(delta_mfccs[i]))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)
    for i in range(12):
        features[f"chroma_{i}_mean"] = float(np.mean(chroma[i]))

    # Tonnetz (harmonic)
    try:
        tonnetz = librosa.feature.tonnetz(y=signal, sr=sr)
        for i in range(6):
            features[f"tonnetz_{i}_mean"] = float(np.mean(tonnetz[i]))
    except Exception:
        for i in range(6):
            features[f"tonnetz_{i}_mean"] = 0.0

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=signal, sr=sr, hop_length=hop_length)
    features["tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # RMS energy
    rms = librosa.feature.rms(y=signal, hop_length=hop_length)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=signal, hop_length=hop_length)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    return features


def process_directory(audio_dir: Path, feat_cfg: dict, logger) -> list:
    """Extract features from all audio files in a directory."""
    import soundfile as sf

    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        audio_files = sorted(audio_dir.glob("*.flac"))

    results = []
    for idx, path in enumerate(audio_files):
        try:
            signal, sr = sf.read(str(path))
            if signal.ndim > 1:
                signal = signal.mean(axis=1)

            feats = extract_features(signal, sr, feat_cfg)
            feats["file_id"] = path.stem
            feats["status"] = "ok"
            results.append(feats)

        except Exception as e:
            logger.error(f"Error extracting features from {path.name}: {e}")
            results.append({"file_id": path.stem, "status": f"error: {e}"})

        if (idx + 1) % 50 == 0 or (idx + 1) == len(audio_files):
            logger.info(f"    [{idx+1}/{len(audio_files)}]")

    return results


def main():
    parser = base_argparser("Extract musicological features")
    parser.add_argument("--catalog_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["analysis"])

    vuln_cfg = cfg.get("vulnerability", {})
    feat_cfg = vuln_cfg.get("musicological", {})
    logger.info(f"Feature config: hop={feat_cfg.get('hop_length', 512)}, "
                f"n_fft={feat_cfg.get('n_fft', 2048)}, "
                f"mfcc_n={feat_cfg.get('mfcc_n', 13)}")

    t0 = time.time()

    # ---- Load generation log for metadata ----
    gen_log_path = Path(args.run_dir) / "manifests" / "generation_log.csv"
    file_meta = {}
    if gen_log_path.exists():
        gen_log = pd.read_csv(gen_log_path)
        gen_log["file_id"] = gen_log["file_path"].apply(lambda p: Path(p).stem)
        file_meta = {row["file_id"]: row for _, row in gen_log.iterrows()}

    # ---- Load artist mapping for catalog ----
    track_to_artist = {}
    manifest_path = Path(args.run_dir) / "manifests" / "sampling_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        id_col = "track_id" if "track_id" in manifest.columns else manifest.columns[0]
        artist_col = "artist_id" if "artist_id" in manifest.columns else "artist_name"
        for _, row in manifest.iterrows():
            track_to_artist[str(row[id_col])] = str(row[artist_col])

    # ---- Process catalog ----
    catalog_dir = Path(args.catalog_dir) if args.catalog_dir else Path(args.run_dir) / "masters_clean"
    logger.info(f"Processing catalog: {catalog_dir}")
    catalog_results = process_directory(catalog_dir, feat_cfg, logger)

    # Add artist metadata to catalog
    for r in catalog_results:
        fid = r.get("file_id", "")
        r["artist_id"] = track_to_artist.get(fid, track_to_artist.get(fid.lstrip("0"), ""))

    if catalog_results:
        cat_df = pd.DataFrame(catalog_results)
        cat_csv = dirs["analysis"] / "features_catalog.csv"
        cat_df.to_csv(cat_csv, index=False)
        logger.info(f"Catalog: {sum(1 for r in catalog_results if r.get('status') == 'ok')} ok")

    # ---- Process generated (all tiers) ----
    all_gen_results = []
    for tier in TIERS:
        gen_dir = Path(args.run_dir) / "generated" / tier
        if not gen_dir.exists():
            logger.warning(f"Dir not found: {gen_dir}")
            continue

        logger.info(f"Processing {tier}: {gen_dir}")
        gen_results = process_directory(gen_dir, feat_cfg, logger)

        # Add metadata
        for r in gen_results:
            fid = r.get("file_id", "")
            r["tier"] = tier
            fmeta = file_meta.get(fid, {})
            raw_aid = fmeta.get("artist_id", "")
            r["artist_id"] = str(int(float(raw_aid))) if pd.notna(raw_aid) and raw_aid != "" else ""
            r["genre"] = str(fmeta.get("genre", ""))

        all_gen_results.extend(gen_results)
        ok = sum(1 for r in gen_results if r.get("status") == "ok")
        logger.info(f"  [{tier}] {ok} ok")

    if all_gen_results:
        gen_df = pd.DataFrame(all_gen_results)
        gen_csv = dirs["analysis"] / "features_generated.csv"
        gen_df.to_csv(gen_csv, index=False)
        logger.info(f"Total generated features: {len(all_gen_results)}")

    # ---- Per-artist feature similarity ----
    per_artist_sim = []
    if catalog_results and all_gen_results:
        cat_df = pd.DataFrame([r for r in catalog_results if r.get("status") == "ok"])
        gen_df = pd.DataFrame([r for r in all_gen_results if r.get("status") == "ok"])

        # Identify numeric feature columns
        exclude_cols = {"file_id", "status", "tier", "artist_id", "genre"}
        feature_cols = [c for c in cat_df.columns
                        if c not in exclude_cols and cat_df[c].dtype in [np.float64, np.int64, float]]

        # Get unique artists
        artists = [a for a in cat_df["artist_id"].unique() if a and str(a) != "nan"]

        for artist_id in artists:
            # Catalog features for this artist
            cat_feats = cat_df[cat_df["artist_id"] == artist_id][feature_cols].dropna()
            if cat_feats.empty:
                continue
            cat_mean = cat_feats.mean().values

            # Matched: Tier A/D generations for this artist
            matched_gen = gen_df[(gen_df["artist_id"] == artist_id) &
                                  (gen_df["tier"].isin(["A_artist_proximal", "D_fma_tags"]))
                                  ][feature_cols].dropna()

            # Mismatched: Tier A/D generations for OTHER artists
            mismatched_gen = gen_df[(gen_df["artist_id"] != artist_id) &
                                     (gen_df["artist_id"] != "") &
                                     (gen_df["tier"].isin(["A_artist_proximal", "D_fma_tags"]))
                                     ][feature_cols].dropna()

            # Control: Tier C generations
            control_gen = gen_df[gen_df["tier"] == "C_out_of_distribution"][feature_cols].dropna()

            row = {"artist_id": artist_id}

            if not matched_gen.empty:
                gen_mean = matched_gen.mean().values
                row["matched_sim"] = float(1.0 - cosine(cat_mean, gen_mean))
                row["n_matched"] = len(matched_gen)

            if not mismatched_gen.empty:
                mm_mean = mismatched_gen.mean().values
                row["mismatched_sim"] = float(1.0 - cosine(cat_mean, mm_mean))

            if not control_gen.empty:
                ctrl_mean = control_gen.mean().values
                row["control_sim"] = float(1.0 - cosine(cat_mean, ctrl_mean))

            if "matched_sim" in row and "mismatched_sim" in row:
                row["sim_gap"] = row["matched_sim"] - row["mismatched_sim"]

            per_artist_sim.append(row)

        if per_artist_sim:
            pa_csv = dirs["analysis"] / "features_per_artist.csv"
            pd.DataFrame(per_artist_sim).to_csv(pa_csv, index=False)
            logger.info(f"Per-artist feature similarity: {len(per_artist_sim)} artists")

            pa_df = pd.DataFrame(per_artist_sim)
            if "matched_sim" in pa_df.columns:
                pa_df_sorted = pa_df.dropna(subset=["matched_sim"]).sort_values("matched_sim", ascending=False)
                logger.info(f"\n=== TOP 5 MUSICOLOGICAL SIMILARITY ===")
                for _, r in pa_df_sorted.head(5).iterrows():
                    logger.info(f"  {r['artist_id']}: matched={r['matched_sim']:.4f}, "
                                f"gap={r.get('sim_gap', 0):.4f}")

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    outputs = [str(dirs["analysis"] / f) for f in
               ["features_catalog.csv", "features_generated.csv", "features_per_artist.csv"]]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
