#!/usr/bin/env python3
"""Stage 01: Standardize audio masters before embedding.

Decodes to PCM, resamples to target sample rate, applies channel policy,
and optionally normalizes loudness. Uses multiprocessing for speed.

Outputs:
    <run_dir>/masters_clean/<track_id>.wav
    <run_dir>/manifests/standardization_log.csv
    <run_dir>/logs/standardize_audio.log
    <run_dir>/logs/standardize_audio_meta.json
"""

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "standardize_audio"


def process_track(args_tuple):
    """Process a single track (designed for multiprocessing)."""
    src_path, out_path, track_id, artist_name, track_title, target_sr, target_channels, loudness_norm, loudness_lufs = args_tuple

    warnings.filterwarnings("ignore", message="Possible clipped samples")

    src_path = Path(src_path)
    out_path = Path(out_path)

    if not src_path.exists():
        return {
            "track_id": track_id, "artist_name": artist_name,
            "source_path": str(src_path), "output_path": str(out_path),
            "original_sr": None, "target_sr": target_sr,
            "original_channels": None, "target_channels": target_channels,
            "duration_s": None, "loudness_lufs": None,
            "status": "error: file not found",
        }

    try:
        # 1. Load and resample
        waveform, orig_sr = torchaudio.load(str(src_path))
        orig_channels = waveform.shape[0]
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resampler(waveform)

        # 2. Channel policy
        if waveform.shape[0] != target_channels:
            if target_channels == 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            elif target_channels == 2 and waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            else:
                waveform = waveform[:target_channels]

        # 3. Loudness normalization
        measured_lufs = None
        if loudness_norm:
            import pyloudnorm as pyln
            meter = pyln.Meter(target_sr)
            waveform_np = waveform.numpy().T  # (samples, channels)
            if waveform_np.ndim == 1:
                waveform_np = waveform_np[:, np.newaxis]

            current_lufs = meter.integrated_loudness(waveform_np)
            if current_lufs != float('-inf') and not np.isnan(current_lufs):
                waveform_np = pyln.normalize.loudness(waveform_np, current_lufs, loudness_lufs)
                peak = np.max(np.abs(waveform_np))
                if peak > 1.0:
                    waveform_np = waveform_np / peak * 0.99

            waveform = torch.from_numpy(waveform_np.T).float()

            check_np = waveform.numpy().T
            if check_np.ndim == 1:
                check_np = check_np[:, np.newaxis]
            measured_lufs = meter.integrated_loudness(check_np)

        # 4. Save as WAV
        duration_s = waveform.shape[1] / target_sr
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), waveform.numpy().T, target_sr, subtype='FLOAT')

        return {
            "track_id": track_id, "artist_name": artist_name,
            "source_path": str(src_path), "output_path": str(out_path),
            "original_sr": orig_sr, "target_sr": target_sr,
            "original_channels": orig_channels, "target_channels": target_channels,
            "duration_s": round(duration_s, 2),
            "loudness_lufs": round(measured_lufs, 2) if measured_lufs is not None else None,
            "status": "ok",
        }

    except Exception as e:
        return {
            "track_id": track_id, "artist_name": artist_name,
            "source_path": str(src_path), "output_path": str(out_path),
            "original_sr": None, "target_sr": target_sr,
            "original_channels": None, "target_channels": target_channels,
            "duration_s": None, "loudness_lufs": None,
            "status": f"error: {e}",
        }


def main():
    parser = base_argparser("Standardize audio masters (resample, channel policy, loudness)")
    parser.add_argument("--tracks_csv", type=str, default=None,
                        help="Path to tracks_selected.csv")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory containing downloaded audio")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["masters_clean", "manifests"])

    sr = cfg["data"]["sample_rate"]
    channels = cfg["data"]["channels"]
    loudness_norm = cfg["data"].get("loudness_normalize", False)
    loudness_lufs = cfg["data"].get("loudness_target_lufs", -14.0)
    n_workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    tracks_csv = args.tracks_csv or str(
        Path(args.run_dir) / "manifests" / "tracks_selected.csv"
    )
    audio_dir = args.audio_dir or cfg["data"]["sources"][0].get(
        "audio_local_path",
        str(Path(cfg["paths"]["scratch_base"]) / "fma_audio")
    )
    audio_dir = Path(audio_dir)

    logger.info(f"Target sample rate: {sr} Hz, channels: {channels}")
    logger.info(f"Loudness normalization: {loudness_norm} (target: {loudness_lufs} LUFS)")
    logger.info(f"Audio source: {audio_dir}")
    logger.info(f"Tracks CSV: {tracks_csv}")
    logger.info(f"Workers: {n_workers}")

    # Load track list
    tracks = pd.read_csv(tracks_csv)
    logger.info(f"Tracks to standardize: {len(tracks)}")

    # Build task list
    tasks = []
    for _, row in tracks.iterrows():
        src_path = str(audio_dir / row["fma_path"])
        track_id = row["track_id"]
        out_path = str(dirs["masters_clean"] / f"{track_id:06d}.wav")
        tasks.append((
            src_path, out_path, track_id, row["artist_name"],
            row.get("track_title", ""), sr, channels, loudness_norm, loudness_lufs
        ))

    # Process in parallel
    log_rows = []
    processed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_track, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            log_rows.append(result)
            if result["status"] == "ok":
                processed += 1
            else:
                errors += 1
            total = processed + errors
            if total % 50 == 0 or total == len(tasks):
                logger.info(f"  [{total}/{len(tasks)}] {processed} ok, {errors} errors")

    # Write standardization log
    log_df = pd.DataFrame(log_rows)
    log_csv = dirs["manifests"] / "standardization_log.csv"
    log_df.to_csv(log_csv, index=False)

    # Summary
    ok_df = log_df[log_df["status"] == "ok"]
    total_duration_h = ok_df["duration_s"].sum() / 3600 if len(ok_df) > 0 else 0

    logger.info(f"\n=== STANDARDIZATION SUMMARY ===")
    logger.info(f"Processed: {processed}/{len(tracks)}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total duration: {total_duration_h:.1f} hours")
    if len(ok_df) > 0:
        logger.info(f"Mean duration: {ok_df['duration_s'].mean():.1f}s")
        logger.info(f"Duration range: {ok_df['duration_s'].min():.1f}s - {ok_df['duration_s'].max():.1f}s")
        if loudness_norm and ok_df["loudness_lufs"].notna().any():
            lufs_vals = ok_df["loudness_lufs"].dropna()
            logger.info(f"Loudness: mean={lufs_vals.mean():.1f} LUFS, std={lufs_vals.std():.1f}")

    outputs = [str(dirs["masters_clean"]), str(log_csv)]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
