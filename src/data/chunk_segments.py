#!/usr/bin/env python3
"""Stage 05: Chunk masters into 10-30s training segments.

Splits audio files into fixed-length segments with optional overlap.
Segments shorter than min_segment_length_s are discarded.

Outputs:
    <run_dir>/segments/<track_id>_<start_ms>_<end_ms>.wav
    <run_dir>/manifests/segment_manifest.csv
    <run_dir>/logs/chunk_segments.log
    <run_dir>/logs/chunk_segments_meta.json
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "chunk_segments"


def chunk_signal(signal: np.ndarray, sr: int, segment_s: float,
                 overlap_s: float, min_s: float):
    """Yield (start_sample, end_sample) tuples for chunking."""
    total_samples = len(signal)
    seg_samples = int(segment_s * sr)
    hop_samples = int((segment_s - overlap_s) * sr)
    min_samples = int(min_s * sr)

    start = 0
    while start < total_samples:
        end = min(start + seg_samples, total_samples)
        if (end - start) >= min_samples:
            yield start, end
        start += hop_samples


def main():
    parser = base_argparser("Chunk masters into training segments")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="Dir with master audio (default: <run_dir>/masters_clean)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["segments", "manifests"])

    seg_len = cfg["chunking"]["segment_length_s"]
    overlap = cfg["chunking"]["overlap_s"]
    min_len = cfg["chunking"]["min_segment_length_s"]
    target_sr = cfg["data"].get("sample_rate", 32000)

    logger.info(f"Segment length: {seg_len}s, overlap: {overlap}s, min: {min_len}s")

    source_dir = Path(args.source_dir) if args.source_dir else Path(args.run_dir) / "masters_clean"
    logger.info(f"Source dir: {source_dir}")

    audio_files = sorted(source_dir.glob("*.wav"))
    if not audio_files:
        audio_files = sorted(source_dir.glob("*.flac"))
    logger.info(f"Found {len(audio_files)} audio files")

    if not audio_files:
        logger.error("No audio files found!")
        return

    out_dir = dirs["segments"]
    manifest_rows = []
    total_segments = 0
    t0 = time.time()

    for idx, audio_path in enumerate(audio_files):
        track_id = audio_path.stem
        try:
            signal, sr = sf.read(str(audio_path))
            if signal.ndim > 1:
                signal = signal.mean(axis=1)

            for start, end in chunk_signal(signal, sr, seg_len, overlap, min_len):
                start_ms = int(start / sr * 1000)
                end_ms = int(end / sr * 1000)
                seg_name = f"{track_id}_{start_ms:08d}_{end_ms:08d}.wav"
                seg_path = out_dir / seg_name

                sf.write(str(seg_path), signal[start:end], sr)

                manifest_rows.append({
                    "segment_path": str(seg_path),
                    "segment_name": seg_name,
                    "track_id": track_id,
                    "start_s": round(start / sr, 3),
                    "end_s": round(end / sr, 3),
                    "duration_s": round((end - start) / sr, 3),
                })
                total_segments += 1

        except Exception as e:
            logger.error(f"Error chunking {track_id}: {e}")

        if (idx + 1) % 50 == 0 or (idx + 1) == len(audio_files):
            elapsed = time.time() - t0
            logger.info(f"  [{idx+1}/{len(audio_files)}] segments={total_segments} ({elapsed:.0f}s)")

    # Write manifest
    manifest_csv = dirs["manifests"] / "segment_manifest.csv"
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "segment_path", "segment_name", "track_id",
            "start_s", "end_s", "duration_s",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_duration = sum(r["duration_s"] for r in manifest_rows)
    logger.info(f"Created {total_segments} segments from {len(audio_files)} tracks")
    logger.info(f"Total segment duration: {total_duration:.1f}s ({total_duration/3600:.2f}h)")

    elapsed = time.time() - t0
    logger.info(f"Time: {elapsed:.1f}s")

    outputs = [str(out_dir), str(manifest_csv)]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    meta["total_segments"] = total_segments
    meta["total_duration_s"] = round(total_duration, 1)
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
