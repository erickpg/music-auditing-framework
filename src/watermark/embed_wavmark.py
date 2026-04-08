#!/usr/bin/env python3
"""Stage 04a: Embed WavMark watermark into standardized audio masters (POC).

Uses WavMark off-the-shelf to do a fast end-to-end pipeline test.
WavMark operates at 16kHz with 16 usable payload bits.

Workflow per track:
  1. Load 32kHz master
  2. Resample to 16kHz (WavMark requirement)
  3. Encode 16-bit payload: artist_id (8b) + album_id (4b) + CRC4 (4b)
  4. Embed watermark via WavMark encoder
  5. Resample back to 32kHz
  6. Save watermarked audio

Outputs:
    <run_dir>/masters_watermarked/<track_id>.wav
    <run_dir>/manifests/watermark_payload_log.csv
    <run_dir>/logs/embed_wavmark.log
    <run_dir>/logs/embed_wavmark_meta.json
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "embed_wavmark"

# WavMark internal sample rate
WAVMARK_SR = 16000


# ---------------------------------------------------------------------------
# POC Payload schema (16 bits)
# ---------------------------------------------------------------------------

def encode_poc_payload(artist_id: int, album_id: int) -> np.ndarray:
    """Encode artist/album into a 16-bit WavMark payload.

    Layout (16 bits):
        artist_id:  8 bits (0-255)
        album_id:   4 bits (0-15)
        crc4:       4 bits (integrity check over bits 0-11)
    """
    assert 0 <= artist_id < 256, f"artist_id must be 0-255, got {artist_id}"
    assert 0 <= album_id < 16, f"album_id must be 0-15, got {album_id}"

    # Pack 12 data bits
    data_bits = []
    for i in range(7, -1, -1):
        data_bits.append((artist_id >> i) & 1)
    for i in range(3, -1, -1):
        data_bits.append((album_id >> i) & 1)

    # CRC4 over the 12 data bits (simple XOR-based)
    data_val = (artist_id << 4) | album_id
    crc = 0
    for byte_val in [(data_val >> 4) & 0xFF, data_val & 0x0F]:
        crc ^= byte_val
    crc &= 0x0F

    crc_bits = []
    for i in range(3, -1, -1):
        crc_bits.append((crc >> i) & 1)

    payload = np.array(data_bits + crc_bits, dtype=np.int32)
    assert len(payload) == 16
    return payload


def decode_poc_payload(payload: np.ndarray) -> dict:
    """Decode a 16-bit WavMark payload and verify CRC."""
    assert len(payload) == 16

    # Extract artist_id (bits 0-7)
    artist_id = 0
    for i in range(8):
        artist_id = (artist_id << 1) | int(payload[i])

    # Extract album_id (bits 8-11)
    album_id = 0
    for i in range(8, 12):
        album_id = (album_id << 1) | int(payload[i])

    # Extract CRC4 (bits 12-15)
    crc_received = 0
    for i in range(12, 16):
        crc_received = (crc_received << 1) | int(payload[i])

    # Recompute CRC
    data_val = (artist_id << 4) | album_id
    crc_computed = 0
    for byte_val in [(data_val >> 4) & 0xFF, data_val & 0x0F]:
        crc_computed ^= byte_val
    crc_computed &= 0x0F

    return {
        "artist_id": artist_id,
        "album_id": album_id,
        "crc_valid": crc_received == crc_computed,
        "payload_hex": format(int("".join(str(b) for b in payload), 2), "04x"),
    }


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio signal using librosa."""
    if orig_sr == target_sr:
        return signal
    import librosa
    return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = base_argparser("Embed WavMark watermark into audio masters (POC)")
    parser.add_argument("--source_run_dir", type=str, default=None,
                        help="Run dir containing masters_clean/ (default: same as --run_dir)")
    parser.add_argument("--verify_n", type=int, default=10,
                        help="Number of tracks to verify with detector after embedding")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["masters_watermarked", "manifests"])

    wm_cfg = cfg["watermark"]
    source_run_dir = args.source_run_dir or cfg["data"].get("source_run_dir", args.run_dir)
    masters_dir = Path(source_run_dir) / "masters_clean"
    native_sr = cfg["data"].get("sample_rate", 32000)

    logger.info(f"Method: WavMark (off-the-shelf POC)")
    logger.info(f"Source masters: {masters_dir}")
    logger.info(f"Native sample rate: {native_sr} Hz")
    logger.info(f"WavMark sample rate: {WAVMARK_SR} Hz")
    logger.info(f"Payload bits: {wm_cfg.get('payload_bits', 16)}")

    # Load tracks manifest to get artist_id / album_id mapping
    tracks_csv = Path(source_run_dir) / "manifests" / "tracks_selected.csv"
    if not tracks_csv.exists():
        # Fallback: try run_dir
        tracks_csv = Path(args.run_dir) / "manifests" / "tracks_selected.csv"

    import pandas as pd
    tracks_df = pd.read_csv(tracks_csv)
    logger.info(f"Tracks manifest: {tracks_csv} ({len(tracks_df)} tracks)")

    # Build artist_id -> sequential index (0-255 for POC 8-bit field)
    unique_artists = sorted(tracks_df["artist_id"].unique())
    artist_id_map = {aid: idx for idx, aid in enumerate(unique_artists)}
    logger.info(f"Unique artists: {len(unique_artists)} (mapped to 0-{len(unique_artists)-1})")

    # Build album_id -> sequential index per artist (0-15 for POC 4-bit field)
    album_id_map = {}
    for artist_id in unique_artists:
        artist_albums = sorted(tracks_df[tracks_df["artist_id"] == artist_id]["album_id"].unique())
        for idx, album_id in enumerate(artist_albums):
            album_id_map[(artist_id, album_id)] = idx

    # Load WavMark model
    logger.info("Loading WavMark model...")
    import wavmark
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = wavmark.load_model().to(device)
    logger.info(f"WavMark model loaded on {device}")

    # Process each track
    log_rows = []
    processed = 0
    errors = 0
    t0 = time.time()

    for _, row in tracks_df.iterrows():
        track_id = row["track_id"]
        artist_id_orig = row["artist_id"]
        album_id_orig = row["album_id"]

        src_path = masters_dir / f"{track_id:06d}.wav"
        out_path = dirs["masters_watermarked"] / f"{track_id:06d}.wav"

        if not src_path.exists():
            logger.warning(f"Missing master: {src_path}")
            log_rows.append({
                "track_id": track_id, "artist_id": artist_id_orig,
                "album_id": album_id_orig, "status": "missing",
            })
            errors += 1
            continue

        try:
            # Map IDs to POC range
            poc_artist_id = artist_id_map[artist_id_orig]
            poc_album_id = album_id_map[(artist_id_orig, album_id_orig)]

            # Encode payload
            payload = encode_poc_payload(poc_artist_id, poc_album_id)

            # Load audio
            signal, sr = sf.read(str(src_path))
            if signal.ndim > 1:
                signal = signal.mean(axis=1)  # mono
            duration_s = len(signal) / sr

            # Resample to 16kHz for WavMark
            signal_16k = resample_signal(signal, sr, WAVMARK_SR)

            # Embed watermark
            watermarked_16k, info = wavmark.encode_watermark(
                model, signal_16k, payload, show_progress=False,
            )

            # Resample back to native SR
            watermarked = resample_signal(watermarked_16k, WAVMARK_SR, native_sr)

            # Match length to original (resampling can change length slightly)
            if len(watermarked) > len(signal):
                watermarked = watermarked[:len(signal)]
            elif len(watermarked) < len(signal):
                watermarked = np.pad(watermarked, (0, len(signal) - len(watermarked)))

            # Save
            sf.write(str(out_path), watermarked, native_sr)

            snr = info.get("snr", float("nan"))
            log_rows.append({
                "track_id": track_id,
                "artist_id": artist_id_orig,
                "artist_id_poc": poc_artist_id,
                "album_id": album_id_orig,
                "album_id_poc": poc_album_id,
                "payload_hex": format(int("".join(str(b) for b in payload), 2), "04x"),
                "snr_db": round(snr, 2) if not np.isnan(snr) else None,
                "duration_s": round(duration_s, 1),
                "status": "ok",
            })
            processed += 1

        except Exception as e:
            logger.error(f"Error embedding {track_id}: {e}")
            log_rows.append({
                "track_id": track_id, "artist_id": artist_id_orig,
                "album_id": album_id_orig, "status": f"error: {e}",
            })
            errors += 1

        total = processed + errors
        if total % 50 == 0 or total == len(tracks_df):
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            logger.info(f"  [{total}/{len(tracks_df)}] {processed} ok, {errors} errors "
                        f"({rate:.1f} tracks/s)")

    # Write payload log
    payload_log = dirs["manifests"] / "watermark_payload_log.csv"
    with open(payload_log, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "track_id", "artist_id", "artist_id_poc", "album_id", "album_id_poc",
            "payload_hex", "snr_db", "duration_s", "status",
        ])
        writer.writeheader()
        writer.writerows(log_rows)
    logger.info(f"Payload log: {payload_log}")

    # Verification: decode a few watermarked files
    if args.verify_n > 0:
        ok_rows = [r for r in log_rows if r["status"] == "ok"]
        verify_sample = ok_rows[:args.verify_n]
        logger.info(f"\n=== VERIFICATION ({len(verify_sample)} tracks) ===")

        verify_ok = 0
        for row in verify_sample:
            try:
                wm_path = dirs["masters_watermarked"] / f"{row['track_id']:06d}.wav"
                wm_signal, wm_sr = sf.read(str(wm_path))
                if wm_signal.ndim > 1:
                    wm_signal = wm_signal.mean(axis=1)
                wm_16k = resample_signal(wm_signal, wm_sr, WAVMARK_SR)

                decoded_payload, dec_info = wavmark.decode_watermark(
                    model, wm_16k, show_progress=False,
                )

                if decoded_payload is not None:
                    result = decode_poc_payload(decoded_payload)
                    match = (result["artist_id"] == row["artist_id_poc"]
                             and result["album_id"] == row["album_id_poc"]
                             and result["crc_valid"])
                    logger.info(f"  Track {row['track_id']}: DECODED "
                                f"artist={result['artist_id']} album={result['album_id']} "
                                f"crc={result['crc_valid']} match={match}")
                    if match:
                        verify_ok += 1
                else:
                    logger.warning(f"  Track {row['track_id']}: NO WATERMARK DETECTED")
            except Exception as e:
                logger.error(f"  Track {row['track_id']}: verify error: {e}")

        logger.info(f"Verification: {verify_ok}/{len(verify_sample)} correct payload recovery")

    # Summary
    elapsed_total = time.time() - t0
    logger.info(f"\n=== EMBEDDING SUMMARY ===")
    logger.info(f"Processed: {processed}/{len(tracks_df)}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}m)")
    if processed > 0:
        logger.info(f"Rate: {processed/elapsed_total:.1f} tracks/s")

    outputs = [
        str(dirs["masters_watermarked"]),
        str(payload_log),
    ]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    meta["processed"] = processed
    meta["errors"] = errors
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
