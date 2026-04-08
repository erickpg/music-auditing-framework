#!/usr/bin/env python3
"""Stage 04b: Test WaveVerify watermark survival through EnCodec tokenizer.

Quick test to see if WaveVerify (HILCodec-based, Jul 2025) survives EnCodec
where WavMark failed (6% detection, 0% match).

For each of 50 test tracks:
  1. Load 32kHz master
  2. Resample to 16kHz (WaveVerify requirement)
  3. Embed 16-bit WaveVerify watermark
  4. Verify immediate detection (pre-codec baseline)
  5. Resample to 32kHz, encode through EnCodec, decode back
  6. Resample to 16kHz, run WaveVerify detector
  7. Compare pre vs post codec detection and payload match

Outputs:
    <run_dir>/analysis/waveverify_survival.csv
    <run_dir>/analysis/waveverify_survival_summary.json
    <run_dir>/logs/waveverify_survival.log
"""

import csv
import json
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
from src.watermark.embed_wavmark import encode_poc_payload, decode_poc_payload, resample_signal

STAGE = "waveverify_survival"
WAVEVERIFY_SR = 16000  # WaveVerify operates at 16kHz


def encode_decode_encodec(signal: np.ndarray, sr: int, encodec_model, encodec_processor, device):
    """Pass audio through EnCodec: encode to tokens, decode back to waveform."""
    inputs = encodec_processor(
        raw_audio=signal,
        sampling_rate=sr,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        encoder_outputs = encodec_model.encode(
            inputs["input_values"],
            inputs.get("padding_mask", None),
        )
        decoded = encodec_model.decode(
            encoder_outputs.audio_codes,
            encoder_outputs.audio_scales,
            inputs.get("padding_mask", None),
        )

    reconstructed = decoded.audio_values.squeeze().cpu().numpy()
    return reconstructed


def payload_to_binary_string(payload: np.ndarray) -> str:
    """Convert a numpy payload array to a binary string for WaveVerify."""
    return "".join(str(int(b)) for b in payload)


def binary_string_to_payload(bits: str) -> np.ndarray:
    """Convert a binary string back to numpy payload array."""
    return np.array([int(b) for b in bits], dtype=np.int32)


def main():
    parser = base_argparser("Test WaveVerify survival through EnCodec")
    parser.add_argument("--sample_size", type=int, default=50,
                        help="Number of tracks to test (default: 50)")
    parser.add_argument("--source_run_dir", type=str, default=None,
                        help="Run dir with masters_clean/ and manifests/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["analysis", "masters_waveverify"])

    source_run_dir = args.source_run_dir or cfg["data"].get("source_run_dir",
                                                             args.run_dir)
    masters_dir = Path(source_run_dir) / "masters_clean"
    native_sr = cfg["data"].get("sample_rate", 32000)

    codec_model_name = cfg.get("tokenizer_survival", {}).get(
        "codec_model", "facebook/encodec_32khz"
    )

    logger.info(f"Masters dir: {masters_dir}")
    logger.info(f"EnCodec model: {codec_model_name}")
    logger.info(f"Native SR: {native_sr} Hz, WaveVerify SR: {WAVEVERIFY_SR} Hz")
    logger.info(f"Sample size: {args.sample_size}")

    # Load tracks manifest to get artist/album mapping
    import pandas as pd
    tracks_csv = Path(source_run_dir) / "manifests" / "tracks_selected.csv"
    tracks_df = pd.read_csv(tracks_csv)
    logger.info(f"Tracks manifest: {tracks_csv} ({len(tracks_df)} tracks)")

    # Build artist/album ID maps (same as embed_wavmark.py)
    unique_artists = sorted(tracks_df["artist_id"].unique())
    artist_id_map = {aid: idx for idx, aid in enumerate(unique_artists)}

    album_id_map = {}
    for artist_id in unique_artists:
        artist_albums = sorted(
            tracks_df[tracks_df["artist_id"] == artist_id]["album_id"].unique()
        )
        for idx, album_id in enumerate(artist_albums):
            album_id_map[(artist_id, album_id)] = idx

    # Sample tracks
    if args.sample_size < len(tracks_df):
        sample_df = tracks_df.sample(n=args.sample_size, random_state=42)
    else:
        sample_df = tracks_df
    logger.info(f"Testing {len(sample_df)} tracks")

    # Load EnCodec
    logger.info("Loading EnCodec model...")
    from transformers import EncodecModel, AutoProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encodec_model = EncodecModel.from_pretrained(codec_model_name).to(device)
    encodec_processor = AutoProcessor.from_pretrained(codec_model_name)
    encodec_sr = encodec_processor.sampling_rate
    logger.info(f"EnCodec loaded on {device}, native SR: {encodec_sr}")

    # Load WaveVerify
    logger.info("Loading WaveVerify model...")
    from waveverify import WaveVerify, WatermarkID
    wv = WaveVerify()
    logger.info("WaveVerify model loaded")

    # Process each track
    results = []
    t0 = time.time()

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        track_id = row["track_id"]
        artist_id_orig = row["artist_id"]
        album_id_orig = row["album_id"]
        poc_artist_id = artist_id_map[artist_id_orig]
        poc_album_id = album_id_map[(artist_id_orig, album_id_orig)]

        src_path = masters_dir / f"{track_id:06d}.wav"

        try:
            # Encode payload (same schema as WavMark POC)
            payload = encode_poc_payload(poc_artist_id, poc_album_id)
            payload_bits = payload_to_binary_string(payload)

            # Load audio
            signal, sr = sf.read(str(src_path))
            if signal.ndim > 1:
                signal = signal.mean(axis=1)

            # Resample to 16kHz for WaveVerify
            signal_16k = resample_signal(signal, sr, WAVEVERIFY_SR)

            # Save temp 16kHz file for WaveVerify (it needs file paths)
            tmp_input = dirs["masters_waveverify"] / f"{track_id:06d}_input.wav"
            sf.write(str(tmp_input), signal_16k, WAVEVERIFY_SR)

            # --- Embed watermark ---
            wm_id = WatermarkID.custom(payload_bits)
            tmp_output = dirs["masters_waveverify"] / f"{track_id:06d}_wm.wav"
            wv.embed(str(tmp_input), wm_id, output_path=str(tmp_output))

            # --- Pre-codec detection ---
            pre_wm_id, pre_confidence = wv.detect(str(tmp_output))
            pre_detected = pre_confidence > 0.5 if pre_confidence is not None else False

            pre_match = False
            pre_bit_accuracy = 0.0
            if pre_detected and pre_wm_id is not None:
                # Compare extracted bits with original
                pre_bits = str(pre_wm_id)
                if len(pre_bits) == 16 and len(payload_bits) == 16:
                    matching = sum(a == b for a, b in zip(pre_bits, payload_bits))
                    pre_bit_accuracy = matching / 16.0
                    pre_match = (pre_bits == payload_bits)

            # --- EnCodec roundtrip ---
            # Load watermarked audio, resample to 32kHz, run through EnCodec
            wm_signal_16k, _ = sf.read(str(tmp_output))
            wm_signal_32k = resample_signal(wm_signal_16k, WAVEVERIFY_SR, native_sr)

            # Resample to EnCodec SR if needed
            if native_sr != encodec_sr:
                wm_signal_enc = resample_signal(wm_signal_32k, native_sr, encodec_sr)
            else:
                wm_signal_enc = wm_signal_32k

            reconstructed = encode_decode_encodec(
                wm_signal_enc, encodec_sr, encodec_model, encodec_processor, device
            )

            # --- Post-codec detection ---
            # Resample EnCodec output back to 16kHz for WaveVerify
            recon_16k = resample_signal(reconstructed, encodec_sr, WAVEVERIFY_SR)
            tmp_recon = dirs["masters_waveverify"] / f"{track_id:06d}_recon.wav"
            sf.write(str(tmp_recon), recon_16k, WAVEVERIFY_SR)

            post_wm_id, post_confidence = wv.detect(str(tmp_recon))
            post_detected = post_confidence > 0.5 if post_confidence is not None else False

            post_match = False
            post_bit_accuracy = 0.0
            if post_detected and post_wm_id is not None:
                post_bits = str(post_wm_id)
                if len(post_bits) == 16 and len(payload_bits) == 16:
                    matching = sum(a == b for a, b in zip(post_bits, payload_bits))
                    post_bit_accuracy = matching / 16.0
                    post_match = (post_bits == payload_bits)

            # Clean up temp files (keep only _wm for potential later use)
            tmp_input.unlink(missing_ok=True)
            tmp_recon.unlink(missing_ok=True)

            results.append({
                "track_id": track_id,
                "artist_id_poc": poc_artist_id,
                "album_id_poc": poc_album_id,
                "pre_detected": pre_detected,
                "pre_confidence": round(pre_confidence, 4) if pre_confidence is not None else None,
                "pre_bit_accuracy": round(pre_bit_accuracy, 4),
                "pre_match": pre_match,
                "post_detected": post_detected,
                "post_confidence": round(post_confidence, 4) if post_confidence is not None else None,
                "post_bit_accuracy": round(post_bit_accuracy, 4),
                "post_match": post_match,
                "status": "ok",
            })

        except Exception as e:
            logger.error(f"Error processing track {track_id}: {e}")
            results.append({
                "track_id": track_id,
                "artist_id_poc": poc_artist_id,
                "album_id_poc": poc_album_id,
                "pre_detected": False,
                "pre_confidence": None,
                "pre_bit_accuracy": 0.0,
                "pre_match": False,
                "post_detected": False,
                "post_confidence": None,
                "post_bit_accuracy": 0.0,
                "post_match": False,
                "status": f"error: {e}",
            })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(sample_df):
            elapsed = time.time() - t0
            ok = [r for r in results if r["status"] == "ok"]
            post_det = sum(1 for r in ok if r["post_detected"])
            post_match = sum(1 for r in ok if r["post_match"])
            mean_post_ba = (
                np.mean([r["post_bit_accuracy"] for r in ok])
                if ok else 0.0
            )
            logger.info(
                f"  [{idx+1}/{len(sample_df)}] "
                f"post_det: {post_det}/{len(ok)}, "
                f"post_match: {post_match}/{len(ok)}, "
                f"mean_post_bit_acc: {mean_post_ba:.3f} "
                f"({elapsed:.0f}s)"
            )

    # Write results CSV
    results_csv = dirs["analysis"] / "waveverify_survival.csv"
    fieldnames = [
        "track_id", "artist_id_poc", "album_id_poc",
        "pre_detected", "pre_confidence", "pre_bit_accuracy", "pre_match",
        "post_detected", "post_confidence", "post_bit_accuracy", "post_match",
        "status",
    ]
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    ok_results = [r for r in results if r["status"] == "ok"]
    n = len(ok_results)
    errors = len(results) - n

    pre_det_n = sum(1 for r in ok_results if r["pre_detected"])
    pre_match_n = sum(1 for r in ok_results if r["pre_match"])
    post_det_n = sum(1 for r in ok_results if r["post_detected"])
    post_match_n = sum(1 for r in ok_results if r["post_match"])

    mean_pre_ba = np.mean([r["pre_bit_accuracy"] for r in ok_results]) if ok_results else 0.0
    mean_post_ba = np.mean([r["post_bit_accuracy"] for r in ok_results]) if ok_results else 0.0

    summary = {
        "method": "WaveVerify",
        "total_tested": n,
        "errors": errors,
        "pre_codec_detected": pre_det_n,
        "pre_codec_detected_pct": round(100 * pre_det_n / n, 1) if n > 0 else 0,
        "pre_codec_match": pre_match_n,
        "pre_codec_match_pct": round(100 * pre_match_n / n, 1) if n > 0 else 0,
        "pre_codec_mean_bit_accuracy": round(mean_pre_ba, 4),
        "post_codec_detected": post_det_n,
        "post_codec_detected_pct": round(100 * post_det_n / n, 1) if n > 0 else 0,
        "post_codec_match": post_match_n,
        "post_codec_match_pct": round(100 * post_match_n / n, 1) if n > 0 else 0,
        "post_codec_mean_bit_accuracy": round(mean_post_ba, 4),
        "survival_rate_detection": round(100 * post_det_n / pre_det_n, 1) if pre_det_n > 0 else 0,
        "survival_rate_match": round(100 * post_match_n / pre_match_n, 1) if pre_match_n > 0 else 0,
    }

    summary_path = dirs["analysis"] / "waveverify_survival_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n=== WAVEVERIFY TOKENIZER SURVIVAL SUMMARY ===")
    logger.info(f"Tested: {n} tracks ({errors} errors)")
    logger.info(f"Pre-codec:  detected={pre_det_n}/{n} ({summary['pre_codec_detected_pct']}%), "
                f"match={pre_match_n}/{n} ({summary['pre_codec_match_pct']}%), "
                f"mean_bit_acc={mean_pre_ba:.3f}")
    logger.info(f"Post-codec: detected={post_det_n}/{n} ({summary['post_codec_detected_pct']}%), "
                f"match={post_match_n}/{n} ({summary['post_codec_match_pct']}%), "
                f"mean_bit_acc={mean_post_ba:.3f}")
    logger.info(f"Survival rate (detection): {summary['survival_rate_detection']}%")
    logger.info(f"Survival rate (correct match): {summary['survival_rate_match']}%")

    elapsed_total = time.time() - t0
    logger.info(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}m)")

    outputs = [str(results_csv), str(summary_path)]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    meta["summary"] = summary
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
