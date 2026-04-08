#!/usr/bin/env python3
"""Stage 07 (POC): Test WavMark watermark survival through EnCodec tokenizer.

For each watermarked file:
  1. Load watermarked audio (32kHz)
  2. Encode through EnCodec → discrete tokens
  3. Decode tokens back to waveform
  4. Resample to 16kHz and run WavMark detector
  5. Compare: does the watermark survive the codec bottleneck?

This is the critical gate for the POC. If watermarks don't survive EnCodec,
they won't survive MusicGen fine-tuning either.

Outputs:
    <run_dir>/analysis/tokenizer_survival.csv
    <run_dir>/analysis/tokenizer_survival_summary.json
    <run_dir>/logs/tokenizer_survival.log
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
from src.watermark.embed_wavmark import decode_poc_payload, resample_signal, WAVMARK_SR

STAGE = "tokenizer_survival"


def encode_decode_encodec(signal: np.ndarray, sr: int, encodec_model, encodec_processor, device):
    """Pass audio through EnCodec: encode to tokens, decode back to waveform."""
    # EnCodec expects mono audio at its native sample rate
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


def main():
    parser = base_argparser("Test WavMark survival through EnCodec (POC)")
    parser.add_argument("--sample_size", type=int, default=50,
                        help="Number of tracks to test (default: 50)")
    parser.add_argument("--source_run_dir", type=str, default=None,
                        help="Run dir with masters_watermarked/ (default: same as --run_dir)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["analysis"])

    source_run_dir = args.source_run_dir or args.run_dir
    wm_dir = Path(source_run_dir) / "masters_watermarked"
    native_sr = cfg["data"].get("sample_rate", 32000)

    codec_model_name = cfg.get("tokenizer_survival", {}).get(
        "codec_model", "facebook/encodec_32khz"
    )

    logger.info(f"Watermarked audio dir: {wm_dir}")
    logger.info(f"EnCodec model: {codec_model_name}")
    logger.info(f"Sample size: {args.sample_size}")

    # Load payload log to know expected payloads
    import pandas as pd
    payload_log = Path(source_run_dir) / "manifests" / "watermark_payload_log.csv"
    payload_df = pd.read_csv(payload_log)
    ok_tracks = payload_df[payload_df["status"] == "ok"]
    logger.info(f"Watermarked tracks available: {len(ok_tracks)}")

    # Sample
    if args.sample_size < len(ok_tracks):
        sample_df = ok_tracks.sample(n=args.sample_size, random_state=42)
    else:
        sample_df = ok_tracks
    logger.info(f"Testing {len(sample_df)} tracks")

    # Load EnCodec
    logger.info("Loading EnCodec model...")
    from transformers import EncodecModel, AutoProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encodec_model = EncodecModel.from_pretrained(codec_model_name).to(device)
    encodec_processor = AutoProcessor.from_pretrained(codec_model_name)
    encodec_sr = encodec_processor.sampling_rate
    logger.info(f"EnCodec loaded on {device}, native SR: {encodec_sr}")

    # Load WavMark model
    logger.info("Loading WavMark model...")
    import wavmark
    wm_model = wavmark.load_model().to(device)
    logger.info("WavMark model loaded")

    # Process each track
    results = []
    t0 = time.time()

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        track_id = row["track_id"]
        expected_artist = row["artist_id_poc"]
        expected_album = row["album_id_poc"]
        wm_path = wm_dir / f"{track_id:06d}.wav"

        try:
            # Load watermarked audio
            signal, sr = sf.read(str(wm_path))
            if signal.ndim > 1:
                signal = signal.mean(axis=1)

            # --- Pre-codec detection ---
            signal_16k = resample_signal(signal, sr, WAVMARK_SR)
            pre_payload, pre_info = wavmark.decode_watermark(
                wm_model, signal_16k, show_progress=False,
            )
            pre_detected = pre_payload is not None
            pre_match = False
            if pre_detected:
                pre_result = decode_poc_payload(pre_payload)
                pre_match = (pre_result["artist_id"] == expected_artist
                             and pre_result["album_id"] == expected_album
                             and pre_result["crc_valid"])

            # --- EnCodec encode/decode ---
            # Resample to EnCodec's native SR if needed
            if sr != encodec_sr:
                signal_enc = resample_signal(signal, sr, encodec_sr)
            else:
                signal_enc = signal

            reconstructed = encode_decode_encodec(
                signal_enc, encodec_sr, encodec_model, encodec_processor, device
            )

            # --- Post-codec detection ---
            recon_16k = resample_signal(reconstructed, encodec_sr, WAVMARK_SR)
            post_payload, post_info = wavmark.decode_watermark(
                wm_model, recon_16k, show_progress=False,
            )
            post_detected = post_payload is not None
            post_match = False
            if post_detected:
                post_result = decode_poc_payload(post_payload)
                post_match = (post_result["artist_id"] == expected_artist
                              and post_result["album_id"] == expected_album
                              and post_result["crc_valid"])

            results.append({
                "track_id": track_id,
                "artist_id_poc": expected_artist,
                "album_id_poc": expected_album,
                "pre_detected": pre_detected,
                "pre_match": pre_match,
                "post_detected": post_detected,
                "post_match": post_match,
                "status": "ok",
            })

        except Exception as e:
            logger.error(f"Error processing track {track_id}: {e}")
            results.append({
                "track_id": track_id,
                "artist_id_poc": expected_artist,
                "album_id_poc": expected_album,
                "pre_detected": False,
                "pre_match": False,
                "post_detected": False,
                "post_match": False,
                "status": f"error: {e}",
            })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(sample_df):
            elapsed = time.time() - t0
            post_det = sum(1 for r in results if r["post_detected"])
            post_match = sum(1 for r in results if r["post_match"])
            logger.info(f"  [{idx+1}/{len(sample_df)}] "
                        f"post_detected: {post_det}/{len(results)}, "
                        f"post_match: {post_match}/{len(results)} "
                        f"({elapsed:.0f}s)")

    # Write results CSV
    results_csv = dirs["analysis"] / "tokenizer_survival.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "track_id", "artist_id_poc", "album_id_poc",
            "pre_detected", "pre_match", "post_detected", "post_match", "status",
        ])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    ok_results = [r for r in results if r["status"] == "ok"]
    n = len(ok_results)
    pre_det_n = sum(1 for r in ok_results if r["pre_detected"])
    pre_match_n = sum(1 for r in ok_results if r["pre_match"])
    post_det_n = sum(1 for r in ok_results if r["post_detected"])
    post_match_n = sum(1 for r in ok_results if r["post_match"])

    summary = {
        "total_tested": n,
        "errors": len(results) - n,
        "pre_codec_detected": pre_det_n,
        "pre_codec_detected_pct": round(100 * pre_det_n / n, 1) if n > 0 else 0,
        "pre_codec_match": pre_match_n,
        "pre_codec_match_pct": round(100 * pre_match_n / n, 1) if n > 0 else 0,
        "post_codec_detected": post_det_n,
        "post_codec_detected_pct": round(100 * post_det_n / n, 1) if n > 0 else 0,
        "post_codec_match": post_match_n,
        "post_codec_match_pct": round(100 * post_match_n / n, 1) if n > 0 else 0,
        "survival_rate_detection": round(100 * post_det_n / pre_det_n, 1) if pre_det_n > 0 else 0,
        "survival_rate_match": round(100 * post_match_n / pre_match_n, 1) if pre_match_n > 0 else 0,
    }

    summary_path = dirs["analysis"] / "tokenizer_survival_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n=== TOKENIZER SURVIVAL SUMMARY ===")
    logger.info(f"Tested: {n} tracks ({len(results) - n} errors)")
    logger.info(f"Pre-codec:  detected={pre_det_n}/{n} ({summary['pre_codec_detected_pct']}%), "
                f"match={pre_match_n}/{n} ({summary['pre_codec_match_pct']}%)")
    logger.info(f"Post-codec: detected={post_det_n}/{n} ({summary['post_codec_detected_pct']}%), "
                f"match={post_match_n}/{n} ({summary['post_codec_match_pct']}%)")
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
