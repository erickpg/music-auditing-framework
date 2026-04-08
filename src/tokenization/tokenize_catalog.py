#!/usr/bin/env python3
"""Stage T1: Tokenize catalog audio through EnCodec.

For each master audio file, encode through EnCodec to extract discrete
codec tokens (without decoding back). Saves token arrays as .npy files
for downstream n-gram memorization analysis.

EnCodec 32kHz produces 4 codebooks × 50 Hz = 200 tokens/sec.
A 30s clip → 1500 frames × 4 codebooks = 6000 tokens.

Outputs:
    <run_dir>/tokens_catalog/<track_id>.npy   (int16, shape [n_codebooks, n_frames])
    <run_dir>/manifests/tokenize_catalog_log.csv
    <run_dir>/logs/tokenize_catalog.log
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

STAGE = "tokenize_catalog"


def tokenize_audio(signal: np.ndarray, sr: int, encodec_model, encodec_processor, device):
    """Encode audio through EnCodec and return discrete token codes.

    Returns:
        np.ndarray of shape [n_codebooks, n_frames], dtype int16
    """
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

    # audio_codes shape varies by transformers version:
    #   Old: [batch, n_codebooks, n_frames]
    #   New: [batch, n_codebooks, 1, n_frames]  (extra chunk dim)
    codes = encoder_outputs.audio_codes.cpu().numpy().astype(np.int16)
    # Squeeze until we get [n_codebooks, n_frames]
    while codes.ndim > 2:
        codes = codes.squeeze(0)
    return codes


def main():
    parser = base_argparser("Tokenize catalog audio through EnCodec")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="Directory with master audio files (default: <run_dir>/masters_clean)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["tokens_catalog", "manifests"])

    tok_cfg = cfg.get("tokenization", {})
    codec_model_name = tok_cfg.get("codec_model", "facebook/encodec_32khz")
    native_sr = cfg["data"].get("sample_rate", 32000)

    source_dir = Path(args.source_dir) if args.source_dir else Path(args.run_dir) / "masters_clean"
    logger.info(f"Source audio dir: {source_dir}")
    logger.info(f"EnCodec model: {codec_model_name}")

    # Discover audio files
    audio_files = sorted(source_dir.glob("*.wav"))
    if not audio_files:
        audio_files = sorted(source_dir.glob("*.flac"))
    logger.info(f"Found {len(audio_files)} audio files")

    if not audio_files:
        logger.error("No audio files found!")
        return

    # Load EnCodec
    logger.info("Loading EnCodec model...")
    from transformers import EncodecModel, AutoProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encodec_model = EncodecModel.from_pretrained(codec_model_name).to(device)
    encodec_processor = AutoProcessor.from_pretrained(codec_model_name)
    encodec_sr = encodec_processor.sampling_rate
    logger.info(f"EnCodec loaded on {device}, native SR: {encodec_sr}")

    # Process each file
    results = []
    t0 = time.time()
    out_dir = dirs["tokens_catalog"]

    for idx, audio_path in enumerate(audio_files):
        track_id = audio_path.stem
        try:
            signal, sr = sf.read(str(audio_path))
            if signal.ndim > 1:
                signal = signal.mean(axis=1)

            # Resample to EnCodec SR if needed
            if sr != encodec_sr:
                import librosa
                signal = librosa.resample(signal, orig_sr=sr, target_sr=encodec_sr)

            codes = tokenize_audio(signal, encodec_sr, encodec_model, encodec_processor, device)

            # Save as .npy
            out_path = out_dir / f"{track_id}.npy"
            np.save(str(out_path), codes)

            n_codebooks = codes.shape[0]
            n_frames = codes.shape[-1]
            duration_s = len(signal) / sr

            results.append({
                "track_id": track_id,
                "n_codebooks": n_codebooks,
                "n_frames": n_frames,
                "duration_s": round(duration_s, 2),
                "tokens_per_sec": round(n_frames / duration_s, 1) if duration_s > 0 else 0,
                "status": "ok",
            })

        except Exception as e:
            logger.error(f"Error tokenizing {track_id}: {e}")
            results.append({
                "track_id": track_id,
                "n_codebooks": 0,
                "n_frames": 0,
                "duration_s": 0,
                "tokens_per_sec": 0,
                "status": f"error: {e}",
            })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(audio_files):
            elapsed = time.time() - t0
            ok_count = sum(1 for r in results if r["status"] == "ok")
            logger.info(f"  [{idx+1}/{len(audio_files)}] ok={ok_count} ({elapsed:.0f}s)")

    # Write log CSV
    log_csv = dirs["manifests"] / "tokenize_catalog_log.csv"
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "track_id", "n_codebooks", "n_frames", "duration_s", "tokens_per_sec", "status",
        ])
        writer.writeheader()
        writer.writerows(results)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    total_frames = sum(r["n_frames"] for r in results if r["status"] == "ok")
    logger.info(f"Tokenized {ok_count}/{len(results)} tracks, {total_frames} total frames")

    elapsed_total = time.time() - t0
    logger.info(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}m)")

    outputs = [str(out_dir), str(log_csv)]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    meta["tracks_tokenized"] = ok_count
    meta["total_frames"] = int(total_frames)
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
