#!/usr/bin/env python3
"""Stage T2: Tokenize generated audio through EnCodec.

Tokenizes all generated outputs across tiers (A/B/C/D). Reads
generation_log.csv to discover files and preserve tier/artist metadata.

Directory structure:
    <run_dir>/generated/A_artist_proximal/*.wav
    <run_dir>/generated/B_genre_generic/*.wav
    <run_dir>/generated/C_out_of_distribution/*.wav
    <run_dir>/generated/D_fma_tags/*.wav

Outputs:
    <run_dir>/tokens_generated/<tier>/<filename>.npy
    <run_dir>/manifests/tokenize_generated_log.csv
    <run_dir>/logs/tokenize_generated.log
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
from src.tokenization.tokenize_catalog import tokenize_audio

STAGE = "tokenize_generated"

TIERS = [
    "A_artist_proximal",
    "B_genre_generic",
    "C_out_of_distribution",
    "D_fma_tags",
]


def main():
    parser = base_argparser("Tokenize generated audio through EnCodec")
    parser.add_argument("--tier", type=str, choices=TIERS, default=None,
                        help="Tokenize only a specific tier (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)

    # Create output dirs for each tier
    token_subdirs = [f"tokens_generated/{t}" for t in TIERS]
    dirs = ensure_dirs(args.run_dir, token_subdirs + ["manifests"])

    tok_cfg = cfg.get("tokenization", {})
    codec_model_name = tok_cfg.get("codec_model", "facebook/encodec_32khz")

    # Load EnCodec
    logger.info("Loading EnCodec model...")
    from transformers import EncodecModel, AutoProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encodec_model = EncodecModel.from_pretrained(codec_model_name).to(device)
    encodec_processor = AutoProcessor.from_pretrained(codec_model_name)
    encodec_sr = encodec_processor.sampling_rate
    logger.info(f"EnCodec loaded on {device}, native SR: {encodec_sr}")

    # Determine which tiers to process
    tiers_to_process = [args.tier] if args.tier else TIERS

    results = []
    t0 = time.time()

    for tier in tiers_to_process:
        gen_dir = Path(args.run_dir) / "generated" / tier
        if not gen_dir.exists():
            logger.warning(f"Directory not found: {gen_dir}")
            continue

        audio_files = sorted(gen_dir.glob("*.wav"))
        logger.info(f"[{tier}] Found {len(audio_files)} audio files")

        out_dir = dirs[f"tokens_generated/{tier}"]

        for idx, audio_path in enumerate(audio_files):
            file_id = audio_path.stem
            out_path = out_dir / f"{file_id}.npy"

            # Skip if already tokenized (resume-safe)
            if out_path.exists():
                results.append({
                    "file_id": file_id,
                    "tier": tier,
                    "n_codebooks": 0,
                    "n_frames": 0,
                    "status": "skipped",
                })
                continue

            try:
                signal, sr = sf.read(str(audio_path))
                if signal.ndim > 1:
                    signal = signal.mean(axis=1)

                if sr != encodec_sr:
                    import librosa
                    signal = librosa.resample(signal, orig_sr=sr, target_sr=encodec_sr)

                codes = tokenize_audio(signal, encodec_sr, encodec_model, encodec_processor, device)

                np.save(str(out_path), codes)

                n_codebooks = codes.shape[0]
                n_frames = codes.shape[-1]
                results.append({
                    "file_id": file_id,
                    "tier": tier,
                    "n_codebooks": n_codebooks,
                    "n_frames": n_frames,
                    "status": "ok",
                })

            except Exception as e:
                logger.error(f"Error tokenizing {tier}/{file_id}: {e}")
                results.append({
                    "file_id": file_id,
                    "tier": tier,
                    "n_codebooks": 0,
                    "n_frames": 0,
                    "status": f"error: {e}",
                })

            if (idx + 1) % 100 == 0 or (idx + 1) == len(audio_files):
                elapsed = time.time() - t0
                ok = sum(1 for r in results if r["status"] == "ok")
                logger.info(f"  [{tier}] [{idx+1}/{len(audio_files)}] ok={ok} ({elapsed:.0f}s)")

    # Write log CSV
    log_csv = Path(args.run_dir) / "manifests" / "tokenize_generated_log.csv"
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file_id", "tier", "n_codebooks", "n_frames", "status",
        ])
        writer.writeheader()
        writer.writerows(results)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    skip_count = sum(1 for r in results if r["status"] == "skipped")
    logger.info(f"Tokenized {ok_count} files, skipped {skip_count}")

    elapsed_total = time.time() - t0
    logger.info(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}m)")

    output_dirs = [str(dirs[f"tokens_generated/{t}"]) for t in tiers_to_process]
    meta = log_finish(logger, meta, STAGE, outputs=output_dirs + [str(log_csv)])
    meta["files_tokenized"] = ok_count
    meta["files_skipped"] = skip_count
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
