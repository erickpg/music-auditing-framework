#!/usr/bin/env python3
"""Stage 09: Generate audio outputs from the fine-tuned MusicGen model.

Uses audiocraft API with chavinlo-style lm state_dict checkpoint.
Loads prompts.json (built by build_prompts.py) and generates audio for each
prompt across multiple temperatures and seeds. Supports tiered prompt design:

  Tier A (artist-proximal): uses per_artist_seeds, per_artist temperatures
  Tier B (genre-generic):   uses general seeds and temperatures
  Tier C (OOD control):     uses general seeds and temperatures
  Tier D (FMA sub-genre):   uses per_artist_seeds, per_artist temperatures

Outputs:
    <run_dir>/generated/<tier>/<prompt_id>_t<T>_seed<S>.wav
    <run_dir>/manifests/generation_log.csv
    <run_dir>/logs/generate_outputs.log
    <run_dir>/logs/generate_outputs_meta.json
"""

import csv
import json
import sys
import time
from pathlib import Path

import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "generate_outputs"


def load_model(model_name: str, checkpoint: str = None, device=None, logger=None):
    """Load MusicGen model via audiocraft API, optionally with fine-tuned lm checkpoint."""
    from audiocraft.models import MusicGen

    # audiocraft uses short names like "small", "medium", "large"
    model_id = model_name.replace("facebook/musicgen-", "") if "facebook/" in model_name else model_name
    device_str = str(device) if device else "cuda"

    logger.info(f"Loading MusicGen-{model_id} via audiocraft...")
    model = MusicGen.get_pretrained(model_id, device=device_str)

    if checkpoint:
        logger.info(f"Loading fine-tuned lm state_dict from {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=device)
        model.lm.load_state_dict(state_dict)
        logger.info("Fine-tuned weights loaded")

    model.lm.eval()
    return model


def generate_audio(model, prompt: str, duration_s: float,
                   temperature: float, top_k: int, top_p: float,
                   seed: int):
    """Generate audio from a text prompt with deterministic seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model.set_generation_params(
        duration=duration_s,
        use_sampling=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    wav = model.generate([prompt])
    audio = wav[0, 0].cpu().numpy()
    sr = model.sample_rate
    return audio, sr


def main():
    parser = base_argparser("Generate audio outputs from fine-tuned MusicGen")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned lm state_dict (.pt file)")
    parser.add_argument("--baseline", action="store_true",
                        help="Use vanilla pretrained model (no fine-tuned checkpoint)")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Path to prompts.json (default: <run_dir>/manifests/prompts.json)")
    parser.add_argument("--tier", type=str, default=None,
                        choices=["A_artist_proximal", "B_genre_generic",
                                 "C_out_of_distribution", "D_fma_tags"],
                        help="Generate only a specific tier (default: all)")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit total prompts (for testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, [
        "generated/A_artist_proximal",
        "generated/B_genre_generic",
        "generated/C_out_of_distribution",
        "generated/D_fma_tags",
        "manifests",
    ])

    gen_cfg = cfg["generation"]
    train_cfg = cfg["training"]

    temperatures = gen_cfg.get("temperatures", [1.0])
    if not isinstance(temperatures, list):
        temperatures = [temperatures]

    artist_seeds = gen_cfg.get("per_artist_seeds", [42])
    general_seeds = gen_cfg.get("seeds", [42])
    top_k = gen_cfg.get("top_k", 250)
    top_p = gen_cfg.get("top_p", 0.95)
    max_duration = gen_cfg.get("max_duration_s", 30)

    logger.info(f"Temperatures: {temperatures}")
    logger.info(f"Artist seeds: {artist_seeds}, General seeds: {general_seeds}")
    logger.info(f"Max duration: {max_duration}s, top_k: {top_k}, top_p: {top_p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load prompts
    prompts_file = args.prompts_file or str(Path(args.run_dir) / "manifests" / "prompts.json")
    if not Path(prompts_file).exists():
        logger.error(f"Prompts file not found: {prompts_file}")
        logger.error("Run build_prompts.py first")
        sys.exit(1)

    with open(prompts_file) as f:
        all_prompts = json.load(f)

    # Filter by tier if specified
    if args.tier:
        all_prompts = [p for p in all_prompts if p["tier"] == args.tier]
        logger.info(f"Filtered to tier {args.tier}: {len(all_prompts)} prompts")

    if args.max_prompts:
        all_prompts = all_prompts[:args.max_prompts]

    logger.info(f"Total prompts to process: {len(all_prompts)}")

    # Find checkpoint
    checkpoint = None
    if not args.baseline:
        checkpoint = args.checkpoint
        if not checkpoint:
            ckpt_path = Path(args.run_dir) / "checkpoints" / "lm_final.pt"
            if ckpt_path.exists():
                checkpoint = str(ckpt_path)
        if not checkpoint:
            logger.error("No checkpoint found. Provide --checkpoint, --baseline, "
                         "or ensure <run_dir>/checkpoints/lm_final.pt exists")
            sys.exit(1)
        logger.info(f"Checkpoint: {checkpoint}")
    else:
        logger.info("BASELINE MODE: using vanilla pretrained model (no fine-tuning)")

    # Load model once
    model = load_model(train_cfg["model_name"], checkpoint, device, logger)

    # Generation loop
    gen_log = []
    gen_count = 0
    skip_count = 0
    fail_count = 0
    t0 = time.time()

    for pi, prompt in enumerate(all_prompts):
        tier = prompt["tier"]
        seeds = artist_seeds if tier in ("A_artist_proximal", "D_fma_tags") else general_seeds
        out_dir = Path(args.run_dir) / "generated" / tier

        for temp in temperatures:
            for seed in seeds:
                fname = f"{prompt['id']}_t{temp}_seed{seed}.wav"
                out_path = out_dir / fname

                if out_path.exists():
                    skip_count += 1
                    continue

                try:
                    audio, sr = generate_audio(
                        model, prompt["text"],
                        max_duration, temp, top_k, top_p, seed,
                    )
                    sf.write(str(out_path), audio, sr)

                    gen_log.append({
                        "file_path": str(out_path),
                        "prompt_id": prompt["id"],
                        "tier": tier,
                        "genre": prompt.get("genre", ""),
                        "artist_id": prompt.get("artist_id", ""),
                        "artist_name": prompt.get("artist_name", ""),
                        "prompt_text": prompt["text"],
                        "seed": seed,
                        "temperature": temp,
                        "top_k": top_k,
                        "duration_s": round(len(audio) / sr, 2),
                    })
                    gen_count += 1

                except Exception as e:
                    logger.error(f"Generation failed: {fname}: {e}")
                    fail_count += 1

                # Progress log every 25 generations
                if gen_count > 0 and gen_count % 25 == 0:
                    elapsed = time.time() - t0
                    rate = gen_count / elapsed
                    remaining = (len(all_prompts) * len(temperatures) * len(seeds)
                                 - gen_count - skip_count - fail_count)
                    eta_s = remaining / rate if rate > 0 else 0
                    logger.info(f"  Generated {gen_count} | skipped {skip_count} | "
                                f"failed {fail_count} | {elapsed:.0f}s elapsed | "
                                f"ETA ~{eta_s/60:.0f}min")

        # Checkpoint the log periodically (resume-safe)
        if (pi + 1) % 50 == 0:
            _write_gen_log(gen_log, Path(args.run_dir) / "manifests" / "generation_log.csv")

    # Final log write
    log_csv = Path(args.run_dir) / "manifests" / "generation_log.csv"
    _write_gen_log(gen_log, log_csv)

    elapsed = time.time() - t0
    logger.info(f"Generation complete: {gen_count} generated, {skip_count} skipped, "
                f"{fail_count} failed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    outputs = [
        str(Path(args.run_dir) / "generated"),
        str(log_csv),
    ]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    meta["total_generated"] = gen_count
    meta["total_skipped"] = skip_count
    meta["total_failed"] = fail_count
    save_run_metadata(args.run_dir, STAGE, meta)


def _write_gen_log(gen_log: list, path: Path):
    """Write generation log CSV (idempotent)."""
    fieldnames = [
        "file_path", "prompt_id", "tier", "genre", "artist_id",
        "artist_name", "prompt_text", "seed", "temperature", "top_k",
        "duration_s",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(gen_log)


if __name__ == "__main__":
    main()
