#!/usr/bin/env python3
"""Stage 08: Full fine-tuning of MusicGen-small decoder.

Based on chavinlo/musicgen_trainer training loop logic, but using
HuggingFace transformers MusicGen API (already installed and tested).
No PEFT, no LoRA, no adapters, no Seq2SeqTrainer.

Simple manual loop:
  1. Encode audio through EnCodec (frozen)
  2. Forward pass through decoder LM
  3. CrossEntropyLoss on predicted vs actual codes
  4. Backprop + optimizer step

Saves state_dict checkpoints. Load with:
  model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
  model.decoder.load_state_dict(torch.load("lm_final.pt"))

Outputs:
    <run_dir>/checkpoints/lm_step_<N>.pt   (periodic checkpoints)
    <run_dir>/checkpoints/lm_final.pt      (final model)
    <run_dir>/logs/finetune.log
"""

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    AutoFeatureExtractor,
    MusicgenForConditionalGeneration,
    get_scheduler,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "finetune"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SegmentDataset(Dataset):
    def __init__(self, segment_dir: Path, descriptions: dict, default_desc: str = "music"):
        self.items = []
        for wav_path in sorted(segment_dir.glob("*.wav")):
            track_id = wav_path.stem.split("_")[0]
            desc = descriptions.get(track_id, default_desc)
            self.items.append((str(wav_path), desc))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ---------------------------------------------------------------------------
# Text descriptions from FMA metadata
# ---------------------------------------------------------------------------
def build_track_descriptions(run_dir: Path):
    tracks_csv = run_dir / "manifests" / "tracks_selected.csv"
    tags_json = run_dir / "manifests" / "artist_fma_tags.json"

    track_to_artist = {}
    track_to_genre = {}
    with open(tracks_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["track_id"]
            track_to_artist[tid] = row["artist_name"]
            track_to_genre[tid] = row["genre_top"].lower()

    artist_tags = {}
    if tags_json.exists():
        with open(tags_json) as f:
            raw = json.load(f)
            for artist, info in raw.items():
                artist_tags[artist] = [t.lower() for t in info.get("sub_genres", [])]

    descriptions = {}
    for tid, artist in track_to_artist.items():
        genre = track_to_genre[tid]
        sub_genres = [s for s in artist_tags.get(artist, []) if s != genre]
        if sub_genres:
            desc = f"{genre} music with {', '.join(sub_genres)} influences"
        else:
            desc = f"{genre} music"
        descriptions[tid] = desc

    return descriptions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = base_argparser("Full fine-tuning of MusicGen-small")
    parser.add_argument("--segment_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(log, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["checkpoints", "manifests", "logs"])

    train_cfg = cfg["training"]
    seed = train_cfg["seed"]
    set_seed(seed)

    run_dir = Path(args.run_dir)
    segment_dir = Path(args.segment_dir) if args.segment_dir else run_dir / "segments"
    ckpt_dir = dirs["checkpoints"]
    model_name = train_cfg.get("model_name", "facebook/musicgen-small")

    lr = train_cfg["learning_rate"]
    epochs = train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]
    grad_acc = train_cfg.get("gradient_accumulation_steps", 8)
    warmup_steps = train_cfg.get("warmup_steps", 10)
    save_step = train_cfg.get("checkpoint_every_steps", 250)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    segment_duration = cfg["chunking"].get("segment_length_s", 30)
    native_sr = cfg["data"].get("sample_rate", 32000)

    log.info(f"Model: {model_name}")
    log.info(f"Full fine-tuning (no LoRA, no PEFT)")
    log.info(f"Batch size: {batch_size}, LR: {lr}, Grad accum: {grad_acc}")
    log.info(f"Epochs: {epochs}, Warmup: {warmup_steps}, Seed: {seed}")

    # -----------------------------------------------------------------------
    # 1. Load model + processor
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Loading model...")
    model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model.config.audio_encoder._name_or_path
    )

    num_codebooks = model.decoder.num_codebooks
    pad_token_id = model.generation_config.pad_token_id  # 2048

    # Set decoder_start_token_id (required by shift_tokens_right in forward pass)
    model.config.update({"pad_token_id": pad_token_id, "decoder_start_token_id": pad_token_id})
    model.config.decoder.update({"pad_token_id": pad_token_id, "decoder_start_token_id": pad_token_id})

    log.info(f"Codebooks: {num_codebooks}, pad_token_id: {pad_token_id}")

    # Freeze audio encoder (EnCodec) and text encoder — only train decoder
    model.freeze_audio_encoder()
    model.freeze_text_encoder()

    # Count trainable params (decoder only)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # -----------------------------------------------------------------------
    # 2. Pre-encode all segments through EnCodec (once, saves GPU time)
    # -----------------------------------------------------------------------
    track_descriptions = build_track_descriptions(run_dir)
    log.info(f"Text descriptions for {len(track_descriptions)} tracks")

    segment_files = sorted(segment_dir.glob("*.wav"))
    log.info(f"Found {len(segment_files)} segments")

    max_samples = int(segment_duration * native_sr)

    log.info("Encoding all segments through EnCodec...")
    all_data = []
    for i, path in enumerate(segment_files):
        track_id = path.stem.split("_")[0]
        desc = track_descriptions.get(track_id, "music")

        # Load audio
        signal, sr = sf.read(str(path))
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        if len(signal) > max_samples:
            signal = signal[:max_samples]
        elif len(signal) < max_samples:
            signal = np.pad(signal, (0, max_samples - len(signal)))

        # Encode through EnCodec
        audio_input = feature_extractor(
            signal.astype(np.float32), sampling_rate=native_sr
        )["input_values"]

        with torch.no_grad():
            enc_out = model.audio_encoder.encode(
                torch.tensor(audio_input).to(device)
            )
        codes = enc_out["audio_codes"]  # [frames, B, codebooks, T]
        codes = codes[0]  # first frame: [B, codebooks, T]

        # Apply delay pattern mask (critical for MusicGen multi-codebook training)
        # Add pad token prefix column (BOS-like) — from reference impl
        pad_prefix = torch.ones((1, num_codebooks, 1), dtype=codes.dtype, device=codes.device) * pad_token_id
        codes_with_bos = torch.cat([pad_prefix, codes], dim=-1)  # [1, codebooks, T+1]

        labels_delayed, delay_mask = model.decoder.build_delay_pattern_mask(
            codes_with_bos.squeeze(0),  # [codebooks, T+1]
            pad_token_id,
            codes_with_bos.shape[-1] + num_codebooks,
        )
        labels_delayed = model.decoder.apply_delay_pattern_mask(labels_delayed, delay_mask)
        # Strip first BOS timestamp — from reference
        labels_delayed = labels_delayed[:, 1:]  # [codebooks, T']

        # Tokenize text
        input_ids = processor.tokenizer(desc, return_tensors="pt")["input_ids"]

        all_data.append({
            "codes": labels_delayed.unsqueeze(0).cpu(),  # [1, codebooks, T']
            "input_ids": input_ids.cpu(),   # [1, seq_len]
            "desc": desc,
        })

        if (i + 1) % 500 == 0:
            log.info(f"  Encoded {i+1}/{len(segment_files)}")

    log.info(f"EnCodec done. {len(all_data)} samples ready.")

    # -----------------------------------------------------------------------
    # 3. Training loop
    # -----------------------------------------------------------------------
    model.train()

    # Re-freeze encoders (model.train() may unfreeze)
    model.freeze_audio_encoder()
    model.freeze_text_encoder()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    steps_per_epoch = len(all_data) // batch_size
    total_steps = (steps_per_epoch * epochs) // grad_acc
    scheduler = get_scheduler("cosine", optimizer, warmup_steps, total_steps)

    log.info(f"Steps/epoch: {steps_per_epoch}, Total optim steps: {total_steps}")
    log.info("Starting training...")

    scaler = torch.amp.GradScaler("cuda")
    global_step = 0
    accum_loss = 0.0

    for epoch in range(epochs):
        # Shuffle data each epoch
        import random
        indices = list(range(len(all_data)))
        random.shuffle(indices)

        for batch_start in range(0, len(indices) - batch_size + 1, batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]

            # Gather batch
            batch_codes = torch.cat(
                [all_data[i]["codes"] for i in batch_indices], dim=0
            ).to(device)  # [B, codebooks, T]

            # Pad input_ids to same length
            batch_input_ids = [all_data[i]["input_ids"].squeeze(0) for i in batch_indices]
            max_len = max(ids.shape[0] for ids in batch_input_ids)
            padded_ids = torch.full((len(batch_indices), max_len), processor.tokenizer.pad_token_id)
            attention_mask = torch.zeros(len(batch_indices), max_len, dtype=torch.long)
            for j, ids in enumerate(batch_input_ids):
                padded_ids[j, :ids.shape[0]] = ids
                attention_mask[j, :ids.shape[0]] = 1

            padded_ids = padded_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Labels: [B, codebooks, T] -> [B, T, codebooks] for model's loss
            labels = batch_codes.permute(0, 2, 1)  # [B, T, codebooks]

            # Forward pass with GradScaler for stable fp16
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=padded_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_acc

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if (batch_start // batch_size + 1) % grad_acc == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                log.info(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"Step {global_step}/{total_steps}, "
                    f"Loss: {accum_loss:.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                accum_loss = 0.0

                # Checkpoint
                if global_step % save_step == 0:
                    ckpt_path = ckpt_dir / f"lm_step_{global_step}.pt"
                    torch.save(model.decoder.state_dict(), str(ckpt_path))
                    log.info(f"Checkpoint: {ckpt_path}")

    # -----------------------------------------------------------------------
    # 4. Save final
    # -----------------------------------------------------------------------
    final_path = ckpt_dir / "lm_final.pt"
    torch.save(model.decoder.state_dict(), str(final_path))
    log.info(f"Final model: {final_path}")

    outputs = [str(final_path)]
    meta = log_finish(log, meta, STAGE, outputs=outputs)
    meta["seed"] = seed
    meta["model_name"] = model_name
    meta["method"] = "full_finetune"
    meta["trainable_params"] = trainable
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
