# An Artist-Side Auditing Framework for Detecting Style Absorption in Generative Music Models

**Author:** Erick Portales Gutierrez  
**Program:** BSc Capstone Project, 2026  

## Abstract

This repository implements an artist-side auditing system to detect whether protected music catalogs were ingested into a generative music model. The framework uses a dual-protection approach: C2PA Content Credentials (cryptographically signed provenance metadata) and inaudible neural audio watermarks. We fine-tune MusicGen-small (full decoder fine-tuning) on a 50-artist catalog from the Free Music Archive, generate outputs via a four-tier prompt system, and audit them through watermark recovery (Tier A) and dataset inference with Welch's t-test (Tier B).

Key finding: fine-tuned models absorb artist-specific characteristics without memorizing specific content. Standard n-gram memorization detection fails, but perceptual similarity metrics (CLAP, FAD) quantify per-artist vulnerability. A composite vulnerability score enables black-box artist-side risk assessment without requiring model access.

## Pipeline Overview

```
                              PROTECTION LAYERS
                              =================
Raw Audio ──> [01] Standardize ──> [02] C2PA Embed ──> [03] C2PA Survival Test
                                         │
                                   [04a] WavMark Embed ──> [07] EnCodec Survival Test
                                         │
                                   [05] Chunk Segments (10-30s, 5s overlap)
                                         │
                              FINE-TUNING & GENERATION
                              ========================
                                         │
                                   [08] Fine-tune MusicGen-small (full, 100 epochs)
                                         │
                                   [09] Generate Outputs (4-tier prompts)
                                         │  ├── Tier A: Artist-proximal prompts
                                         │  ├── Tier B: Genre-generic prompts
                                         │  ├── Tier C: OOD negative control
                                         │  └── Tier D: FMA sub-genre tags
                                         │
                              ANALYSIS & AUDITING
                              ====================
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                     [T1-T2] Tokenize          [V1] CLAP Embeddings
                              │                     │
                     [A1] N-gram Search        [V2] Per-artist FAD
                              │                     │
                     [A2] N-gram Stats         [V3] Musicological Features
                              │                     │
                              └──────────┬──────────┘
                                         │
                                   [V4] Vulnerability Score
                                   (2-signal: CLAP gap + FAD)
                                         │
                                   Binary High/Low Tier Classification
```

## Repository Structure

```
submission/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── configs/                  # YAML experiment configurations
│   ├── exp001_minimal.yaml   #   5-artist validation run
│   ├── exp002_full.yaml      #   50-artist production run
│   ├── exp003_wavmark_poc.yaml       # WavMark proof-of-concept
│   ├── exp004_audioseal_train_data.yaml  # AudioSeal training data sampling
│   └── exp005_memorization.yaml      # Memorization + vulnerability pipeline
├── src/                      # Core pipeline modules
│   ├── utils.py              #   Shared utilities (config, logging, argparse)
│   ├── data/                 #   Data pipeline
│   │   ├── sample_artists.py #     Stage 00: FMA artist sampling
│   │   ├── download_audio.py #     Download FMA tracks
│   │   ├── standardize_audio.py  #  Stage 01: Decode/resample/normalize
│   │   └── chunk_segments.py #     Stage 05: Split into training segments
│   ├── c2pa/                 #   C2PA Content Credentials
│   │   ├── embed_c2pa.py     #     Stage 02: Embed provenance metadata
│   │   └── c2pa_survival_matrix.py  # Stage 03: Codec survival testing
│   ├── watermark/            #   Audio watermarking
│   │   ├── payload_schema.py #     Payload bit-packing (artist/album/CRC)
│   │   ├── embed_wavmark.py  #     Stage 04a: WavMark embedding
│   │   ├── tokenizer_survival_poc.py  # Stage 07: EnCodec survival test
│   │   └── waveverify_survival_test.py  # WaveVerify durability test
│   ├── training/             #   MusicGen fine-tuning
│   │   └── finetune_musicgen_full.py  # Stage 08: Full decoder fine-tuning
│   ├── generation/           #   Audio generation
│   │   ├── build_prompts.py  #     Four-tier prompt construction
│   │   └── generate_outputs.py  #  Stage 09: Generate from fine-tuned model
│   ├── tokenization/         #   EnCodec tokenization
│   │   ├── tokenize_catalog.py  #  Tokenize catalog audio
│   │   └── tokenize_generated.py  # Tokenize generated audio
│   └── analysis/             #   Multi-metric analysis
│       ├── compute_clap_embeddings.py  # CLAP semantic similarity
│       ├── per_artist_fad.py          # Frechet Audio Distance
│       ├── musicological_features.py  # Librosa feature extraction
│       ├── vulnerability_score.py     # Composite vulnerability scoring
│       ├── ngram_search.py            # N-gram memorization search
│       └── ngram_stats.py             # Statistical tests + FDR correction
├── scripts/                  # Analysis & evaluation scripts
│   ├── generate_thesis_plots.py       # All thesis figures
│   ├── three_tier_full_results.py     # 3-tier vulnerability classification
│   ├── paired_delta_test.py           # Paired delta test (fine-tuned vs base)
│   ├── v1_v2_comparison.py            # Cross-version result comparison
│   ├── icc_and_bootstrap_3tier.py     # ICC + bootstrap confidence intervals
│   ├── cv_3tier.py                    # Cross-validation of tier classification
│   └── [20+ additional analysis scripts]
├── jobs/                     # Slurm cluster job scripts
│   ├── _preamble.sh          #   Shared environment setup
│   ├── run_sampling.sh       #   Stage 00: Artist sampling
│   ├── run_standardize.sh    #   Stage 01: Audio standardization
│   ├── run_c2pa.sh           #   Stages 02-03: C2PA pipeline
│   ├── run_chavinlo_v2.sh    #   Stage 08: Full fine-tuning
│   ├── run_generate_v2.sh    #   Stage 09: Generation
│   ├── run_analysis_v2_full.sh  # Full analysis pipeline
│   ├── run_thesis_plots.sh   #   Publication figure generation
│   └── [additional job scripts]
```

## Environment Setup

### Requirements

- Python 3.11
- NVIDIA GPU with 48 GB VRAM (tested on RTX 6000 Ada)
- Slurm-based HPC cluster (scripts assume `sbatch`/`srun`)
- ~200 GB scratch storage for datasets, checkpoints, and generated audio

### Installation

```bash
# 1. Create conda environment
conda create --prefix /scratch/$USER/capstone_env python=3.11 -y
source activate /scratch/$USER/capstone_env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set cache directories (critical for cluster storage policy)
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache

# 4. Generate C2PA signing key (required for Stage 02):
#    The C2PA embed stage needs an Ed25519 certificate chain.
#    See src/c2pa/embed_c2pa.py for auto-generation, or provide your own
#    at the path specified in your config under c2pa.sign_key_path
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0 | Core ML framework |
| transformers | >= 4.35 | MusicGen model loading |
| scikit-learn | >= 1.0 | Covariance estimation (FAD) |
| audiocraft | >= 1.0 | MusicGen model + EnCodec |
| librosa | >= 0.10 | Audio feature extraction |
| scipy | >= 1.11 | Statistical testing (Welch's t-test, FDR) |
| c2pa-python | >= 0.28 | C2PA Content Credentials |

## Reproduction Guide

### Run Naming Convention

All runs use `RUN_ID=YYYY-MM-DD_<shortdesc>`. Every script accepts `--config`, `--run_id`, and `--run_dir`.

### Step-by-Step Pipeline Execution

Each stage can be run standalone or via Slurm. Below shows both methods.

#### Stage 0: Data Preparation

```bash
export RUN_ID=2026-03-10_full
export CONFIG=configs/exp002_full.yaml
export RUN_DIR=/scratch/$USER/runs/$RUN_ID

# Sample 50 eligible artists from FMA
sbatch jobs/run_sampling.sh
# OR: python src/data/sample_artists.py --config $CONFIG --run_id $RUN_ID --run_dir $RUN_DIR

# Download tracks
sbatch jobs/run_download.sh
```

#### Stage 1: Audio Standardization

```bash
# Decode, resample to 32kHz mono, loudness normalize
sbatch jobs/run_standardize.sh
# OR: python src/data/standardize_audio.py --config $CONFIG --run_id $RUN_ID --run_dir $RUN_DIR
```

#### Stages 2-3: C2PA Content Credentials

```bash
# Embed C2PA provenance metadata + test survival across codecs
sbatch jobs/run_c2pa.sh
```

#### Stage 4: Watermark Embedding (POC)

```bash
# WavMark POC embedding
sbatch jobs/04a_wavmark_embed.slurm
```

#### Stage 5: Segment Chunking

```bash
# Split watermarked masters into 10-30s training segments
python src/data/chunk_segments.py --config $CONFIG --run_id $RUN_ID --run_dir $RUN_DIR
```

#### Stage 7: Tokenizer Survival Test

```bash
# Test watermark survival through EnCodec
python src/watermark/tokenizer_survival_poc.py --config $CONFIG --run_id $RUN_ID --run_dir $RUN_DIR
```

#### Stage 8: Fine-tuning

```bash
# MusicGen-small full decoder fine-tuning (GPU required, ~5 min/epoch)
sbatch jobs/run_chavinlo_v2.sh
# OR: sbatch jobs/run_finetune_full.sh

# Key hyperparameters (in config):
#   model: facebook/musicgen-small (300M params, full decoder fine-tuning)
#   epochs: 100, lr: 1e-4, batch_size: 4 + grad_accum: 8
#   segment length: 30s with 5s overlap
```

#### Stage 9: Generation

```bash
# Generate audio from fine-tuned + baseline models
sbatch jobs/run_generate_v2.sh

# Generates: 510 prompts x 2 temperatures (0.7, 1.0) x 3 seeds = 3,060 WAVs
# Four prompt tiers: artist-proximal, genre-generic, OOD control, FMA-tag
```

#### Stages T1-T2, A1-A2, V1-V4: Analysis

```bash
# Full analysis pipeline (tokenization, n-gram, CLAP, FAD, vulnerability)
sbatch jobs/run_analysis_v2_full.sh

# Or run individual stages:
python src/tokenization/tokenize_catalog.py --config $CONFIG --run_id $RUN_ID --run_dir $RUN_DIR
python src/tokenization/tokenize_generated.py --config $CONFIG --run_id $RUN_ID --run_dir $RUN_DIR
python src/analysis/ngram_search.py --config $CONFIG --run_id $RUN_ID --run_dir $RUN_DIR
python src/analysis/ngram_stats.py --config $CONFIG --run_id $RUN_ID --run_dir $RUN_DIR
python src/analysis/compute_clap_embeddings.py --config $CONFIG --run_id $RUN_ID --run_dir $RUN_DIR
python src/analysis/per_artist_fad.py --config $CONFIG --run_id $RUN_ID --run_dir $RUN_DIR
python src/analysis/vulnerability_score.py --config $CONFIG --run_id $RUN_ID --run_dir $RUN_DIR
```

#### Thesis Figures

```bash
# Generate all publication-ready figures
sbatch jobs/run_thesis_plots.sh
# OR: python scripts/generate_thesis_plots.py --results_dir $RUN_DIR/analysis --output_dir $RUN_DIR/plots
```

## Expected Outputs

Each run produces outputs in `<run_dir>/` with the following subdirectories:

| Directory | Contents |
|-----------|----------|
| `data/` | Sampled artist manifests, FMA metadata |
| `masters_clean/` | Standardized 32kHz mono WAVs |
| `masters_c2pa/` | C2PA-signed masters |
| `masters_watermarked/` | Watermarked masters |
| `segments/` | 10-30s training segments |
| `checkpoints/` | Decoder checkpoints (every 250 steps) |
| `generated/` | 3,060 generated WAV files |
| `analysis/` | CSV results, embeddings, statistical test outputs |
| `plots/` | Publication-ready figures (PNG/PDF) |
| `logs/` | Per-stage execution logs |

### Key Result Files

- `analysis/ngram_stats.csv` -- N-gram memorization test results (Welch's t-test per artist)
- `analysis/clap_similarity_gap.csv` -- Per-artist CLAP matched vs mismatched similarity
- `analysis/per_artist_fad.csv` -- Frechet Audio Distance per artist
- `analysis/vulnerability_scores.csv` -- Composite 2-signal vulnerability ranking
- `analysis/three_tier_results.csv` -- Binary High/Low tier classification

## Experiment Configurations

| Config | Purpose | Artists | Key Settings |
|--------|---------|---------|-------------|
| `exp001_minimal.yaml` | Quick validation | 5 | MusicGen-small, 5 epochs, 10s segments |
| `exp002_full.yaml` | Production run | 50 | MusicGen-small, 100 epochs, 30s segments, full fine-tuning |
| `exp005_memorization.yaml` | Memorization + vulnerability | 50 | N-gram [3-8], CLAP+FAD+musicological, 1000 bootstrap |

## Key Results Summary

- **No memorization detected**: N-gram token matching finds no significant overlap (all p > 0.05 after FDR)
- **Style absorption confirmed**: CLAP similarity gap = 0.083 (V1) / 0.032 (V2), both p < 0.001
- **Central paradox**: Better training (100 epochs, 20% loss drop) produces LESS artist-specific outputs (61% CLAP gap drop)
- **Optimal vulnerability signal**: 2-signal CLAP+FAD composite (rho=0.629 cross-version stability)
- **Binary tiers recommended**: High/Low classification achieves 82% cross-version agreement (kappa=0.59)

## Analysis & Evaluation Scripts

The `scripts/` directory contains 27 standalone analysis scripts used to produce thesis figures and supplementary analyses. These scripts read results from completed pipeline runs.

**Note:** Many scripts have result paths configured at the top of the file. Before running, update the path variables (e.g., `RESULTS_DIR`, `V1_DIR`, `V2_DIR`) to point to your run output directories.

| Script | Purpose |
|--------|---------|
| `generate_thesis_plots.py` | Generate all publication-ready thesis figures |
| `three_tier_full_results.py` | Compute 3-tier vulnerability classification (High/Mid/Low) |
| `paired_delta_test.py` | Paired delta test: fine-tuned vs baseline similarity gap |
| `vulnerability_bootstrap_ci.py` | Bootstrap 95% CI for per-artist vulnerability scores |
| `icc_and_bootstrap_3tier.py` | Intraclass correlation + bootstrap CI for tier stability |
| `cv_3tier.py` | Cross-validation of tier classification |
| `ablation_tier_agreement.py` | Ablation: which signal combinations yield best tier agreement |
| `genre_control.py` | Genre-controlled vulnerability (z-score + genre-matched) |
| `v1_v2_comparison.py` | V1 vs V2 comparative analysis (7 metrics) |
| `v1_v2_metric_stability.py` | Per-metric stability across training versions |
| `v1_v2_threshold_stability.py` | Tier threshold robustness across versions |
| `v1_v2_recompute_vuln.py` | Vulnerability under different signal weightings |
| `v2_vs_baseline_comparison.py` | V2 fine-tuned vs untrained baseline comparison |
| `tier_stability_per_tier.py` | Per-tier migration rates across versions |
| `temporal_split_analysis.py` | Seen vs unseen track similarity (generalization test) |
| `temporal_step{1-4}_*.py` | Temporal split pipeline (download, standardize, CLAP, FAD) |
| `muq_mulan_validation.py` | Cross-embedding validation with MuQ-MuLan |
| `muq_tier_analysis.py` | 3-tier agreement across MuQ-MuLan and CLAP |
| `catalog_map.py` | UMAP visualization of artist CLAP embeddings |
| `catalog_map_by_genre.py` | Per-genre UMAP maps |
| `catalog_map_muq.py` | UMAP visualization with MuQ-MuLan embeddings |
| `baseline_catalog_property.py` | Baseline catalog property analysis |
| `run_four_analyses.py` | Combined supplementary analyses (4 in one) |
| `check_extra_tracks.py` | Verify held-out unseen tracks per artist |

## Data Source

Audio data is sourced from the [Free Music Archive (FMA)](https://github.com/mdeff/fma), a public dataset of Creative Commons-licensed music. The 50-artist test catalog is sampled via stratified selection from FMA metadata (see `src/data/sample_artists.py`).

## Citation

If you use this code or methodology, please cite:

```
Portales Gutierrez, E. (2026). An Artist-Side Auditing Framework for
Detecting Style Absorption in Generative Music Models. BSc Capstone.
```
