#!/usr/bin/env python3
"""Build structured text prompts for MusicGen generation.

Four-tier experimental design:
  Tier A (artist-proximal): Per-artist style descriptors for vulnerability assessment.
         Built from genre × instrumentation × texture × tempo vocabulary.
  Tier B (genre-generic):   Broad genre prompts for n-gram memorization detection.
         Not artist-specific — tests overall catalog leakage.
  Tier C (out-of-distribution): Genres absent from training data (negative control).
         Should show no memorization; validates methodology.
  Tier D (FMA sub-genre tags): Per-artist prompts derived from actual FMA metadata
         sub-genre labels. Data-grounded — uses the dataset's own taxonomy.

Tiers A-C use a fixed musical vocabulary (reproducible, no cherry-picking).
Tier D uses FMA's sub-genre tags directly, adding ecological validity.

Outputs:
    <run_dir>/manifests/prompts.json
    <run_dir>/logs/build_prompts.log
    <run_dir>/logs/build_prompts_meta.json
"""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "build_prompts"

# ═══════════════════════════════════════════════════════════════════════
# Fixed musical vocabulary — documented, reproducible, genre-grounded
# ═══════════════════════════════════════════════════════════════════════

# Instruments typical of each FMA top-level genre
GENRE_INSTRUMENTS = {
    "Rock": [
        "electric guitar", "bass guitar", "drums", "distorted guitar",
        "overdriven guitar", "power chords", "drum kit", "cymbals",
    ],
    "Folk": [
        "acoustic guitar", "fingerpicked guitar", "fiddle", "mandolin",
        "banjo", "harmonica", "upright bass", "hand percussion",
    ],
    "Hip-Hop": [
        "808 bass", "hi-hat", "sampled drums", "synthesizer",
        "vinyl scratching", "drum machine", "sub bass", "snare",
    ],
    "Pop": [
        "synth pad", "pop drums", "electric piano", "bass synth",
        "clean guitar", "hand claps", "bright synth", "tambourine",
    ],
    "Jazz": [
        "piano", "upright bass", "brush drums", "saxophone",
        "trumpet", "vibraphone", "walking bass", "ride cymbal",
    ],
    "Classical": [
        "violin", "cello", "viola", "piano",
        "flute", "oboe", "string ensemble", "harpsichord",
    ],
    "Country": [
        "steel guitar", "acoustic guitar", "fiddle", "banjo",
        "dobro", "bass guitar", "brush snare", "pedal steel",
    ],
    "Soul-RnB": [
        "electric piano", "bass guitar", "hi-hat", "horns",
        "organ", "wah guitar", "congas", "strings",
    ],
    "Blues": [
        "blues guitar", "harmonica", "bass guitar", "shuffle drums",
        "slide guitar", "piano", "organ", "brass",
    ],
}

# Coherent mood-tempo pairs — avoids contradictions like "energetic + slow"
# Each pair: (mood, tempo) that make musical sense together
MOOD_TEMPO_PAIRS = [
    ("melancholic", "slow"),
    ("melancholic", "mid-tempo"),
    ("calm", "slow"),
    ("calm", "mid-tempo"),
    ("warm", "mid-tempo"),
    ("warm", "slow"),
    ("dark", "slow"),
    ("dark", "mid-tempo"),
    ("dark", "fast"),
    ("energetic", "upbeat"),
    ("energetic", "fast"),
    ("intense", "fast"),
    ("intense", "upbeat"),
    ("intense", "mid-tempo"),
    ("driving", "fast"),
    ("driving", "upbeat"),
    ("groovy", "mid-tempo"),
    ("groovy", "upbeat"),
    ("dreamy", "slow"),
    ("dreamy", "mid-tempo"),
    ("aggressive", "fast"),
    ("aggressive", "upbeat"),
    ("reflective", "slow"),
    ("reflective", "mid-tempo"),
]

# Production/texture descriptors
TEXTURES = ["lo-fi", "polished", "raw", "layered", "minimal", "dense"]

# Prompt templates — varied to avoid template-specific artifacts
TEMPLATES = [
    "{mood} {genre} with {inst1} and {inst2}, {tempo} tempo",
    "{genre} track featuring {inst1} and {inst2}, {mood} and {texture}",
    "{tempo} {genre} piece with {inst1}, {inst2}, {texture} production",
    "{mood} {texture} {genre} featuring {inst1} and {inst2}",
    "{genre} with {inst1}, {inst2}, and {inst3}, {mood} {tempo}",
]

# ═══════════════════════════════════════════════════════════════════════
# Out-of-distribution genres (NONE of these appear in our training data)
# ═══════════════════════════════════════════════════════════════════════

# NOTE: Removed genres that overlap with MusicGen's pretraining data or our
# catalog's genre space (EDM, grunge, disco, trap, synthwave, gospel, ska).
# Remaining genres are culturally/geographically distinct from the Western
# popular music that dominates the training catalog.
# Uses same template format as Tiers A/B to avoid prompt-format confounds.
OOD_STYLES = [
    # Latin / Caribbean
    {"genre": "reggaeton", "desc": "reggaeton beat with dembow rhythm and Latin percussion"},
    {"genre": "bossa nova", "desc": "bossa nova with nylon guitar and soft brushes"},
    {"genre": "mariachi", "desc": "mariachi with trumpet, guitarron, and vihuela"},
    {"genre": "cumbia", "desc": "cumbia with accordion, guiro, and conga drums"},
    {"genre": "son cubano", "desc": "son cubano with tres guitar, bongos, and maracas"},
    {"genre": "bachata", "desc": "bachata with requinto guitar and bongos, romantic mood"},
    # African / Afro-diasporic
    {"genre": "Afrobeat", "desc": "Afrobeat with polyrhythmic drums and brass"},
    {"genre": "Afro-Cuban", "desc": "Afro-Cuban jazz with clave rhythm and congas"},
    {"genre": "highlife", "desc": "highlife music with palm wine guitar and brass horns"},
    {"genre": "mbaqanga", "desc": "South African mbaqanga with groovy bass and vocal harmony"},
    {"genre": "soukous", "desc": "Congolese soukous with fast fingerpicked guitar and sebene"},
    # Asian
    {"genre": "K-pop", "desc": "K-pop track with bright synths and dance beat"},
    {"genre": "Indian classical", "desc": "Indian classical raga with sitar and tabla"},
    {"genre": "Bollywood", "desc": "Bollywood film score with sitar, tabla, and strings"},
    {"genre": "gamelan", "desc": "Javanese gamelan with metalophone and gong ensemble"},
    {"genre": "Carnatic", "desc": "Carnatic classical with veena, mridangam, and violin"},
    {"genre": "Chinese opera", "desc": "Chinese opera with erhu, pipa, and gong percussion"},
    {"genre": "Japanese koto", "desc": "Japanese koto with shamisen and shakuhachi flute"},
    {"genre": "qawwali", "desc": "qawwali devotional music with tabla and harmonium"},
    # European traditional
    {"genre": "flamenco", "desc": "flamenco guitar with palmas and cajon"},
    {"genre": "Celtic", "desc": "Celtic folk with tin whistle, bodhran, and fiddle"},
    {"genre": "fado", "desc": "Portuguese fado with guitarra portuguesa and melancholic vocals"},
    {"genre": "Balkan brass", "desc": "Balkan brass band with tuba, trumpet, and accordion"},
    {"genre": "rebetiko", "desc": "Greek rebetiko with bouzouki and baglamas"},
    {"genre": "klezmer", "desc": "klezmer with clarinet, violin, and accordion"},
    # Middle Eastern / Central Asian
    {"genre": "Arabic maqam", "desc": "Arabic maqam music with oud, qanun, and darbuka"},
    {"genre": "Persian classical", "desc": "Persian classical with tar, santur, and tombak"},
    {"genre": "Turkish makam", "desc": "Turkish classical with ney flute and kanun"},
    # Oceanian / Pacific
    {"genre": "Aboriginal didgeridoo", "desc": "Aboriginal Australian music with didgeridoo drone and clapsticks"},
    {"genre": "Polynesian chant", "desc": "Polynesian rhythmic chant with log drums and conch shell"},
    # Other distinct styles
    {"genre": "dubstep", "desc": "dubstep with wobble bass and half-time drums"},
    {"genre": "ambient electronic", "desc": "ambient electronic with granular textures and pad drones"},
    {"genre": "noise", "desc": "harsh noise with distorted feedback and dense textures"},
    {"genre": "drone", "desc": "drone music with sustained tones and slow harmonic movement"},
    {"genre": "musique concrete", "desc": "musique concrete with manipulated found sounds and tape loops"},
    {"genre": "throat singing", "desc": "Tuvan throat singing with overtone harmonics and drone"},
    # Additional OOD to reach 50+ with template variety
    {"genre": "gagaku", "desc": "Japanese gagaku court music with sho, hichiriki, and ryuteki"},
    {"genre": "gnawa", "desc": "Moroccan gnawa with guembri bass lute and metal castanets"},
    {"genre": "dangdut", "desc": "Indonesian dangdut with tabla, flute, and synth bass"},
    {"genre": "tango", "desc": "Argentine tango with bandoneon, violin, and piano"},
    {"genre": "mbalax", "desc": "Senegalese mbalax with sabar drums and electric guitar"},
    {"genre": "mbira", "desc": "Shona mbira music with thumb piano and hosho shakers"},
    {"genre": "steel pan", "desc": "Trinidad steel pan calypso with steel drums and percussion"},
    {"genre": "dub", "desc": "dub reggae with heavy reverb, delay, and deep bass"},
    {"genre": "tuareg desert blues", "desc": "Tuareg desert blues with electric guitar and calabash percussion"},
    {"genre": "berimbau", "desc": "Brazilian capoeira music with berimbau, atabaque, and pandeiro"},
    {"genre": "bhangra", "desc": "Punjabi bhangra with dhol drum, tumbi, and chimta"},
    {"genre": "chamame", "desc": "Argentine chamame with accordion, guitar, and double bass"},
    {"genre": "maracatu", "desc": "Brazilian maracatu with alfaia drums, agbe, and ganza"},
]


def build_artist_prompts(artists: list[dict], n_per_artist: int,
                         rng: random.Random) -> list[dict]:
    """Tier A: Per-artist style-descriptive prompts.

    For each artist, generate n prompts that describe their genre's musical
    characteristics. The prompts capture the style space without naming the
    artist — MusicGen responds to style descriptions, not artist names.

    The cross-artist comparison is the internal control: if artist X's catalog
    is more similar to X-prompted outputs than to Y-prompted outputs, that
    indicates artist-specific memorization.
    """
    prompts = []
    prompt_idx = 0

    for artist in artists:
        genre = artist["genre_top"]
        instruments = GENRE_INSTRUMENTS.get(genre, GENRE_INSTRUMENTS["Rock"])
        artist_id = artist["artist_id"]
        artist_name = artist["artist_name"]

        for i in range(n_per_artist):
            template = rng.choice(TEMPLATES)
            mood, tempo = rng.choice(MOOD_TEMPO_PAIRS)
            texture = rng.choice(TEXTURES)

            # Sample instruments without replacement for variety
            n_inst = template.count("{inst")
            selected_inst = rng.sample(instruments, min(n_inst, len(instruments)))
            while len(selected_inst) < n_inst:
                selected_inst.append(rng.choice(instruments))

            inst_map = {f"inst{j+1}": selected_inst[j] for j in range(n_inst)}

            text = template.format(
                genre=genre.lower(),
                mood=mood,
                tempo=tempo,
                texture=texture,
                **inst_map,
            )

            prompts.append({
                "id": f"a{prompt_idx:04d}",
                "text": text,
                "tier": "A_artist_proximal",
                "artist_id": artist_id,
                "artist_name": artist_name,
                "genre": genre,
                "attributes": {
                    "mood": mood,
                    "tempo": tempo,
                    "texture": texture,
                    "instruments": selected_inst,
                },
            })
            prompt_idx += 1

    return prompts


def build_genre_prompts(genres: list[str], n_per_genre: int,
                        rng: random.Random) -> list[dict]:
    """Tier B: Genre-generic prompts for n-gram memorization detection.

    Broad genre descriptions without artist-specific characteristics.
    Tests whether the model leaks catalog token patterns even with
    non-specific prompts.
    """
    prompts = []
    prompt_idx = 0

    for genre in genres:
        instruments = GENRE_INSTRUMENTS.get(genre, GENRE_INSTRUMENTS["Rock"])

        for i in range(n_per_genre):
            template = rng.choice(TEMPLATES)
            mood, tempo = rng.choice(MOOD_TEMPO_PAIRS)
            texture = rng.choice(TEXTURES)

            n_inst = template.count("{inst")
            selected_inst = rng.sample(instruments, min(n_inst, len(instruments)))
            while len(selected_inst) < n_inst:
                selected_inst.append(rng.choice(instruments))

            inst_map = {f"inst{j+1}": selected_inst[j] for j in range(n_inst)}

            text = template.format(
                genre=genre.lower(),
                mood=mood,
                tempo=tempo,
                texture=texture,
                **inst_map,
            )

            prompts.append({
                "id": f"g{prompt_idx:04d}",
                "text": text,
                "tier": "B_genre_generic",
                "artist_id": None,
                "artist_name": None,
                "genre": genre,
                "attributes": {
                    "mood": mood,
                    "tempo": tempo,
                    "texture": texture,
                    "instruments": selected_inst,
                },
            })
            prompt_idx += 1

    return prompts


# ═══════════════════════════════════════════════════════════════════════
# FMA sub-genre → natural-language description mapping
# Used by Tier D to convert dataset taxonomy labels into MusicGen prompts
# ═══════════════════════════════════════════════════════════════════════

SUBGENRE_DESCRIPTIONS = {
    # Rock sub-genres
    "Indie-Rock": "indie rock with guitar-driven melodies",
    "Post-Rock": "post-rock with atmospheric guitars and building crescendos",
    "Psych-Rock": "psychedelic rock with reverb-heavy guitars and swirling effects",
    "Garage": "garage rock with raw distorted guitars and lo-fi energy",
    "Surf": "surf rock with reverb-drenched twangy guitar",
    "Punk": "punk rock with fast power chords and driving drums",
    "Hardcore": "hardcore punk with aggressive distortion and fast tempo",
    "Post-Punk": "post-punk with angular guitars and dark atmosphere",
    "Goth": "gothic rock with dark brooding bass and atmospheric reverb",
    "Metal": "heavy metal with distorted guitars and double bass drums",
    "Progressive": "progressive rock with complex time signatures and layered arrangements",
    "Shoegaze": "shoegaze with dense walls of guitar feedback and ethereal textures",
    "New Wave": "new wave with synths and clean guitar, upbeat rhythm",
    "Noise-Rock": "noise rock with abrasive distortion and dissonant textures",
    "Loud-Rock": "loud rock with heavy amplified guitars",
    "Lo-Fi": "lo-fi rock with intentionally rough recording quality",
    "Power-Pop": "power pop with catchy hooks and jangly guitars",
    "Instrumental": "instrumental music with no vocals",
    # Other genres
    "Hip-Hop Beats": "hip-hop instrumental beats with sampled drums",
    "Synth Pop": "synth pop with electronic keyboards and pop melodies",
    "Experimental Pop": "experimental pop with unconventional song structures",
    "Big Band/Swing": "big band swing with brass section and upright bass",
    "20th Century Classical": "20th century classical with modern harmonies",
    "Composed Music": "composed orchestral music",
    "Rockabilly": "rockabilly with slap bass and twangy guitar",
    "Disco": "disco with four-on-the-floor beat and funky bass",
    "Dance": "dance music with electronic beat",
    "Electronic": "electronic music with synthesizers",
    "Musical Theater": "musical theater style orchestration",
    "Soundtrack": "cinematic soundtrack music",
}

# Templates for Tier D — use sub-genre descriptions directly
FMA_TAG_TEMPLATES = [
    "{subgenre_desc}",
    "{subgenre_desc}, {mood} atmosphere",
    "{subgenre_desc} with {texture} production",
    "{subgenre_desc}, {tempo} tempo",
    "{subgenre_desc}, {mood} and {texture}",
]


def build_fma_tag_prompts(artists: list[dict], fma_tags: dict,
                          n_per_artist: int,
                          rng: random.Random) -> list[dict]:
    """Tier D: Per-artist prompts from FMA sub-genre tags.

    Uses the actual sub-genre labels from FMA metadata to build prompts.
    Each artist's tracks have sub-genre annotations (e.g., "Garage, Surf",
    "Post-Rock, Shoegaze") — these are converted to natural-language
    descriptions and used as prompts.

    This tier adds ecological validity: prompts are derived from the
    dataset's own taxonomy, not from a hand-coded vocabulary.
    """
    prompts = []
    prompt_idx = 0

    for artist in artists:
        artist_name = artist["artist_name"]
        artist_id = artist["artist_id"]
        genre_top = artist["genre_top"]

        # Get this artist's FMA sub-genre tags
        tag_data = fma_tags.get(artist_name, {})
        sub_genres = tag_data.get("sub_genres", [genre_top])

        # Filter to sub-genres we have descriptions for, excluding top-level
        specific_tags = [sg for sg in sub_genres
                         if sg in SUBGENRE_DESCRIPTIONS and sg != genre_top]

        # Fall back to top-level if no specific sub-genres
        if not specific_tags:
            specific_tags = [genre_top]

        for i in range(n_per_artist):
            # Pick a sub-genre tag (cycle through available tags)
            tag = specific_tags[i % len(specific_tags)]
            desc = SUBGENRE_DESCRIPTIONS.get(
                tag,
                f"{tag.lower()} music"  # fallback for unmapped tags
            )

            template = rng.choice(FMA_TAG_TEMPLATES)
            mood, tempo = rng.choice(MOOD_TEMPO_PAIRS)
            texture = rng.choice(TEXTURES)

            text = template.format(
                subgenre_desc=desc,
                mood=mood,
                tempo=tempo,
                texture=texture,
            )

            prompts.append({
                "id": f"d{prompt_idx:04d}",
                "text": text,
                "tier": "D_fma_tags",
                "artist_id": artist_id,
                "artist_name": artist_name,
                "genre": genre_top,
                "fma_subgenre": tag,
                "attributes": {
                    "mood": mood,
                    "tempo": tempo,
                    "texture": texture,
                    "fma_tags": specific_tags,
                },
            })
            prompt_idx += 1

    return prompts


def build_ood_prompts(rng: random.Random) -> list[dict]:
    """Tier C: Out-of-distribution prompts (negative control).

    Genres completely absent from the training catalog. If the model
    shows high n-gram overlap or similarity scores for these, our
    methodology has a false-positive problem.

    Uses template-based generation matching Tier A/B format to avoid
    prompt-format confounds (a plain description vs a structured template
    could behave differently regardless of memorization).
    """
    # Templates matching the Tier A/B format
    ood_templates = [
        "{mood} {genre} with {desc}, {tempo} tempo",
        "{genre} track, {desc}, {mood} and {texture}",
        "{tempo} {genre} piece, {desc}, {texture} production",
        "{mood} {texture} {genre}, {desc}",
    ]

    prompts = []
    for i, style in enumerate(OOD_STYLES):
        template = rng.choice(ood_templates)
        mood, tempo = rng.choice(MOOD_TEMPO_PAIRS)
        texture = rng.choice(TEXTURES)

        text = template.format(
            genre=style["genre"].lower(),
            desc=style["desc"],
            mood=mood,
            tempo=tempo,
            texture=texture,
        )

        prompts.append({
            "id": f"o{i:04d}",
            "text": text,
            "tier": "C_out_of_distribution",
            "artist_id": None,
            "artist_name": None,
            "genre": style["genre"],
            "attributes": {
                "mood": mood,
                "tempo": tempo,
                "texture": texture,
            },
        })
    return prompts


def main():
    parser = base_argparser("Build structured prompts for audio generation")
    parser.add_argument("--artists_csv", type=str, default=None,
                        help="Path to artists_selected.csv (default: <run_dir>/manifests/artists_selected.csv)")
    parser.add_argument("--fma_tags", type=str, default=None,
                        help="Path to artist_fma_tags.json (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt generation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["manifests"])

    gen_cfg = cfg["generation"]
    rng = random.Random(args.seed)

    # Load artist metadata
    artists_csv = args.artists_csv
    if not artists_csv:
        # Try run_dir first, then local data dir
        candidates = [
            Path(args.run_dir) / "manifests" / "artists_selected.csv",
            Path("data/manifests_test_catalog/artists_selected.csv"),
        ]
        for c in candidates:
            if c.exists():
                artists_csv = str(c)
                break

    if not artists_csv or not Path(artists_csv).exists():
        logger.error(f"Cannot find artists_selected.csv. Tried: {candidates}")
        sys.exit(1)

    artists = []
    with open(artists_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            artists.append(row)
    logger.info(f"Loaded {len(artists)} artists from {artists_csv}")

    # Genre distribution
    genres_present = sorted(set(a["genre_top"] for a in artists))
    genre_counts = {}
    for a in artists:
        genre_counts[a["genre_top"]] = genre_counts.get(a["genre_top"], 0) + 1
    logger.info(f"Genres in catalog: {genre_counts}")

    # Load FMA sub-genre tags for Tier D
    fma_tags = {}
    fma_tags_path = args.fma_tags
    if not fma_tags_path:
        candidates = [
            Path(args.run_dir) / "manifests" / "artist_fma_tags.json",
            Path("data/manifests_test_catalog/artist_fma_tags.json"),
        ]
        for c in candidates:
            if c.exists():
                fma_tags_path = str(c)
                break

    if fma_tags_path and Path(fma_tags_path).exists():
        with open(fma_tags_path) as f:
            fma_tags = json.load(f)
        logger.info(f"Loaded FMA tags for {len(fma_tags)} artists from {fma_tags_path}")
    else:
        logger.warning("No FMA tags file found — Tier D will use top-level genres only")

    # Build prompts per tier
    n_per_artist = 3  # Tier A: fewer — Tier D (FMA tags) is more targeted
    n_per_genre = 10  # fixed: 10 generic prompts per genre
    n_fma_per_artist = 5  # Tier D: primary per-artist tier, data-grounded

    logger.info(f"Building Tier A: {n_per_artist} prompts × {len(artists)} artists")
    tier_a = build_artist_prompts(artists, n_per_artist, rng)

    logger.info(f"Building Tier B: {n_per_genre} prompts × {len(genres_present)} genres")
    tier_b = build_genre_prompts(genres_present, n_per_genre, rng)

    logger.info(f"Building Tier C: {len(OOD_STYLES)} OOD prompts")
    tier_c = build_ood_prompts(rng)

    logger.info(f"Building Tier D: {n_fma_per_artist} prompts × {len(artists)} artists (FMA tags)")
    tier_d = build_fma_tag_prompts(artists, fma_tags, n_fma_per_artist, rng)

    all_prompts = tier_a + tier_b + tier_c + tier_d

    # Summary
    logger.info(f"Total prompts: {len(all_prompts)}")
    logger.info(f"  Tier A (artist-proximal): {len(tier_a)}")
    logger.info(f"  Tier B (genre-generic):   {len(tier_b)}")
    logger.info(f"  Tier C (OOD control):     {len(tier_c)}")
    logger.info(f"  Tier D (FMA sub-genre):   {len(tier_d)}")

    # Generation count estimate
    temps = gen_cfg.get("temperatures", [1.0])
    artist_seeds = gen_cfg.get("per_artist_seeds", [42])
    general_seeds = gen_cfg.get("seeds", [42])

    tier_a_gens = len(tier_a) * len(temps) * len(artist_seeds)
    tier_b_gens = len(tier_b) * len(temps) * len(general_seeds)
    tier_c_gens = len(tier_c) * len(temps) * len(general_seeds)
    tier_d_gens = len(tier_d) * len(temps) * len(artist_seeds)
    total_gens = tier_a_gens + tier_b_gens + tier_c_gens + tier_d_gens

    logger.info(f"Estimated generations: {total_gens}")
    logger.info(f"  Tier A: {len(tier_a)} × {len(temps)} temps × {len(artist_seeds)} seeds = {tier_a_gens}")
    logger.info(f"  Tier B: {len(tier_b)} × {len(temps)} temps × {len(general_seeds)} seeds = {tier_b_gens}")
    logger.info(f"  Tier C: {len(tier_c)} × {len(temps)} temps × {len(general_seeds)} seeds = {tier_c_gens}")
    logger.info(f"  Tier D: {len(tier_d)} × {len(temps)} temps × {len(artist_seeds)} seeds = {tier_d_gens}")
    logger.info(f"  Estimated GPU time: ~{total_gens * 10 / 3600:.1f}h (at ~10s/generation)")

    # Write prompts
    prompts_file = dirs["manifests"] / "prompts.json"
    with open(prompts_file, "w") as f:
        json.dump(all_prompts, f, indent=2)
    logger.info(f"Prompts written to {prompts_file}")

    # Write summary CSV for quick inspection
    summary_csv = dirs["manifests"] / "prompts_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "tier", "genre", "artist_id", "artist_name", "text",
        ])
        writer.writeheader()
        for p in all_prompts:
            writer.writerow({
                "id": p["id"],
                "tier": p["tier"],
                "genre": p["genre"],
                "artist_id": p.get("artist_id", ""),
                "artist_name": p.get("artist_name", ""),
                "text": p["text"],
            })

    outputs = [str(prompts_file), str(summary_csv)]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    meta["num_prompts"] = len(all_prompts)
    meta["tier_counts"] = {
        "A_artist_proximal": len(tier_a),
        "B_genre_generic": len(tier_b),
        "C_out_of_distribution": len(tier_c),
        "D_fma_tags": len(tier_d),
    }
    meta["estimated_generations"] = total_gens
    meta["seed"] = args.seed
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
