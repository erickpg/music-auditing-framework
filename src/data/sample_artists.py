#!/usr/bin/env python3
"""Step 00: Build eligible artist pool via album-level selection.

Reads FMA metadata, scores albums by listener engagement, and selects
the best album per artist via stratified sampling across commercial genres.

Also supports "compilation" mode: for artists without a qualifying album,
selects their top tracks by listens as a best-of compilation.

Outputs:
    <run_dir>/manifests/artists_selected.csv
    <run_dir>/manifests/tracks_selected.csv
    <run_dir>/manifests/sampling_summary.json
    <run_dir>/logs/sample_artists.log
    <run_dir>/logs/sample_artists_meta.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "sample_artists"

COMMERCIAL_GENRES = [
    "Pop", "Rock", "Hip-Hop", "Soul-RnB", "Folk",
    "Country", "Blues", "Jazz", "Classical",
]


def load_fma_tracks(metadata_dir: str) -> pd.DataFrame:
    """Load FMA tracks.csv with its multi-level header."""
    tracks_path = Path(metadata_dir) / "fma_metadata" / "tracks.csv"
    if not tracks_path.exists():
        tracks_path = Path(metadata_dir) / "tracks.csv"

    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    tracks.columns = [
        f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
        for col in tracks.columns
    ]
    return tracks


def score_albums(tracks: pd.DataFrame, logger) -> pd.DataFrame:
    """Score albums by composite engagement metric.

    Score = 0.5 * mean_listens + 0.3 * artist_avg_listens + 0.2 * max_listens
    """
    commercial = tracks[tracks["track_genre_top"].isin(COMMERCIAL_GENRES)].copy()
    logger.info(f"Tracks in commercial genres: {len(commercial)}")

    # Album-level stats (avoid slow lambda — compute genre separately)
    album_stats = commercial.groupby(["artist_id", "album_id"]).agg(
        artist_name=("artist_name", "first"),
        album_title=("album_title", "first"),
        n_tracks=("track_listens", "count"),
        total_listens=("track_listens", "sum"),
        mean_listens=("track_listens", "mean"),
        max_listens=("track_listens", "max"),
        genre_top=("track_genre_top", "first"),
        total_duration_s=("track_duration", "sum"),
        mean_duration_s=("track_duration", "mean"),
    ).reset_index()

    # Filter: reasonable albums
    album_stats = album_stats[
        (album_stats["n_tracks"] >= 5) &
        (album_stats["mean_duration_s"] >= 60) &
        (album_stats["mean_duration_s"] <= 600)
    ]
    logger.info(f"Albums after filtering (≥5 tracks, 1-10min avg): {len(album_stats)}")

    # Artist-level aggregate for scoring
    artist_totals = commercial.groupby("artist_id").agg(
        artist_total_listens=("track_listens", "sum"),
        artist_total_tracks=("track_listens", "count"),
    ).reset_index()

    album_stats = album_stats.merge(artist_totals, on="artist_id")

    # Composite score
    artist_avg = (album_stats["artist_total_listens"]
                  / album_stats["artist_total_tracks"])
    album_stats["score"] = (
        album_stats["mean_listens"] * 0.5
        + artist_avg * 0.3
        + album_stats["max_listens"] * 0.2
    )

    return album_stats


def build_compilations(tracks: pd.DataFrame, excluded_artist_ids: set,
                       min_tracks: int, max_tracks: int,
                       logger) -> pd.DataFrame:
    """For artists without qualifying albums, build best-of compilations.

    Selects top tracks by listens for artists who have enough individual
    tracks but no single album meeting the threshold.
    """
    commercial = tracks[tracks["track_genre_top"].isin(COMMERCIAL_GENRES)].copy()

    # Artists not already selected via album
    remaining = commercial[~commercial["artist_id"].isin(excluded_artist_ids)]

    # Group by artist
    artist_stats = remaining.groupby("artist_id").agg(
        artist_name=("artist_name", "first"),
        n_tracks=("track_listens", "count"),
        total_listens=("track_listens", "sum"),
        mean_listens=("track_listens", "mean"),
        max_listens=("track_listens", "max"),
        genre_top=("track_genre_top", lambda x: x.mode().iloc[0]
                   if len(x.mode()) > 0 else "Unknown"),
    ).reset_index()

    # Eligible for compilation: enough tracks spread across albums
    eligible = artist_stats[
        (artist_stats["n_tracks"] >= min_tracks) &
        (artist_stats["n_tracks"] <= max_tracks)
    ].sort_values("total_listens", ascending=False)

    logger.info(f"Compilation-eligible artists (no qualifying album): {len(eligible)}")
    return eligible


def stratified_select(candidates: pd.DataFrame, num_artists: int,
                      seed: int, prefer_top: bool, logger) -> pd.DataFrame:
    """Stratified selection across genres, preferring highest-scored."""
    rng = np.random.RandomState(seed)

    genre_counts = candidates["genre_top"].value_counts()
    # Proportional allocation with minimum 1 per genre
    raw_alloc = (genre_counts / genre_counts.sum() * num_artists)
    genre_allocation = raw_alloc.round().astype(int).clip(lower=1)

    # If more genres than target artists, keep only top genres by candidate count
    if len(genre_allocation) > num_artists:
        top_genres = genre_counts.head(num_artists).index
        genre_allocation = pd.Series(1, index=top_genres)
    else:
        # Adjust to hit target
        while genre_allocation.sum() > num_artists:
            largest = genre_allocation.idxmax()
            genre_allocation[largest] -= 1
            if genre_allocation[largest] == 0:
                genre_allocation = genre_allocation.drop(largest)
        while genre_allocation.sum() < num_artists:
            for g in genre_counts.index:
                if genre_allocation.sum() >= num_artists:
                    break
                genre_allocation[g] = genre_allocation.get(g, 0) + 1

    selected = []
    for genre, n_pick in genre_allocation.items():
        pool = candidates[candidates["genre_top"] == genre]
        if prefer_top:
            picked = pool.head(n_pick)
        else:
            n_actual = min(n_pick, len(pool))
            picked = pool.sample(n=n_actual, random_state=rng)
        selected.append(picked)
        logger.info(f"  {genre}: {len(picked)}/{len(pool)} candidates")

    return pd.concat(selected, ignore_index=True)


def get_album_tracks(tracks: pd.DataFrame,
                     selected: pd.DataFrame,
                     max_per_artist: int = 10) -> pd.DataFrame:
    """Get tracks belonging to selected artist+album pairs, capped per artist."""
    keys = selected[["artist_id", "album_id"]].drop_duplicates()
    keys["_selected"] = True
    merged = tracks.join(
        keys.set_index(["artist_id", "album_id"]),
        on=["artist_id", "album_id"],
        how="left",
    )
    sel = tracks[merged["_selected"] == True].copy()
    # Cap tracks per artist: keep top by listens
    sel = sel.sort_values("track_listens", ascending=False)
    sel = sel.groupby("artist_id", group_keys=False).head(max_per_artist)
    return _build_track_df(sel)


def get_compilation_tracks(tracks: pd.DataFrame,
                           selected_artists: pd.DataFrame,
                           max_per_artist: int = 20) -> pd.DataFrame:
    """Get top tracks by listens for compilation artists."""
    results = []
    for _, artist in selected_artists.iterrows():
        artist_tracks = tracks[tracks["artist_id"] == artist["artist_id"]]
        artist_tracks = artist_tracks.sort_values("track_listens", ascending=False)
        results.append(artist_tracks.head(max_per_artist))

    if not results:
        return pd.DataFrame()
    sel = pd.concat(results)
    return _build_track_df(sel)


def _build_track_df(sel: pd.DataFrame) -> pd.DataFrame:
    """Build standardized output dataframe from selected tracks."""
    return pd.DataFrame({
        "track_id": sel.index,
        "artist_id": sel["artist_id"].values,
        "artist_name": sel["artist_name"].values,
        "album_id": sel["album_id"].values,
        "album_title": sel["album_title"].values,
        "track_title": sel["track_title"].values,
        "genre_top": sel["track_genre_top"].values,
        "duration_s": sel["track_duration"].values,
        "listens": sel["track_listens"].values,
        "fma_path": [
            f"{tid:06d}"[:3] + "/" + f"{tid:06d}.mp3"
            for tid in sel.index
        ],
    })


def main():
    parser = base_argparser("Sample artists from FMA dataset (album-level)")
    parser.add_argument("--num_artists", type=int, default=None,
                        help="Override number of artists from config")
    parser.add_argument("--metadata_dir", type=str, default=None,
                        help="Path to FMA metadata directory")
    parser.add_argument("--min_tracks", type=int, default=5,
                        help="Min tracks per album/compilation (default: 5)")
    parser.add_argument("--max_tracks", type=int, default=100,
                        help="Max tracks per artist for compilations (default: 100)")
    parser.add_argument("--target_tracks", type=int, default=None,
                        help="Target tracks per artist (default: from config or 10)")
    parser.add_argument("--compilation_tracks", type=int, default=15,
                        help="Max tracks to pick per compilation artist (default: 15)")
    parser.add_argument("--prefer_top", action="store_true", default=True,
                        help="Prefer top-scored artists instead of random sampling")
    parser.add_argument("--exclude_artists", type=str, default=None,
                        help="Path to CSV with artist_id column to exclude from selection")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["manifests"])

    num_artists = args.num_artists or cfg["data"]["num_artists"]
    target_tracks = args.target_tracks or cfg["data"].get("target_tracks_per_artist", 10)
    seed = cfg.get("training", {}).get("seed", 42)
    metadata_dir = args.metadata_dir or str(
        Path(cfg["paths"]["scratch_base"]) / "fma_metadata"
    )

    logger.info(f"Metadata dir: {metadata_dir}")
    logger.info(f"Target artists: {num_artists}")
    logger.info(f"Commercial genres: {COMMERCIAL_GENRES}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Prefer top-scored: {args.prefer_top}")

    # 1. Load metadata
    logger.info("Loading FMA tracks metadata...")
    tracks = load_fma_tracks(metadata_dir)
    logger.info(f"Loaded {len(tracks)} tracks")

    # 1b. Exclude artists if requested
    if args.exclude_artists:
        exclude_df = pd.read_csv(args.exclude_artists)
        exclude_ids = set(exclude_df["artist_id"].unique())
        before = len(tracks)
        tracks = tracks[~tracks["artist_id"].isin(exclude_ids)]
        logger.info(f"Excluded {len(exclude_ids)} artists ({before - len(tracks)} tracks removed, {len(tracks)} remaining)")

    # 2. Score albums
    logger.info("Scoring albums by engagement...")
    album_scores = score_albums(tracks, logger)

    # Best album per artist
    best_albums = (album_scores
                   .sort_values("score", ascending=False)
                   .drop_duplicates("artist_id"))
    logger.info(f"Unique artists with qualifying albums: {len(best_albums)}")

    # 3. Allocate: mostly album-based, rest from compilations
    n_album = min(num_artists, len(best_albums))
    n_compilation = num_artists - n_album

    # 4. Select album-based artists
    logger.info(f"\n=== Album-based selection ({n_album} artists) ===")
    album_selected = stratified_select(
        best_albums, n_album, seed, args.prefer_top, logger
    )

    # 5. If needed, fill remaining with compilation artists
    compilation_selected = pd.DataFrame()
    if n_compilation > 0:
        logger.info(f"\n=== Compilation selection ({n_compilation} artists) ===")
        excluded_ids = set(album_selected["artist_id"])
        comp_candidates = build_compilations(
            tracks, excluded_ids, args.min_tracks, args.max_tracks, logger
        )
        compilation_selected = stratified_select(
            comp_candidates, n_compilation, seed + 1, args.prefer_top, logger
        )

    # 6. Gather tracks
    logger.info("\nGathering tracks for selected artists...")
    album_tracks = get_album_tracks(tracks, album_selected, max_per_artist=target_tracks)
    logger.info(f"Album tracks: {len(album_tracks)} (capped at {target_tracks}/artist)")

    comp_tracks = pd.DataFrame()
    if len(compilation_selected) > 0:
        comp_tracks = get_compilation_tracks(
            tracks, compilation_selected, target_tracks
        )
        logger.info(f"Compilation tracks: {len(comp_tracks)}")

    all_tracks = pd.concat([album_tracks, comp_tracks], ignore_index=True)

    # 7. Build artist summary
    all_artists = []
    for _, row in album_selected.iterrows():
        artist_tracks = all_tracks[all_tracks["artist_id"] == row["artist_id"]]
        all_artists.append({
            "artist_id": row["artist_id"],
            "artist_name": row["artist_name"],
            "album_id": row["album_id"],
            "album_title": row["album_title"],
            "genre_top": row["genre_top"],
            "selection_type": "album",
            "num_tracks": len(artist_tracks),
            "total_listens": int(artist_tracks["listens"].sum()),
            "mean_listens": int(artist_tracks["listens"].mean()),
            "total_duration_s": float(artist_tracks["duration_s"].sum()),
            "score": float(row["score"]),
        })
    for _, row in compilation_selected.iterrows():
        artist_tracks = all_tracks[all_tracks["artist_id"] == row["artist_id"]]
        all_artists.append({
            "artist_id": row["artist_id"],
            "artist_name": row["artist_name"],
            "album_id": "compilation",
            "album_title": f"Best of {row['artist_name']}",
            "genre_top": row["genre_top"],
            "selection_type": "compilation",
            "num_tracks": len(artist_tracks),
            "total_listens": int(artist_tracks["listens"].sum()) if len(artist_tracks) > 0 else 0,
            "mean_listens": int(artist_tracks["listens"].mean()) if len(artist_tracks) > 0 else 0,
            "total_duration_s": float(artist_tracks["duration_s"].sum()) if len(artist_tracks) > 0 else 0,
            "score": float(row.get("total_listens", 0)),
        })
    artists_df = pd.DataFrame(all_artists)

    # 8. Write outputs
    artists_csv = dirs["manifests"] / "artists_selected.csv"
    tracks_csv = dirs["manifests"] / "tracks_selected.csv"
    summary_json = dirs["manifests"] / "sampling_summary.json"

    artists_df.to_csv(artists_csv, index=False)
    all_tracks.to_csv(tracks_csv, index=False)

    # Summary
    genre_dist = artists_df["genre_top"].value_counts().to_dict()
    summary = {
        "num_artists": len(artists_df),
        "num_tracks": len(all_tracks),
        "total_duration_hours": round(all_tracks["duration_s"].sum() / 3600, 1),
        "mean_tracks_per_artist": round(len(all_tracks) / max(len(artists_df), 1), 1),
        "genre_distribution": genre_dist,
        "selection_types": artists_df["selection_type"].value_counts().to_dict(),
        "commercial_genres": COMMERCIAL_GENRES,
        "seed": seed,
        "prefer_top": args.prefer_top,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nWrote {artists_csv} ({len(artists_df)} artists)")
    logger.info(f"Wrote {tracks_csv} ({len(all_tracks)} tracks)")
    logger.info(f"Wrote {summary_json}")

    # Print summary
    logger.info("\n=== SELECTION SUMMARY ===")
    logger.info(f"Artists: {len(artists_df)}")
    logger.info(f"Tracks: {len(all_tracks)}")
    logger.info(f"Total duration: {all_tracks['duration_s'].sum() / 3600:.1f} hours")
    logger.info(f"Genre distribution:")
    for genre, count in sorted(genre_dist.items(), key=lambda x: -x[1]):
        logger.info(f"  {genre}: {count} artists")
    logger.info(f"Selection types: {summary['selection_types']}")

    # Per-artist detail
    logger.info("\n=== PER-ARTIST DETAIL ===")
    for _, a in artists_df.sort_values("score", ascending=False).iterrows():
        logger.info(
            f"  {a['artist_name'][:30]:30s} | {a['genre_top']:10s} | "
            f"{a['num_tracks']:2d} trk | {a['mean_listens']:6d} avg | "
            f"{a['total_duration_s']/60:.0f} min | {a['selection_type']}"
        )

    outputs = [str(artists_csv), str(tracks_csv), str(summary_json)]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
