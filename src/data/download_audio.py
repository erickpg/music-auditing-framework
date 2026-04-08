#!/usr/bin/env python3
"""Download selected FMA tracks to scratch using HTTP range requests.

Reads tracks_selected.csv and downloads each track from the remote
fma_full.zip without downloading the entire 879GB archive.

Outputs:
    <audio_dir>/<fma_path>  (e.g., /scratch/$USER/fma_audio/010/010805.mp3)
    <run_dir>/logs/download_audio.log
    <run_dir>/logs/download_audio_meta.json
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "download_audio"


def main():
    parser = base_argparser("Download selected FMA tracks via HTTP range requests")
    parser.add_argument("--tracks_csv", type=str, default=None,
                        help="Path to tracks_selected.csv")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Output directory for audio files")
    parser.add_argument("--audio_url", type=str, default=None,
                        help="URL to fma_full.zip")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)

    tracks_csv = args.tracks_csv or str(
        Path(args.run_dir) / "manifests" / "tracks_selected.csv"
    )
    audio_dir = args.audio_dir or cfg["data"]["sources"][0].get(
        "audio_local_path", str(Path(cfg["paths"]["scratch_base"]) / "fma_audio")
    )
    audio_url = args.audio_url or cfg["data"]["sources"][0].get(
        "audio_base_url", "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
    )

    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Tracks CSV: {tracks_csv}")
    logger.info(f"Audio output dir: {audio_dir}")
    logger.info(f"Remote ZIP URL: {audio_url}")

    # Load track list
    tracks = pd.read_csv(tracks_csv)
    logger.info(f"Tracks to download: {len(tracks)}")

    # Check which are already downloaded
    to_download = []
    for _, row in tracks.iterrows():
        out_path = audio_dir / row["fma_path"]
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        to_download.append(row)

    logger.info(f"Already on disk: {len(tracks) - len(to_download)}")
    logger.info(f"Need to download: {len(to_download)}")

    if not to_download:
        logger.info("All tracks already downloaded. Nothing to do.")
        meta = log_finish(logger, meta, STAGE, outputs=[str(audio_dir)])
        save_run_metadata(args.run_dir, STAGE, meta)
        return

    try:
        from remotezip import RemoteZip
    except ImportError:
        logger.error("remotezip not installed. Run: pip install remotezip")
        sys.exit(1)

    logger.info(f"Opening remote ZIP index (this may take a minute)...")
    downloaded = 0
    errors = 0

    with RemoteZip(audio_url) as rz:
        for row in to_download:
            fma_path = row["fma_path"]
            zip_path = f"fma_full/{fma_path}"
            out_path = audio_dir / fma_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                data = rz.read(zip_path)
                out_path.write_bytes(data)
                downloaded += 1
                size_kb = len(data) / 1024
                logger.info(
                    f"  [{downloaded}/{len(to_download)}] {fma_path} "
                    f"({size_kb:.0f} KB) - {row['artist_name']}: {row['track_title']}"
                )
            except Exception as e:
                errors += 1
                logger.error(f"  FAILED {fma_path}: {e}")

    logger.info(f"\nDownload complete: {downloaded} OK, {errors} errors")
    logger.info(f"Audio directory: {audio_dir}")

    meta = log_finish(logger, meta, STAGE, outputs=[str(audio_dir)])
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
