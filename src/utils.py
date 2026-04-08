"""Shared utilities for all pipeline stages.

Every CLI script imports from here to guarantee consistent:
  - argument parsing (--config, --run_id, --run_dir)
  - logging (file + stderr, with timestamps)
  - reproducibility preamble (git commit, hostname, config hash, seed, timestamps)
  - directory creation under the run directory
  - config loading with environment variable expansion
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config with environment variable expansion."""
    with open(config_path) as f:
        raw = f.read()
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)


def config_hash(config_path: str) -> str:
    """Return SHA-256 hex digest of the raw config file (before env expansion)."""
    with open(config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit_hash() -> str:
    """Return the current short git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def base_argparser(description: str) -> argparse.ArgumentParser:
    """Create a base argument parser with the three mandatory arguments.

    Every pipeline script receives:
      --config   Path to experiment YAML config
      --run_id   Run identifier (YYYY-MM-DD_<shortdesc>)
      --run_dir  Root output directory for the run on scratch
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run identifier (format: YYYY-MM-DD_<shortdesc>)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Root directory for this run (e.g., /scratch/$USER/runs/$RUN_ID)",
    )
    return parser


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(run_dir: str, stage_name: str) -> logging.Logger:
    """Configure logging to file and stderr.

    Log file: <run_dir>/logs/<stage_name>.log
    """
    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{stage_name}.log"

    logger = logging.getLogger(stage_name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # File handler -- DEBUG level (everything)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(fh)

    # Stderr handler -- INFO level
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Reproducibility preamble
# ---------------------------------------------------------------------------

def log_preamble(logger: logging.Logger, args: argparse.Namespace,
                 stage_name: str) -> dict:
    """Log a full reproducibility preamble and return the metadata dict.

    Logs (and returns):
      - stage name
      - run_id, run_dir
      - config path + SHA-256 hash
      - git commit hash + dirty flag
      - hostname, user, platform
      - start timestamp (UTC + local)
      - Python version
      - seed (if present in config)

    The returned dict is suitable for writing to a JSON sidecar file.
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = datetime.datetime.now()

    commit = git_commit_hash()
    dirty = git_is_dirty()
    cfg_hash = config_hash(args.config) if Path(args.config).exists() else "N/A"

    meta = {
        "stage": stage_name,
        "run_id": args.run_id,
        "run_dir": args.run_dir,
        "config_path": str(Path(args.config).resolve()),
        "config_hash": cfg_hash,
        "git_commit": commit,
        "git_dirty": dirty,
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER", "unknown"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "start_utc": now_utc.isoformat(),
        "start_local": now_local.isoformat(),
    }

    logger.info("=" * 60)
    logger.info(f"STAGE: {stage_name}")
    logger.info(f"RUN_ID: {args.run_id}")
    logger.info(f"RUN_DIR: {args.run_dir}")
    logger.info(f"CONFIG: {args.config} (sha256:{cfg_hash})")
    logger.info(f"GIT COMMIT: {commit}{'  *** DIRTY ***' if dirty else ''}")
    logger.info(f"HOST: {meta['hostname']} | USER: {meta['user']}")
    logger.info(f"START (UTC): {now_utc.isoformat()}")
    logger.info(f"PYTHON: {meta['python']}")
    logger.info("=" * 60)

    return meta


def log_finish(logger: logging.Logger, meta: dict, stage_name: str,
               outputs: list[str] | None = None) -> dict:
    """Log a completion footer with timing info.

    Args:
        logger: the stage logger
        meta: the dict returned by log_preamble
        stage_name: name of the stage
        outputs: optional list of output file/dir paths produced

    Returns:
        Updated meta dict with end time and duration.
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    start = datetime.datetime.fromisoformat(meta["start_utc"])
    duration = now_utc - start

    meta["end_utc"] = now_utc.isoformat()
    meta["duration_s"] = round(duration.total_seconds(), 1)
    meta["outputs"] = outputs or []

    logger.info("=" * 60)
    logger.info(f"STAGE COMPLETE: {stage_name}")
    logger.info(f"DURATION: {duration} ({meta['duration_s']}s)")
    if outputs:
        logger.info(f"OUTPUTS ({len(outputs)}):")
        for o in outputs:
            logger.info(f"  {o}")
    logger.info("=" * 60)

    return meta


def save_run_metadata(run_dir: str, stage_name: str, meta: dict) -> Path:
    """Write the run metadata dict to a JSON sidecar in the logs directory.

    File: <run_dir>/logs/<stage_name>_meta.json
    """
    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_path = log_dir / f"{stage_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs(run_dir: str, subdirs: list[str]) -> dict[str, Path]:
    """Create stage-specific subdirectories under run_dir."""
    paths = {}
    for subdir in subdirs:
        p = Path(run_dir) / subdir
        p.mkdir(parents=True, exist_ok=True)
        paths[subdir] = p
    return paths


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def export_results_pack(run_dir: str, run_id: str,
                        export_base: str, files: list[str]) -> Path:
    """Copy small result files to durable storage.

    Destination: <export_base>/<run_id>/
    Only copies files that exist. Skips silently if source is missing.

    Returns:
        Path to the export directory.
    """
    export_dir = Path(export_base) / run_id
    export_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        src = Path(f)
        if src.exists():
            dst = export_dir / src.name
            shutil.copy2(src, dst)

    # Always copy the stage metadata JSONs
    logs_dir = Path(run_dir) / "logs"
    for meta_json in logs_dir.glob("*_meta.json"):
        shutil.copy2(meta_json, export_dir / meta_json.name)

    return export_dir
