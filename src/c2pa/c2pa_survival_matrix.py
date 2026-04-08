#!/usr/bin/env python3
"""Stage 03: C2PA durability study -- survival matrix.

Tests whether C2PA Content Credentials survive common audio transforms.
Expected result: 0% survival (motivating the watermark approach).

Outputs:
    <run_dir>/analysis/c2pa_survival.csv
    <run_dir>/analysis/c2pa_survival_summary.json
    <run_dir>/logs/c2pa_survival.log
    <run_dir>/logs/c2pa_survival_meta.json
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "c2pa_survival"


def apply_transform(src_path: str, pipeline: dict, tmp_dir: str) -> str:
    """Apply an audio transform pipeline and return path to transformed file."""
    name = pipeline["name"]
    codec = pipeline.get("codec", "")
    bitrate = pipeline.get("bitrate", 128)

    src = Path(src_path)

    if name == "strip_metadata":
        out_path = f"{tmp_dir}/{src.stem}_stripped.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src),
            "-map_metadata", "-1", "-c:a", "copy",
            out_path
        ], check=True, capture_output=True)
        return out_path

    if codec == "mp3":
        out_path = f"{tmp_dir}/{src.stem}_{name}.mp3"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src),
            "-codec:a", "libmp3lame", "-b:a", f"{bitrate}k",
            out_path
        ], check=True, capture_output=True)
        return out_path

    if codec == "aac":
        out_path = f"{tmp_dir}/{src.stem}_{name}.m4a"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src),
            "-codec:a", "aac", "-b:a", f"{bitrate}k",
            out_path
        ], check=True, capture_output=True)
        return out_path

    raise ValueError(f"Unknown pipeline: {name}")


def check_c2pa(file_path: str) -> dict:
    """Check if a file has valid C2PA Content Credentials."""
    try:
        from c2pa import Reader
        reader = Reader(file_path)
        manifest_store = json.loads(reader.json())
        active = manifest_store.get("active_manifest", "")
        return {
            "manifest_present": True,
            "active_manifest": active,
            "validation_status": "valid",
        }
    except Exception as e:
        err_str = str(e)
        if "not found" in err_str.lower() or "no manifest" in err_str.lower() or "JumbfNotFound" in err_str:
            return {
                "manifest_present": False,
                "active_manifest": None,
                "validation_status": "no_manifest",
            }
        return {
            "manifest_present": False,
            "active_manifest": None,
            "validation_status": f"error: {err_str[:100]}",
        }


def main():
    parser = base_argparser("Run C2PA survival matrix across transform pipelines")
    parser.add_argument("--sample_size", type=int, default=20,
                        help="Number of files to test (default: 20)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["analysis"])

    pipelines = cfg.get("c2pa", {}).get("survival_matrix", {}).get("pipelines", [])
    logger.info(f"Testing {len(pipelines)} pipelines on {args.sample_size} files")

    # Get C2PA-embedded files
    c2pa_dir = Path(args.run_dir) / "masters_c2pa"
    all_files = sorted(c2pa_dir.glob("*.wav"))
    sample_files = all_files[:args.sample_size]
    logger.info(f"Found {len(all_files)} C2PA files, sampling {len(sample_files)}")

    if not sample_files:
        logger.error("No C2PA files found. Run Stage 02 first.")
        return

    # First verify C2PA is present in originals
    logger.info("\n=== Verifying originals ===")
    originals_valid = 0
    for f in sample_files[:3]:
        result = check_c2pa(str(f))
        logger.info(f"  {f.name}: manifest={result['manifest_present']}, status={result['validation_status']}")
        if result["manifest_present"]:
            originals_valid += 1
    logger.info(f"  Originals with valid C2PA: {originals_valid}/3 (spot check)")

    # Run survival matrix
    logger.info("\n=== Running survival matrix ===")
    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for pipeline in pipelines:
            survived = 0
            tested = 0

            for src_file in sample_files:
                try:
                    transformed_path = apply_transform(str(src_file), pipeline, tmp_dir)
                    c2pa_result = check_c2pa(transformed_path)

                    results.append({
                        "source_file": src_file.name,
                        "pipeline_name": pipeline["name"],
                        "codec": pipeline.get("codec", "none"),
                        "bitrate": pipeline.get("bitrate", ""),
                        "manifest_present": c2pa_result["manifest_present"],
                        "validation_status": c2pa_result["validation_status"],
                    })

                    if c2pa_result["manifest_present"]:
                        survived += 1
                    tested += 1

                except Exception as e:
                    results.append({
                        "source_file": src_file.name,
                        "pipeline_name": pipeline["name"],
                        "codec": pipeline.get("codec", "none"),
                        "bitrate": pipeline.get("bitrate", ""),
                        "manifest_present": False,
                        "validation_status": f"transform_error: {e}",
                    })
                    tested += 1

            survival_rate = survived / tested * 100 if tested > 0 else 0
            logger.info(f"  {pipeline['name']:30s}: {survived}/{tested} survived ({survival_rate:.0f}%)")

    # Write results
    results_df = pd.DataFrame(results)
    results_csv = dirs["analysis"] / "c2pa_survival.csv"
    results_df.to_csv(results_csv, index=False)

    # Summary
    summary = {}
    for pipeline in pipelines:
        name = pipeline["name"]
        pipe_results = results_df[results_df["pipeline_name"] == name]
        survived = pipe_results["manifest_present"].sum()
        total = len(pipe_results)
        summary[name] = {
            "survived": int(survived),
            "total": int(total),
            "survival_rate": round(survived / total * 100, 1) if total > 0 else 0,
        }

    overall_survived = results_df["manifest_present"].sum()
    overall_total = len(results_df)
    summary["overall"] = {
        "survived": int(overall_survived),
        "total": int(overall_total),
        "survival_rate": round(overall_survived / overall_total * 100, 1) if overall_total > 0 else 0,
    }

    summary_path = dirs["analysis"] / "c2pa_survival_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n=== C2PA SURVIVAL SUMMARY ===")
    logger.info(f"Overall: {overall_survived}/{overall_total} survived ({summary['overall']['survival_rate']}%)")
    for name, stats in summary.items():
        if name != "overall":
            logger.info(f"  {name:30s}: {stats['survived']}/{stats['total']} ({stats['survival_rate']}%)")

    outputs = [str(results_csv), str(summary_path)]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
