#!/usr/bin/env python3
"""Stage 02: Embed C2PA Content Credentials into each track master.

Creates a CA + end-entity certificate chain, builds a C2PA manifest with
artist attribution, and embeds it into each standardized WAV file.

Outputs:
    <run_dir>/masters_c2pa/<track_id>.wav
    <run_dir>/manifests/c2pa_embed_log.csv
    <run_dir>/logs/embed_c2pa.log
    <run_dir>/logs/embed_c2pa_meta.json
"""

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "embed_c2pa"


def generate_cert_chain(cert_dir: Path):
    """Generate a CA + end-entity ed25519 certificate chain for C2PA signing.

    C2PA requires a proper certificate chain (not self-signed) with
    digitalSignature key usage and emailProtection EKU.

    Returns (ee_key_path, chain_path) where chain is EE cert + CA cert concatenated.
    """
    cert_dir.mkdir(parents=True, exist_ok=True)

    ca_key = cert_dir / "ca_key.pem"
    ca_cert = cert_dir / "ca_cert.pem"
    ee_key = cert_dir / "ee_key.pem"
    ee_csr = cert_dir / "ee_csr.pem"
    ee_cert = cert_dir / "ee_cert.pem"
    chain_path = cert_dir / "chain.pem"

    if ee_key.exists() and chain_path.exists():
        return str(ee_key), str(chain_path)

    # CA extensions config
    ca_ext = cert_dir / "v3_ca.ext"
    ca_ext.write_text("[v3_ca]\nbasicConstraints = critical, CA:TRUE, pathlen:0\nkeyUsage = critical, keyCertSign, cRLSign\n")

    # EE extensions config (C2PA requires digitalSignature + emailProtection)
    ee_ext = cert_dir / "v3_ee.ext"
    ee_ext.write_text("[v3_ee]\nbasicConstraints = CA:FALSE\nkeyUsage = critical, digitalSignature\nextendedKeyUsage = emailProtection\n")

    # Generate CA
    subprocess.run([
        "openssl", "genpkey", "-algorithm", "ed25519", "-out", str(ca_key)
    ], check=True, capture_output=True)
    subprocess.run([
        "openssl", "req", "-new", "-x509", "-key", str(ca_key), "-out", str(ca_cert),
        "-days", "365", "-subj", "/CN=Capstone Root CA/O=Capstone Research",
        "-extensions", "v3_ca", "-config", str(ca_ext)
    ], check=True, capture_output=True)

    # Generate end-entity key + CSR
    subprocess.run([
        "openssl", "genpkey", "-algorithm", "ed25519", "-out", str(ee_key)
    ], check=True, capture_output=True)
    subprocess.run([
        "openssl", "req", "-new", "-key", str(ee_key), "-out", str(ee_csr),
        "-subj", "/CN=Capstone Auditor/O=Capstone Research"
    ], check=True, capture_output=True)

    # Sign EE cert with CA
    subprocess.run([
        "openssl", "x509", "-req", "-in", str(ee_csr), "-CA", str(ca_cert),
        "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(ee_cert),
        "-days", "365", "-extfile", str(ee_ext), "-extensions", "v3_ee"
    ], check=True, capture_output=True)

    # Build chain file: EE cert + CA cert
    chain_data = ee_cert.read_bytes() + ca_cert.read_bytes()
    chain_path.write_bytes(chain_data)

    return str(ee_key), str(chain_path)


TSA_URL = "http://timestamp.digicert.com"


def embed_track_c2pa(args_tuple):
    """Embed C2PA manifest into a single track."""
    src_path, out_path, track_id, artist_name, track_title, claim_generator, key_path, chain_path = args_tuple

    src_path = Path(src_path)
    out_path = Path(out_path)

    if not src_path.exists():
        return {
            "track_id": track_id, "artist_name": artist_name,
            "source_path": str(src_path), "output_path": str(out_path),
            "manifest_embedded": False, "status": "error: source not found",
        }

    try:
        from c2pa import Builder, Reader, Signer, C2paSignerInfo, C2paSigningAlg

        # Build manifest JSON
        manifest_json = json.dumps({
            "claim_generator": claim_generator,
            "title": f"{artist_name} - {track_title}" if track_title else f"Track {track_id}",
            "assertions": [
                {
                    "label": "stds.schema-org.CreativeWork",
                    "data": {
                        "@type": "CreativeWork",
                        "author": [{"@type": "Person", "name": artist_name}],
                    }
                },
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "softwareAgent": claim_generator,
                            }
                        ]
                    }
                }
            ]
        })

        # Create builder and signer
        builder = Builder.from_json(manifest_json)
        chain_cert = open(chain_path, "rb").read()
        private_key = open(key_path, "rb").read()
        signer_info = C2paSignerInfo(C2paSigningAlg.ED25519, chain_cert, private_key, TSA_URL)
        signer = Signer.from_info(signer_info)

        # Sign and embed (arg order: source, dest, signer)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        builder.sign_file(str(src_path), str(out_path), signer)

        # Verify
        reader = Reader(str(out_path))
        manifest_store = json.loads(reader.json())
        active_manifest = manifest_store.get("active_manifest", "")

        return {
            "track_id": track_id, "artist_name": artist_name,
            "source_path": str(src_path), "output_path": str(out_path),
            "manifest_embedded": True,
            "active_manifest": active_manifest,
            "file_size_kb": round(out_path.stat().st_size / 1024, 1),
            "status": "ok",
        }

    except Exception as e:
        # Fallback: copy file without C2PA if embedding fails
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_path), str(out_path))
        return {
            "track_id": track_id, "artist_name": artist_name,
            "source_path": str(src_path), "output_path": str(out_path),
            "manifest_embedded": False,
            "active_manifest": None,
            "file_size_kb": round(out_path.stat().st_size / 1024, 1),
            "status": f"error: {e}",
        }


def main():
    parser = base_argparser("Embed C2PA Content Credentials into audio masters")
    parser.add_argument("--tracks_csv", type=str, default=None,
                        help="Path to tracks_selected.csv")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["masters_c2pa", "manifests"])

    c2pa_cfg = cfg["c2pa"]
    claim_generator = c2pa_cfg["claim_generator"]
    n_workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    logger.info(f"C2PA enabled: {c2pa_cfg['enabled']}")
    logger.info(f"Claim generator: {claim_generator}")
    logger.info(f"Workers: {n_workers}")

    # Generate signing credentials (CA chain, not self-signed)
    secrets_dir = Path(args.run_dir) / "secrets"
    key_path, chain_path = generate_cert_chain(secrets_dir)
    logger.info(f"Signing key: {key_path}")
    logger.info(f"Certificate chain: {chain_path}")
    logger.info(f"Timestamp authority: {TSA_URL}")

    # Load track list
    tracks_csv = args.tracks_csv or str(
        Path(args.run_dir) / "manifests" / "tracks_selected.csv"
    )
    tracks = pd.read_csv(tracks_csv)
    logger.info(f"Tracks to embed: {len(tracks)}")

    # Build task list
    masters_clean = Path(args.run_dir) / "masters_clean"
    tasks = []
    for _, row in tracks.iterrows():
        track_id = row["track_id"]
        src_path = str(masters_clean / f"{track_id:06d}.wav")
        out_path = str(dirs["masters_c2pa"] / f"{track_id:06d}.wav")
        tasks.append((
            src_path, out_path, track_id,
            row["artist_name"], row.get("track_title", ""),
            claim_generator, key_path, chain_path
        ))

    # Process in parallel
    log_rows = []
    embedded = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(embed_track_c2pa, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            log_rows.append(result)
            if result["status"] == "ok":
                embedded += 1
            else:
                errors += 1
            total = embedded + errors
            if total % 50 == 0 or total == len(tasks):
                logger.info(f"  [{total}/{len(tasks)}] {embedded} ok, {errors} errors")

    # Write log
    log_df = pd.DataFrame(log_rows)
    log_csv = dirs["manifests"] / "c2pa_embed_log.csv"
    log_df.to_csv(log_csv, index=False)

    logger.info(f"\n=== C2PA EMBED SUMMARY ===")
    logger.info(f"Embedded: {embedded}/{len(tracks)}")
    logger.info(f"Errors: {errors}")
    if errors > 0:
        error_rows = log_df[log_df["status"] != "ok"]
        for _, row in error_rows.head(5).iterrows():
            logger.info(f"  Error: {row['track_id']} - {row['status']}")

    outputs = [str(dirs["masters_c2pa"]), str(log_csv)]
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
