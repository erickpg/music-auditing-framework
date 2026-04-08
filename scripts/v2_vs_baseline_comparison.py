#!/usr/bin/env python3
"""
V2 vs Baseline comparison.
Mirrors the V1 vs baseline analysis (ft_vs_bl_comparison) for V2.
Compares per-artist CLAP, FAD, vulnerability, and n-gram metrics.
"""

import csv
import json
import os
import sys
import numpy as np
from scipy import stats

RESULTS = os.environ.get("RESULTS_DIR", "/Users/erickpg/capstone/results")
V2_DIR = f"{RESULTS}/v2/analysis"
BL_DIR = f"{RESULTS}/baseline/analysis"
OUT_DIR = f"{RESULTS}/v2/comparison"
os.makedirs(OUT_DIR, exist_ok=True)


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(v == k for k, v in row.items()):
                continue
            rows.append(row)
    return rows


def safe_float(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# --- Load V2 per-artist vulnerability scores ---
v2_vuln = load_csv(f"{V2_DIR}/vulnerability_scores.csv")
bl_vuln = load_csv(f"{BL_DIR}/vulnerability_scores.csv")

v2_dict = {}
for row in v2_vuln:
    aid = row.get("artist_id", "")
    if not aid or aid == "artist_id":
        continue
    v2_dict[aid] = {
        "name": row.get("artist_name", ""),
        "genre": row.get("genre", ""),
        "clap": safe_float(row.get("clap_similarity")),
        "fad": safe_float(row.get("fad")),
        "musico": safe_float(row.get("musicological_similarity")),
        "vuln": safe_float(row.get("vulnerability_score")),
        "ngram": safe_float(row.get("ngram_match_rate")),
    }

bl_dict = {}
for row in bl_vuln:
    aid = row.get("artist_id", "")
    if not aid or aid == "artist_id":
        continue
    bl_dict[aid] = {
        "name": row.get("artist_name", ""),
        "genre": row.get("genre", ""),
        "clap": safe_float(row.get("clap_similarity")),
        "fad": safe_float(row.get("fad")),
        "musico": safe_float(row.get("musicological_similarity")),
        "vuln": safe_float(row.get("vulnerability_score")),
        "ngram": safe_float(row.get("ngram_match_rate")),
    }

common = sorted(set(v2_dict.keys()) & set(bl_dict.keys()))
print(f"Common artists: {len(common)}")

# --- Paired comparisons ---
metrics = ["clap", "fad", "vuln", "ngram"]
metric_labels = {
    "clap": "CLAP Similarity",
    "fad": "FAD",
    "vuln": "Vulnerability Score",
    "ngram": "N-gram Rate",
}

results_summary = {}

for m in metrics:
    v2_vals = []
    bl_vals = []
    for aid in common:
        v2_v = v2_dict[aid].get(m)
        bl_v = bl_dict[aid].get(m)
        if v2_v is not None and bl_v is not None:
            v2_vals.append(v2_v)
            bl_vals.append(bl_v)

    if len(v2_vals) < 3:
        print(f"\n{metric_labels[m]}: insufficient data ({len(v2_vals)} pairs)")
        continue

    v2_arr = np.array(v2_vals)
    bl_arr = np.array(bl_vals)
    delta = v2_arr - bl_arr

    t_stat, p_val = stats.ttest_rel(v2_arr, bl_arr)
    pooled_std = np.sqrt((np.std(v2_arr)**2 + np.std(bl_arr)**2) / 2)
    d = np.mean(delta) / pooled_std if pooled_std > 0 else 0

    print(f"\n{metric_labels[m]}:")
    print(f"  V2 mean: {np.mean(v2_arr):.6f}")
    print(f"  BL mean: {np.mean(bl_arr):.6f}")
    print(f"  Delta:   {np.mean(delta):.6f}")
    print(f"  Paired t: {t_stat:.4f}, p={p_val:.4f}")
    print(f"  Cohen's d: {d:.4f}")

    results_summary[metric_labels[m]] = {
        "v2_mean": round(float(np.mean(v2_arr)), 6),
        "bl_mean": round(float(np.mean(bl_arr)), 6),
        "delta": round(float(np.mean(delta)), 6),
        "paired_t": round(float(t_stat), 4),
        "p_value": round(float(p_val), 4),
        "effect_size_d": round(float(d), 4),
    }

# --- CLAP per-artist gap comparison ---
v2_clap = load_csv(f"{V2_DIR}/clap_per_artist.csv")
bl_clap = load_csv(f"{BL_DIR}/clap_per_artist.csv")

v2_gaps = {}
for row in v2_clap:
    aid = row.get("artist_id", "")
    gap = safe_float(row.get("mean_sim_gap"))
    if aid and gap is not None:
        v2_gaps[aid] = gap

bl_gaps = {}
for row in bl_clap:
    aid = row.get("artist_id", "")
    gap = safe_float(row.get("mean_sim_gap"))
    if aid and gap is not None:
        bl_gaps[aid] = gap

common_gap = sorted(set(v2_gaps.keys()) & set(bl_gaps.keys()))
if common_gap:
    v2_g = np.array([v2_gaps[a] for a in common_gap])
    bl_g = np.array([bl_gaps[a] for a in common_gap])
    gap_delta = v2_g - bl_g
    t_gap, p_gap = stats.ttest_rel(v2_g, bl_g)
    d_gap = np.mean(gap_delta) / np.std(gap_delta) if np.std(gap_delta) > 0 else 0

    print(f"\nCLAP Gap (matched - mismatched):")
    print(f"  V2 mean gap: {np.mean(v2_g):.6f}")
    print(f"  BL mean gap: {np.mean(bl_g):.6f}")
    print(f"  Delta:       {np.mean(gap_delta):.6f}")
    print(f"  Paired t:    {t_gap:.4f}, p={p_gap:.4f}")
    print(f"  Cohen's d_z: {d_gap:.4f}")

    results_summary["CLAP Gap"] = {
        "v2_mean": round(float(np.mean(v2_g)), 6),
        "bl_mean": round(float(np.mean(bl_g)), 6),
        "delta": round(float(np.mean(gap_delta)), 6),
        "paired_t": round(float(t_gap), 4),
        "p_value": round(float(p_gap), 4),
        "effect_size_dz": round(float(d_gap), 4),
    }

# --- Per-artist absorption (V2 vuln - BL vuln) ---
absorption_data = []
for aid in common:
    v2_v = v2_dict[aid].get("vuln")
    bl_v = bl_dict[aid].get("vuln")
    if v2_v is not None and bl_v is not None:
        absorption_data.append({
            "artist_id": aid,
            "artist_name": v2_dict[aid]["name"],
            "genre": v2_dict[aid]["genre"],
            "v2_vuln": v2_v,
            "bl_vuln": bl_v,
            "absorption": v2_v - bl_v,
        })

absorption_data.sort(key=lambda x: x["absorption"], reverse=True)

absorptions = [d["absorption"] for d in absorption_data]
n_pos = sum(1 for a in absorptions if a > 0)
n_neg = sum(1 for a in absorptions if a < 0)
t_abs, p_abs = stats.ttest_1samp(absorptions, 0)

print(f"\nV2 Absorption (V2_vuln - BL_vuln):")
print(f"  Mean: {np.mean(absorptions):+.4f}")
print(f"  Positive (V2 > BL): {n_pos}/{len(absorptions)}")
print(f"  Negative (V2 < BL): {n_neg}/{len(absorptions)}")
print(f"  t vs 0: {t_abs:.4f}, p={p_abs:.4f}")

results_summary["Absorption"] = {
    "mean": round(float(np.mean(absorptions)), 4),
    "n_positive": n_pos,
    "n_negative": n_neg,
    "t_vs_zero": round(float(t_abs), 4),
    "p_value": round(float(p_abs), 4),
}

print(f"\nTop 5 absorption (V2 gained most vs baseline):")
for d in absorption_data[:5]:
    print(f"  {d['artist_name']}: {d['absorption']:+.4f} (V2={d['v2_vuln']:.4f}, BL={d['bl_vuln']:.4f})")

print(f"\nBottom 5 absorption (V2 lost most vs baseline):")
for d in absorption_data[-5:]:
    print(f"  {d['artist_name']}: {d['absorption']:+.4f} (V2={d['v2_vuln']:.4f}, BL={d['bl_vuln']:.4f})")

# --- Save ---
with open(f"{OUT_DIR}/v2_vs_bl_comparison_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

with open(f"{OUT_DIR}/v2_vs_bl_absorption.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["rank", "artist_id", "artist_name", "genre", "v2_vuln", "bl_vuln", "absorption"])
    for i, d in enumerate(absorption_data, 1):
        w.writerow([i, d["artist_id"], d["artist_name"], d["genre"],
                     f"{d['v2_vuln']:.6f}", f"{d['bl_vuln']:.6f}", f"{d['absorption']:.6f}"])

print(f"\n[SAVED] {OUT_DIR}/v2_vs_bl_comparison_summary.json")
print(f"[SAVED] {OUT_DIR}/v2_vs_bl_absorption.csv")
