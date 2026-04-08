#!/usr/bin/env python3
"""
Baseline Catalog Property Analysis.

Tests whether vulnerability is a property of the catalog's position in CLAP space
rather than a result of model-specific absorption from fine-tuning.

Computes 2-signal (CLAP + FAD) vulnerability for V1, V2, and Baseline,
then correlates per-artist scores across all three.
"""

import csv
import json
import os
import sys
import numpy as np
import math


def spearmanr(x, y):
    """Pure numpy Spearman rank correlation."""
    x, y = np.array(x), np.array(y)
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    rho = 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))
    # t-test for significance
    if abs(rho) < 1:
        t = rho * math.sqrt((n - 2) / (1 - rho**2))
        # Two-tailed p-value approximation using t-distribution
        # For n >= 20 this is very close
        from math import atan, pi
        df = n - 2
        x_val = df / (df + t**2)
        # Regularized incomplete beta approximation (good enough for p < 0.001)
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))  # normal approx
    else:
        p = 0.0
    return rho, p


def pearsonr(x, y):
    """Pure numpy Pearson correlation."""
    x, y = np.array(x), np.array(y)
    n = len(x)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    r = np.mean((x - mx) * (y - my)) / (sx * sy) if sx > 0 and sy > 0 else 0
    if abs(r) < 1:
        t = r * math.sqrt((n - 2) / (1 - r**2))
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    else:
        p = 0.0
    return r, p

RESULTS = os.environ.get("RESULTS_DIR", "/Users/erickpg/capstone/results")
OUT_DIR = os.path.join(RESULTS, "robustness")
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


def sf(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def compute_2signal_vuln(vuln_csv_path):
    """Recompute vulnerability using only CLAP similarity and FAD (2-signal)."""
    rows = load_csv(vuln_csv_path)
    artists = {}
    for r in rows:
        aid = r.get('artist_id', '')
        if not aid or aid == 'artist_id':
            continue
        artists[aid] = {
            'name': r.get('artist_name', ''),
            'genre': r.get('genre', ''),
            'n_tracks': sf(r.get('n_catalog_tracks'), 0),
            'clap': sf(r.get('clap_similarity')),
            'fad': sf(r.get('fad')),
        }

    # Min-max normalize CLAP and FAD across this version's artists
    clap_vals = [a['clap'] for a in artists.values() if a['clap'] is not None]
    fad_vals = [a['fad'] for a in artists.values() if a['fad'] is not None]

    clap_min, clap_max = min(clap_vals), max(clap_vals)
    fad_min, fad_max = min(fad_vals), max(fad_vals)

    for aid, a in artists.items():
        if a['clap'] is not None:
            a['clap_norm'] = (a['clap'] - clap_min) / (clap_max - clap_min + 1e-10)
        else:
            a['clap_norm'] = 0.5

        if a['fad'] is not None:
            # Invert FAD: lower FAD = higher vulnerability
            a['fad_norm'] = 1.0 - (a['fad'] - fad_min) / (fad_max - fad_min + 1e-10)
        else:
            a['fad_norm'] = 0.5

        # Equal weight 2-signal
        a['vuln_2sig'] = 0.5 * a['clap_norm'] + 0.5 * a['fad_norm']

    return artists


# Load per-artist CLAP gap data too
def load_clap_gap(clap_per_artist_path):
    """Load per-artist CLAP gap (matched - mismatched)."""
    rows = load_csv(clap_per_artist_path)
    gaps = {}
    # This file has per-file data, need to aggregate per artist
    artist_matched = {}
    artist_mismatched = {}
    for r in rows:
        aid = r.get('artist_id', '')
        matched = sf(r.get('matched_mean_sim'))
        mismatched = sf(r.get('mismatched_mean_sim'))
        if aid and matched is not None and mismatched is not None:
            artist_matched.setdefault(aid, []).append(matched)
            artist_mismatched.setdefault(aid, []).append(mismatched)

    for aid in artist_matched:
        m = np.mean(artist_matched[aid])
        mm = np.mean(artist_mismatched[aid])
        gaps[aid] = m - mm

    return gaps


# ─── Load all three versions ────────────────────────────────────────────────
print("Loading vulnerability scores...")

versions = {}
for label, subdir in [('V1', 'v1'), ('V2', 'v2'), ('Baseline', 'baseline')]:
    vuln_path = f"{RESULTS}/{subdir}/analysis/vulnerability_scores.csv"
    if not os.path.exists(vuln_path):
        print(f"  WARNING: {vuln_path} not found, skipping {label}")
        continue
    versions[label] = compute_2signal_vuln(vuln_path)
    print(f"  {label}: {len(versions[label])} artists")

    # Also load CLAP gaps
    clap_path = f"{RESULTS}/{subdir}/analysis/clap_per_artist.csv"
    if os.path.exists(clap_path):
        gaps = load_clap_gap(clap_path)
        for aid in versions[label]:
            versions[label][aid]['clap_gap'] = gaps.get(aid, None)

# ─── Find common artists ────────────────────────────────────────────────────
common = sorted(set.intersection(*[set(v.keys()) for v in versions.values()]))
print(f"\nCommon artists across all versions: {len(common)}")

# ─── Pairwise correlations ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2-SIGNAL VULNERABILITY SCORE CORRELATIONS")
print("=" * 70)

results = {}
pairs = [('Baseline', 'V1'), ('Baseline', 'V2'), ('V1', 'V2')]
for a_label, b_label in pairs:
    a_scores = [versions[a_label][aid]['vuln_2sig'] for aid in common]
    b_scores = [versions[b_label][aid]['vuln_2sig'] for aid in common]

    rho, p_rho = spearmanr(a_scores, b_scores)
    r, p_r = pearsonr(a_scores, b_scores)

    # Tier agreement (0.5 threshold)
    a_tiers = ['High' if s > 0.5 else 'Low' for s in a_scores]
    b_tiers = ['High' if s > 0.5 else 'Low' for s in b_scores]
    tier_agree = sum(1 for at, bt in zip(a_tiers, b_tiers) if at == bt)
    tier_pct = 100 * tier_agree / len(common)

    # Cohen's kappa
    # 2x2 contingency
    hh = sum(1 for at, bt in zip(a_tiers, b_tiers) if at == 'High' and bt == 'High')
    hl = sum(1 for at, bt in zip(a_tiers, b_tiers) if at == 'High' and bt == 'Low')
    lh = sum(1 for at, bt in zip(a_tiers, b_tiers) if at == 'Low' and bt == 'High')
    ll = sum(1 for at, bt in zip(a_tiers, b_tiers) if at == 'Low' and bt == 'Low')
    n = len(common)
    po = (hh + ll) / n
    pe = ((hh + hl) * (hh + lh) + (lh + ll) * (hl + ll)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0

    print(f"\n{a_label} vs {b_label}:")
    print(f"  Spearman ρ = {rho:.4f} (p = {p_rho:.6f})")
    print(f"  Pearson r  = {r:.4f} (p = {p_r:.6f})")
    print(f"  Tier agreement: {tier_agree}/{n} ({tier_pct:.0f}%), κ = {kappa:.3f}")
    print(f"  Contingency: HH={hh} HL={hl} LH={lh} LL={ll}")

    results[f"{a_label}_vs_{b_label}"] = {
        "spearman_rho": round(rho, 4),
        "spearman_p": round(float(p_rho), 6),
        "pearson_r": round(r, 4),
        "pearson_p": round(float(p_r), 6),
        "tier_agreement": f"{tier_agree}/{n}",
        "tier_agreement_pct": round(tier_pct, 1),
        "cohens_kappa": round(kappa, 3),
        "contingency": {"HH": hh, "HL": hl, "LH": lh, "LL": ll},
    }

# ─── CLAP gap correlations ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLAP GAP CORRELATIONS (per-artist matched - mismatched)")
print("=" * 70)

for a_label, b_label in pairs:
    a_gaps = [versions[a_label][aid].get('clap_gap') for aid in common]
    b_gaps = [versions[b_label][aid].get('clap_gap') for aid in common]
    valid = [(a, b) for a, b in zip(a_gaps, b_gaps) if a is not None and b is not None]
    if len(valid) < 5:
        continue
    a_v, b_v = zip(*valid)
    rho, p = spearmanr(a_v, b_v)
    print(f"  {a_label} vs {b_label}: ρ = {rho:.4f} (p = {p:.6f}), n = {len(valid)}")
    results[f"{a_label}_vs_{b_label}"]["clap_gap_rho"] = round(rho, 4)
    results[f"{a_label}_vs_{b_label}"]["clap_gap_p"] = round(float(p), 6)

# ─── Per-artist table ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PER-ARTIST SCORES (2-signal, sorted by baseline)")
print("=" * 70)

table = []
for aid in common:
    bl = versions['Baseline'][aid]
    v1 = versions['V1'][aid]
    v2 = versions['V2'][aid]
    table.append({
        'artist_id': aid,
        'name': bl['name'],
        'genre': bl['genre'],
        'n_tracks': bl['n_tracks'],
        'bl_vuln': bl['vuln_2sig'],
        'v1_vuln': v1['vuln_2sig'],
        'v2_vuln': v2['vuln_2sig'],
        'bl_tier': 'High' if bl['vuln_2sig'] > 0.5 else 'Low',
        'v1_tier': 'High' if v1['vuln_2sig'] > 0.5 else 'Low',
        'v2_tier': 'High' if v2['vuln_2sig'] > 0.5 else 'Low',
    })

table.sort(key=lambda x: x['bl_vuln'], reverse=True)

print(f"\n{'Artist':<28} {'BL':>6} {'V1':>6} {'V2':>6}  {'BL':>4} {'V1':>4} {'V2':>4}  Stable?")
print("-" * 78)
for t in table:
    stable = "✓" if t['bl_tier'] == t['v1_tier'] == t['v2_tier'] else "✗"
    print(f"{t['name'][:27]:<28} {t['bl_vuln']:.3f} {t['v1_vuln']:.3f} {t['v2_vuln']:.3f}  "
          f"{t['bl_tier']:>4} {t['v1_tier']:>4} {t['v2_tier']:>4}  {stable}")

# Count stability
all_same = sum(1 for t in table if t['bl_tier'] == t['v1_tier'] == t['v2_tier'])
print(f"\nAll three agree: {all_same}/{len(table)} ({100*all_same/len(table):.0f}%)")

always_high = [t for t in table if t['bl_tier'] == t['v1_tier'] == t['v2_tier'] == 'High']
always_low = [t for t in table if t['bl_tier'] == t['v1_tier'] == t['v2_tier'] == 'Low']
print(f"Always High: {len(always_high)}")
print(f"Always Low:  {len(always_low)}")
print(f"Changed:     {len(table) - all_same}")

results["three_way_tier_agreement"] = {
    "all_agree": all_same,
    "total": len(table),
    "pct": round(100 * all_same / len(table), 1),
    "always_high": len(always_high),
    "always_low": len(always_low),
    "changed": len(table) - all_same,
    "always_high_artists": [t['name'] for t in always_high],
    "always_low_artists": [t['name'] for t in always_low],
}

# ─── Save ────────────────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "baseline_catalog_property.json"), "w") as f:
    json.dump(results, f, indent=2)

with open(os.path.join(OUT_DIR, "baseline_catalog_property_per_artist.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["rank", "artist_id", "artist_name", "genre", "n_tracks",
                "bl_vuln_2sig", "v1_vuln_2sig", "v2_vuln_2sig",
                "bl_tier", "v1_tier", "v2_tier", "all_agree"])
    for i, t in enumerate(table, 1):
        agree = "Yes" if t['bl_tier'] == t['v1_tier'] == t['v2_tier'] else "No"
        w.writerow([i, t['artist_id'], t['name'], t['genre'], int(t['n_tracks']),
                     f"{t['bl_vuln']:.6f}", f"{t['v1_vuln']:.6f}", f"{t['v2_vuln']:.6f}",
                     t['bl_tier'], t['v1_tier'], t['v2_tier'], agree])

print(f"\n[SAVED] {OUT_DIR}/baseline_catalog_property.json")
print(f"[SAVED] {OUT_DIR}/baseline_catalog_property_per_artist.csv")
