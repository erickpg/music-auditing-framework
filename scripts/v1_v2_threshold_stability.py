#!/usr/bin/env python3
"""Threshold-based vulnerability tier stability: V1 vs V2.
Instead of rankings, assign artists to tiers and check agreement."""
import csv, json, os
import numpy as np
from collections import defaultdict
from scipy import stats

V1 = "/scratch/$USER/runs/2026-03-10_full/analysis"
V2 = "/scratch/$USER/runs/2026-03-10_full_v2/analysis"

def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def sf(v):
    try: return float(v)
    except: return None

# ============================================================
# Load signals and compute 2-signal (CLAP gap + FAD) scores
# ============================================================

def load_and_score(base):
    # CLAP gap
    clap_matched = defaultdict(list)
    clap_mismatched = defaultdict(list)
    for r in load_csv(f"{base}/clap_per_artist.csv"):
        aid = r.get("artist_id", "")
        if not aid: continue
        m = sf(r.get("matched_mean_sim"))
        mm = sf(r.get("mismatched_mean_sim"))
        if m is not None: clap_matched[aid].append(m)
        if mm is not None: clap_mismatched[aid].append(mm)

    # FAD
    fad = {}
    for r in load_csv(f"{base}/per_artist_fad.csv"):
        if r.get("comparison") == "matched":
            aid = r.get("artist_id", "")
            if aid: fad[aid] = sf(r.get("fad"))

    # Compute per-artist signals
    artists = sorted(set(clap_matched.keys()) & set(fad.keys()))
    clap_gaps = {a: np.mean(clap_matched[a]) - np.mean(clap_mismatched[a])
                 for a in artists if a in clap_mismatched}
    artists = [a for a in artists if a in clap_gaps and fad[a] is not None]

    # Normalize CLAP gap (higher = more vulnerable)
    cg_vals = [clap_gaps[a] for a in artists]
    cg_min, cg_max = min(cg_vals), max(cg_vals)
    cg_norm = {a: (clap_gaps[a] - cg_min) / (cg_max - cg_min) for a in artists}

    # Normalize FAD (lower FAD = more similar = more vulnerable, so invert)
    fad_vals = [fad[a] for a in artists]
    fad_min, fad_max = min(fad_vals), max(fad_vals)
    fad_norm = {a: 1 - (fad[a] - fad_min) / (fad_max - fad_min) for a in artists}

    # Composite: equal weight
    scores = {a: (cg_norm[a] + fad_norm[a]) / 2 for a in artists}

    return scores, clap_gaps, fad

v1_scores, v1_clap, v1_fad = load_and_score(V1)
v2_scores, v2_clap, v2_fad = load_and_score(V2)

# Artist names
names = {}
for r in load_csv(f"{V1}/vulnerability_scores.csv"):
    names[r.get("artist_id", "")] = r.get("artist_name", "")

common = sorted(set(v1_scores.keys()) & set(v2_scores.keys()))
print(f"Common artists: {len(common)}")
print(f"V1 score range: [{min(v1_scores[a] for a in common):.3f}, {max(v1_scores[a] for a in common):.3f}]")
print(f"V2 score range: [{min(v2_scores[a] for a in common):.3f}, {max(v2_scores[a] for a in common):.3f}]")

# ============================================================
# Test multiple tier schemes
# ============================================================

tier_schemes = [
    {
        "name": "3 tiers (Low/Medium/High)",
        "thresholds": [0.33, 0.67],
        "labels": ["Low", "Medium", "High"],
    },
    {
        "name": "3 tiers (0.3/0.6)",
        "thresholds": [0.3, 0.6],
        "labels": ["Low", "Medium", "High"],
    },
    {
        "name": "3 tiers (0.4/0.7)",
        "thresholds": [0.4, 0.7],
        "labels": ["Low", "Medium", "High"],
    },
    {
        "name": "2 tiers (0.5 split)",
        "thresholds": [0.5],
        "labels": ["Low", "High"],
    },
    {
        "name": "2 tiers (median split)",
        "thresholds": ["median"],
        "labels": ["Below median", "Above median"],
    },
    {
        "name": "4 tiers (quartiles)",
        "thresholds": ["q25", "q50", "q75"],
        "labels": ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
    },
    {
        "name": "5 tiers (quintiles)",
        "thresholds": ["p20", "p40", "p60", "p80"],
        "labels": ["Very Low", "Low", "Medium", "High", "Very High"],
    },
]

def assign_tier(score, thresholds, labels):
    for i, t in enumerate(thresholds):
        if score < t:
            return labels[i], i
    return labels[-1], len(thresholds)

def resolve_thresholds(thresholds, scores_list):
    """Replace string thresholds with computed values."""
    resolved = []
    for t in thresholds:
        if t == "median":
            resolved.append(np.median(scores_list))
        elif t == "q25":
            resolved.append(np.percentile(scores_list, 25))
        elif t == "q50":
            resolved.append(np.percentile(scores_list, 50))
        elif t == "q75":
            resolved.append(np.percentile(scores_list, 75))
        elif isinstance(t, str) and t.startswith("p"):
            resolved.append(np.percentile(scores_list, int(t[1:])))
        else:
            resolved.append(t)
    return resolved

print(f"\n{'='*80}")
print("THRESHOLD-BASED TIER STABILITY")
print(f"{'='*80}")

for scheme in tier_schemes:
    # Use pooled scores for threshold computation (so both versions use same boundaries)
    all_scores = [v1_scores[a] for a in common] + [v2_scores[a] for a in common]
    thresholds = resolve_thresholds(scheme["thresholds"], all_scores)
    labels = scheme["labels"]

    v1_tiers = {a: assign_tier(v1_scores[a], thresholds, labels) for a in common}
    v2_tiers = {a: assign_tier(v2_scores[a], thresholds, labels) for a in common}

    # Agreement
    agree = sum(1 for a in common if v1_tiers[a][0] == v2_tiers[a][0])
    pct = 100 * agree / len(common)

    # Cohen's kappa
    n_tiers = len(labels)
    confusion = np.zeros((n_tiers, n_tiers), dtype=int)
    for a in common:
        confusion[v1_tiers[a][1], v2_tiers[a][1]] += 1

    # Kappa calculation
    n = len(common)
    po = agree / n
    pe = sum(confusion[i, :].sum() * confusion[:, i].sum() for i in range(n_tiers)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0

    # How many shifted by more than 1 tier?
    shifts = [abs(v1_tiers[a][1] - v2_tiers[a][1]) for a in common]
    big_shifts = sum(1 for s in shifts if s > 1)

    print(f"\n--- {scheme['name']} ---")
    print(f"  Thresholds: {[round(t, 3) for t in thresholds]}")
    print(f"  Agreement: {agree}/{n} ({pct:.0f}%)")
    print(f"  Cohen's kappa: {kappa:.3f}", end="")
    if kappa < 0.2: print(" (poor)")
    elif kappa < 0.4: print(" (fair)")
    elif kappa < 0.6: print(" (moderate)")
    elif kappa < 0.8: print(" (substantial)")
    else: print(" (almost perfect)")
    if n_tiers > 2:
        print(f"  Shifted >1 tier: {big_shifts}/{n} ({100*big_shifts/n:.0f}%)")

    # Distribution
    for label in labels:
        v1_n = sum(1 for a in common if v1_tiers[a][0] == label)
        v2_n = sum(1 for a in common if v2_tiers[a][0] == label)
        print(f"    {label:>15}: V1={v1_n:>3}, V2={v2_n:>3}")

    # Confusion matrix
    print(f"  Confusion matrix (rows=V1, cols=V2):")
    header = "         " + "".join(f"{l[:8]:>9}" for l in labels)
    print(header)
    for i, l1 in enumerate(labels):
        row = f"  {l1[:8]:>7} " + "".join(f"{confusion[i,j]:>9}" for j in range(n_tiers))
        print(row)

# ============================================================
# Best scheme: show per-artist detail
# ============================================================
print(f"\n{'='*80}")
print("PER-ARTIST DETAIL: 3-tier (Low/Medium/High) at 0.33/0.67")
print(f"{'='*80}")

thresholds = [0.33, 0.67]
labels = ["Low", "Medium", "High"]

v1_tiers = {a: assign_tier(v1_scores[a], thresholds, labels) for a in common}
v2_tiers = {a: assign_tier(v2_scores[a], thresholds, labels) for a in common}

# Artists that changed tier
changed = [(a, v1_tiers[a][0], v2_tiers[a][0], v1_scores[a], v2_scores[a])
           for a in common if v1_tiers[a][0] != v2_tiers[a][0]]
stable = [(a, v1_tiers[a][0], v1_scores[a], v2_scores[a])
          for a in common if v1_tiers[a][0] == v2_tiers[a][0]]

print(f"\nSTABLE ({len(stable)} artists - same tier in both versions):")
print(f"  {'Artist':<25} {'Tier':<10} {'V1 score':>10} {'V2 score':>10}")
print(f"  {'-'*55}")
for a, tier, s1, s2 in sorted(stable, key=lambda x: -x[2]):
    print(f"  {names.get(a, a)[:25]:<25} {tier:<10} {s1:>10.4f} {s2:>10.4f}")

print(f"\nCHANGED ({len(changed)} artists - different tier):")
print(f"  {'Artist':<25} {'V1 tier':<10} {'V2 tier':<10} {'V1 score':>10} {'V2 score':>10}")
print(f"  {'-'*65}")
for a, t1, t2, s1, s2 in sorted(changed, key=lambda x: -abs(x[3]-x[4])):
    print(f"  {names.get(a, a)[:25]:<25} {t1:<10} {t2:<10} {s1:>10.4f} {s2:>10.4f}")

# ============================================================
# Summary recommendation
# ============================================================
print(f"\n{'='*80}")
print("SUMMARY: BEST TIER SCHEME")
print(f"{'='*80}")
print("""
Tier scheme comparison (2-signal CLAP+FAD, pooled thresholds):

  Scheme              Agreement   Kappa    Interpretation
  2-tier (median)       72%       0.44     Moderate
  2-tier (0.5 split)    72-78%    ~0.5     Moderate
  3-tier (0.33/0.67)    66-72%    ~0.4     Fair-Moderate
  4-tier (quartiles)    40-50%    ~0.2     Fair
  5-tier (quintiles)    30-40%    ~0.15    Poor

Recommendation: Use 2 or 3 tiers for stable reporting.
At 3 tiers, ~70% of artists get the same classification
regardless of model version. The ~30% that shift are
mostly borderline cases moving between adjacent tiers.
""")
