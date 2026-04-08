#!/usr/bin/env python3
"""Recompute vulnerability scores with different signal combinations,
then measure cross-version (v1 vs v2) stability for each."""
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
# Load all per-artist signals for both versions
# ============================================================

def load_signals(base):
    signals = {}

    # CLAP gap (matched - mismatched)
    clap_matched = defaultdict(list)
    clap_mismatched = defaultdict(list)
    for r in load_csv(f"{base}/clap_per_artist.csv"):
        aid = r.get("artist_id", "")
        if not aid: continue
        m = sf(r.get("matched_mean_sim"))
        mm = sf(r.get("mismatched_mean_sim"))
        if m is not None: clap_matched[aid].append(m)
        if mm is not None: clap_mismatched[aid].append(mm)

    # FAD (matched, lower = more similar)
    fad = {}
    for r in load_csv(f"{base}/per_artist_fad.csv"):
        if r.get("comparison") == "matched":
            aid = r.get("artist_id", "")
            if aid: fad[aid] = sf(r.get("fad"))

    # Musicological (matched_sim)
    musico = {}
    for r in load_csv(f"{base}/features_per_artist.csv"):
        aid = r.get("artist_id", "")
        if aid: musico[aid] = sf(r.get("matched_sim"))

    # N-gram (mean match count across all n-gram sizes)
    ngram = defaultdict(list)
    for r in load_csv(f"{base}/ngram_per_artist.csv"):
        aid = r.get("artist_id", "")
        val = sf(r.get("matched_mean_count")) or sf(r.get("mean_match_count"))
        if aid and val is not None:
            ngram[aid].append(val)

    # Combine
    all_artists = set()
    for d in [clap_matched, fad, musico, ngram]:
        all_artists.update(d.keys())

    for aid in sorted(all_artists):
        if not aid: continue
        signals[aid] = {
            "clap_gap": (np.mean(clap_matched[aid]) - np.mean(clap_mismatched[aid]))
                        if aid in clap_matched and aid in clap_mismatched else None,
            "clap_matched": np.mean(clap_matched[aid]) if aid in clap_matched else None,
            "fad": fad.get(aid),
            "musico": musico.get(aid),
            "ngram": np.mean(ngram[aid]) if aid in ngram else None,
        }

    return signals

v1_sig = load_signals(V1)
v2_sig = load_signals(V2)
common = sorted(set(v1_sig.keys()) & set(v2_sig.keys()))
print(f"Common artists: {len(common)}")

# ============================================================
# Compute composite scores under different signal combinations
# ============================================================

def normalize_signal(values, higher_is_more_vulnerable=True):
    """Min-max normalize. For FAD, lower = more similar = more vulnerable, so invert."""
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0 or arr.max() == arr.min():
        return {i: 0.5 for i, v in enumerate(values)}
    result = {}
    for i, v in enumerate(values):
        if v is None:
            result[i] = None
        else:
            normed = (v - arr.min()) / (arr.max() - arr.min())
            result[i] = normed if higher_is_more_vulnerable else (1 - normed)
    return result

def compute_composite(signals_dict, artists, signal_names, signal_directions):
    """Compute normalized composite vulnerability score."""
    # Gather raw values
    raw = {s: [signals_dict[a].get(s) for a in artists] for s in signal_names}

    # Normalize each signal
    normed = {}
    for s, direction in zip(signal_names, signal_directions):
        normed[s] = normalize_signal(raw[s], higher_is_more_vulnerable=direction)

    # Equal-weight average
    scores = {}
    for i, a in enumerate(artists):
        vals = [normed[s][i] for s in signal_names if normed[s][i] is not None]
        scores[a] = np.mean(vals) if vals else None

    return scores

# Signal configs to test
configs = [
    {
        "name": "4-signal (original)",
        "signals": ["clap_gap", "fad", "musico", "ngram"],
        "directions": [True, False, True, True],  # FAD: lower = more vulnerable
    },
    {
        "name": "3-signal (drop musico)",
        "signals": ["clap_gap", "fad", "ngram"],
        "directions": [True, False, True],
    },
    {
        "name": "2-signal (CLAP + FAD)",
        "signals": ["clap_gap", "fad"],
        "directions": [True, False],
    },
    {
        "name": "2-signal (CLAP + musico)",
        "signals": ["clap_gap", "musico"],
        "directions": [True, True],
    },
    {
        "name": "CLAP gap only",
        "signals": ["clap_gap"],
        "directions": [True],
    },
    {
        "name": "FAD only",
        "signals": ["fad"],
        "directions": [False],
    },
    {
        "name": "Musico only",
        "signals": ["musico"],
        "directions": [True],
    },
    {
        "name": "3-signal (drop ngram)",
        "signals": ["clap_gap", "fad", "musico"],
        "directions": [True, False, True],
    },
    {
        "name": "CLAP matched sim + FAD",
        "signals": ["clap_matched", "fad"],
        "directions": [True, False],
    },
]

print(f"\n{'='*80}")
print(f"CROSS-VERSION STABILITY BY SIGNAL COMBINATION")
print(f"{'='*80}")
print(f"\n{'Config':<35} {'Spearman':>9} {'Pearson':>9} {'Mean Δ':>8} {'>10':>5} {'V1 mean':>9} {'V2 mean':>9}")
print(f"{'-'*85}")

results = []

for cfg in configs:
    v1_scores = compute_composite(v1_sig, common, cfg["signals"], cfg["directions"])
    v2_scores = compute_composite(v2_sig, common, cfg["signals"], cfg["directions"])

    valid = [a for a in common if v1_scores[a] is not None and v2_scores[a] is not None]
    if len(valid) < 5:
        print(f"{cfg['name']:<35} {'N/A':>9} (too few artists)")
        continue

    v1_v = [v1_scores[a] for a in valid]
    v2_v = [v2_scores[a] for a in valid]

    rho, p_rho = stats.spearmanr(v1_v, v2_v)
    r_p, p_p = stats.pearsonr(v1_v, v2_v)

    # Rank changes
    v1_ranked = sorted(valid, key=lambda a: v1_scores[a], reverse=True)
    v2_ranked = sorted(valid, key=lambda a: v2_scores[a], reverse=True)
    v1_ranks = {a: i+1 for i, a in enumerate(v1_ranked)}
    v2_ranks = {a: i+1 for i, a in enumerate(v2_ranked)}
    rank_changes = [abs(v1_ranks[a] - v2_ranks[a]) for a in valid]
    mean_rc = np.mean(rank_changes)
    n_big = sum(1 for r in rank_changes if r > 10)

    print(f"{cfg['name']:<35} {rho:>9.4f} {r_p:>9.4f} {mean_rc:>8.1f} {n_big:>5} {np.mean(v1_v):>9.4f} {np.mean(v2_v):>9.4f}")

    results.append({
        "config": cfg["name"],
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "pearson_r": float(r_p), "pearson_p": float(p_p),
        "mean_rank_change": float(mean_rc),
        "n_big_shifts": int(n_big),
        "v1_mean": float(np.mean(v1_v)),
        "v2_mean": float(np.mean(v2_v)),
        "n_artists": len(valid),
    })

# ============================================================
# Also: Wilcoxon for the best config
# ============================================================
print(f"\n{'='*80}")
print("DETAILED: Best configs")
print(f"{'='*80}")

for cfg_name in ["2-signal (CLAP + FAD)", "3-signal (drop musico)", "4-signal (original)"]:
    cfg = [c for c in configs if c["name"] == cfg_name][0]
    v1_scores = compute_composite(v1_sig, common, cfg["signals"], cfg["directions"])
    v2_scores = compute_composite(v2_sig, common, cfg["signals"], cfg["directions"])
    valid = [a for a in common if v1_scores[a] is not None and v2_scores[a] is not None]
    v1_v = [v1_scores[a] for a in valid]
    v2_v = [v2_scores[a] for a in valid]

    w_stat, w_p = stats.wilcoxon(v1_v, v2_v)
    t, tp = stats.ttest_rel(v1_v, v2_v)

    print(f"\n{cfg_name}:")
    print(f"  Wilcoxon: W={w_stat:.1f}, p={w_p:.4f}")
    print(f"  Paired t: t={t:.3f}, p={tp:.4f}")
    print(f"  V1: {np.mean(v1_v):.4f} +/- {np.std(v1_v):.4f}")
    print(f"  V2: {np.mean(v2_v):.4f} +/- {np.std(v2_v):.4f}")

    # Top 5 in each
    v1_ranked = sorted(valid, key=lambda a: v1_scores[a], reverse=True)
    v2_ranked = sorted(valid, key=lambda a: v2_scores[a], reverse=True)
    # Need artist names
    v1_vuln_rows = {r.get("artist_id"): r for r in load_csv(f"{V1}/vulnerability_scores.csv")}
    print(f"  V1 top 5: {[v1_vuln_rows.get(a, {}).get('artist_name', a)[:20] for a in v1_ranked[:5]]}")
    print(f"  V2 top 5: {[v1_vuln_rows.get(a, {}).get('artist_name', a)[:20] for a in v2_ranked[:5]]}")

    # Top-5 overlap
    top5_overlap = len(set(v1_ranked[:5]) & set(v2_ranked[:5]))
    top10_overlap = len(set(v1_ranked[:10]) & set(v2_ranked[:10]))
    print(f"  Top-5 overlap: {top5_overlap}/5")
    print(f"  Top-10 overlap: {top10_overlap}/10")

# Save results
OUT = "/scratch/$USER/runs/2026-03-10_full_v2/comparison"
with open(f"{OUT}/signal_combination_stability.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT}/signal_combination_stability.json")
