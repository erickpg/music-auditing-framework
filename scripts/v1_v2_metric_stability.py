#!/usr/bin/env python3
"""Per-metric stability analysis: V1 vs V2 for each individual signal."""
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
# 1. CLAP similarity (per-artist matched sim)
# ============================================================
print("=" * 60)
print("PER-METRIC STABILITY: V1 vs V2")
print("=" * 60)

v1_clap = load_csv(f"{V1}/clap_per_artist.csv")
v2_clap = load_csv(f"{V2}/clap_per_artist.csv")

def get_artist_clap(rows):
    matched = defaultdict(list)
    mismatched = defaultdict(list)
    for r in rows:
        aid = r.get("artist_id", "")
        if not aid: continue
        m = sf(r.get("matched_mean_sim"))
        mm = sf(r.get("mismatched_mean_sim"))
        if m is not None: matched[aid].append(m)
        if mm is not None: mismatched[aid].append(mm)
    result = {}
    for aid in matched:
        result[aid] = {
            "matched": np.mean(matched[aid]),
            "mismatched": np.mean(mismatched[aid]) if aid in mismatched else None,
            "gap": np.mean(matched[aid]) - (np.mean(mismatched[aid]) if aid in mismatched else 0),
        }
    return result

v1_c = get_artist_clap(v1_clap)
v2_c = get_artist_clap(v2_clap)
common = sorted(set(v1_c.keys()) & set(v2_c.keys()))

print(f"\n--- CLAP Matched Similarity (per-artist) ---")
v1_vals = [v1_c[a]["matched"] for a in common]
v2_vals = [v2_c[a]["matched"] for a in common]
rho, p = stats.spearmanr(v1_vals, v2_vals)
r_pearson, p_pearson = stats.pearsonr(v1_vals, v2_vals)
print(f"  Spearman rho: {rho:.4f} (p={p:.6f})")
print(f"  Pearson r:    {r_pearson:.4f} (p={p_pearson:.6f})")
print(f"  V1 mean: {np.mean(v1_vals):.4f} +/- {np.std(v1_vals):.4f}")
print(f"  V2 mean: {np.mean(v2_vals):.4f} +/- {np.std(v2_vals):.4f}")
# Paired test
t, tp = stats.ttest_rel(v1_vals, v2_vals)
print(f"  Paired t-test: t={t:.3f}, p={tp:.6f}")
# Rank changes
v1_rank = {a: i+1 for i, a in enumerate(sorted(common, key=lambda x: v1_c[x]["matched"], reverse=True))}
v2_rank = {a: i+1 for i, a in enumerate(sorted(common, key=lambda x: v2_c[x]["matched"], reverse=True))}
rank_changes = [abs(v1_rank[a] - v2_rank[a]) for a in common]
print(f"  Mean rank change: {np.mean(rank_changes):.1f} positions")
print(f"  Max rank change:  {max(rank_changes)} positions")
print(f"  Artists moving >10 ranks: {sum(1 for r in rank_changes if r > 10)}/50")

print(f"\n--- CLAP Gap (matched - mismatched) ---")
v1_gaps = [v1_c[a]["gap"] for a in common]
v2_gaps = [v2_c[a]["gap"] for a in common]
rho, p = stats.spearmanr(v1_gaps, v2_gaps)
r_pearson, p_pearson = stats.pearsonr(v1_gaps, v2_gaps)
print(f"  Spearman rho: {rho:.4f} (p={p:.6f})")
print(f"  Pearson r:    {r_pearson:.4f} (p={p_pearson:.6f})")
print(f"  V1 mean gap: {np.mean(v1_gaps):.4f} +/- {np.std(v1_gaps):.4f}")
print(f"  V2 mean gap: {np.mean(v2_gaps):.4f} +/- {np.std(v2_gaps):.4f}")
t, tp = stats.ttest_rel(v1_gaps, v2_gaps)
print(f"  Paired t-test: t={t:.3f}, p={tp:.6f}")
v1_rank_gap = {a: i+1 for i, a in enumerate(sorted(common, key=lambda x: v1_c[x]["gap"], reverse=True))}
v2_rank_gap = {a: i+1 for i, a in enumerate(sorted(common, key=lambda x: v2_c[x]["gap"], reverse=True))}
rank_changes_gap = [abs(v1_rank_gap[a] - v2_rank_gap[a]) for a in common]
print(f"  Mean rank change: {np.mean(rank_changes_gap):.1f} positions")
print(f"  Max rank change:  {max(rank_changes_gap)} positions")
print(f"  Artists moving >10 ranks: {sum(1 for r in rank_changes_gap if r > 10)}/50")

# ============================================================
# 2. FAD (per-artist)
# ============================================================
print(f"\n--- FAD (per-artist, matched only) ---")
v1_fad_rows = {r.get("artist_id"): sf(r.get("fad")) for r in load_csv(f"{V1}/per_artist_fad.csv") if r.get("comparison") == "matched"}
v2_fad_rows = {r.get("artist_id"): sf(r.get("fad")) for r in load_csv(f"{V2}/per_artist_fad.csv") if r.get("comparison") == "matched"}
common_fad = sorted(set(v1_fad_rows.keys()) & set(v2_fad_rows.keys()) - {None, ""})
common_fad = [a for a in common_fad if v1_fad_rows[a] is not None and v2_fad_rows[a] is not None]

v1_fvals = [v1_fad_rows[a] for a in common_fad]
v2_fvals = [v2_fad_rows[a] for a in common_fad]
rho, p = stats.spearmanr(v1_fvals, v2_fvals)
r_pearson, p_pearson = stats.pearsonr(v1_fvals, v2_fvals)
print(f"  Spearman rho: {rho:.4f} (p={p:.6f})")
print(f"  Pearson r:    {r_pearson:.4f} (p={p_pearson:.6f})")
print(f"  V1 mean: {np.mean(v1_fvals):.4f} +/- {np.std(v1_fvals):.4f}")
print(f"  V2 mean: {np.mean(v2_fvals):.4f} +/- {np.std(v2_fvals):.4f}")
t, tp = stats.ttest_rel(v1_fvals, v2_fvals)
print(f"  Paired t-test: t={t:.3f}, p={tp:.6f}")
v1_rank_f = {a: i+1 for i, a in enumerate(sorted(common_fad, key=lambda x: v1_fad_rows[x]))}
v2_rank_f = {a: i+1 for i, a in enumerate(sorted(common_fad, key=lambda x: v2_fad_rows[x]))}
rank_changes_f = [abs(v1_rank_f[a] - v2_rank_f[a]) for a in common_fad]
print(f"  Mean rank change: {np.mean(rank_changes_f):.1f} positions")
print(f"  Max rank change:  {max(rank_changes_f)} positions")
print(f"  Artists moving >10 ranks: {sum(1 for r in rank_changes_f if r > 10)}/50")

# ============================================================
# 3. Musicological similarity (per-artist)
# ============================================================
print(f"\n--- Musicological Similarity (per-artist) ---")
v1_feat = {r.get("artist_id"): sf(r.get("cosine_similarity")) for r in load_csv(f"{V1}/features_per_artist.csv")}
v2_feat = {r.get("artist_id"): sf(r.get("cosine_similarity")) for r in load_csv(f"{V2}/features_per_artist.csv")}
common_feat = sorted(set(v1_feat.keys()) & set(v2_feat.keys()) - {None, ""})
common_feat = [a for a in common_feat if v1_feat[a] is not None and v2_feat[a] is not None]

if common_feat:
    v1_mvals = [v1_feat[a] for a in common_feat]
    v2_mvals = [v2_feat[a] for a in common_feat]
    rho, p = stats.spearmanr(v1_mvals, v2_mvals)
    r_pearson, p_pearson = stats.pearsonr(v1_mvals, v2_mvals)
    print(f"  Spearman rho: {rho:.4f} (p={p:.6f})")
    print(f"  Pearson r:    {r_pearson:.4f} (p={p_pearson:.6f})")
    print(f"  V1 mean: {np.mean(v1_mvals):.4f} +/- {np.std(v1_mvals):.4f}")
    print(f"  V2 mean: {np.mean(v2_mvals):.4f} +/- {np.std(v2_mvals):.4f}")
    t, tp = stats.ttest_rel(v1_mvals, v2_mvals)
    print(f"  Paired t-test: t={t:.3f}, p={tp:.6f}")
    v1_rank_m = {a: i+1 for i, a in enumerate(sorted(common_feat, key=lambda x: v1_feat[x], reverse=True))}
    v2_rank_m = {a: i+1 for i, a in enumerate(sorted(common_feat, key=lambda x: v2_feat[x], reverse=True))}
    rank_changes_m = [abs(v1_rank_m[a] - v2_rank_m[a]) for a in common_feat]
    print(f"  Mean rank change: {np.mean(rank_changes_m):.1f} positions")
    print(f"  Max rank change:  {max(rank_changes_m)} positions")
    print(f"  Artists moving >10 ranks: {sum(1 for r in rank_changes_m if r > 10)}/50")
else:
    print("  No common artists with musicological data")

# ============================================================
# 4. N-gram match rate (per-artist)
# ============================================================
print(f"\n--- N-gram Match Rate (per-artist, 3-gram) ---")
v1_ngram = {}
v2_ngram = {}
for r in load_csv(f"{V1}/ngram_per_artist.csv"):
    aid = r.get("artist_id", "")
    if r.get("ngram_size") == "3" and aid:
        v1_ngram[aid] = sf(r.get("matched_mean_count")) or sf(r.get("mean_match_count")) or 0
for r in load_csv(f"{V2}/ngram_per_artist.csv"):
    aid = r.get("artist_id", "")
    if r.get("ngram_size") == "3" and aid:
        v2_ngram[aid] = sf(r.get("matched_mean_count")) or sf(r.get("mean_match_count")) or 0

common_ng = sorted(set(v1_ngram.keys()) & set(v2_ngram.keys()) - {""})
if common_ng:
    v1_nvals = [v1_ngram[a] for a in common_ng]
    v2_nvals = [v2_ngram[a] for a in common_ng]
    rho, p = stats.spearmanr(v1_nvals, v2_nvals)
    print(f"  Spearman rho: {rho:.4f} (p={p:.6f})")
    print(f"  V1 mean: {np.mean(v1_nvals):.4f} +/- {np.std(v1_nvals):.4f}")
    print(f"  V2 mean: {np.mean(v2_nvals):.4f} +/- {np.std(v2_nvals):.4f}")
    v1_rank_n = {a: i+1 for i, a in enumerate(sorted(common_ng, key=lambda x: v1_ngram[x], reverse=True))}
    v2_rank_n = {a: i+1 for i, a in enumerate(sorted(common_ng, key=lambda x: v2_ngram[x], reverse=True))}
    rank_changes_n = [abs(v1_rank_n[a] - v2_rank_n[a]) for a in common_ng]
    print(f"  Mean rank change: {np.mean(rank_changes_n):.1f} positions")
    print(f"  Max rank change:  {max(rank_changes_n)} positions")
    print(f"  Artists moving >10 ranks: {sum(1 for r in rank_changes_n if r > 10)}/50")
else:
    print("  No common artists with n-gram data")

# ============================================================
# 5. Vulnerability composite
# ============================================================
print(f"\n--- Vulnerability Score (composite) ---")
v1_vuln = {r.get("artist_id"): sf(r.get("vulnerability_score")) for r in load_csv(f"{V1}/vulnerability_scores.csv")}
v2_vuln = {r.get("artist_id"): sf(r.get("vulnerability_score")) for r in load_csv(f"{V2}/vulnerability_scores.csv")}
common_v = sorted(set(v1_vuln.keys()) & set(v2_vuln.keys()) - {None, ""})
common_v = [a for a in common_v if v1_vuln[a] is not None and v2_vuln[a] is not None]
v1_vvals = [v1_vuln[a] for a in common_v]
v2_vvals = [v2_vuln[a] for a in common_v]
rho, p = stats.spearmanr(v1_vvals, v2_vvals)
r_pearson, p_pearson = stats.pearsonr(v1_vvals, v2_vvals)
print(f"  Spearman rho: {rho:.4f} (p={p:.6f})")
print(f"  Pearson r:    {r_pearson:.4f} (p={p_pearson:.6f})")
print(f"  V1 mean: {np.mean(v1_vvals):.4f} +/- {np.std(v1_vvals):.4f}")
print(f"  V2 mean: {np.mean(v2_vvals):.4f} +/- {np.std(v2_vvals):.4f}")
v1_rank_v = {a: i+1 for i, a in enumerate(sorted(common_v, key=lambda x: v1_vuln[x], reverse=True))}
v2_rank_v = {a: i+1 for i, a in enumerate(sorted(common_v, key=lambda x: v2_vuln[x], reverse=True))}
rank_changes_v = [abs(v1_rank_v[a] - v2_rank_v[a]) for a in common_v]
print(f"  Mean rank change: {np.mean(rank_changes_v):.1f} positions")
print(f"  Max rank change:  {max(rank_changes_v)} positions")
print(f"  Artists moving >10 ranks: {sum(1 for r in rank_changes_v if r > 10)}/50")

# ============================================================
# Summary table
# ============================================================
print(f"\n{'='*60}")
print(f"STABILITY SUMMARY")
print(f"{'='*60}")
print(f"{'Metric':<30} {'Spearman':>10} {'Mean rank Δ':>12} {'>10 ranks':>10}")
print(f"{'-'*62}")
print(f"{'CLAP matched sim':<30} {rho:.4f}      ... see above")
# Re-collect for summary
metrics = []

# CLAP matched
v1_v = [v1_c[a]["matched"] for a in common]
v2_v = [v2_c[a]["matched"] for a in common]
rho_cm, _ = stats.spearmanr(v1_v, v2_v)
v1r = {a: i+1 for i, a in enumerate(sorted(common, key=lambda x: v1_c[x]["matched"], reverse=True))}
v2r = {a: i+1 for i, a in enumerate(sorted(common, key=lambda x: v2_c[x]["matched"], reverse=True))}
rc = [abs(v1r[a]-v2r[a]) for a in common]
metrics.append(("CLAP matched sim", rho_cm, np.mean(rc), sum(1 for r in rc if r > 10)))

# CLAP gap
rho_cg, _ = stats.spearmanr(v1_gaps, v2_gaps)
rc_g = rank_changes_gap
metrics.append(("CLAP gap", rho_cg, np.mean(rc_g), sum(1 for r in rc_g if r > 10)))

# FAD
rho_f, _ = stats.spearmanr(v1_fvals, v2_fvals)
metrics.append(("FAD", rho_f, np.mean(rank_changes_f), sum(1 for r in rank_changes_f if r > 10)))

# Musicological
if common_feat:
    rho_m, _ = stats.spearmanr(v1_mvals, v2_mvals)
    metrics.append(("Musicological sim", rho_m, np.mean(rank_changes_m), sum(1 for r in rank_changes_m if r > 10)))

# N-gram
if common_ng:
    rho_n, _ = stats.spearmanr(v1_nvals, v2_nvals)
    metrics.append(("N-gram match rate", rho_n, np.mean(rank_changes_n), sum(1 for r in rank_changes_n if r > 10)))

# Composite
rho_cv, _ = stats.spearmanr(v1_vvals, v2_vvals)
metrics.append(("Vulnerability (composite)", rho_cv, np.mean(rank_changes_v), sum(1 for r in rank_changes_v if r > 10)))

print(f"\n{'Metric':<28} {'Spearman':>9} {'Mean Δ rank':>12} {'>10 ranks':>10}")
print(f"{'-'*60}")
for name, rho, mean_rc, n_big in metrics:
    print(f"{name:<28} {rho:>9.4f} {mean_rc:>12.1f} {n_big:>10}/50")
