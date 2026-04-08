#!/usr/bin/env python3
"""Phase 3: V1 vs V2 Comparative Analysis (C1-C7)"""
import csv, json, os, sys
import numpy as np
from collections import defaultdict
from scipy import stats

V1 = "/scratch/$USER/runs/2026-03-10_full/analysis"
V2 = "/scratch/$USER/runs/2026-03-10_full_v2/analysis"
OUT = "/scratch/$USER/runs/2026-03-10_full_v2/comparison"
os.makedirs(OUT, exist_ok=True)

def load_csv(path):
    rows = []
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def sf(v):
    try: return float(v)
    except: return None

# Load vulnerability scores
v1_vuln = {r.get("artist_id", ""): r for r in load_csv(f"{V1}/vulnerability_scores.csv")}
v2_vuln = {r.get("artist_id", ""): r for r in load_csv(f"{V2}/vulnerability_scores.csv")}
common_artists = sorted(set(v1_vuln.keys()) & set(v2_vuln.keys()) - {""})
print(f"Common artists: {len(common_artists)}")

# Load CLAP per-artist
v1_clap = load_csv(f"{V1}/clap_per_artist.csv")
v2_clap = load_csv(f"{V2}/clap_per_artist.csv")

def aggregate_clap(rows):
    matched = defaultdict(list)
    mismatched = defaultdict(list)
    for r in rows:
        aid = r.get("artist_id", "")
        if not aid: continue
        m = sf(r.get("matched_mean_sim"))
        mm = sf(r.get("mismatched_mean_sim"))
        if m is not None: matched[aid].append(m)
        if mm is not None: mismatched[aid].append(mm)
    return matched, mismatched

v1_m, v1_mm = aggregate_clap(v1_clap)
v2_m, v2_mm = aggregate_clap(v2_clap)

# FAD — column is "fad", filter to matched comparisons
v1_fad = {r.get("artist_id", ""): sf(r.get("fad")) for r in load_csv(f"{V1}/per_artist_fad.csv") if r.get("comparison") == "matched"}
v2_fad = {r.get("artist_id", ""): sf(r.get("fad")) for r in load_csv(f"{V2}/per_artist_fad.csv") if r.get("comparison") == "matched"}

# N-gram verdicts
v1_ngram_json = json.load(open(f"{V1}/memorization_verdict.json")) if os.path.exists(f"{V1}/memorization_verdict.json") else {}
v2_ngram_json = json.load(open(f"{V2}/memorization_verdict.json")) if os.path.exists(f"{V2}/memorization_verdict.json") else {}

# ============================================================
# C1: Aggregate Comparison Table
# ============================================================
print("\n=== C1: Aggregate Comparison ===")
v1_vulns = [sf(v1_vuln[a].get("vulnerability_score")) for a in common_artists if sf(v1_vuln[a].get("vulnerability_score")) is not None]
v2_vulns = [sf(v2_vuln[a].get("vulnerability_score")) for a in common_artists if sf(v2_vuln[a].get("vulnerability_score")) is not None]

v1_gaps = [np.mean(v1_m.get(a, [0])) - np.mean(v1_mm.get(a, [0])) for a in common_artists if a in v1_m and a in v1_mm]
v2_gaps = [np.mean(v2_m.get(a, [0])) - np.mean(v2_mm.get(a, [0])) for a in common_artists if a in v2_m and a in v2_mm]

v1_fads_list = [v1_fad.get(a) for a in common_artists if v1_fad.get(a) is not None]
v2_fads_list = [v2_fad.get(a) for a in common_artists if v2_fad.get(a) is not None]

agg = {
    "v1": {
        "mean_vulnerability": float(np.mean(v1_vulns)) if v1_vulns else None,
        "mean_clap_gap": float(np.mean(v1_gaps)) if v1_gaps else None,
        "mean_fad": float(np.mean(v1_fads_list)) if v1_fads_list else None,
        "training_loss": 4.22,
        "n_significant_ngram": v1_ngram_json.get("n_significant", 0),
    },
    "v2": {
        "mean_vulnerability": float(np.mean(v2_vulns)) if v2_vulns else None,
        "mean_clap_gap": float(np.mean(v2_gaps)) if v2_gaps else None,
        "mean_fad": float(np.mean(v2_fads_list)) if v2_fads_list else None,
        "training_loss": 4.07,
        "n_significant_ngram": v2_ngram_json.get("n_significant", 0),
    },
    "n_common_artists": len(common_artists),
}

# Paired Wilcoxon on vulnerability
paired_v1 = [sf(v1_vuln[a].get("vulnerability_score")) for a in common_artists]
paired_v2 = [sf(v2_vuln[a].get("vulnerability_score")) for a in common_artists]
valid_pairs = [(a, b) for a, b in zip(paired_v1, paired_v2) if a is not None and b is not None]
if len(valid_pairs) >= 5:
    w_stat, w_p = stats.wilcoxon([a for a, b in valid_pairs], [b for a, b in valid_pairs])
    agg["wilcoxon_vulnerability"] = {"statistic": float(w_stat), "p": float(w_p)}
    print(f"  Wilcoxon vulnerability: W={w_stat:.1f}, p={w_p:.4f}")

for k in ["v1", "v2"]:
    v = agg[k]
    vuln_str = f"{v['mean_vulnerability']:.4f}" if v['mean_vulnerability'] is not None else "N/A"
    gap_str = f"{v['mean_clap_gap']:.4f}" if v['mean_clap_gap'] is not None else "N/A"
    fad_str = f"{v['mean_fad']:.4f}" if v['mean_fad'] is not None else "N/A"
    print(f"  {k}: vuln={vuln_str}, clap_gap={gap_str}, fad={fad_str}")

with open(f"{OUT}/aggregate_comparison.json", "w") as f:
    json.dump(agg, f, indent=2)

# ============================================================
# C2: Per-Artist Delta
# ============================================================
print("\n=== C2: Per-Artist Delta ===")
delta_rows = []
for a in common_artists:
    v1_v = sf(v1_vuln[a].get("vulnerability_score"))
    v2_v = sf(v2_vuln[a].get("vulnerability_score"))
    n_tracks = sf(v1_vuln[a].get("n_catalog_tracks")) or sf(v2_vuln[a].get("n_catalog_tracks"))
    if v1_v is not None and v2_v is not None:
        delta_rows.append({
            "artist_id": a,
            "artist_name": v1_vuln[a].get("artist_name", ""),
            "genre": v1_vuln[a].get("genre", ""),
            "v1_vulnerability": round(v1_v, 6),
            "v2_vulnerability": round(v2_v, 6),
            "delta": round(v2_v - v1_v, 6),
            "n_catalog_tracks": int(n_tracks) if n_tracks else "",
        })

delta_rows.sort(key=lambda x: x["delta"], reverse=True)
with open(f"{OUT}/per_artist_delta.csv", "w") as f:
    w = csv.DictWriter(f, ["artist_id", "artist_name", "genre", "v1_vulnerability", "v2_vulnerability", "delta", "n_catalog_tracks"])
    w.writeheader()
    w.writerows(delta_rows)

deltas = [r["delta"] for r in delta_rows]
n_tracks_list = [r["n_catalog_tracks"] for r in delta_rows if r["n_catalog_tracks"] != ""]
if deltas and n_tracks_list and len(n_tracks_list) >= 5:
    rho, p = stats.spearmanr(deltas[:len(n_tracks_list)], n_tracks_list)
    print(f"  Delta vs n_tracks: rho={rho:.3f}, p={p:.4f}")

gained = sum(1 for d in deltas if d > 0)
lost = sum(1 for d in deltas if d < 0)
print(f"  Gained vulnerability: {gained}, Lost: {lost}, Mean delta: {np.mean(deltas):.4f}")

# ============================================================
# C3: N-gram Memorization Comparison
# ============================================================
print("\n=== C3: N-gram Comparison ===")
v1_ns = load_csv(f"{V1}/ngram_statistical_tests.csv") if os.path.exists(f"{V1}/ngram_statistical_tests.csv") else []
v2_ns = load_csv(f"{V2}/ngram_statistical_tests.csv") if os.path.exists(f"{V2}/ngram_statistical_tests.csv") else []
v1_sig = [r for r in v1_ns if r.get("significant_fdr", "").lower() == "true"]
v2_sig = [r for r in v2_ns if r.get("significant_fdr", "").lower() == "true"]
ngram_comp = {"v1_tests": len(v1_ns), "v2_tests": len(v2_ns), "v1_significant": len(v1_sig), "v2_significant": len(v2_sig)}
print(f"  V1: {len(v1_sig)}/{len(v1_ns)} significant, V2: {len(v2_sig)}/{len(v2_ns)} significant")
with open(f"{OUT}/ngram_comparison.json", "w") as f:
    json.dump(ngram_comp, f, indent=2)

# ============================================================
# C4: CLAP Gap Improvement
# ============================================================
print("\n=== C4: CLAP Gap Improvement ===")
pg1, pg2 = [], []
for a in common_artists:
    if a in v1_m and a in v1_mm and a in v2_m and a in v2_mm:
        pg1.append(np.mean(v1_m[a]) - np.mean(v1_mm[a]))
        pg2.append(np.mean(v2_m[a]) - np.mean(v2_mm[a]))

if len(pg1) >= 5:
    t, p = stats.ttest_rel(pg2, pg1)
    imp = np.mean(pg2) - np.mean(pg1)
    clap_comp = {"v1_mean_gap": float(np.mean(pg1)), "v2_mean_gap": float(np.mean(pg2)),
                 "improvement": float(imp), "paired_t": float(t), "p": float(p), "n": len(pg1)}
    print(f"  V1 gap: {np.mean(pg1):.4f}, V2 gap: {np.mean(pg2):.4f}")
    print(f"  Improvement: {imp:+.4f}, t={t:.3f}, p={p:.4f}")
else:
    clap_comp = {"error": "too few paired artists"}
with open(f"{OUT}/clap_gap_comparison.json", "w") as f:
    json.dump(clap_comp, f, indent=2)

# ============================================================
# C5: Training Loss vs Vulnerability
# ============================================================
print("\n=== C5: Training Loss vs Vulnerability ===")
loss_comp = {
    "v1_final_loss": 4.22, "v2_final_loss": 4.07,
    "loss_improvement_pct": round((4.22 - 4.07) / 4.22 * 100, 1),
    "v1_mean_vuln": float(np.mean(v1_vulns)) if v1_vulns else None,
    "v2_mean_vuln": float(np.mean(v2_vulns)) if v2_vulns else None,
}
if v1_vulns and v2_vulns:
    loss_comp["vuln_change_pct"] = round((np.mean(v2_vulns) - np.mean(v1_vulns)) / np.mean(v1_vulns) * 100, 1)
v1_mv = f"{loss_comp['v1_mean_vuln']:.4f}" if loss_comp['v1_mean_vuln'] else "N/A"
v2_mv = f"{loss_comp['v2_mean_vuln']:.4f}" if loss_comp['v2_mean_vuln'] else "N/A"
print(f"  Loss: 4.22 -> 4.07 ({loss_comp['loss_improvement_pct']}% better)")
print(f"  Vuln: {v1_mv} -> {v2_mv}")
with open(f"{OUT}/training_loss_comparison.json", "w") as f:
    json.dump(loss_comp, f, indent=2)

# ============================================================
# C6: Rank Stability Across Fine-tuning Levels
# ============================================================
print("\n=== C6: Rank Stability (V1 vs V2 rankings) ===")
v1_ranked = sorted(common_artists, key=lambda a: sf(v1_vuln[a].get("vulnerability_score")) or 0, reverse=True)
v2_ranked = sorted(common_artists, key=lambda a: sf(v2_vuln[a].get("vulnerability_score")) or 0, reverse=True)
v1_ranks = {a: i+1 for i, a in enumerate(v1_ranked)}
v2_ranks = {a: i+1 for i, a in enumerate(v2_ranked)}

rho, p = stats.spearmanr([v1_ranks[a] for a in common_artists], [v2_ranks[a] for a in common_artists])
rank_stability = {"spearman_rho": float(rho), "p": float(p), "n": len(common_artists)}
print(f"  Spearman rho: {rho:.3f}, p={p:.4f}")
print(f"  V1 top 5: {[v1_vuln[a].get('artist_name', a)[:20] for a in v1_ranked[:5]]}")
print(f"  V2 top 5: {[v2_vuln[a].get('artist_name', a)[:20] for a in v2_ranked[:5]]}")
with open(f"{OUT}/rank_stability.json", "w") as f:
    json.dump(rank_stability, f, indent=2)

# ============================================================
# C7: Effect Size Comparison
# ============================================================
print("\n=== C7: Effect Size Comparison ===")
v2_supp = "/scratch/$USER/runs/2026-03-10_full_v2/supplementary"
v1_effects, v2_effects = [], []
if os.path.exists(f"{v2_supp}/effect_sizes.csv"):
    for r in load_csv(f"{v2_supp}/effect_sizes.csv"):
        d = sf(r.get("cohens_d"))
        if d is not None: v2_effects.append(d)

for aid in sorted(set(v1_m.keys()) & set(v1_mm.keys())):
    m_arr = np.array(v1_m[aid])
    mm_arr = np.array(v1_mm[aid])
    if len(m_arr) >= 2 and len(mm_arr) >= 2:
        pooled = np.sqrt((np.var(m_arr, ddof=1) + np.var(mm_arr, ddof=1)) / 2)
        if pooled > 0:
            v1_effects.append((np.mean(m_arr) - np.mean(mm_arr)) / pooled)

effect_comp = {
    "v1_mean_d": float(np.mean(v1_effects)) if v1_effects else None,
    "v2_mean_d": float(np.mean(v2_effects)) if v2_effects else None,
    "v1_n": len(v1_effects), "v2_n": len(v2_effects),
}
if v1_effects and v2_effects:
    if len(v1_effects) == len(v2_effects):
        t, p = stats.ttest_rel(v2_effects, v1_effects)
        effect_comp["paired_t"] = float(t)
        effect_comp["p"] = float(p)
    print(f"  V1 mean d: {np.mean(v1_effects):.4f} (n={len(v1_effects)}), V2 mean d: {np.mean(v2_effects):.4f} (n={len(v2_effects)})")
with open(f"{OUT}/effect_size_comparison.json", "w") as f:
    json.dump(effect_comp, f, indent=2)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)

with open(f"{OUT}/full_comparison_summary.json", "w") as f:
    json.dump({
        "aggregate": agg,
        "clap_gap": clap_comp,
        "ngram": ngram_comp,
        "loss": loss_comp,
        "rank_stability": rank_stability,
        "effect_sizes": effect_comp,
        "per_artist_delta": {
            "n_gained": gained, "n_lost": lost,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
        },
    }, f, indent=2)

print(f"Files: {os.listdir(OUT)}")
