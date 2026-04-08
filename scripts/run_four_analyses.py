#!/usr/bin/env python3
"""
Four supplementary analyses for thesis:
1. Artist Replicability Score (baseline-only vulnerability)
2. Cross-signal Convergent Validity (correlate CLAP delta with n-gram delta)
3. Prompt Specificity Analysis (Tier A vs D vs B similarity)
4. Artist Distinctiveness Metric (genre/catalog-size correlations)
"""

import csv
import json
import os
import sys
from collections import defaultdict
from scipy import stats
import numpy as np

RESULTS = "/Users/erickpg/capstone/results"
FT_DIR = f"{RESULTS}/2026-03-10_full/analysis"
BL_DIR = f"{RESULTS}/2026-03-10_baseline/analysis"
COMP_DIR = f"{RESULTS}/comparison"
OUT_DIR = f"{RESULTS}/supplementary"
os.makedirs(OUT_DIR, exist_ok=True)


def load_csv(path):
    """Load CSV robustly, skipping duplicate header rows."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip duplicate headers from srun cat
            if any(v == k for k, v in row.items()):
                continue
            rows.append(row)
    return rows


def safe_float(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ============================================================
# Analysis 1: Artist Replicability Score
# How easily can ANY model replicate an artist from prompts alone?
# Uses baseline data exclusively.
# ============================================================
print("=" * 60)
print("ANALYSIS 1: Artist Replicability Score")
print("=" * 60)

bl_vuln = load_csv(f"{BL_DIR}/vulnerability_scores.csv")
bl_fad = load_csv(f"{BL_DIR}/per_artist_fad.csv")
bl_clap = load_csv(f"{BL_DIR}/clap_per_artist.csv")

# Build per-artist baseline metrics
bl_metrics = {}
for row in bl_vuln:
    aid = row.get("artist_id", "")
    if not aid or aid == "artist_id":
        continue
    bl_metrics[aid] = {
        "artist_name": row.get("artist_name", ""),
        "genre": row.get("genre", ""),
        "n_catalog": safe_float(row.get("n_catalog_tracks", 0), 0),
        "bl_clap_sim": safe_float(row.get("clap_similarity")),
        "bl_fad": safe_float(row.get("fad")),
        "bl_musico": safe_float(row.get("musicological_similarity")),
        "bl_vuln": safe_float(row.get("vulnerability_score")),
    }

# Supplement with CLAP per-artist matched/mismatched gap
for row in bl_clap:
    aid = row.get("artist_id", "")
    if aid in bl_metrics:
        bl_metrics[aid]["bl_clap_gap"] = safe_float(row.get("mean_sim_gap"))

# Supplement with FAD
for row in bl_fad:
    aid = row.get("artist_id", "")
    if aid in bl_metrics and row.get("comparison") == "matched":
        bl_metrics[aid]["bl_fad_matched"] = safe_float(row.get("fad"))

# Compute replicability score: high CLAP sim + low FAD + high musico = high replicability
# Normalize each component to [0,1] then average
clap_vals = [m["bl_clap_sim"] for m in bl_metrics.values() if m["bl_clap_sim"] is not None]
fad_vals = [m.get("bl_fad_matched") or m.get("bl_fad") for m in bl_metrics.values()
            if (m.get("bl_fad_matched") or m.get("bl_fad")) is not None]
musico_vals = [m["bl_musico"] for m in bl_metrics.values() if m["bl_musico"] is not None]

if clap_vals and fad_vals and musico_vals:
    clap_min, clap_max = min(clap_vals), max(clap_vals)
    fad_min, fad_max = min(fad_vals), max(fad_vals)
    musico_min, musico_max = min(musico_vals), max(musico_vals)

    for aid, m in bl_metrics.items():
        clap_norm = (m["bl_clap_sim"] - clap_min) / (clap_max - clap_min + 1e-10) if m["bl_clap_sim"] is not None else 0.5
        fad_raw = m.get("bl_fad_matched") or m.get("bl_fad")
        fad_norm = 1.0 - (fad_raw - fad_min) / (fad_max - fad_min + 1e-10) if fad_raw is not None else 0.5  # lower FAD = more similar
        musico_norm = (m["bl_musico"] - musico_min) / (musico_max - musico_min + 1e-10) if m["bl_musico"] is not None else 0.5

        m["replicability_score"] = 0.4 * clap_norm + 0.3 * fad_norm + 0.3 * musico_norm

    # Rank
    sorted_artists = sorted(bl_metrics.items(), key=lambda x: x[1].get("replicability_score", 0), reverse=True)

    print(f"\nTop 10 most replicable artists (baseline only — ANY model can replicate):")
    print(f"{'Rank':<5} {'Artist':<30} {'Genre':<12} {'Score':<8} {'CLAP':<8} {'FAD':<8} {'Musico':<8}")
    for i, (aid, m) in enumerate(sorted_artists[:10], 1):
        print(f"{i:<5} {m['artist_name'][:29]:<30} {m['genre'][:11]:<12} {m['replicability_score']:.4f}  {m['bl_clap_sim'] or 0:.4f}  {m.get('bl_fad', 0) or 0:.4f}  {m['bl_musico'] or 0:.4f}")

    print(f"\nBottom 5 least replicable artists:")
    for i, (aid, m) in enumerate(sorted_artists[-5:], len(sorted_artists)-4):
        print(f"{i:<5} {m['artist_name'][:29]:<30} {m['genre'][:11]:<12} {m['replicability_score']:.4f}  {m['bl_clap_sim'] or 0:.4f}  {m.get('bl_fad', 0) or 0:.4f}  {m['bl_musico'] or 0:.4f}")

    # Save
    with open(f"{OUT_DIR}/artist_replicability_scores.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["rank", "artist_id", "artist_name", "genre", "n_catalog",
                     "replicability_score", "bl_clap_sim", "bl_fad", "bl_musico", "bl_vuln_score"])
        for i, (aid, m) in enumerate(sorted_artists, 1):
            w.writerow([i, aid, m["artist_name"], m["genre"], int(m["n_catalog"]),
                        f"{m['replicability_score']:.6f}",
                        f"{m['bl_clap_sim']:.6f}" if m["bl_clap_sim"] else "",
                        f"{m.get('bl_fad', '')}" if m.get("bl_fad") else "",
                        f"{m['bl_musico']:.6f}" if m["bl_musico"] else "",
                        f"{m['bl_vuln']:.6f}" if m["bl_vuln"] else ""])

    # Commercial insight: genre-level replicability
    genre_scores = defaultdict(list)
    for aid, m in bl_metrics.items():
        g = m["genre"] or "Unknown"
        if m.get("replicability_score") is not None:
            genre_scores[g].append(m["replicability_score"])

    print(f"\nGenre-level replicability (baseline):")
    print(f"{'Genre':<15} {'Mean Score':<12} {'Std':<10} {'N':<5}")
    for g in sorted(genre_scores.keys(), key=lambda x: np.mean(genre_scores[x]), reverse=True):
        scores = genre_scores[g]
        print(f"{g:<15} {np.mean(scores):.4f}      {np.std(scores):.4f}    {len(scores)}")

    print("\n[SAVED] artist_replicability_scores.csv")
else:
    print("ERROR: Missing data for replicability score computation")

# ============================================================
# Analysis 2: Cross-signal Convergent Validity
# Do CLAP delta and n-gram delta agree per artist?
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 2: Cross-signal Convergent Validity")
print("=" * 60)

clap_paired = load_csv(f"{COMP_DIR}/clap_paired_per_artist.csv")
ngram_paired = load_csv(f"{COMP_DIR}/ngram_paired_per_artist.csv")

# Build per-artist deltas
clap_deltas = {}
for row in clap_paired:
    aid = row.get("artist_id", "")
    if aid:
        clap_deltas[aid] = {
            "delta_gap": safe_float(row.get("delta_gap")),
            "delta_matched": safe_float(row.get("delta_matched_mean")),
            "genre": row.get("genre", ""),
        }

# N-gram deltas (use ngram_size=4 as representative)
ngram_deltas = {}
for row in ngram_paired:
    aid = row.get("artist_id", "")
    ns = row.get("ngram_size", "")
    if aid and ns == "4":
        ngram_deltas[aid] = {
            "delta_matched_rate": safe_float(row.get("delta_matched_rate")),
            "delta_ratio": safe_float(row.get("delta_ratio")),
        }

# Also load FT and BL vulnerability scores for delta
ft_vuln = load_csv(f"{FT_DIR}/vulnerability_scores.csv")
ft_vuln_dict = {}
for row in ft_vuln:
    aid = row.get("artist_id", "")
    if aid and aid != "artist_id":
        ft_vuln_dict[aid] = safe_float(row.get("vulnerability_score"))

bl_vuln_dict = {}
for row in bl_vuln:
    aid = row.get("artist_id", "")
    if aid and aid != "artist_id":
        bl_vuln_dict[aid] = safe_float(row.get("vulnerability_score"))

# Pair: CLAP delta vs n-gram delta
common_artists = set(clap_deltas.keys()) & set(ngram_deltas.keys())
print(f"\nArtists with both CLAP and n-gram deltas: {len(common_artists)}")

if len(common_artists) >= 5:
    clap_d = []
    ngram_d = []
    vuln_delta = []
    labels = []

    for aid in sorted(common_artists):
        cd = clap_deltas[aid]["delta_matched"]
        nd = ngram_deltas[aid]["delta_matched_rate"]
        if cd is not None and nd is not None:
            clap_d.append(cd)
            ngram_d.append(nd)
            labels.append(aid)

            ft_v = ft_vuln_dict.get(aid)
            bl_v = bl_vuln_dict.get(aid)
            if ft_v is not None and bl_v is not None:
                vuln_delta.append(ft_v - bl_v)

    # Spearman correlation: CLAP delta vs n-gram delta
    r_cn, p_cn = stats.spearmanr(clap_d, ngram_d)
    print(f"\nSpearman correlation (CLAP delta vs N-gram delta):")
    print(f"  rho = {r_cn:.4f}, p = {p_cn:.4f} {'***' if p_cn < 0.001 else '**' if p_cn < 0.01 else '*' if p_cn < 0.05 else 'ns'}")

    # Pearson too
    r_cn_p, p_cn_p = stats.pearsonr(clap_d, ngram_d)
    print(f"  Pearson r = {r_cn_p:.4f}, p = {p_cn_p:.4f}")

    # CLAP delta vs vulnerability delta
    if len(vuln_delta) >= 5:
        r_cv, p_cv = stats.spearmanr(clap_d[:len(vuln_delta)], vuln_delta)
        print(f"\nSpearman (CLAP delta vs Vulnerability delta):")
        print(f"  rho = {r_cv:.4f}, p = {p_cv:.4f} {'***' if p_cv < 0.001 else '**' if p_cv < 0.01 else '*' if p_cv < 0.05 else 'ns'}")

    # Concordance: do they agree on direction?
    agree = sum(1 for c, n in zip(clap_d, ngram_d) if (c > 0) == (n > 0))
    print(f"\nDirection concordance: {agree}/{len(clap_d)} artists ({100*agree/len(clap_d):.1f}%) agree on FT > BL direction")

    # Save
    with open(f"{OUT_DIR}/cross_signal_validity.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["artist_id", "genre", "clap_delta_matched", "ngram_delta_matched_rate", "vuln_delta"])
        for i, aid in enumerate(sorted(common_artists)):
            cd = clap_deltas[aid]["delta_matched"]
            nd = ngram_deltas[aid]["delta_matched_rate"]
            vd = vuln_delta[i] if i < len(vuln_delta) else ""
            w.writerow([aid, clap_deltas[aid].get("genre", ""),
                        f"{cd:.6f}" if cd else "", f"{nd:.10f}" if nd else "",
                        f"{vd:.6f}" if isinstance(vd, float) else ""])

    summary = {
        "n_artists": len(common_artists),
        "spearman_clap_ngram": {"rho": round(r_cn, 4), "p": round(p_cn, 4)},
        "pearson_clap_ngram": {"r": round(r_cn_p, 4), "p": round(p_cn_p, 4)},
        "direction_concordance": f"{agree}/{len(clap_d)} ({100*agree/len(clap_d):.1f}%)",
        "interpretation": "positive rho = signals agree" if r_cn > 0 else "negative rho = signals disagree"
    }
    if len(vuln_delta) >= 5:
        summary["spearman_clap_vuln"] = {"rho": round(r_cv, 4), "p": round(p_cv, 4)}

    with open(f"{OUT_DIR}/cross_signal_validity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[SAVED] cross_signal_validity.csv, cross_signal_validity_summary.json")

# ============================================================
# Analysis 3: Prompt Specificity Analysis
# Tier A (artist-proximal) vs Tier D (FMA sub-genre tags) vs Tier B (genre-generic)
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 3: Prompt Specificity Analysis")
print("=" * 60)

# Load per-file CLAP data for both FT and BL
ft_clap_files = load_csv(f"{FT_DIR}/clap_similarity.csv")
bl_clap_files = load_csv(f"{BL_DIR}/clap_similarity.csv")

# Separate by tier
def tier_stats(rows, label):
    tiers = defaultdict(list)
    for row in rows:
        tier = row.get("tier", "")
        gap = safe_float(row.get("sim_gap"))
        matched = safe_float(row.get("matched_mean_sim"))
        if tier and gap is not None:
            tiers[tier].append({"gap": gap, "matched": matched})

    print(f"\n{label}:")
    print(f"{'Tier':<25} {'N':<6} {'Mean Gap':<12} {'Std Gap':<12} {'Mean Matched':<14}")
    for t in sorted(tiers.keys()):
        vals = tiers[t]
        gaps = [v["gap"] for v in vals if v["gap"] is not None]
        matched = [v["matched"] for v in vals if v["matched"] is not None]
        if gaps:
            print(f"{t:<25} {len(gaps):<6} {np.mean(gaps):.6f}    {np.std(gaps):.6f}    {np.mean(matched):.6f}")
    return tiers

ft_tiers = tier_stats(ft_clap_files, "Fine-tuned model")
bl_tiers = tier_stats(bl_clap_files, "Baseline model")

# Statistical tests between tiers
print("\n--- Statistical Tests (Fine-tuned) ---")
tier_names = sorted(ft_tiers.keys())
for i in range(len(tier_names)):
    for j in range(i+1, len(tier_names)):
        t1, t2 = tier_names[i], tier_names[j]
        g1 = [v["gap"] for v in ft_tiers[t1] if v["gap"] is not None]
        g2 = [v["gap"] for v in ft_tiers[t2] if v["gap"] is not None]
        if len(g1) >= 3 and len(g2) >= 3:
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
            d = (np.mean(g1) - np.mean(g2)) / np.sqrt((np.std(g1)**2 + np.std(g2)**2) / 2)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f"  {t1} vs {t2}: t={t_stat:.3f}, p={p_val:.4f} {sig}, d={d:.3f}")

# Key comparison: does FT change the tier pattern vs baseline?
print("\n--- FT vs BL per tier (matched similarity) ---")
tier_comparison = {}
for tier in sorted(set(list(ft_tiers.keys()) + list(bl_tiers.keys()))):
    ft_matched = [v["matched"] for v in ft_tiers.get(tier, []) if v["matched"] is not None]
    bl_matched = [v["matched"] for v in bl_tiers.get(tier, []) if v["matched"] is not None]
    if ft_matched and bl_matched:
        t_stat, p_val = stats.ttest_ind(ft_matched, bl_matched, equal_var=False)
        d = (np.mean(ft_matched) - np.mean(bl_matched)) / np.sqrt((np.std(ft_matched)**2 + np.std(bl_matched)**2) / 2)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"  {tier}: FT={np.mean(ft_matched):.4f}, BL={np.mean(bl_matched):.4f}, delta={np.mean(ft_matched)-np.mean(bl_matched):.4f}, t={t_stat:.3f}, p={p_val:.4f} {sig}, d={d:.3f}")
        tier_comparison[tier] = {
            "ft_mean_matched": round(np.mean(ft_matched), 6),
            "bl_mean_matched": round(np.mean(bl_matched), 6),
            "delta": round(np.mean(ft_matched) - np.mean(bl_matched), 6),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_val, 4),
            "cohens_d": round(d, 4),
            "n_ft": len(ft_matched),
            "n_bl": len(bl_matched),
        }

with open(f"{OUT_DIR}/prompt_specificity.json", "w") as f:
    json.dump(tier_comparison, f, indent=2)

# Save detailed tier stats
with open(f"{OUT_DIR}/prompt_specificity_tiers.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["model", "tier", "n", "mean_gap", "std_gap", "mean_matched", "std_matched"])
    for model, tiers in [("fine_tuned", ft_tiers), ("baseline", bl_tiers)]:
        for t in sorted(tiers.keys()):
            vals = tiers[t]
            gaps = [v["gap"] for v in vals if v["gap"] is not None]
            matched = [v["matched"] for v in vals if v["matched"] is not None]
            if gaps:
                w.writerow([model, t, len(gaps), f"{np.mean(gaps):.6f}", f"{np.std(gaps):.6f}",
                           f"{np.mean(matched):.6f}" if matched else "",
                           f"{np.std(matched):.6f}" if matched else ""])

print("\n[SAVED] prompt_specificity.json, prompt_specificity_tiers.csv")

# ============================================================
# Analysis 4: Artist Distinctiveness Metric
# Which artists are hardest/easiest to distinguish? Correlate with genre, catalog size.
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 4: Artist Distinctiveness Metric")
print("=" * 60)

# Compute: FT absorption = FT_vuln - BL_vuln (positive = FT made more vulnerable)
# Correlate with catalog size, genre
paired_data = []
for aid in bl_metrics:
    ft_v = ft_vuln_dict.get(aid)
    bl_v = bl_vuln_dict.get(aid)
    m = bl_metrics[aid]
    if ft_v is not None and bl_v is not None:
        paired_data.append({
            "artist_id": aid,
            "artist_name": m["artist_name"],
            "genre": m["genre"] or "Unknown",
            "n_catalog": m["n_catalog"],
            "ft_vuln": ft_v,
            "bl_vuln": bl_v,
            "absorption": ft_v - bl_v,  # positive = FT increased vulnerability
            "bl_replicability": m.get("replicability_score", 0),
        })

if paired_data:
    # Sort by absorption
    paired_data.sort(key=lambda x: x["absorption"], reverse=True)

    print(f"\nArtists with highest FT absorption (FT vulnerability - BL vulnerability):")
    print(f"{'Artist':<30} {'Genre':<12} {'N':<4} {'FT Vuln':<9} {'BL Vuln':<9} {'Absorb':<9} {'BL Replic':<9}")
    for d in paired_data[:10]:
        print(f"{d['artist_name'][:29]:<30} {d['genre'][:11]:<12} {int(d['n_catalog']):<4} {d['ft_vuln']:.4f}   {d['bl_vuln']:.4f}   {d['absorption']:+.4f}  {d['bl_replicability']:.4f}")

    print(f"\nArtists with lowest FT absorption (FT made LESS vulnerable):")
    for d in paired_data[-5:]:
        print(f"{d['artist_name'][:29]:<30} {d['genre'][:11]:<12} {int(d['n_catalog']):<4} {d['ft_vuln']:.4f}   {d['bl_vuln']:.4f}   {d['absorption']:+.4f}  {d['bl_replicability']:.4f}")

    # Correlations
    absorptions = [d["absorption"] for d in paired_data]
    catalog_sizes = [d["n_catalog"] for d in paired_data]
    bl_replic = [d["bl_replicability"] for d in paired_data]

    r_abs_cat, p_abs_cat = stats.spearmanr(absorptions, catalog_sizes)
    r_abs_rep, p_abs_rep = stats.spearmanr(absorptions, bl_replic)

    print(f"\nCorrelations:")
    print(f"  Absorption vs Catalog size: rho={r_abs_cat:.4f}, p={p_abs_cat:.4f} {'*' if p_abs_cat < 0.05 else 'ns'}")
    print(f"  Absorption vs BL Replicability: rho={r_abs_rep:.4f}, p={p_abs_rep:.4f} {'*' if p_abs_rep < 0.05 else 'ns'}")

    # Genre-level absorption
    genre_absorb = defaultdict(list)
    for d in paired_data:
        genre_absorb[d["genre"]].append(d["absorption"])

    print(f"\nGenre-level FT absorption:")
    print(f"{'Genre':<15} {'Mean Absorb':<14} {'Std':<10} {'N':<5} {'1-sample t':<12} {'p':<8}")
    for g in sorted(genre_absorb.keys(), key=lambda x: np.mean(genre_absorb[x]), reverse=True):
        vals = genre_absorb[g]
        if len(vals) >= 2:
            t_stat, p_val = stats.ttest_1samp(vals, 0)
            sig = '*' if p_val < 0.05 else 'ns'
            print(f"{g:<15} {np.mean(vals):+.4f}       {np.std(vals):.4f}    {len(vals):<5} {t_stat:.3f}       {p_val:.4f} {sig}")
        else:
            print(f"{g:<15} {np.mean(vals):+.4f}       --        {len(vals):<5} --")

    # Overall: is mean absorption significantly different from 0?
    t_overall, p_overall = stats.ttest_1samp(absorptions, 0)
    print(f"\nOverall mean absorption: {np.mean(absorptions):+.4f} (t={t_overall:.3f}, p={p_overall:.4f})")

    # Save
    with open(f"{OUT_DIR}/artist_distinctiveness.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["rank", "artist_id", "artist_name", "genre", "n_catalog",
                     "ft_vulnerability", "bl_vulnerability", "absorption", "bl_replicability"])
        for i, d in enumerate(paired_data, 1):
            w.writerow([i, d["artist_id"], d["artist_name"], d["genre"], int(d["n_catalog"]),
                        f"{d['ft_vuln']:.6f}", f"{d['bl_vuln']:.6f}",
                        f"{d['absorption']:.6f}", f"{d['bl_replicability']:.6f}"])

    summary4 = {
        "n_artists": len(paired_data),
        "mean_absorption": round(np.mean(absorptions), 4),
        "std_absorption": round(np.std(absorptions), 4),
        "t_test_vs_zero": {"t": round(t_overall, 4), "p": round(p_overall, 4)},
        "correlation_absorption_catalog_size": {"rho": round(r_abs_cat, 4), "p": round(p_abs_cat, 4)},
        "correlation_absorption_bl_replicability": {"rho": round(r_abs_rep, 4), "p": round(p_abs_rep, 4)},
        "n_positive_absorption": sum(1 for a in absorptions if a > 0),
        "n_negative_absorption": sum(1 for a in absorptions if a < 0),
        "interpretation": "positive absorption = fine-tuning increased vulnerability relative to baseline"
    }

    with open(f"{OUT_DIR}/artist_distinctiveness_summary.json", "w") as f:
        json.dump(summary4, f, indent=2)

    print("\n[SAVED] artist_distinctiveness.csv, artist_distinctiveness_summary.json")

print("\n" + "=" * 60)
print("ALL 4 ANALYSES COMPLETE")
print(f"Results saved to: {OUT_DIR}/")
print("=" * 60)
