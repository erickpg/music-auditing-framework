#!/usr/bin/env python3
"""
Paired delta test: fine-tuned gap minus baseline gap per artist.

For each artist, computes:
  delta_CLAP = mean_sim_gap(fine-tuned) - mean_sim_gap(baseline)
  delta_FAD  = FAD(fine-tuned) - FAD(baseline)  [inverted: negative = closer to catalog]

Tests whether fine-tuning produces model-specific absorption beyond
what the pretrained model already exhibits.

Runs for both V1 and V2 fine-tuned models against the shared baseline.
Reports results for all 50 artists and the 12 "always high" subgroup.
"""

import csv
import json
import os
import sys
import numpy as np
from scipy import stats

RESULTS = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "results"))
BL_DIR = os.path.join(RESULTS, "baseline", "analysis")
V1_DIR = os.path.join(RESULTS, "v1", "analysis")
V2_DIR = os.path.join(RESULTS, "v2", "analysis")
OUT_DIR = os.path.join(RESULTS, "paired_delta_test")
os.makedirs(OUT_DIR, exist_ok=True)

# The 12 artists that remained "Always High" (vuln > 0.67) across both V1 and V2
ALWAYS_HIGH_NAMES = {
    "Mesmerists", "Comfort Fit", "David Szesztay", "Plusplus",
    "Montana Skies", "Ga'an", "Ben von Wildenhaus", "Fhernando",
    "Sleep Out", "Wann", "Lamprey", "Movie Theater",
}


def load_clap_per_file(path):
    """Load per-file CLAP rows, return dict: artist_id -> list of sim_gap floats."""
    artist_gaps = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            aid = row.get("artist_id", "")
            tier = row.get("tier", "")
            gap = row.get("sim_gap", "")
            if not aid or aid == "artist_id" or tier not in ("A_artist_proximal", "D_fma_tags"):
                continue
            try:
                artist_gaps.setdefault(aid, []).append(float(gap))
            except ValueError:
                continue
    return artist_gaps


def load_fad_matched(path):
    """Load per-artist FAD (matched comparison only)."""
    artist_fad = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            aid = row.get("artist_id", "")
            comp = row.get("comparison", "")
            if not aid or aid == "artist_id" or comp != "matched":
                continue
            try:
                artist_fad[aid] = {
                    "fad": float(row["fad"]),
                    "name": row.get("artist_name", ""),
                }
            except (ValueError, KeyError):
                continue
    return artist_fad


def aggregate_clap_gaps(per_file_gaps):
    """Aggregate per-file sim_gap to per-artist mean sim_gap."""
    return {aid: np.mean(gaps) for aid, gaps in per_file_gaps.items()}


def paired_test(ft_vals, bl_vals, label=""):
    """Run paired t-test and Wilcoxon signed-rank test on matched arrays."""
    delta = ft_vals - bl_vals
    n = len(delta)
    mean_delta = np.mean(delta)
    std_delta = np.std(delta, ddof=1)

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(ft_vals, bl_vals)

    # Wilcoxon signed-rank (non-parametric)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(delta)
    except ValueError:
        # All zeros
        w_stat, p_wilcoxon = 0.0, 1.0

    # Cohen's d_z (effect size for paired design)
    d_z = mean_delta / std_delta if std_delta > 0 else 0.0

    # Sign test: how many positive vs negative deltas
    n_pos = int(np.sum(delta > 0))
    n_neg = int(np.sum(delta < 0))
    n_zero = int(np.sum(delta == 0))

    # 95% CI for mean delta
    se = std_delta / np.sqrt(n)
    ci_lo = mean_delta - 1.96 * se
    ci_hi = mean_delta + 1.96 * se

    return {
        "n": n,
        "ft_mean": float(np.mean(ft_vals)),
        "bl_mean": float(np.mean(bl_vals)),
        "mean_delta": float(mean_delta),
        "std_delta": float(std_delta),
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "t_stat": float(t_stat),
        "p_ttest": float(p_ttest),
        "w_stat": float(w_stat),
        "p_wilcoxon": float(p_wilcoxon),
        "cohens_dz": float(d_z),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n_zero,
    }


def run_analysis(ft_label, ft_dir):
    """Run paired delta analysis for one fine-tuned version vs baseline."""
    print(f"\n{'='*70}")
    print(f"  PAIRED DELTA TEST: {ft_label} vs Baseline")
    print(f"{'='*70}")

    # --- Load data ---
    bl_clap_raw = load_clap_per_file(os.path.join(BL_DIR, "clap_per_artist.csv"))
    ft_clap_raw = load_clap_per_file(os.path.join(ft_dir, "clap_per_artist.csv"))
    bl_clap = aggregate_clap_gaps(bl_clap_raw)
    ft_clap = aggregate_clap_gaps(ft_clap_raw)

    bl_fad_data = load_fad_matched(os.path.join(BL_DIR, "per_artist_fad.csv"))
    ft_fad_data = load_fad_matched(os.path.join(ft_dir, "per_artist_fad.csv"))

    # Build name lookup from FAD data (has artist_name)
    name_lookup = {}
    for aid, d in {**bl_fad_data, **ft_fad_data}.items():
        name_lookup[aid] = d["name"]

    # --- Common artists ---
    common_clap = sorted(set(ft_clap.keys()) & set(bl_clap.keys()))
    common_fad = sorted(set(ft_fad_data.keys()) & set(bl_fad_data.keys()))

    print(f"\nArtists with CLAP data: {len(common_clap)}")
    print(f"Artists with FAD data:  {len(common_fad)}")

    # Identify always-high subset by name
    always_high_ids = set()
    for aid in set(common_clap) | set(common_fad):
        name = name_lookup.get(aid, "")
        if name in ALWAYS_HIGH_NAMES:
            always_high_ids.add(aid)
    print(f"Always-high artists matched: {len(always_high_ids)}/{len(ALWAYS_HIGH_NAMES)}")

    results = {}

    # --- CLAP sim_gap delta ---
    for group_label, artist_ids in [("all_artists", common_clap),
                                      ("always_high_12", sorted(always_high_ids & set(common_clap)))]:
        if len(artist_ids) < 3:
            print(f"\n  CLAP gap ({group_label}): insufficient data ({len(artist_ids)})")
            continue

        ft_vals = np.array([ft_clap[a] for a in artist_ids])
        bl_vals = np.array([bl_clap[a] for a in artist_ids])
        res = paired_test(ft_vals, bl_vals)
        key = f"clap_gap_{group_label}"
        results[key] = res

        print(f"\n  CLAP Sim Gap ({group_label}, n={res['n']}):")
        print(f"    {ft_label} mean gap:  {res['ft_mean']:.6f}")
        print(f"    Baseline mean gap: {res['bl_mean']:.6f}")
        print(f"    Mean delta (FT-BL): {res['mean_delta']:+.6f}  95% CI [{res['ci_95_lo']:+.6f}, {res['ci_95_hi']:+.6f}]")
        print(f"    Paired t-test:  t={res['t_stat']:.4f}, p={res['p_ttest']:.6f}")
        print(f"    Wilcoxon:       W={res['w_stat']:.1f}, p={res['p_wilcoxon']:.6f}")
        print(f"    Cohen's d_z:    {res['cohens_dz']:.4f}")
        print(f"    Sign: {res['n_positive']} positive, {res['n_negative']} negative")
        sig = "***" if res["p_ttest"] < 0.001 else "**" if res["p_ttest"] < 0.01 else "*" if res["p_ttest"] < 0.05 else "ns"
        print(f"    Interpretation: {sig} {'Fine-tuning INCREASED gap (model-specific absorption)' if res['mean_delta'] > 0 else 'Fine-tuning DECREASED gap (no model-specific absorption)' if res['mean_delta'] < 0 else 'No change'}")

    # --- FAD delta ---
    for group_label, artist_ids in [("all_artists", common_fad),
                                      ("always_high_12", sorted(always_high_ids & set(common_fad)))]:
        if len(artist_ids) < 3:
            print(f"\n  FAD ({group_label}): insufficient data ({len(artist_ids)})")
            continue

        ft_vals = np.array([ft_fad_data[a]["fad"] for a in artist_ids])
        bl_vals = np.array([bl_fad_data[a]["fad"] for a in artist_ids])
        res = paired_test(ft_vals, bl_vals)
        key = f"fad_{group_label}"
        results[key] = res

        print(f"\n  FAD ({group_label}, n={res['n']}):")
        print(f"    {ft_label} mean FAD:  {res['ft_mean']:.6f}")
        print(f"    Baseline mean FAD: {res['bl_mean']:.6f}")
        print(f"    Mean delta (FT-BL): {res['mean_delta']:+.6f}  95% CI [{res['ci_95_lo']:+.6f}, {res['ci_95_hi']:+.6f}]")
        print(f"    Paired t-test:  t={res['t_stat']:.4f}, p={res['p_ttest']:.6f}")
        print(f"    Wilcoxon:       W={res['w_stat']:.1f}, p={res['p_wilcoxon']:.6f}")
        print(f"    Cohen's d_z:    {res['cohens_dz']:.4f}")
        print(f"    Sign: {res['n_positive']} positive, {res['n_negative']} negative")
        sig = "***" if res["p_ttest"] < 0.001 else "**" if res["p_ttest"] < 0.01 else "*" if res["p_ttest"] < 0.05 else "ns"
        # For FAD: lower = more similar to catalog. Negative delta = FT closer to catalog.
        print(f"    Interpretation: {sig} {'Fine-tuning INCREASED distance (less absorption)' if res['mean_delta'] > 0 else 'Fine-tuning DECREASED distance (model-specific absorption)' if res['mean_delta'] < 0 else 'No change'}")

    # --- Per-artist delta table ---
    per_artist = []
    for aid in sorted(set(common_clap) & set(common_fad)):
        name = name_lookup.get(aid, "")
        clap_delta = ft_clap.get(aid, 0) - bl_clap.get(aid, 0)
        fad_delta = ft_fad_data[aid]["fad"] - bl_fad_data[aid]["fad"] if aid in ft_fad_data and aid in bl_fad_data else None
        is_high = name in ALWAYS_HIGH_NAMES
        per_artist.append({
            "artist_id": aid,
            "artist_name": name,
            "always_high": is_high,
            "ft_clap_gap": ft_clap.get(aid, None),
            "bl_clap_gap": bl_clap.get(aid, None),
            "clap_delta": clap_delta,
            "ft_fad": ft_fad_data[aid]["fad"] if aid in ft_fad_data else None,
            "bl_fad": bl_fad_data[aid]["fad"] if aid in bl_fad_data else None,
            "fad_delta": fad_delta,
        })

    # Sort by CLAP delta descending (most absorption first)
    per_artist.sort(key=lambda x: x["clap_delta"] or 0, reverse=True)

    # Print top/bottom 10
    print(f"\n  --- Per-Artist CLAP Gap Delta (top 10 absorption) ---")
    print(f"  {'Artist':<30s} {'FT gap':>8s} {'BL gap':>8s} {'Delta':>8s} {'High?':>5s}")
    for d in per_artist[:10]:
        flag = "YES" if d["always_high"] else ""
        print(f"  {d['artist_name']:<30s} {d['ft_clap_gap']:8.4f} {d['bl_clap_gap']:8.4f} {d['clap_delta']:+8.4f} {flag:>5s}")

    print(f"\n  --- Per-Artist CLAP Gap Delta (bottom 10) ---")
    print(f"  {'Artist':<30s} {'FT gap':>8s} {'BL gap':>8s} {'Delta':>8s} {'High?':>5s}")
    for d in per_artist[-10:]:
        flag = "YES" if d["always_high"] else ""
        print(f"  {d['artist_name']:<30s} {d['ft_clap_gap']:8.4f} {d['bl_clap_gap']:8.4f} {d['clap_delta']:+8.4f} {flag:>5s}")

    # --- Subgroup comparison: always-high vs rest ---
    high_deltas = [d["clap_delta"] for d in per_artist if d["always_high"] and d["clap_delta"] is not None]
    rest_deltas = [d["clap_delta"] for d in per_artist if not d["always_high"] and d["clap_delta"] is not None]
    if len(high_deltas) >= 3 and len(rest_deltas) >= 3:
        t_sub, p_sub = stats.ttest_ind(high_deltas, rest_deltas, equal_var=False)
        mwu, p_mwu = stats.mannwhitneyu(high_deltas, rest_deltas, alternative="two-sided")
        print(f"\n  --- Subgroup Comparison: Always-High vs Rest (CLAP delta) ---")
        print(f"    Always-high (n={len(high_deltas)}): mean delta = {np.mean(high_deltas):+.6f}")
        print(f"    Rest (n={len(rest_deltas)}):        mean delta = {np.mean(rest_deltas):+.6f}")
        print(f"    Welch's t: t={t_sub:.4f}, p={p_sub:.6f}")
        print(f"    Mann-Whitney U: U={mwu:.1f}, p={p_mwu:.6f}")
        results["subgroup_clap_delta"] = {
            "high_mean": float(np.mean(high_deltas)),
            "rest_mean": float(np.mean(rest_deltas)),
            "welch_t": float(t_sub),
            "p_welch": float(p_sub),
            "mwu": float(mwu),
            "p_mwu": float(p_mwu),
        }

    # Same for FAD
    high_fad_deltas = [d["fad_delta"] for d in per_artist if d["always_high"] and d["fad_delta"] is not None]
    rest_fad_deltas = [d["fad_delta"] for d in per_artist if not d["always_high"] and d["fad_delta"] is not None]
    if len(high_fad_deltas) >= 3 and len(rest_fad_deltas) >= 3:
        t_sub_f, p_sub_f = stats.ttest_ind(high_fad_deltas, rest_fad_deltas, equal_var=False)
        mwu_f, p_mwu_f = stats.mannwhitneyu(high_fad_deltas, rest_fad_deltas, alternative="two-sided")
        print(f"\n  --- Subgroup Comparison: Always-High vs Rest (FAD delta) ---")
        print(f"    Always-high (n={len(high_fad_deltas)}): mean delta = {np.mean(high_fad_deltas):+.6f}")
        print(f"    Rest (n={len(rest_fad_deltas)}):        mean delta = {np.mean(rest_fad_deltas):+.6f}")
        print(f"    Welch's t: t={t_sub_f:.4f}, p={p_sub_f:.6f}")
        print(f"    Mann-Whitney U: U={mwu_f:.1f}, p={p_mwu_f:.6f}")
        results["subgroup_fad_delta"] = {
            "high_mean": float(np.mean(high_fad_deltas)),
            "rest_mean": float(np.mean(rest_fad_deltas)),
            "welch_t": float(t_sub_f),
            "p_welch": float(p_sub_f),
            "mwu": float(mwu_f),
            "p_mwu": float(p_mwu_f),
        }

    return results, per_artist


def main():
    all_results = {}

    # V1 vs Baseline
    v1_results, v1_per_artist = run_analysis("V1", V1_DIR)
    all_results["v1_vs_baseline"] = v1_results

    # V2 vs Baseline
    v2_results, v2_per_artist = run_analysis("V2", V2_DIR)
    all_results["v2_vs_baseline"] = v2_results

    # --- Cross-version consistency of delta ---
    print(f"\n{'='*70}")
    print(f"  CROSS-VERSION DELTA CONSISTENCY")
    print(f"{'='*70}")

    # Match artists across v1 and v2 per-artist tables
    v1_by_id = {d["artist_id"]: d for d in v1_per_artist}
    v2_by_id = {d["artist_id"]: d for d in v2_per_artist}
    common = sorted(set(v1_by_id.keys()) & set(v2_by_id.keys()))

    v1_clap_deltas = np.array([v1_by_id[a]["clap_delta"] for a in common if v1_by_id[a]["clap_delta"] is not None])
    v2_clap_deltas = np.array([v2_by_id[a]["clap_delta"] for a in common if v2_by_id[a]["clap_delta"] is not None])

    if len(v1_clap_deltas) == len(v2_clap_deltas) and len(v1_clap_deltas) > 3:
        rho, p_rho = stats.spearmanr(v1_clap_deltas, v2_clap_deltas)
        r, p_r = stats.pearsonr(v1_clap_deltas, v2_clap_deltas)
        print(f"\n  CLAP gap delta correlation (V1 delta vs V2 delta across artists):")
        print(f"    Spearman rho: {rho:.4f}, p={p_rho:.6f}")
        print(f"    Pearson r:    {r:.4f}, p={p_r:.6f}")
        print(f"    Interpretation: {'Strong' if abs(rho) > 0.6 else 'Moderate' if abs(rho) > 0.3 else 'Weak'} consistency — "
              f"{'artists with high absorption in V1 also show it in V2' if rho > 0.3 else 'absorption patterns differ across training levels'}")

        all_results["cross_version"] = {
            "clap_delta_spearman_rho": float(rho),
            "clap_delta_spearman_p": float(p_rho),
            "clap_delta_pearson_r": float(r),
            "clap_delta_pearson_p": float(p_r),
        }

    v1_fad_deltas = np.array([v1_by_id[a]["fad_delta"] for a in common if v1_by_id[a]["fad_delta"] is not None])
    v2_fad_deltas = np.array([v2_by_id[a]["fad_delta"] for a in common if v2_by_id[a]["fad_delta"] is not None])

    if len(v1_fad_deltas) == len(v2_fad_deltas) and len(v1_fad_deltas) > 3:
        rho_f, p_rho_f = stats.spearmanr(v1_fad_deltas, v2_fad_deltas)
        r_f, p_r_f = stats.pearsonr(v1_fad_deltas, v2_fad_deltas)
        print(f"\n  FAD delta correlation (V1 delta vs V2 delta across artists):")
        print(f"    Spearman rho: {rho_f:.4f}, p={p_rho_f:.6f}")
        print(f"    Pearson r:    {r_f:.4f}, p={p_r_f:.6f}")

        all_results["cross_version"]["fad_delta_spearman_rho"] = float(rho_f)
        all_results["cross_version"]["fad_delta_spearman_p"] = float(p_rho_f)

    # --- Save all results ---
    out_json = os.path.join(OUT_DIR, "paired_delta_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {out_json}")

    # Save per-artist CSVs
    for label, per_artist in [("v1", v1_per_artist), ("v2", v2_per_artist)]:
        out_csv = os.path.join(OUT_DIR, f"{label}_per_artist_delta.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["artist_id", "artist_name", "always_high",
                         f"{label}_clap_gap", "bl_clap_gap", "clap_delta",
                         f"{label}_fad", "bl_fad", "fad_delta"])
            for d in per_artist:
                w.writerow([
                    d["artist_id"], d["artist_name"], d["always_high"],
                    f"{d['ft_clap_gap']:.6f}" if d["ft_clap_gap"] is not None else "",
                    f"{d['bl_clap_gap']:.6f}" if d["bl_clap_gap"] is not None else "",
                    f"{d['clap_delta']:.6f}" if d["clap_delta"] is not None else "",
                    f"{d['ft_fad']:.6f}" if d["ft_fad"] is not None else "",
                    f"{d['bl_fad']:.6f}" if d["bl_fad"] is not None else "",
                    f"{d['fad_delta']:.6f}" if d["fad_delta"] is not None else "",
                ])
        print(f"[SAVED] {out_csv}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
