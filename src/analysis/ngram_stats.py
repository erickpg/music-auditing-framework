#!/usr/bin/env python3
"""Stage A2: Statistical analysis of n-gram match rates.

Performs multiple hypothesis tests using the tier-based experimental design:

  T1: Tier A+D (artist-linked) vs Tier C (OOD control)
      → Tests whether artist-linked prompts trigger more catalog memorization
  T2: Tier B (genre-generic) vs Tier C (OOD control)
      → Tests whether even generic genre prompts leak catalog content
  T3: Tier A vs Tier B (artist-specificity)
      → Tests whether artist-specific prompts produce more memorization than genre-level
  T3b: Tier A vs Tier D (prompt type comparison)
  T4: Per-artist matched vs mismatched (from ngram_per_artist.csv)
      → Paired test: same files, matched vs same-genre mismatched catalog
  T5: Per-genre comparison
  T6: Training exposure correlation (Spearman)

Statistical methods:
  - Welch's t-test (parametric) for unpaired comparisons (T1-T3, T5)
  - Paired t-test for matched vs mismatched (T4)
  - Mann-Whitney U (non-parametric unpaired) for T1-T3, T5
  - Wilcoxon signed-rank (non-parametric paired) for T4
  - Cohen's d (unpaired) for T1-T3, T5; Cohen's d_z (paired) for T4
  - Permutation test for T4 (exact p-value, distribution-free)
  - BH FDR correction applied WITHIN test families:
      Family 1: T1-T4 (primary hypotheses)
      Family 2: T5 (per-genre, exploratory)
      Family 3: T6 (correlation, separate)
  - Pseudoreplication correction: averages across seeds per prompt before testing

Outputs:
    <run_dir>/analysis/ngram_statistical_tests.csv
    <run_dir>/analysis/ngram_null_distribution.json
    <run_dir>/analysis/memorization_verdict.json
    <run_dir>/analysis/ngram_per_genre_stats.csv
    <run_dir>/logs/ngram_stats.log
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    base_argparser, ensure_dirs, load_config, log_finish,
    log_preamble, save_run_metadata, setup_logging,
)

STAGE = "ngram_stats"


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d for unpaired (independent) samples."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def cohens_d_z(diffs: np.ndarray) -> float:
    """Cohen's d_z for paired samples: mean(diff) / std(diff).

    This is the correct effect size for paired designs (Lakens, 2013).
    """
    if len(diffs) < 2:
        return 0.0
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diffs) / sd)


def permutation_test_paired(x: np.ndarray, y: np.ndarray, n_perms: int = 10000,
                             seed: int = 42) -> float:
    """Two-sided permutation test for paired samples.

    Under H0, the sign of each difference is equally likely to be positive
    or negative. We permute signs and compute the proportion of permuted
    mean-diffs as extreme as the observed one.
    """
    rng = np.random.RandomState(seed)
    diffs = x - y
    observed = np.abs(np.mean(diffs))
    n = len(diffs)

    count = 0
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=n)
        perm_mean = np.abs(np.mean(diffs * signs))
        if perm_mean >= observed:
            count += 1

    return (count + 1) / (n_perms + 1)  # +1 for continuity correction


def collapse_pseudoreplicates(df: pd.DataFrame, group_col: str = "prompt_id") -> pd.DataFrame:
    """Average match rates across seeds for the same prompt to avoid pseudoreplication.

    If prompt_id is not available, attempts to infer it from file_id by stripping
    seed/temperature suffixes.
    """
    if group_col not in df.columns:
        # Infer prompt_id: strip _s42_t1.0 style suffixes
        df = df.copy()
        df["prompt_id"] = df["file_id"].str.replace(r"_s\d+_t[\d.]+$", "", regex=True)
        group_col = "prompt_id"

    # Keep tier, artist_id, genre from first occurrence; average match_rate
    agg = df.groupby([group_col, "tier", "ngram_size"], dropna=False).agg(
        match_rate=("match_rate", "mean"),
        artist_id=("artist_id", "first"),
        genre=("genre", "first"),
        n_seeds=("match_rate", "count"),
    ).reset_index()
    agg["file_id"] = agg[group_col]  # for downstream compat
    return agg


def run_unpaired_test(group1: np.ndarray, group2: np.ndarray, label1: str,
                      label2: str, test_name: str, ngram_size: int) -> dict:
    """Run Welch's t-test + Mann-Whitney U and return a result dict."""
    if len(group1) < 2 or len(group2) < 2:
        return None

    t_stat, p_welch = stats.ttest_ind(group1, group2, equal_var=False)
    u_stat, p_mann = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    d = cohens_d(group1, group2)

    return {
        "test_name": test_name,
        "ngram_size": ngram_size,
        "group1": label1,
        "group2": label2,
        "n_group1": len(group1),
        "n_group2": len(group2),
        "mean_group1": float(np.mean(group1)),
        "mean_group2": float(np.mean(group2)),
        "std_group1": float(np.std(group1, ddof=1)),
        "std_group2": float(np.std(group2, ddof=1)),
        "t_statistic": float(t_stat),
        "p_welch": float(p_welch),
        "u_statistic": float(u_stat),
        "p_mann_whitney": float(p_mann),
        "cohens_d": float(d),
        "test_type": "unpaired",
    }


def run_paired_test(matched: np.ndarray, mismatched: np.ndarray,
                    test_name: str, ngram_size: int,
                    n_perms: int = 10000) -> dict:
    """Run paired t-test + Wilcoxon signed-rank + permutation test."""
    if len(matched) < 2:
        return None

    diffs = matched - mismatched

    t_stat, p_paired = stats.ttest_rel(matched, mismatched)

    # Wilcoxon signed-rank (handles zero diffs gracefully)
    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) >= 5:
        w_stat, p_wilcox = stats.wilcoxon(nonzero_diffs, alternative="two-sided")
    else:
        w_stat, p_wilcox = float("nan"), float("nan")

    d_z = cohens_d_z(diffs)
    p_perm = permutation_test_paired(matched, mismatched, n_perms=n_perms)

    return {
        "test_name": test_name,
        "ngram_size": ngram_size,
        "group1": "matched_artist",
        "group2": "mismatched_same_genre",
        "n_group1": len(matched),
        "n_group2": len(mismatched),
        "mean_group1": float(np.mean(matched)),
        "mean_group2": float(np.mean(mismatched)),
        "std_group1": float(np.std(matched, ddof=1)),
        "std_group2": float(np.std(mismatched, ddof=1)),
        "t_statistic": float(t_stat),
        "p_paired_t": float(p_paired),
        "w_statistic": float(w_stat),
        "p_wilcoxon": float(p_wilcox),
        "p_permutation": float(p_perm),
        "cohens_d_z": float(d_z),
        "test_type": "paired",
    }


def bootstrap_null_distribution(catalog_dir: Path, n_samples: int,
                                ngram_sizes: list, codebooks: list,
                                num_frames: int, seed: int, logger):
    """Generate null distribution of n-gram match rates by sampling from
    the empirical token frequency distribution."""
    from src.analysis.ngram_search import build_catalog_index, count_ngram_matches

    rng = np.random.RandomState(seed)

    # Compute empirical token distribution from catalog
    catalog_files = sorted(catalog_dir.glob("*.npy"))
    token_counts = {}
    for cb in codebooks:
        token_counts[cb] = np.zeros(2048, dtype=np.int64)

    for fpath in catalog_files:
        codes = np.load(str(fpath))
        for cb in codebooks:
            if cb < codes.shape[0]:
                for t in codes[cb]:
                    token_counts[cb][t] += 1

    token_probs = {}
    for cb in codebooks:
        total = token_counts[cb].sum()
        token_probs[cb] = token_counts[cb] / total if total > 0 else np.ones(2048) / 2048

    catalog_index = build_catalog_index(catalog_dir, codebooks, ngram_sizes, logger)

    null_dist = {}
    for cb in codebooks:
        for n in ngram_sizes:
            rates = []
            for _ in range(n_samples):
                random_tokens = rng.choice(2048, size=num_frames, p=token_probs[cb])
                matches = count_ngram_matches(random_tokens, catalog_index[(cb, n)], n)
                total = max(num_frames - n + 1, 0)
                rates.append(matches / total if total > 0 else 0)

            null_dist[f"cb{cb}_n{n}"] = {
                "mean": float(np.mean(rates)),
                "std": float(np.std(rates)),
                "p5": float(np.percentile(rates, 5)),
                "p95": float(np.percentile(rates, 95)),
                "p99": float(np.percentile(rates, 99)),
                "max": float(np.max(rates)),
            }
            logger.info(f"  Null dist cb{cb} {n}-gram: "
                        f"mean={null_dist[f'cb{cb}_n{n}']['mean']:.8f}, "
                        f"p99={null_dist[f'cb{cb}_n{n}']['p99']:.8f}")

    return null_dist


def apply_fdr_by_family(test_results: list, alpha: float, logger):
    """Apply BH FDR correction separately within test families.

    Family 1: T1-T4 (primary hypotheses)
    Family 2: T5 (per-genre, exploratory)
    Family 3: T6 (handled separately — correlation, not hypothesis test)
    """
    from statsmodels.stats.multitest import multipletests

    families = {
        "primary": ["T1_memorization", "T2_genre_leakage", "T3_artist_specificity",
                     "T3b_prompt_type", "T4_matched_vs_mismatched"],
        "exploratory": [],  # T5_genre_* tests
    }

    for r in test_results:
        if r["test_name"].startswith("T5_genre_"):
            families["exploratory"].append(r["test_name"])

    for family_name, test_names in families.items():
        family_tests = [r for r in test_results if r["test_name"] in test_names]
        if not family_tests:
            continue

        # Use the most appropriate p-value for FDR
        p_values = []
        for r in family_tests:
            if r.get("test_type") == "paired":
                p_values.append(r.get("p_paired_t", r.get("p_value", 1.0)))
            else:
                p_values.append(r.get("p_welch", r.get("p_value", 1.0)))

        rejected, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

        for i, r in enumerate(family_tests):
            r["fdr_family"] = family_name
            r["p_adjusted"] = float(p_adjusted[i])
            r["significant"] = bool(rejected[i])

            # Effect significance: both statistically significant AND meaningful effect
            if r.get("test_type") == "paired":
                effect = abs(r.get("cohens_d_z", 0))
            else:
                effect = abs(r.get("cohens_d", 0))
            r["effect_significant"] = bool(rejected[i] and effect >= 0.2)

        logger.info(f"  FDR family '{family_name}': {len(family_tests)} tests, "
                    f"{sum(1 for r in family_tests if r.get('significant'))} significant")


def main():
    parser = base_argparser("Statistical analysis of n-gram match rates")
    parser.add_argument("--skip_null", action="store_true",
                        help="Skip null distribution computation (faster)")
    parser.add_argument("--n_perms", type=int, default=10000,
                        help="Number of permutations for permutation test (default: 10000)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(args.run_dir, STAGE)
    meta = log_preamble(logger, args, STAGE)
    dirs = ensure_dirs(args.run_dir, ["analysis"])

    mem_cfg = cfg.get("memorization", {})
    tok_cfg = cfg.get("tokenization", {})
    ngram_sizes = mem_cfg.get("ngram_sizes", [3, 4, 5, 6, 8])
    alpha = mem_cfg.get("alpha", 0.05)
    num_codebooks = tok_cfg.get("num_codebooks", 4)
    codebooks_cfg = mem_cfg.get("codebooks", None)
    codebooks = list(range(num_codebooks)) if codebooks_cfg is None else codebooks_cfg

    # ---- Load n-gram match results ----
    matches_csv = dirs["analysis"] / "ngram_matches.csv"
    if not matches_csv.exists():
        logger.error(f"N-gram matches file not found: {matches_csv}")
        logger.error("Run ngram_search.py first!")
        return

    df = pd.read_csv(matches_csv)
    logger.info(f"Loaded {len(df)} match records")

    # Aggregate match rates per file (summing across codebooks)
    # dropna=False is critical: Tier B/C files have NaN artist_id/genre
    file_rates = df.groupby(["file_id", "tier", "artist_id", "genre", "ngram_size"],
                             dropna=False).agg(
        total_matches=("matches", "sum"),
        total_ngrams=("total_ngrams", "sum"),
    ).reset_index()
    file_rates["match_rate"] = file_rates["total_matches"] / file_rates["total_ngrams"]

    # ---- Collapse pseudoreplicates (average across seeds per prompt) ----
    logger.info("Collapsing pseudoreplicates (averaging across seeds per prompt)...")
    prompt_rates = collapse_pseudoreplicates(file_rates)
    logger.info(f"  {len(file_rates)} file-level rows → {len(prompt_rates)} prompt-level rows")

    # ---- Run all statistical tests ----
    test_results = []

    for n in ngram_sizes:
        n_rates = prompt_rates[prompt_rates["ngram_size"] == n]

        tier_a = n_rates[n_rates["tier"] == "A_artist_proximal"]["match_rate"].values
        tier_b = n_rates[n_rates["tier"] == "B_genre_generic"]["match_rate"].values
        tier_c = n_rates[n_rates["tier"] == "C_out_of_distribution"]["match_rate"].values
        tier_d = n_rates[n_rates["tier"] == "D_fma_tags"]["match_rate"].values
        tier_ad = np.concatenate([tier_a, tier_d]) if len(tier_a) > 0 or len(tier_d) > 0 else np.array([])

        # T1: Tier A+D vs Tier C (main memorization test)
        r = run_unpaired_test(tier_ad, tier_c, "A+D (artist-linked)", "C (OOD control)",
                              "T1_memorization", n)
        if r:
            test_results.append(r)

        # T2: Tier B vs Tier C (genre leakage)
        r = run_unpaired_test(tier_b, tier_c, "B (genre-generic)", "C (OOD control)",
                              "T2_genre_leakage", n)
        if r:
            test_results.append(r)

        # T3: Tier A vs Tier B (artist-specificity)
        r = run_unpaired_test(tier_a, tier_b, "A (artist-proximal)", "B (genre-generic)",
                              "T3_artist_specificity", n)
        if r:
            test_results.append(r)

        # T3b: Tier A vs Tier D (prompt type comparison)
        r = run_unpaired_test(tier_a, tier_d, "A (artist-proximal)", "D (FMA tags)",
                              "T3b_prompt_type", n)
        if r:
            test_results.append(r)

    # ---- T4: Per-artist matched vs mismatched (PAIRED) ----
    pa_csv = dirs["analysis"] / "ngram_per_artist.csv"
    if pa_csv.exists():
        pa_df = pd.read_csv(pa_csv)
        logger.info(f"Loaded {len(pa_df)} per-artist rows")

        # Collapse pseudoreplicates for paired data too
        if "file_id" in pa_df.columns:
            pa_df["prompt_id"] = pa_df["file_id"].str.replace(r"_s\d+_t[\d.]+$", "", regex=True)
            pa_collapsed = pa_df.groupby(["prompt_id", "artist_id", "genre", "ngram_size"]).agg(
                matched_rate=("matched_rate", "mean"),
                mismatched_rate_per_artist=("mismatched_rate_per_artist", "mean"),
            ).reset_index()
        else:
            pa_collapsed = pa_df

        for n in ngram_sizes:
            n_df = pa_collapsed[pa_collapsed["ngram_size"] == n]
            if n_df.empty or len(n_df) < 2:
                continue

            matched = n_df["matched_rate"].values
            mismatched = n_df["mismatched_rate_per_artist"].values

            r = run_paired_test(matched, mismatched,
                                "T4_matched_vs_mismatched", n,
                                n_perms=args.n_perms)
            if r:
                test_results.append(r)

    # ---- T5: Per-genre comparison ----
    genre_results = []
    genres = prompt_rates["genre"].dropna().unique()
    genres = [g for g in genres if g and str(g) != "" and str(g) != "nan"]

    if len(genres) >= 2:
        for n in ngram_sizes:
            n_rates = prompt_rates[prompt_rates["ngram_size"] == n]
            tier_c = n_rates[n_rates["tier"] == "C_out_of_distribution"]["match_rate"].values

            for genre in genres:
                genre_rates = n_rates[(n_rates["genre"] == genre) &
                                      (n_rates["tier"].isin(["A_artist_proximal", "D_fma_tags"]))
                                      ]["match_rate"].values
                if len(genre_rates) < 2:
                    continue

                r = run_unpaired_test(genre_rates, tier_c, f"genre:{genre}", "C (OOD control)",
                                      f"T5_genre_{genre}", n)
                if r:
                    genre_results.append(r)
                    test_results.append(r)

    # ---- T6: Training exposure correlation ----
    exposure_results = []
    if pa_csv.exists():
        pa_df = pd.read_csv(pa_csv)
        manifest_path = Path(args.run_dir) / "manifests" / "sampling_manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            artist_col = "artist_id" if "artist_id" in manifest.columns else "artist_name"
            track_counts = manifest.groupby(artist_col).size().reset_index(name="n_tracks")

            for n in ngram_sizes:
                n_df = pa_df[pa_df["ngram_size"] == n]
                artist_rates = n_df.groupby("artist_id")["matched_rate"].mean().reset_index()
                artist_rates["artist_id"] = artist_rates["artist_id"].astype(str)
                track_counts[artist_col] = track_counts[artist_col].astype(str)

                merged = artist_rates.merge(track_counts, left_on="artist_id",
                                             right_on=artist_col, how="inner")
                if len(merged) >= 5:
                    rho, p_val = stats.spearmanr(merged["n_tracks"], merged["matched_rate"])
                    exposure_results.append({
                        "ngram_size": n,
                        "spearman_rho": float(rho),
                        "p_value": float(p_val),
                        "n_artists": len(merged),
                    })
                    logger.info(f"  T6 exposure correlation {n}-gram: "
                                f"rho={rho:.3f}, p={p_val:.4f} (n={len(merged)})")

    # ---- Apply FDR correction by family ----
    if test_results:
        apply_fdr_by_family(test_results, alpha, logger)

    # ---- Write test results ----
    tests_csv = dirs["analysis"] / "ngram_statistical_tests.csv"
    if test_results:
        pd.DataFrame(test_results).to_csv(tests_csv, index=False)

    # Genre stats separately
    if genre_results:
        genre_csv = dirs["analysis"] / "ngram_per_genre_stats.csv"
        pd.DataFrame(genre_results).to_csv(genre_csv, index=False)

    # ---- Log results ----
    logger.info(f"\n{'='*60}")
    logger.info(f"N-GRAM STATISTICAL TESTS (alpha={alpha})")
    logger.info(f"  Pseudoreplication: collapsed across seeds per prompt")
    logger.info(f"  FDR: BH correction applied within test families")
    logger.info(f"{'='*60}")

    for test_name in ["T1_memorization", "T2_genre_leakage", "T3_artist_specificity",
                      "T3b_prompt_type"]:
        test_rows = [r for r in test_results if r["test_name"] == test_name]
        if test_rows:
            logger.info(f"\n--- {test_name} (unpaired: Welch + Mann-Whitney) ---")
            for r in test_rows:
                sig = "***" if r.get("effect_significant") else ("*" if r.get("significant") else "ns")
                logger.info(
                    f"  {r['ngram_size']}-gram: "
                    f"{r['group1']}={r['mean_group1']:.8f} vs "
                    f"{r['group2']}={r['mean_group2']:.8f}, "
                    f"Welch p={r['p_welch']:.4f}, MW p={r['p_mann_whitney']:.4f}, "
                    f"p_adj={r.get('p_adjusted', 'N/A')}, "
                    f"d={r['cohens_d']:.3f} [{sig}]"
                )

    # T4 (paired)
    t4_rows = [r for r in test_results if r["test_name"] == "T4_matched_vs_mismatched"]
    if t4_rows:
        logger.info(f"\n--- T4_matched_vs_mismatched (paired: t-test + Wilcoxon + permutation) ---")
        for r in t4_rows:
            sig = "***" if r.get("effect_significant") else ("*" if r.get("significant") else "ns")
            logger.info(
                f"  {r['ngram_size']}-gram: "
                f"matched={r['mean_group1']:.8f} vs mismatched={r['mean_group2']:.8f}, "
                f"paired-t p={r['p_paired_t']:.4f}, Wilcoxon p={r['p_wilcoxon']:.4f}, "
                f"perm p={r['p_permutation']:.4f}, "
                f"p_adj={r.get('p_adjusted', 'N/A')}, "
                f"d_z={r['cohens_d_z']:.3f} [{sig}]"
            )

    if exposure_results:
        logger.info(f"\n--- T6: Training Exposure Correlation (not FDR-corrected) ---")
        for r in exposure_results:
            logger.info(f"  {r['ngram_size']}-gram: rho={r['spearman_rho']:.3f}, "
                        f"p={r['p_value']:.4f}")

    # ---- Null distribution ----
    null_dist = None
    if not args.skip_null:
        catalog_dir = Path(args.run_dir) / "tokens_catalog"
        if catalog_dir.exists():
            logger.info("\nComputing null distribution via bootstrap...")
            median_frames = int(file_rates["total_ngrams"].median()) + max(ngram_sizes)
            null_dist = bootstrap_null_distribution(
                catalog_dir,
                n_samples=mem_cfg.get("null_bootstrap_samples", 1000),
                ngram_sizes=ngram_sizes,
                codebooks=codebooks,
                num_frames=median_frames,
                seed=mem_cfg.get("null_seed", 42),
                logger=logger,
            )
            null_path = dirs["analysis"] / "ngram_null_distribution.json"
            with open(null_path, "w") as f:
                json.dump(null_dist, f, indent=2)

    # ---- Verdict ----
    t1_significant = [r for r in test_results
                      if r["test_name"] == "T1_memorization" and r.get("effect_significant")]
    t4_significant = [r for r in test_results
                      if r["test_name"] == "T4_matched_vs_mismatched" and r.get("effect_significant")]

    memorization_detected = len(t1_significant) > 0 or len(t4_significant) > 0

    verdict = {
        "memorization_detected": memorization_detected,
        "evidence": {
            "T1_tier_comparison": {
                "significant": len(t1_significant) > 0,
                "significant_ngram_sizes": [r["ngram_size"] for r in t1_significant],
                "description": "Artist-linked prompts (A+D) vs OOD control (C)",
            },
            "T2_genre_leakage": {
                "significant": any(r.get("effect_significant") for r in test_results
                                   if r["test_name"] == "T2_genre_leakage"),
                "description": "Genre-generic prompts (B) vs OOD control (C)",
            },
            "T3_artist_specificity": {
                "significant": any(r.get("effect_significant") for r in test_results
                                   if r["test_name"] == "T3_artist_specificity"),
                "description": "Artist-proximal (A) vs genre-generic (B)",
            },
            "T4_matched_vs_mismatched": {
                "significant": len(t4_significant) > 0,
                "significant_ngram_sizes": [r["ngram_size"] for r in t4_significant],
                "description": "Per-artist: matched catalog vs same-genre mismatched catalog (paired)",
                "note": "Same-genre control eliminates genre confound",
            },
            "T6_exposure_correlation": {
                "results": exposure_results,
                "description": "Spearman correlation: n_training_tracks vs match rate",
            },
        },
        "methodology": {
            "pseudoreplication": "Averaged across seeds per prompt before testing",
            "parametric_tests": "Welch's t-test (unpaired), paired t-test",
            "nonparametric_tests": "Mann-Whitney U (unpaired), Wilcoxon signed-rank (paired)",
            "permutation_test": f"{args.n_perms} permutations for T4",
            "effect_sizes": "Cohen's d (unpaired), Cohen's d_z (paired, Lakens 2013)",
            "fdr_correction": "BH within families: primary (T1-T4), exploratory (T5), correlation (T6)",
        },
        "alpha": alpha,
        "n_tests_total": len(test_results),
        "n_tests_significant": sum(1 for r in test_results if r.get("significant")),
        "n_tests_effect_significant": sum(1 for r in test_results if r.get("effect_significant")),
    }

    verdict_path = dirs["analysis"] / "memorization_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"VERDICT: Memorization {'DETECTED' if memorization_detected else 'NOT detected'}")
    if memorization_detected:
        if t1_significant:
            logger.info(f"  T1 significant at n-gram sizes: {[r['ngram_size'] for r in t1_significant]}")
        if t4_significant:
            logger.info(f"  T4 significant at n-gram sizes: {[r['ngram_size'] for r in t4_significant]}")
    logger.info(f"  Total tests: {len(test_results)}, significant: "
                f"{sum(1 for r in test_results if r.get('effect_significant'))}")
    logger.info(f"{'='*60}")

    outputs = [str(tests_csv), str(verdict_path)]
    if null_dist:
        outputs.append(str(dirs["analysis"] / "ngram_null_distribution.json"))
    if genre_results:
        outputs.append(str(dirs["analysis"] / "ngram_per_genre_stats.csv"))
    meta = log_finish(logger, meta, STAGE, outputs=outputs)
    meta["verdict"] = verdict
    if exposure_results:
        meta["exposure_correlation"] = exposure_results
    save_run_metadata(args.run_dir, STAGE, meta)


if __name__ == "__main__":
    main()
