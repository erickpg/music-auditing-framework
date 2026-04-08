#!/usr/bin/env python3
"""
Compute ICC for all pairs (V1-V2, V1-BL, V2-BL) and
recompute bootstrap CIs with 3-tier thresholds.
"""
import csv, json, os, random, math

RESULTS = os.environ.get("RESULTS_DIR", "/Users/erickpg/capstone/results")
OUT_DIR = os.path.join(RESULTS, "robustness")
os.makedirs(OUT_DIR, exist_ok=True)


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if any(v == k for k, v in r.items()):
                continue
            rows.append(r)
    return rows


def sf(v):
    try:
        return float(v)
    except:
        return None


def tier3(score):
    if score >= 0.67:
        return "High"
    elif score <= 0.33:
        return "Low"
    else:
        return "Intermediate"


def compute_2sig(data, artists):
    clap_vals = [data[a]["clap"] for a in artists if data[a]["clap"] is not None]
    fad_vals = [data[a]["fad"] for a in artists if data[a]["fad"] is not None]
    c_min, c_max = min(clap_vals), max(clap_vals)
    f_min, f_max = min(fad_vals), max(fad_vals)
    scores = {}
    for a in artists:
        cn = (data[a]["clap"] - c_min) / (c_max - c_min + 1e-10)
        fn = 1.0 - (data[a]["fad"] - f_min) / (f_max - f_min + 1e-10)
        scores[a] = 0.5 * cn + 0.5 * fn
    return scores


def icc_a1(x, y):
    """ICC(A,1) - two-way mixed, single measures, absolute agreement."""
    n = len(x)
    grand_mean = sum(x[i] + y[i] for i in range(n)) / (2 * n)

    # Between-subjects SS
    row_means = [(x[i] + y[i]) / 2 for i in range(n)]
    ssb = 2 * sum((rm - grand_mean) ** 2 for rm in row_means)

    # Within-subjects SS
    ssw = sum((x[i] - row_means[i]) ** 2 + (y[i] - row_means[i]) ** 2 for i in range(n))

    # Between-measures SS (column effect)
    col_means = [sum(x) / n, sum(y) / n]
    ssc = n * sum((cm - grand_mean) ** 2 for cm in col_means)

    # Error SS
    sse = ssw - ssc

    # Mean squares
    msb = ssb / (n - 1)
    msw = ssw / n  # within MS
    mse = sse / (n - 1)
    msc = ssc / 1  # k-1 = 1

    # ICC(A,1) = (MSB - MSW) / (MSB + MSW + 2*(MSC - MSE)/n)
    # Simplified for k=2:
    # ICC(A,1) = (MSB - MSE) / (MSB + MSE + 2*(MSC - MSE)/n)
    denom = msb + mse + 2 * (msc - mse) / n
    if denom == 0:
        return 0, 1.0
    icc = (msb - mse) / denom

    # F-test
    f_val = msb / msw if msw > 0 else 0
    # Approximate p-value using normal approximation of F
    df1 = n - 1
    df2 = n
    if f_val > 1:
        t_approx = math.sqrt(f_val) * math.sqrt(df2 / (df2 + df1))
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t_approx) / math.sqrt(2))))
    else:
        p = 1.0

    return round(icc, 4), round(p, 6)


def bootstrap_ci_3tier(clap_per_artist_path, fad_csv_path, version_label,
                       n_boot=1000, seed=42):
    """Bootstrap 95% CIs for 2-signal vulnerability, interpreted with 3-tier."""
    rng = random.Random(seed)

    # Load per-file CLAP similarity
    clap_rows = load_csv(clap_per_artist_path)
    artist_files = {}  # aid -> list of (matched_sim, mismatched_sim)
    for r in clap_rows:
        aid = r.get("artist_id", "")
        m = sf(r.get("matched_mean_sim"))
        mm = sf(r.get("mismatched_mean_sim"))
        if aid and m is not None and mm is not None:
            artist_files.setdefault(aid, []).append((m, mm))

    # Load per-artist FAD (fixed, not bootstrapped)
    fad_rows = load_csv(fad_csv_path)
    artist_fad = {}
    for r in fad_rows:
        aid = r.get("artist_id", "")
        fad_val = sf(r.get("fad"))
        comp = r.get("comparison", "matched")
        if aid and fad_val is not None and comp == "matched":
            artist_fad[aid] = fad_val

    # Only artists with both CLAP files and FAD
    artists = sorted(set(artist_files.keys()) & set(artist_fad.keys()))
    if not artists:
        print(f"  WARNING: No artists with both CLAP and FAD for {version_label}")
        return []

    # For each bootstrap iteration: resample per-file CLAP observations,
    # recompute per-artist mean CLAP sim, combine with FAD, normalize, score
    all_fad = [artist_fad[a] for a in artists]
    fad_min, fad_max = min(all_fad), max(all_fad)

    results = []
    boot_scores = {a: [] for a in artists}

    for b in range(n_boot):
        # Resample CLAP files per artist
        clap_means = {}
        for a in artists:
            files = artist_files[a]
            resampled = [rng.choice(files) for _ in range(len(files))]
            clap_means[a] = sum(m for m, mm in resampled) / len(resampled)

        # Normalize CLAP
        cv = list(clap_means.values())
        c_min, c_max = min(cv), max(cv)

        for a in artists:
            cn = (clap_means[a] - c_min) / (c_max - c_min + 1e-10)
            fn = 1.0 - (artist_fad[a] - fad_min) / (fad_max - fad_min + 1e-10)
            boot_scores[a].append(0.5 * cn + 0.5 * fn)

    # Compute CIs and tier classification
    for a in artists:
        scores = sorted(boot_scores[a])
        mean_score = sum(scores) / len(scores)
        ci_lower = scores[int(0.025 * n_boot)]
        ci_upper = scores[int(0.975 * n_boot)]
        ci_width = ci_upper - ci_lower

        # 3-tier classification
        if ci_lower >= 0.67:
            classification = "Clearly High"
        elif ci_upper <= 0.33:
            classification = "Clearly Low"
        elif ci_lower >= 0.33 and ci_upper <= 0.67:
            classification = "Clearly Intermediate"
        else:
            classification = "Ambiguous"

        # Also check 2-tier
        if ci_lower > 0.5:
            class_2tier = "Clearly High"
        elif ci_upper < 0.5:
            class_2tier = "Clearly Low"
        else:
            class_2tier = "Ambiguous"

        results.append({
            "artist_id": a,
            "version": version_label,
            "score": round(mean_score, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "ci_width": round(ci_width, 4),
            "tier_point": tier3(mean_score),
            "classification_3tier": classification,
            "classification_2tier": class_2tier,
            "n_files": len(artist_files[a]),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════

def load_version(subdir):
    rows = load_csv(f"{RESULTS}/{subdir}/analysis/vulnerability_scores.csv")
    data = {}
    for r in rows:
        aid = r.get("artist_id", "")
        if not aid or aid == "artist_id":
            continue
        data[aid] = {
            "clap": sf(r.get("clap_similarity")),
            "fad": sf(r.get("fad")),
            "name": r.get("artist_name", ""),
        }
    return data


v1 = load_version("v1")
v2 = load_version("v2")
bl = load_version("baseline")
common = sorted(set(v1.keys()) & set(v2.keys()) & set(bl.keys()))

v1_scores = compute_2sig(v1, common)
v2_scores = compute_2sig(v2, common)
bl_scores = compute_2sig(bl, common)


# ═══════════════════════════════════════════════════════════════════════
# ICC for all pairs
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ICC(A,1) — 2-signal CLAP+FAD vulnerability scores")
print("=" * 60)

icc_results = {}
for label, sa, sb in [("V1_vs_V2", v1_scores, v2_scores),
                       ("V1_vs_BL", v1_scores, bl_scores),
                       ("V2_vs_BL", v2_scores, bl_scores)]:
    x = [sa[a] for a in common]
    y = [sb[a] for a in common]
    icc, p = icc_a1(x, y)
    # Also Pearson and Spearman for reference
    mx, my = sum(x)/len(x), sum(y)/len(y)
    sx = (sum((xi-mx)**2 for xi in x)/(len(x)-1))**0.5
    sy = (sum((yi-my)**2 for yi in y)/(len(y)-1))**0.5
    pearson_r = sum((xi-mx)*(yi-my) for xi, yi in zip(x,y)) / ((len(x)-1)*sx*sy) if sx>0 and sy>0 else 0
    # Spearman
    rx = sorted(range(len(x)), key=lambda i: x[i])
    ry = sorted(range(len(y)), key=lambda i: y[i])
    rank_x = [0]*len(x)
    rank_y = [0]*len(y)
    for i, idx in enumerate(rx):
        rank_x[idx] = i
    for i, idx in enumerate(ry):
        rank_y[idx] = i
    d2 = sum((rank_x[i]-rank_y[i])**2 for i in range(len(x)))
    n = len(x)
    spearman = 1 - 6*d2/(n*(n**2-1))

    # Mean absolute difference
    mad = sum(abs(x[i]-y[i]) for i in range(n)) / n

    print(f"\n{label}:")
    print(f"  ICC(A,1): {icc}")
    print(f"  Pearson r: {round(pearson_r, 4)}")
    print(f"  Spearman ρ: {round(spearman, 4)}")
    print(f"  Mean abs diff: {round(mad, 4)}")

    icc_results[label] = {
        "icc_a1": icc,
        "pearson_r": round(pearson_r, 4),
        "spearman_rho": round(spearman, 4),
        "mean_abs_diff": round(mad, 4),
        "n": n,
    }


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap CIs with 3-tier thresholds
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("BOOTSTRAP 95% CIs (1,000 resamples, 3-tier interpretation)")
print("=" * 60)

all_boot = []
for version, subdir in [("V1", "v1"), ("V2", "v2"), ("Baseline", "baseline")]:
    clap_path = f"{RESULTS}/{subdir}/analysis/clap_per_artist.csv"
    fad_path = f"{RESULTS}/{subdir}/analysis/per_artist_fad.csv"

    if not os.path.exists(clap_path) or not os.path.exists(fad_path):
        # Try clap_similarity.csv which has per-file data
        clap_path = f"{RESULTS}/{subdir}/analysis/clap_similarity.csv"

    if not os.path.exists(clap_path):
        print(f"  Skipping {version} — no CLAP data at {clap_path}")
        continue

    print(f"\nComputing bootstrap for {version}...")
    boot = bootstrap_ci_3tier(clap_path, fad_path, version)
    all_boot.extend(boot)

    # Summary
    c3 = {}
    for r in boot:
        c = r["classification_3tier"]
        c3[c] = c3.get(c, 0) + 1
    print(f"  {version}: {c3}")

# Save bootstrap results
boot_csv = os.path.join(OUT_DIR, "bootstrap_ci_3tier.csv")
if all_boot:
    with open(boot_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_boot[0].keys())
        w.writeheader()
        w.writerows(all_boot)
    print(f"\n[SAVED] {boot_csv}")

# Save ICC results
icc_path = os.path.join(OUT_DIR, "icc_all_pairs.json")
with open(icc_path, "w") as f:
    json.dump(icc_results, f, indent=2)
print(f"[SAVED] {icc_path}")

# Summary table for the document
print("\n\n" + "=" * 60)
print("SUMMARY FOR DOCUMENT")
print("=" * 60)

print("\nICC Table:")
print(f"{'Pair':<15} {'ICC(A,1)':>10} {'Pearson r':>10} {'Spearman ρ':>12} {'MAD':>8}")
print("-" * 58)
for label in ["V1_vs_V2", "V1_vs_BL", "V2_vs_BL"]:
    r = icc_results[label]
    print(f"{label:<15} {r['icc_a1']:>10} {r['pearson_r']:>10} {r['spearman_rho']:>12} {r['mean_abs_diff']:>8}")

print("\nBootstrap 3-tier classification:")
for version in ["V1", "V2", "Baseline"]:
    v_boot = [r for r in all_boot if r["version"] == version]
    if not v_boot:
        continue
    c3 = {}
    for r in v_boot:
        c = r["classification_3tier"]
        c3[c] = c3.get(c, 0) + 1
    print(f"  {version}: Clearly High={c3.get('Clearly High',0)}, "
          f"Clearly Intermediate={c3.get('Clearly Intermediate',0)}, "
          f"Clearly Low={c3.get('Clearly Low',0)}, "
          f"Ambiguous={c3.get('Ambiguous',0)}")
