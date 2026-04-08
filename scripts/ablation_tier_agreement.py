#!/usr/bin/env python3
"""Ablation study: tier agreement (3-tier) across signal configurations, V1 vs V2."""
import csv, json, os

RESULTS = os.environ.get("RESULTS_DIR", "/Users/erickpg/capstone/results")


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


def tier3(score, lo=0.33, hi=0.67):
    if score >= hi:
        return "High"
    elif score <= lo:
        return "Low"
    else:
        return "Intermediate"


def compute_agreement(tiers_a, tiers_b):
    n = len(tiers_a)
    exact = sum(1 for a, b in zip(tiers_a, tiers_b) if a == b)
    catastrophic = sum(1 for a, b in zip(tiers_a, tiers_b) if {a, b} == {"High", "Low"})
    cats = ["High", "Intermediate", "Low"]
    po = exact / n
    pe = sum(
        (sum(1 for t in tiers_a if t == c) / n) * (sum(1 for t in tiers_b if t == c) / n)
        for c in cats
    )
    k = (po - pe) / (1 - pe) if pe < 1 else 0
    return {
        "exact": exact,
        "pct": round(100 * exact / n, 1),
        "kappa": round(k, 3),
        "catastrophic": catastrophic,
    }


# Load per-artist raw signals for V1 and V2
v1_vuln = load_csv(f"{RESULTS}/v1/analysis/vulnerability_scores.csv")
v2_vuln = load_csv(f"{RESULTS}/v2/analysis/vulnerability_scores.csv")

# Build per-artist signal dictionaries
v1_data = {}
for r in v1_vuln:
    aid = r.get("artist_id", "")
    if not aid or aid == "artist_id":
        continue
    v1_data[aid] = {
        "clap": sf(r.get("clap_similarity")),
        "fad": sf(r.get("fad")),
        "musico": sf(r.get("musicological_similarity")),
        "ngram": sf(r.get("ngram_match_rate")),
        "name": r.get("artist_name", ""),
    }

v2_data = {}
for r in v2_vuln:
    aid = r.get("artist_id", "")
    if not aid or aid == "artist_id":
        continue
    v2_data[aid] = {
        "clap": sf(r.get("clap_similarity")),
        "fad": sf(r.get("fad")),
        "musico": sf(r.get("musicological_similarity")),
        "ngram": sf(r.get("ngram_match_rate")),
        "name": r.get("artist_name", ""),
    }

common = sorted(set(v1_data.keys()) & set(v2_data.keys()))
print(f"Common artists: {len(common)}")


def compute_vuln(data, signals, artists):
    """Compute vulnerability score from specified signals for a set of artists."""
    # Gather raw values
    raw = {sig: [] for sig in signals}
    for aid in artists:
        for sig in signals:
            v = data[aid].get(sig)
            if v is not None:
                raw[sig].append(v)

    # Min-max bounds
    bounds = {}
    for sig in signals:
        vals = raw[sig]
        bounds[sig] = (min(vals), max(vals)) if vals else (0, 1)

    # Compute per-artist score
    scores = {}
    for aid in artists:
        components = []
        for sig in signals:
            v = data[aid].get(sig)
            if v is None:
                continue
            lo, hi = bounds[sig]
            rng = hi - lo + 1e-10
            norm = (v - lo) / rng
            if sig == "fad":
                norm = 1.0 - norm  # lower FAD = higher vulnerability
            components.append(norm)
        scores[aid] = sum(components) / len(components) if components else 0.5
    return scores


# Define configurations
configs = [
    ("2-signal (CLAP + FAD)", ["clap", "fad"]),
    ("3-signal (drop musico)", ["clap", "fad", "ngram"]),
    ("3-signal (drop ngram)", ["clap", "fad", "musico"]),
    ("4-signal (original)", ["clap", "fad", "musico", "ngram"]),
    ("CLAP only", ["clap"]),
    ("FAD only", ["fad"]),
    ("Musico only", ["musico"]),
    ("2-signal (CLAP + musico)", ["clap", "musico"]),
]

print(f"\n{'Config':<28} {'Agree':>6} {'%':>6} {'κ':>7} {'Catastr':>8}")
print("-" * 60)

results = []
for name, signals in configs:
    v1_scores = compute_vuln(v1_data, signals, common)
    v2_scores = compute_vuln(v2_data, signals, common)

    v1_tiers = [tier3(v1_scores[a]) for a in common]
    v2_tiers = [tier3(v2_scores[a]) for a in common]

    ag = compute_agreement(v1_tiers, v2_tiers)
    print(f"{name:<28} {ag['exact']:>3}/50 {ag['pct']:>5}% {ag['kappa']:>6.3f} {ag['catastrophic']:>8}")

    # Also compute V1 summary stats for the config
    v1_vals = [v1_scores[a] for a in common]
    v2_vals = [v2_scores[a] for a in common]

    results.append({
        "config": name,
        "signals": signals,
        "v1_mean": round(sum(v1_vals) / len(v1_vals), 4),
        "v2_mean": round(sum(v2_vals) / len(v2_vals), 4),
        "tier_agreement": ag["exact"],
        "tier_agreement_pct": ag["pct"],
        "kappa": ag["kappa"],
        "catastrophic": ag["catastrophic"],
        "v1_n_high": sum(1 for t in v1_tiers if t == "High"),
        "v1_n_intermediate": sum(1 for t in v1_tiers if t == "Intermediate"),
        "v1_n_low": sum(1 for t in v1_tiers if t == "Low"),
    })

# Also do 2-tier (0.5) for comparison
print(f"\n\n{'--- 2-TIER (0.5 threshold) ---':}")
print(f"\n{'Config':<28} {'Agree':>6} {'%':>6} {'κ':>7} {'Catastr':>8}")
print("-" * 60)

for name, signals in configs:
    v1_scores = compute_vuln(v1_data, signals, common)
    v2_scores = compute_vuln(v2_data, signals, common)

    v1_tiers = ["High" if v1_scores[a] > 0.5 else "Low" for a in common]
    v2_tiers = ["High" if v2_scores[a] > 0.5 else "Low" for a in common]

    n = len(common)
    exact = sum(1 for a, b in zip(v1_tiers, v2_tiers) if a == b)
    hh = sum(1 for a, b in zip(v1_tiers, v2_tiers) if a == "High" and b == "High")
    hl = sum(1 for a, b in zip(v1_tiers, v2_tiers) if a == "High" and b == "Low")
    lh = sum(1 for a, b in zip(v1_tiers, v2_tiers) if a == "Low" and b == "High")
    ll = sum(1 for a, b in zip(v1_tiers, v2_tiers) if a == "Low" and b == "Low")
    po = (hh + ll) / n
    pe = ((hh + hl) * (hh + lh) + (lh + ll) * (hl + ll)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else 0
    print(f"{name:<28} {exact:>3}/50 {100*exact/n:>5.1f}% {k:>6.3f}")

# ─── V1 vs Baseline ──────────────────────────────────────────────────
bl_vuln = load_csv(f"{RESULTS}/baseline/analysis/vulnerability_scores.csv")
bl_data = {}
for r in bl_vuln:
    aid = r.get("artist_id", "")
    if not aid or aid == "artist_id":
        continue
    bl_data[aid] = {
        "clap": sf(r.get("clap_similarity")),
        "fad": sf(r.get("fad")),
        "musico": sf(r.get("musicological_similarity")),
        "ngram": sf(r.get("ngram_match_rate")),
        "name": r.get("artist_name", ""),
    }

common_bl = sorted(set(v1_data.keys()) & set(bl_data.keys()))
print(f"\n\n{'='*60}")
print(f"V1 vs BASELINE — 3-TIER (0.33/0.67)")
print(f"{'='*60}")
print(f"Common artists: {len(common_bl)}")
print(f"\n{'Config':<28} {'Agree':>6} {'%':>6} {'κ':>7} {'Catastr':>8}")
print("-" * 60)

for name, signals in configs:
    v1_scores = compute_vuln(v1_data, signals, common_bl)
    bl_scores = compute_vuln(bl_data, signals, common_bl)
    v1_tiers = [tier3(v1_scores[a]) for a in common_bl]
    bl_tiers = [tier3(bl_scores[a]) for a in common_bl]
    ag = compute_agreement(v1_tiers, bl_tiers)
    print(f"{name:<28} {ag['exact']:>3}/50 {ag['pct']:>5}% {ag['kappa']:>6.3f} {ag['catastrophic']:>8}")

print(f"\n{'--- V1 vs BASELINE — 2-TIER (0.5) ---':}")
print(f"\n{'Config':<28} {'Agree':>6} {'%':>6} {'κ':>7}")
print("-" * 60)

for name, signals in configs:
    v1_scores = compute_vuln(v1_data, signals, common_bl)
    bl_scores = compute_vuln(bl_data, signals, common_bl)
    v1_tiers = ["High" if v1_scores[a] > 0.5 else "Low" for a in common_bl]
    bl_tiers = ["High" if bl_scores[a] > 0.5 else "Low" for a in common_bl]
    n = len(common_bl)
    exact = sum(1 for a, b in zip(v1_tiers, bl_tiers) if a == b)
    hh = sum(1 for a, b in zip(v1_tiers, bl_tiers) if a == "High" and b == "High")
    hl = sum(1 for a, b in zip(v1_tiers, bl_tiers) if a == "High" and b == "Low")
    lh = sum(1 for a, b in zip(v1_tiers, bl_tiers) if a == "Low" and b == "High")
    ll = sum(1 for a, b in zip(v1_tiers, bl_tiers) if a == "Low" and b == "Low")
    po = (hh + ll) / n
    pe = ((hh + hl) * (hh + lh) + (lh + ll) * (hl + ll)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else 0
    print(f"{name:<28} {exact:>3}/50 {100*exact/n:>5.1f}% {k:>6.3f}")

# ─── V2 vs Baseline ──────────────────────────────────────────────────
common_v2bl = sorted(set(v2_data.keys()) & set(bl_data.keys()))
print(f"\n\n{'='*60}")
print(f"V2 vs BASELINE — 3-TIER (0.33/0.67)")
print(f"{'='*60}")
print(f"\n{'Config':<28} {'Agree':>6} {'%':>6} {'κ':>7} {'Catastr':>8}")
print("-" * 60)

for name, signals in configs:
    v2_scores = compute_vuln(v2_data, signals, common_v2bl)
    bl_scores = compute_vuln(bl_data, signals, common_v2bl)
    v2_tiers = [tier3(v2_scores[a]) for a in common_v2bl]
    bl_tiers = [tier3(bl_scores[a]) for a in common_v2bl]
    ag = compute_agreement(v2_tiers, bl_tiers)
    print(f"{name:<28} {ag['exact']:>3}/50 {ag['pct']:>5}% {ag['kappa']:>6.3f} {ag['catastrophic']:>8}")

print(f"\n{'--- V2 vs BASELINE — 2-TIER (0.5) ---':}")
print(f"\n{'Config':<28} {'Agree':>6} {'%':>6} {'κ':>7}")
print("-" * 60)

for name, signals in configs:
    v2_scores = compute_vuln(v2_data, signals, common_v2bl)
    bl_scores = compute_vuln(bl_data, signals, common_v2bl)
    v2_tiers = ["High" if v2_scores[a] > 0.5 else "Low" for a in common_v2bl]
    bl_tiers = ["High" if bl_scores[a] > 0.5 else "Low" for a in common_v2bl]
    n = len(common_v2bl)
    exact = sum(1 for a, b in zip(v2_tiers, bl_tiers) if a == b)
    hh = sum(1 for a, b in zip(v2_tiers, bl_tiers) if a == "High" and b == "High")
    hl = sum(1 for a, b in zip(v2_tiers, bl_tiers) if a == "High" and b == "Low")
    lh = sum(1 for a, b in zip(v2_tiers, bl_tiers) if a == "Low" and b == "High")
    ll = sum(1 for a, b in zip(v2_tiers, bl_tiers) if a == "Low" and b == "Low")
    po = (hh + ll) / n
    pe = ((hh + hl) * (hh + lh) + (lh + ll) * (hl + ll)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else 0
    print(f"{name:<28} {exact:>3}/50 {100*exact/n:>5.1f}% {k:>6.3f}")

# Save
out_path = os.path.join(RESULTS, "robustness", "ablation_tier_agreement.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[SAVED] {out_path}")
