#!/usr/bin/env python3
"""Per-tier stability: what % of High artists stay High, Low stay Low, etc."""
import csv, os

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


def compute_vuln(data, signals, artists):
    raw = {sig: [] for sig in signals}
    for aid in artists:
        for sig in signals:
            v = data[aid].get(sig)
            if v is not None:
                raw[sig].append(v)
    bounds = {}
    for sig in signals:
        vals = raw[sig]
        bounds[sig] = (min(vals), max(vals)) if vals else (0, 1)
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
                norm = 1.0 - norm
            components.append(norm)
        scores[aid] = sum(components) / len(components) if components else 0.5
    return scores


# Load data
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


v1_data = load_version("v1")
v2_data = load_version("v2")
bl_data = load_version("baseline")

signals = ["clap", "fad"]
common = sorted(set(v1_data.keys()) & set(v2_data.keys()) & set(bl_data.keys()))


def per_tier_stability(scores_a, scores_b, artists, label):
    tiers_a = {a: tier3(scores_a[a]) for a in artists}
    tiers_b = {a: tier3(scores_b[a]) for a in artists}

    print(f"\n{label}:")
    print(f"{'Tier in A':<15} {'N':>4}  {'→High':>6} {'→Inter':>6} {'→Low':>6}  {'Stay%':>6}")
    print("-" * 55)

    for tier in ["High", "Intermediate", "Low"]:
        in_tier = [a for a in artists if tiers_a[a] == tier]
        n = len(in_tier)
        if n == 0:
            continue
        to_high = sum(1 for a in in_tier if tiers_b[a] == "High")
        to_inter = sum(1 for a in in_tier if tiers_b[a] == "Intermediate")
        to_low = sum(1 for a in in_tier if tiers_b[a] == "Low")
        stay = sum(1 for a in in_tier if tiers_b[a] == tier)
        print(f"{tier:<15} {n:>4}  {to_high:>6} {to_inter:>6} {to_low:>6}  {100*stay/n:>5.0f}%")

        # List the ones that moved
        moved = [(a, v1_data[a]["name"], tiers_b[a]) for a in in_tier if tiers_b[a] != tier]
        if moved:
            for aid, name, new_tier in moved:
                print(f"    {name[:30]:<30} → {new_tier}")


v1_scores = compute_vuln(v1_data, signals, common)
v2_scores = compute_vuln(v2_data, signals, common)
bl_scores = compute_vuln(bl_data, signals, common)

per_tier_stability(v1_scores, v2_scores, common, "V1 → V2 (2-signal CLAP+FAD, 3-tier)")
per_tier_stability(v1_scores, bl_scores, common, "V1 → Baseline (2-signal CLAP+FAD, 3-tier)")
per_tier_stability(v2_scores, bl_scores, common, "V2 → Baseline (2-signal CLAP+FAD, 3-tier)")

# Also do it for the baseline catalog property CSV which has pre-computed 2-signal scores
print("\n\n" + "=" * 60)
print("FROM PRE-COMPUTED 2-SIGNAL SCORES (baseline_catalog_property)")
print("=" * 60)

bp_rows = load_csv(f"{RESULTS}/robustness/baseline_catalog_property_per_artist.csv")


def per_tier_from_precomputed(col_a, col_b, label):
    tiers_a = {r["artist_id"]: tier3(float(r[col_a])) for r in bp_rows}
    tiers_b = {r["artist_id"]: tier3(float(r[col_b])) for r in bp_rows}
    names = {r["artist_id"]: r["artist_name"] for r in bp_rows}
    artists = list(tiers_a.keys())

    print(f"\n{label}:")
    print(f"{'Tier in A':<15} {'N':>4}  {'→High':>6} {'→Inter':>6} {'→Low':>6}  {'Stay%':>6}")
    print("-" * 55)

    for tier in ["High", "Intermediate", "Low"]:
        in_tier = [a for a in artists if tiers_a[a] == tier]
        n = len(in_tier)
        if n == 0:
            continue
        to_high = sum(1 for a in in_tier if tiers_b[a] == "High")
        to_inter = sum(1 for a in in_tier if tiers_b[a] == "Intermediate")
        to_low = sum(1 for a in in_tier if tiers_b[a] == "Low")
        stay = sum(1 for a in in_tier if tiers_b[a] == tier)
        print(f"{tier:<15} {n:>4}  {to_high:>6} {to_inter:>6} {to_low:>6}  {100*stay/n:>5.0f}%")

        moved = [(a, names[a], tiers_b[a]) for a in in_tier if tiers_b[a] != tier]
        if moved:
            for aid, name, new_tier in moved:
                print(f"    {name[:30]:<30} → {new_tier}")


per_tier_from_precomputed("v1_vuln_2sig", "v2_vuln_2sig", "V1 → V2")
per_tier_from_precomputed("v1_vuln_2sig", "bl_vuln_2sig", "V1 → Baseline")
per_tier_from_precomputed("v2_vuln_2sig", "bl_vuln_2sig", "V2 → Baseline")
per_tier_from_precomputed("bl_vuln_2sig", "v1_vuln_2sig", "Baseline → V1")
per_tier_from_precomputed("bl_vuln_2sig", "v2_vuln_2sig", "Baseline → V2")
