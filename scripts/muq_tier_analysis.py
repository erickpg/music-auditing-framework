#!/usr/bin/env python3
"""3-tier agreement analysis across MuQ-MuLan and CLAP."""
import csv, json, os

RESULTS = os.environ.get("RESULTS_DIR", "/Users/erickpg/capstone/results")
MUQ_DIR = os.environ.get("MUQ_DIR", os.path.join(RESULTS, "muq_validation"))
ROB_DIR = os.environ.get("ROB_DIR", os.path.join(RESULTS, "robustness"))

rows = []
with open(os.path.join(MUQ_DIR, "muq_per_artist_comparison.csv")) as f:
    for r in csv.DictReader(f):
        rows.append(r)

bl_rows = []
with open(os.path.join(ROB_DIR, "baseline_catalog_property_per_artist.csv")) as f:
    for r in csv.DictReader(f):
        bl_rows.append(r)

clap_bl_lookup = {r["artist_id"]: float(r["bl_vuln_2sig"]) for r in bl_rows}
clap_v1_lookup = {r["artist_id"]: float(r["v1_vuln_2sig"]) for r in bl_rows}

# Also get artist names
name_lookup = {r["artist_id"]: r.get("artist_name", r["artist_id"]) for r in bl_rows}


def tier3(score, lo=0.33, hi=0.67):
    if score >= hi:
        return "High"
    elif score <= lo:
        return "Low"
    else:
        return "Intermediate"


def agreement_stats(tiers_a, tiers_b, label):
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

    print(f"\n{label}:")
    print(f"  Exact agreement: {exact}/{n} ({100*exact/n:.0f}%)")
    print(f"  Catastrophic (High<->Low): {catastrophic}/{n}")
    print(f"  Kappa: {k:.3f}")

    header = "            " + "  ".join(f"{c:>12}" for c in cats)
    print(header)
    for ca in cats:
        row_vals = [sum(1 for a, b in zip(tiers_a, tiers_b) if a == ca and b == cb) for cb in cats]
        print(f"  {ca:>10}: " + "  ".join(f"{v:>12}" for v in row_vals))

    return {"exact": exact, "pct": round(100 * exact / n, 1),
            "catastrophic": catastrophic, "kappa": round(k, 3)}


for lo, hi, scheme in [(0.33, 0.67, "0.33/0.67"), (0.4, 0.6, "0.40/0.60")]:
    print("\n" + "=" * 70)
    print(f"3-TIER SCHEME: {scheme}")
    print("=" * 70)

    muq_v1 = [tier3(float(r["muq_vuln_v1"]), lo, hi) for r in rows]
    clap_v1 = [tier3(clap_v1_lookup.get(r["artist_id"], 0.5), lo, hi) for r in rows]
    muq_bl = [tier3(float(r["muq_vuln_bl"]), lo, hi) for r in rows]
    clap_bl = [tier3(clap_bl_lookup.get(r["artist_id"], 0.5), lo, hi) for r in rows]

    # Within-embedding comparisons
    agreement_stats(muq_v1, muq_bl, "Within MuQ: V1 vs Baseline")
    agreement_stats(clap_v1, clap_bl, "Within CLAP: V1 vs Baseline")

    # Cross-embedding comparisons
    agreement_stats(muq_v1, clap_v1, "Cross-embedding: MuQ-V1 vs CLAP-V1")
    agreement_stats(muq_bl, clap_bl, "Cross-embedding: MuQ-BL vs CLAP-BL")

    # Four-way
    all4 = sum(1 for a, b, c, d in zip(muq_v1, muq_bl, clap_v1, clap_bl) if a == b == c == d)
    ah = sum(1 for a, b, c, d in zip(muq_v1, muq_bl, clap_v1, clap_bl) if a == b == c == d == "High")
    ai = sum(1 for a, b, c, d in zip(muq_v1, muq_bl, clap_v1, clap_bl) if a == b == c == d == "Intermediate")
    al = sum(1 for a, b, c, d in zip(muq_v1, muq_bl, clap_v1, clap_bl) if a == b == c == d == "Low")
    print(f"\nFour-way consensus: {all4}/50 ({100*all4/50:.0f}%)")
    print(f"  Always High: {ah}")
    print(f"  Always Intermediate: {ai}")
    print(f"  Always Low: {al}")
    print(f"  Disagreed: {50 - all4}")

    print("\nConsensus High artists:")
    for r, a, b, c, d in zip(rows, muq_v1, muq_bl, clap_v1, clap_bl):
        if a == b == c == d == "High":
            print(f"  {name_lookup.get(r['artist_id'], r['artist_id'])}")

    print("\nConsensus Low artists:")
    for r, a, b, c, d in zip(rows, muq_v1, muq_bl, clap_v1, clap_bl):
        if a == b == c == d == "Low":
            print(f"  {name_lookup.get(r['artist_id'], r['artist_id'])}")

    print("\nConsensus Intermediate artists:")
    for r, a, b, c, d in zip(rows, muq_v1, muq_bl, clap_v1, clap_bl):
        if a == b == c == d == "Intermediate":
            print(f"  {name_lookup.get(r['artist_id'], r['artist_id'])}")
