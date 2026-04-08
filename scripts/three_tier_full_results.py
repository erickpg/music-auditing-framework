#!/usr/bin/env python3
"""Compute all results using 3-tier (0.33/0.67) scheme for the new reference document."""
import csv, json, os

RESULTS = "/Users/erickpg/capstone/results"


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


def kappa_3(tiers_a, tiers_b):
    n = len(tiers_a)
    exact = sum(1 for a, b in zip(tiers_a, tiers_b) if a == b)
    cats = ["High", "Intermediate", "Low"]
    po = exact / n
    pe = sum(
        (sum(1 for t in tiers_a if t == c) / n) * (sum(1 for t in tiers_b if t == c) / n)
        for c in cats
    )
    k = (po - pe) / (1 - pe) if pe < 1 else 0
    catastrophic = sum(1 for a, b in zip(tiers_a, tiers_b) if {a, b} == {"High", "Low"})
    return {"exact": exact, "pct": round(100 * exact / n, 1), "kappa": round(k, 3),
            "catastrophic": catastrophic, "n": n}


def compute_2sig_vuln(data, artists):
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


# Load all versions
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
            "genre": r.get("genre", ""),
            "n_tracks": sf(r.get("n_catalog_tracks")),
        }
    return data


v1 = load_version("v1")
v2 = load_version("v2")
bl = load_version("baseline")
common = sorted(set(v1.keys()) & set(v2.keys()) & set(bl.keys()))

v1_scores = compute_2sig_vuln(v1, common)
v2_scores = compute_2sig_vuln(v2, common)
bl_scores = compute_2sig_vuln(bl, common)

v1_tiers = {a: tier3(v1_scores[a]) for a in common}
v2_tiers = {a: tier3(v2_scores[a]) for a in common}
bl_tiers = {a: tier3(bl_scores[a]) for a in common}

out = {}

# === 1. Score summary per version ===
for label, scores in [("V1", v1_scores), ("V2", v2_scores), ("Baseline", bl_scores)]:
    vals = [scores[a] for a in common]
    tiers = [tier3(s) for s in vals]
    out[f"summary_{label}"] = {
        "mean": round(sum(vals)/len(vals), 4),
        "std": round((sum((v - sum(vals)/len(vals))**2 for v in vals) / (len(vals)-1))**0.5, 4),
        "median": round(sorted(vals)[len(vals)//2], 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "n_high": sum(1 for t in tiers if t == "High"),
        "n_intermediate": sum(1 for t in tiers if t == "Intermediate"),
        "n_low": sum(1 for t in tiers if t == "Low"),
    }
    print(f"\n{label}: {out[f'summary_{label}']}")

# === 2. Pairwise tier agreement ===
pairs = [("V1", "V2", v1_tiers, v2_tiers),
         ("V1", "Baseline", v1_tiers, bl_tiers),
         ("V2", "Baseline", v2_tiers, bl_tiers)]

print("\n\n=== PAIRWISE TIER AGREEMENT (3-tier, 0.33/0.67) ===")
for la, lb, ta, tb in pairs:
    tiers_a = [ta[a] for a in common]
    tiers_b = [tb[a] for a in common]
    ag = kappa_3(tiers_a, tiers_b)
    out[f"agreement_{la}_{lb}"] = ag
    print(f"{la} vs {lb}: {ag['exact']}/50 ({ag['pct']}%), κ={ag['kappa']}, catastrophic={ag['catastrophic']}")

# === 3. Per-tier retention (BL→V2 focus) ===
print("\n\n=== PER-TIER RETENTION: Baseline → V2 ===")
for tier in ["High", "Intermediate", "Low"]:
    in_tier = [a for a in common if bl_tiers[a] == tier]
    n = len(in_tier)
    if n == 0:
        continue
    to = {t: sum(1 for a in in_tier if v2_tiers[a] == t) for t in ["High", "Intermediate", "Low"]}
    stay = to[tier]
    print(f"  {tier}: {n} artists, {stay} stay ({100*stay/n:.0f}%), → H:{to['High']} I:{to['Intermediate']} L:{to['Low']}")

# === 4. Three-way agreement ===
print("\n\n=== THREE-WAY AGREEMENT ===")
all3 = sum(1 for a in common if v1_tiers[a] == v2_tiers[a] == bl_tiers[a])
ah = [a for a in common if v1_tiers[a] == v2_tiers[a] == bl_tiers[a] == "High"]
ai = [a for a in common if v1_tiers[a] == v2_tiers[a] == bl_tiers[a] == "Intermediate"]
al = [a for a in common if v1_tiers[a] == v2_tiers[a] == bl_tiers[a] == "Low"]
print(f"All agree: {all3}/50 ({100*all3/50:.0f}%)")
print(f"Always High ({len(ah)}): {[v1[a]['name'] for a in ah]}")
print(f"Always Intermediate ({len(ai)}): {[v1[a]['name'] for a in ai]}")
print(f"Always Low ({len(al)}): {[v1[a]['name'] for a in al]}")

out["three_way"] = {
    "all_agree": all3,
    "always_high": len(ah),
    "always_high_artists": [v1[a]["name"] for a in ah],
    "always_intermediate": len(ai),
    "always_intermediate_artists": [v1[a]["name"] for a in ai],
    "always_low": len(al),
    "always_low_artists": [v1[a]["name"] for a in al],
}

# === 5. Top/bottom artists per version (3-tier) ===
for label, scores in [("V1", v1_scores), ("V2", v2_scores), ("BL", bl_scores)]:
    ranked = sorted(common, key=lambda a: scores[a], reverse=True)
    print(f"\n{label} Top 5: {[(v1[a]['name'], round(scores[a],3), tier3(scores[a])) for a in ranked[:5]]}")
    print(f"{label} Bottom 5: {[(v1[a]['name'], round(scores[a],3), tier3(scores[a])) for a in ranked[-5:]]}")

# === 6. Bootstrap CIs — reinterpret for 3-tier ===
bp_path = f"{RESULTS}/robustness/bootstrap_ci.csv"
if os.path.exists(bp_path):
    bp_rows = load_csv(bp_path)
    print("\n\n=== BOOTSTRAP CIs (3-tier interpretation) ===")
    clearly_high = 0
    clearly_low = 0
    clearly_inter = 0
    ambiguous = 0
    for r in bp_rows:
        lo = sf(r.get("ci_lower") or r.get("lower") or r.get("ci_low"))
        hi = sf(r.get("ci_upper") or r.get("upper") or r.get("ci_high"))
        score = sf(r.get("score") or r.get("vulnerability_score") or r.get("mean"))
        if lo is None or hi is None:
            continue
        if lo >= 0.67:
            clearly_high += 1
        elif hi <= 0.33:
            clearly_low += 1
        elif lo >= 0.33 and hi <= 0.67:
            clearly_inter += 1
        else:
            ambiguous += 1
    print(f"Clearly High (CI lower ≥ 0.67): {clearly_high}")
    print(f"Clearly Intermediate (CI within 0.33-0.67): {clearly_inter}")
    print(f"Clearly Low (CI upper ≤ 0.33): {clearly_low}")
    print(f"Ambiguous (CI crosses tier boundary): {ambiguous}")
    out["bootstrap_3tier"] = {
        "clearly_high": clearly_high,
        "clearly_intermediate": clearly_inter,
        "clearly_low": clearly_low,
        "ambiguous": ambiguous,
    }

# === 7. Cross-validation — reinterpret for 3-tier ===
cv_path = f"{RESULTS}/robustness/cross_validation.json"
if os.path.exists(cv_path):
    cv = json.load(open(cv_path))
    print(f"\n=== CROSS-VALIDATION ===")
    print(json.dumps(cv, indent=2))

# Save
with open(f"{RESULTS}/robustness/three_tier_full_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[SAVED] {RESULTS}/robustness/three_tier_full_results.json")
