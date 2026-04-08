#!/usr/bin/env python3
"""Cross-validation with 3-tier thresholds (0.33/0.67)."""
import csv, json, os, random

RESULTS = os.environ.get("RESULTS_DIR", "/Users/erickpg/capstone/results")
OUT_DIR = os.path.join(RESULTS, "robustness")
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS = 5
SEED = 42


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


def compute_2sig(clap_dict, fad_dict, artists):
    """Compute 2-signal vuln from per-artist CLAP sim and FAD."""
    clap_vals = [clap_dict[a] for a in artists if a in clap_dict]
    fad_vals = [fad_dict[a] for a in artists if a in fad_dict]
    c_min, c_max = min(clap_vals), max(clap_vals)
    f_min, f_max = min(fad_vals), max(fad_vals)
    scores = {}
    for a in artists:
        if a not in clap_dict or a not in fad_dict:
            continue
        cn = (clap_dict[a] - c_min) / (c_max - c_min + 1e-10)
        fn = 1.0 - (fad_dict[a] - f_min) / (f_max - f_min + 1e-10)
        scores[a] = max(0, min(1, 0.5 * cn + 0.5 * fn))
    return scores


for version, subdir in [("V1", "v1"), ("V2", "v2")]:
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION: {version} (3-tier, 0.33/0.67)")
    print(f"{'='*60}")

    vuln_rows = load_csv(f"{RESULTS}/{subdir}/analysis/vulnerability_scores.csv")

    clap = {}
    fad = {}
    names = {}
    for r in vuln_rows:
        aid = r.get("artist_id", "")
        if not aid or aid == "artist_id":
            continue
        clap[aid] = sf(r.get("clap_similarity"))
        fad[aid] = sf(r.get("fad"))
        names[aid] = r.get("artist_name", aid)

    artists = sorted(clap.keys())
    n = len(artists)

    # Full-sample scores and tiers
    full_scores = compute_2sig(clap, fad, artists)
    full_tiers = {a: tier3(full_scores[a]) for a in artists}

    # Shuffle and split into folds
    rng = random.Random(SEED)
    shuffled = list(artists)
    rng.shuffle(shuffled)
    fold_size = n // N_FOLDS

    cv_predictions = {}
    fold_agreements_3tier = []
    fold_agreements_2tier = []

    for fold_idx in range(N_FOLDS):
        start = fold_idx * fold_size
        end = start + fold_size if fold_idx < N_FOLDS - 1 else n
        test_set = set(shuffled[start:end])
        train_set = [a for a in artists if a not in test_set]

        # Recompute normalization using only training set
        train_scores = compute_2sig(clap, fad, train_set)

        # Score test set using train normalization bounds
        train_clap = [clap[a] for a in train_set]
        train_fad = [fad[a] for a in train_set]
        c_min, c_max = min(train_clap), max(train_clap)
        f_min, f_max = min(train_fad), max(train_fad)

        for a in test_set:
            cn = (clap[a] - c_min) / (c_max - c_min + 1e-10)
            fn = 1.0 - (fad[a] - f_min) / (f_max - f_min + 1e-10)
            cn = max(0, min(1, cn))
            fn = max(0, min(1, fn))
            cv_score = 0.5 * cn + 0.5 * fn

            cv_predictions[a] = {
                "cv_score": cv_score,
                "cv_tier_3": tier3(cv_score),
                "cv_tier_2": "High" if cv_score >= 0.5 else "Low",
                "full_score": full_scores[a],
                "full_tier_3": full_tiers[a],
                "full_tier_2": "High" if full_scores[a] >= 0.5 else "Low",
                "fold": fold_idx,
            }

        # Fold agreement
        fold_3 = sum(1 for a in test_set if cv_predictions[a]["cv_tier_3"] == full_tiers[a])
        fold_2 = sum(1 for a in test_set if cv_predictions[a]["cv_tier_2"] == cv_predictions[a]["full_tier_2"])
        fold_agreements_3tier.append(fold_3 / len(test_set))
        fold_agreements_2tier.append(fold_2 / len(test_set))

    # Overall
    agree_3 = sum(1 for a in artists if cv_predictions[a]["cv_tier_3"] == full_tiers[a])
    agree_2 = sum(1 for a in artists if cv_predictions[a]["cv_tier_2"] == cv_predictions[a]["full_tier_2"])

    # Score MAE
    mae = sum(abs(cv_predictions[a]["cv_score"] - full_scores[a]) for a in artists) / n

    # Flipped artists (3-tier)
    flipped_3 = [(a, cv_predictions[a]) for a in artists
                 if cv_predictions[a]["cv_tier_3"] != full_tiers[a]]
    flipped_2 = [(a, cv_predictions[a]) for a in artists
                 if cv_predictions[a]["cv_tier_2"] != cv_predictions[a]["full_tier_2"]]

    print(f"\n  3-tier agreement: {agree_3}/{n} ({100*agree_3/n:.0f}%)")
    print(f"  2-tier agreement: {agree_2}/{n} ({100*agree_2/n:.0f}%)")
    print(f"  MAE: {mae:.4f}")
    print(f"  Per-fold 3-tier: {[f'{a:.0%}' for a in fold_agreements_3tier]}")
    print(f"  Per-fold 2-tier: {[f'{a:.0%}' for a in fold_agreements_2tier]}")

    if flipped_3:
        print(f"\n  Flipped (3-tier): {len(flipped_3)} artists")
        for a, p in sorted(flipped_3, key=lambda x: -abs(x[1]["cv_score"] - x[1]["full_score"])):
            print(f"    {names[a][:25]:<25} full={p['full_score']:.3f} ({p['full_tier_3']}) "
                  f"-> cv={p['cv_score']:.3f} ({p['cv_tier_3']})")
    else:
        print(f"\n  Flipped (3-tier): 0 artists")

    if flipped_2:
        print(f"  Flipped (2-tier): {len(flipped_2)} artists")
        for a, p in sorted(flipped_2, key=lambda x: -abs(x[1]["cv_score"] - x[1]["full_score"])):
            print(f"    {names[a][:25]:<25} full={p['full_score']:.3f} ({p['full_tier_2']}) "
                  f"-> cv={p['cv_score']:.3f} ({p['cv_tier_2']})")
