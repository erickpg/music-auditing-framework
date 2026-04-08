#!/usr/bin/env python3
"""Genre-controlled vulnerability analysis: Options B (z-score) + C (genre-matched)."""
import csv, json, os
from collections import defaultdict
from scipy import stats
import numpy as np

SCRATCH = "/scratch/$USER/runs"
FT_DIR = f"{SCRATCH}/2026-03-10_full/analysis"
BL_DIR = f"{SCRATCH}/2026-03-10_baseline/analysis"
OUT_DIR = f"{SCRATCH}/2026-03-10_full/supplementary"
os.makedirs(OUT_DIR, exist_ok=True)

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(v == k for k, v in row.items()):
                continue
            rows.append(row)
    return rows

def sf(v):
    try: return float(v)
    except: return None

# Get genre from ngram data
ft_ngram_rows = load_csv(f"{FT_DIR}/ngram_per_artist.csv")
artist_genre = {}
for row in ft_ngram_rows:
    aid = row.get("artist_id", "")
    g = row.get("genre", "")
    if aid and g:
        artist_genre[aid] = g

# Load vulnerability scores
def load_vuln(path):
    rows = load_csv(path)
    artists = {}
    for row in rows:
        aid = row.get("artist_id", "")
        if not aid: continue
        artists[aid] = {
            "name": row.get("artist_name", ""),
            "genre": artist_genre.get(aid, row.get("genre", "") or "Unknown"),
            "n_catalog": sf(row.get("n_catalog_tracks", 0)) or 0,
            "clap": sf(row.get("clap_similarity")),
            "fad": sf(row.get("fad")),
            "musico": sf(row.get("musicological_similarity")),
            "ngram": sf(row.get("ngram_match_rate")),
            "vuln": sf(row.get("vulnerability_score")),
        }
    return artists

ft = load_vuln(f"{FT_DIR}/vulnerability_scores.csv")
bl = load_vuln(f"{BL_DIR}/vulnerability_scores.csv")
print(f"FT: {len(ft)}, BL: {len(bl)}")

# Load per-file CLAP
ft_clap = load_csv(f"{FT_DIR}/clap_per_artist.csv")
bl_clap = load_csv(f"{BL_DIR}/clap_per_artist.csv")

# ============================================================
# OPTION C: N-gram genre-matched analysis
# ============================================================
print("=" * 60)
print("OPTION C: N-gram Genre-Matched Comparison")
print("=" * 60)

def ngram_genre_analysis(rows, label):
    artist_data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        aid = row.get("artist_id", "")
        ns = row.get("ngram_size", "")
        matched = sf(row.get("matched_rate"))
        mismatched_genre = sf(row.get("mismatched_rate"))
        mismatched_all = sf(row.get("mismatched_all_rate"))
        genre = row.get("genre", "") or artist_genre.get(aid, "Unknown")
        if aid and ns and matched is not None:
            artist_data[aid][ns].append({
                "matched": matched, "mismatched_genre": mismatched_genre,
                "mismatched_all": mismatched_all, "genre": genre,
            })

    print(f"\n{label}:")
    for ns in ["3", "4", "5"]:
        matched_rates, mg_rates, ma_rates, genres = [], [], [], []
        for aid in sorted(artist_data.keys()):
            if ns in artist_data[aid]:
                entries = artist_data[aid][ns]
                m = np.mean([e["matched"] for e in entries if e["matched"] is not None])
                mg = np.mean([e["mismatched_genre"] for e in entries if e["mismatched_genre"] is not None])
                ma = np.mean([e["mismatched_all"] for e in entries if e["mismatched_all"] is not None])
                matched_rates.append(m)
                mg_rates.append(mg)
                ma_rates.append(ma)
                genres.append(entries[0]["genre"])

        if matched_rates:
            gap_genre = [m - mg for m, mg in zip(matched_rates, mg_rates)]
            gap_all = [m - ma for m, ma in zip(matched_rates, ma_rates)]

            print(f"\n  N-gram size {ns} (n={len(matched_rates)}):")
            print(f"    Mean matched rate:            {np.mean(matched_rates):.8f}")
            print(f"    Mean mismatched (same genre):  {np.mean(mg_rates):.8f}")
            print(f"    Mean mismatched (all artists): {np.mean(ma_rates):.8f}")
            print(f"    Gap (matched - same_genre):    {np.mean(gap_genre):+.8f}")
            print(f"    Gap (matched - all):           {np.mean(gap_all):+.8f}")

            if np.std(gap_genre) > 0 and np.std(gap_all) > 0:
                t, p = stats.ttest_rel(gap_genre, gap_all)
                print(f"    Paired t-test (genre gap vs all gap): t={t:.3f}, p={p:.4f}")

            # By genre (n >= 3 only)
            genre_gaps = defaultdict(list)
            for g, gg in zip(genres, gap_genre):
                genre_gaps[g].append(gg)
            print(f"    Per-genre gap (matched - same_genre):")
            for g in sorted(genre_gaps.keys()):
                vals = genre_gaps[g]
                if len(vals) >= 3:
                    print(f"      {g:12s} (n={len(vals):2d}): mean={np.mean(vals):+.8f}, std={np.std(vals):.8f}")
                else:
                    print(f"      {g:12s} (n={len(vals):2d}): mean={np.mean(vals):+.8f} [too few]")

ngram_genre_analysis(load_csv(f"{FT_DIR}/ngram_per_artist.csv"), "Fine-tuned")
ngram_genre_analysis(load_csv(f"{BL_DIR}/ngram_per_artist.csv"), "Baseline")

# ============================================================
# OPTION B: Within-Genre Z-Scores
# ============================================================
print("\n" + "=" * 60)
print("OPTION B: Within-Genre Z-Score Normalization")
print("=" * 60)

for label, artists in [("FT", ft), ("BL", bl)]:
    print(f"\n--- {label} ---")
    genre_groups = defaultdict(list)
    for aid, data in artists.items():
        genre_groups[data["genre"]].append((aid, data))

    z_scored = []
    for genre, group in sorted(genre_groups.items()):
        vulns = [(aid, d["vuln"]) for aid, d in group if d["vuln"] is not None]
        claps = [(aid, d["clap"]) for aid, d in group if d["clap"] is not None]

        if len(vulns) >= 3:
            mean_v = np.mean([v for _, v in vulns])
            std_v = np.std([v for _, v in vulns])
            mean_c = np.mean([c for _, c in claps])
            std_c = np.std([c for _, c in claps])

            print(f"\n  {genre} (n={len(vulns)}, vuln mean={mean_v:.4f}, std={std_v:.4f}):")
            print(f"    {'Artist':<30s} {'Raw Vuln':>10s} {'Z-Score':>10s} {'Raw CLAP':>10s} {'CLAP Z':>10s}")

            artist_z = []
            for aid, v in vulns:
                z_v = (v - mean_v) / (std_v + 1e-10)
                c = artists[aid]["clap"] or 0
                z_c = (c - mean_c) / (std_c + 1e-10) if std_c > 0 else 0
                artist_z.append((aid, artists[aid]["name"], v, z_v, c, z_c, genre))

            artist_z.sort(key=lambda x: x[3], reverse=True)
            for aid, name, v, z_v, c, z_c, g in artist_z:
                marker = " <<<" if abs(z_v) > 1.5 else ""
                print(f"    {name[:29]:<30s} {v:10.4f} {z_v:+10.3f} {c:10.4f} {z_c:+10.3f}{marker}")
                z_scored.append({"artist_id": aid, "name": name, "genre": g,
                    "raw_vuln": v, "vuln_z": z_v, "raw_clap": c, "clap_z": z_c})
        else:
            print(f"\n  {genre} (n={len(vulns)}): SKIPPED (n < 3)")
            for aid, d in group:
                z_scored.append({"artist_id": aid, "name": d["name"], "genre": genre,
                    "raw_vuln": d["vuln"], "vuln_z": None, "raw_clap": d["clap"], "clap_z": None})

    fname = f"genre_zscore_{label.lower()}.csv"
    with open(f"{OUT_DIR}/{fname}", "w") as f:
        w = csv.writer(f)
        w.writerow(["artist_id","artist_name","genre","raw_vulnerability","vulnerability_z_score","raw_clap","clap_z_score"])
        for d in sorted(z_scored, key=lambda x: x.get("vuln_z") or -999, reverse=True):
            w.writerow([d["artist_id"], d["name"], d["genre"],
                f"{d['raw_vuln']:.6f}" if d["raw_vuln"] is not None else "",
                f"{d['vuln_z']:.4f}" if d["vuln_z"] is not None else "N/A",
                f"{d['raw_clap']:.6f}" if d["raw_clap"] is not None else "",
                f"{d['clap_z']:.4f}" if d["clap_z"] is not None else "N/A"])
    print(f"\n  [SAVED] {fname}")

# ============================================================
# OPTION C: CLAP within-genre rankings
# ============================================================
print("\n" + "=" * 60)
print("OPTION C: CLAP Within-Genre Rankings")
print("=" * 60)

for label, clap_rows in [("FT", ft_clap), ("BL", bl_clap)]:
    print(f"\n--- {label} ---")
    artist_matched = defaultdict(list)
    artist_gap = defaultdict(list)
    artist_genres = {}

    for row in clap_rows:
        aid = row.get("artist_id", "")
        if not aid: continue
        m = sf(row.get("matched_mean_sim"))
        g = sf(row.get("sim_gap"))
        genre = row.get("genre", "") or artist_genre.get(aid, "Unknown")
        if m is not None: artist_matched[aid].append(m)
        if g is not None: artist_gap[aid].append(g)
        artist_genres[aid] = genre

    genre_data = defaultdict(list)
    for aid in artist_matched:
        g = artist_genres.get(aid, "Unknown")
        genre_data[g].append({
            "artist_id": aid,
            "name": ft.get(aid, bl.get(aid, {})).get("name", aid),
            "mean_matched": np.mean(artist_matched[aid]),
            "mean_gap": np.mean(artist_gap[aid]) if aid in artist_gap else None,
        })

    for genre in sorted(genre_data.keys()):
        group = genre_data[genre]
        if len(group) >= 3:
            group.sort(key=lambda x: x["mean_matched"], reverse=True)
            vals = [d["mean_matched"] for d in group]
            mean_m, std_m = np.mean(vals), np.std(vals)

            print(f"\n  {genre} (n={len(group)}, mean_matched={mean_m:.4f}, std={std_m:.4f}):")
            print(f"    {'Artist':<30s} {'Matched':>10s} {'Gap':>10s} {'Z-Score':>10s}")
            for d in group:
                z = (d["mean_matched"] - mean_m) / (std_m + 1e-10)
                marker = " <<<" if abs(z) > 1.5 else ""
                print(f"    {d['name'][:29]:<30s} {d['mean_matched']:10.4f} {d['mean_gap'] or 0:+10.4f} {z:+10.3f}{marker}")

# ============================================================
# STABILITY: Raw vs Genre-Controlled
# ============================================================
print("\n" + "=" * 60)
print("STABILITY: Raw vs Genre-Controlled Rankings (Rock)")
print("=" * 60)

for label, artists in [("FT", ft), ("BL", bl)]:
    rock = [(aid, d) for aid, d in artists.items() if d["genre"] == "Rock" and d["vuln"] is not None]
    if len(rock) >= 5:
        raw_sorted = sorted(rock, key=lambda x: x[1]["vuln"], reverse=True)
        raw_ranks = {aid: i+1 for i, (aid, _) in enumerate(raw_sorted)}
        clap_sorted = sorted(rock, key=lambda x: x[1]["clap"] or 0, reverse=True)
        clap_ranks = {aid: i+1 for i, (aid, _) in enumerate(clap_sorted)}

        aids = sorted(raw_ranks.keys())
        rho, p = stats.spearmanr([raw_ranks[a] for a in aids], [clap_ranks[a] for a in aids])

        print(f"\n  {label} Rock (n={len(rock)}):")
        print(f"    Vuln ranking vs CLAP-only ranking: rho={rho:.3f}, p={p:.4f}")
        print(f"    Top 5:")
        for i, (aid, d) in enumerate(raw_sorted[:5], 1):
            print(f"      {i}. {d['name'][:30]} vuln={d['vuln']:.4f} clap={d['clap'] or 0:.4f}")
        print(f"    Bottom 5:")
        for aid, d in raw_sorted[-5:]:
            print(f"      {raw_ranks[aid]}. {d['name'][:30]} vuln={d['vuln']:.4f} clap={d['clap'] or 0:.4f}")

# ============================================================
# Rock vs Non-Rock
# ============================================================
print("\n" + "=" * 60)
print("Rock vs Non-Rock Comparison")
print("=" * 60)

for label, artists in [("FT", ft), ("BL", bl)]:
    rock_v = [d["vuln"] for d in artists.values() if d["genre"] == "Rock" and d["vuln"] is not None]
    nonrock_v = [d["vuln"] for d in artists.values() if d["genre"] != "Rock" and d["vuln"] is not None]
    if rock_v and nonrock_v:
        t, p = stats.ttest_ind(rock_v, nonrock_v, equal_var=False)
        d_eff = (np.mean(rock_v) - np.mean(nonrock_v)) / np.sqrt((np.std(rock_v)**2 + np.std(nonrock_v)**2) / 2)
        print(f"\n  {label}: Rock (n={len(rock_v)}, mean={np.mean(rock_v):.4f}) vs Non-Rock (n={len(nonrock_v)}, mean={np.mean(nonrock_v):.4f})")
        print(f"    Welch's t={t:.3f}, p={p:.4f}, Cohen's d={d_eff:.3f}")

print("\nDONE")
