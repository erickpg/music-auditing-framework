#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=v2_resume
#SBATCH --output=/home/$USER/slurm_analysis_v2_resume_%j.out

# Resume V2 analysis from Step 4 (CLAP) onward
# Steps 1-3 already completed by job 1073

set -euo pipefail

source $HOME/miniforge3/bin/activate /scratch/$USER/capstone_env6
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache

REPO=/scratch/$USER/capstone-repo
V1_DIR=/scratch/$USER/runs/2026-03-10_full
V2_DIR=/scratch/$USER/runs/2026-03-10_full_v2
CONFIG=$REPO/configs/exp005_memorization.yaml
RUN_ID=2026-03-10_full_v2

# Sync fixed CLAP script
cp /home/$USER/analysis_compute_clap_embeddings.py $REPO/src/analysis/compute_clap_embeddings.py
cp /home/$USER/utils.py $REPO/src/utils.py
cp /home/$USER/exp005_memorization.yaml $CONFIG
for f in /home/$USER/analysis_*.py; do
    base=$(echo $f | sed 's|.*/analysis_||')
    cp "$f" "$REPO/src/analysis/$base"
done

cd $REPO

echo "============================================================"
echo "RESUMING V2 Analysis from Step 4"
echo "Started: $(date)"
echo "============================================================"

# Step 4: CLAP embeddings
echo "[$(date)] Step 4: CLAP embeddings..."
python src/analysis/compute_clap_embeddings.py \
    --config "$CONFIG" --run_id "$RUN_ID" --run_dir "$V2_DIR"

# Step 5: Musicological features
echo "[$(date)] Step 5: Musicological features..."
python src/analysis/musicological_features.py \
    --config "$CONFIG" --run_id "$RUN_ID" --run_dir "$V2_DIR"

# Step 6: Per-artist FAD
echo "[$(date)] Step 6: Per-artist FAD..."
python src/analysis/per_artist_fad.py \
    --config "$CONFIG" --run_id "$RUN_ID" --run_dir "$V2_DIR"

# Step 7: Vulnerability score
echo "[$(date)] Step 7: Vulnerability score..."
python src/analysis/vulnerability_score.py \
    --config "$CONFIG" --run_id "$RUN_ID" --run_dir "$V2_DIR"

echo "[$(date)] Phase 1 complete. Checking outputs..."
ls -la $V2_DIR/analysis/ | head -30

echo ""
echo "============================================================"
echo "PHASE 2: Supplementary Analyses (S1-S7)"
echo "Started: $(date)"
echo "============================================================"

python3 << 'SUPPLEMENTARY_EOF'
import csv, json, os, sys
import numpy as np
from collections import defaultdict
from scipy import stats

V2 = "/scratch/$USER/runs/2026-03-10_full_v2/analysis"
OUT = "/scratch/$USER/runs/2026-03-10_full_v2/supplementary"
os.makedirs(OUT, exist_ok=True)

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

# ============================================================
# S1: Effect Sizes (Cohen's d per artist)
# ============================================================
print("S1: Effect sizes...")
clap_rows = load_csv(f"{V2}/clap_per_artist.csv")
artist_matched = defaultdict(list)
artist_mismatched = defaultdict(list)
for row in clap_rows:
    aid = row.get("artist_id", "")
    if not aid: continue
    m = sf(row.get("matched_mean_sim"))
    mm = sf(row.get("mismatched_mean_sim"))
    if m is not None: artist_matched[aid].append(m)
    if mm is not None: artist_mismatched[aid].append(mm)

effect_rows = []
ds = []
for aid in sorted(artist_matched.keys()):
    if aid not in artist_mismatched or len(artist_matched[aid]) < 2 or len(artist_mismatched[aid]) < 2:
        continue
    m_arr = np.array(artist_matched[aid])
    mm_arr = np.array(artist_mismatched[aid])
    pooled_std = np.sqrt((np.var(m_arr, ddof=1) + np.var(mm_arr, ddof=1)) / 2)
    d = (np.mean(m_arr) - np.mean(mm_arr)) / pooled_std if pooled_std > 0 else 0
    boot_ds = []
    rng = np.random.default_rng(42)
    for _ in range(1000):
        bm = rng.choice(m_arr, len(m_arr), replace=True)
        bmm = rng.choice(mm_arr, len(mm_arr), replace=True)
        ps = np.sqrt((np.var(bm, ddof=1) + np.var(bmm, ddof=1)) / 2)
        boot_ds.append((np.mean(bm) - np.mean(bmm)) / ps if ps > 0 else 0)
    ci_lo, ci_hi = np.percentile(boot_ds, [2.5, 97.5])
    effect_rows.append({"artist_id": aid, "cohens_d": d, "ci_lower": ci_lo, "ci_upper": ci_hi,
                         "n_matched": len(m_arr), "n_mismatched": len(mm_arr)})
    ds.append(d)

with open(f"{OUT}/effect_sizes.csv", "w") as f:
    w = csv.DictWriter(f, ["artist_id", "cohens_d", "ci_lower", "ci_upper", "n_matched", "n_mismatched"])
    w.writeheader()
    w.writerows(effect_rows)
with open(f"{OUT}/effect_sizes_summary.json", "w") as f:
    json.dump({"mean_d": float(np.mean(ds)), "median_d": float(np.median(ds)),
               "n_large": int(sum(1 for d in ds if abs(d) > 0.8)),
               "n_artists": len(ds)}, f, indent=2)
print(f"  Mean Cohen's d: {np.mean(ds):.4f}, n_large(>0.8): {sum(1 for d in ds if abs(d) > 0.8)}")

# ============================================================
# S2: Permutation Test (10,000x per artist)
# ============================================================
print("S2: Permutation test...")
perm_rows = []
n_sig = 0
rng = np.random.default_rng(42)
for aid in sorted(artist_matched.keys()):
    if aid not in artist_mismatched: continue
    m_arr = np.array(artist_matched[aid])
    mm_arr = np.array(artist_mismatched[aid])
    obs_gap = np.mean(m_arr) - np.mean(mm_arr)
    combined = np.concatenate([m_arr, mm_arr])
    n_m = len(m_arr)
    count = 0
    for _ in range(10000):
        rng.shuffle(combined)
        perm_gap = np.mean(combined[:n_m]) - np.mean(combined[n_m:])
        if perm_gap >= obs_gap:
            count += 1
    p = count / 10000
    sig = p < 0.05
    if sig: n_sig += 1
    perm_rows.append({"artist_id": aid, "observed_gap": obs_gap, "perm_p": p, "significant": sig})

with open(f"{OUT}/permutation_test_per_artist.csv", "w") as f:
    w = csv.DictWriter(f, ["artist_id", "observed_gap", "perm_p", "significant"])
    w.writeheader()
    w.writerows(perm_rows)
with open(f"{OUT}/permutation_test_summary.json", "w") as f:
    json.dump({"n_significant": n_sig, "n_artists": len(perm_rows),
               "pct_significant": round(100*n_sig/len(perm_rows), 1) if perm_rows else 0}, f, indent=2)
print(f"  {n_sig}/{len(perm_rows)} artists significant (p<0.05)")

# ============================================================
# S3: Vulnerability Ablation
# ============================================================
print("S3: Vulnerability ablation...")
vuln_rows = load_csv(f"{V2}/vulnerability_scores.csv")
signals = ["clap_similarity", "fad", "musicological_similarity", "ngram_match_rate"]
artist_signals = {}
for row in vuln_rows:
    aid = row.get("artist_id", "")
    if not aid: continue
    artist_signals[aid] = {s: sf(row.get(s)) for s in signals}
    artist_signals[aid]["full_score"] = sf(row.get("vulnerability_score"))

ablation = {}
for drop in signals:
    keep = [s for s in signals if s != drop]
    scores = {}
    for aid, sigs in artist_signals.items():
        vals = [sigs[s] for s in keep if sigs[s] is not None]
        scores[aid] = np.mean(vals) if vals else None
    aids = [a for a in scores if scores[a] is not None and artist_signals[a]["full_score"] is not None]
    if len(aids) >= 5:
        rho, p = stats.spearmanr([scores[a] for a in aids], [artist_signals[a]["full_score"] for a in aids])
        ablation[f"drop_{drop}"] = {"spearman_rho": float(rho), "p": float(p), "n": len(aids)}

with open(f"{OUT}/vulnerability_ablation.json", "w") as f:
    json.dump(ablation, f, indent=2)
print(f"  Ablation results: {json.dumps({k: round(v['spearman_rho'], 3) for k, v in ablation.items()})}")

# ============================================================
# S4: Vulnerability Rank Stability
# ============================================================
print("S4: Rank stability...")
weight_schemes = [
    {"name": "equal", "w": [0.25, 0.25, 0.25, 0.25]},
    {"name": "clap_heavy", "w": [0.5, 0.2, 0.2, 0.1]},
    {"name": "fad_heavy", "w": [0.2, 0.5, 0.2, 0.1]},
    {"name": "musico_heavy", "w": [0.2, 0.2, 0.5, 0.1]},
    {"name": "ngram_heavy", "w": [0.1, 0.2, 0.2, 0.5]},
    {"name": "perceptual_only", "w": [0.5, 0.5, 0.0, 0.0]},
]
scheme_ranks = {}
for scheme in weight_schemes:
    scores = {}
    for aid, sigs in artist_signals.items():
        vals = [sigs.get(s) for s in signals]
        if all(v is not None for v in vals):
            scores[aid] = sum(w * v for w, v in zip(scheme["w"], vals))
    ranked = sorted(scores.keys(), key=lambda a: scores[a], reverse=True)
    scheme_ranks[scheme["name"]] = {a: i+1 for i, a in enumerate(ranked)}

stability = {}
names = list(scheme_ranks.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        common = set(scheme_ranks[names[i]].keys()) & set(scheme_ranks[names[j]].keys())
        if len(common) >= 5:
            rho, p = stats.spearmanr(
                [scheme_ranks[names[i]][a] for a in common],
                [scheme_ranks[names[j]][a] for a in common])
            stability[f"{names[i]}_vs_{names[j]}"] = {"rho": float(rho), "p": float(p)}

with open(f"{OUT}/vulnerability_rank_stability.json", "w") as f:
    json.dump(stability, f, indent=2)
print(f"  Rank stability pairs computed: {len(stability)}")

# ============================================================
# S5: Genre-controlled Z-scores
# ============================================================
print("S5: Genre-controlled z-scores...")
genre_groups = defaultdict(list)
for row in vuln_rows:
    aid = row.get("artist_id", "")
    genre = row.get("genre", "Unknown")
    vuln = sf(row.get("vulnerability_score"))
    clap = sf(row.get("clap_similarity"))
    if aid and vuln is not None:
        genre_groups[genre].append({"aid": aid, "vuln": vuln, "clap": clap or 0, "genre": genre,
                                    "name": row.get("artist_name", "")})

z_rows = []
for genre, group in sorted(genre_groups.items()):
    vulns = [g["vuln"] for g in group]
    mean_v, std_v = np.mean(vulns), np.std(vulns)
    for g in group:
        z = (g["vuln"] - mean_v) / (std_v + 1e-10)
        z_rows.append({"artist_id": g["aid"], "artist_name": g["name"], "genre": genre,
                       "raw_vulnerability": g["vuln"], "vulnerability_z_score": round(z, 4),
                       "raw_clap": g["clap"], "genre_n": len(group)})

with open(f"{OUT}/genre_zscore_v2.csv", "w") as f:
    w = csv.DictWriter(f, ["artist_id", "artist_name", "genre", "raw_vulnerability",
                           "vulnerability_z_score", "raw_clap", "genre_n"])
    w.writeheader()
    w.writerows(sorted(z_rows, key=lambda x: x.get("vulnerability_z_score", -999), reverse=True))
print(f"  {len(z_rows)} artists z-scored across {len(genre_groups)} genres")

# ============================================================
# S6: Top-k Retrieval
# ============================================================
print("S6: Top-k retrieval...")
topk_rows = []
for aid in sorted(artist_matched.keys()):
    if aid not in artist_mismatched: continue
    m_mean = np.mean(artist_matched[aid])
    all_means = []
    for other_aid in artist_matched.keys():
        if other_aid in artist_mismatched:
            all_means.append((other_aid, np.mean(artist_matched[other_aid])))
    all_means.sort(key=lambda x: x[1], reverse=True)
    rank = next((i+1 for i, (a, _) in enumerate(all_means) if a == aid), -1)
    topk_rows.append({"artist_id": aid, "self_rank": rank, "total_artists": len(all_means),
                       "top1_correct": rank == 1, "top5_correct": rank <= 5})

n_top1 = sum(1 for r in topk_rows if r["top1_correct"])
n_top5 = sum(1 for r in topk_rows if r["top5_correct"])
with open(f"{OUT}/topk_retrieval_per_artist.csv", "w") as f:
    w = csv.DictWriter(f, ["artist_id", "self_rank", "total_artists", "top1_correct", "top5_correct"])
    w.writeheader()
    w.writerows(topk_rows)
with open(f"{OUT}/topk_retrieval_summary.json", "w") as f:
    json.dump({"top1_precision": round(n_top1/len(topk_rows), 4) if topk_rows else 0,
               "top5_precision": round(n_top5/len(topk_rows), 4) if topk_rows else 0,
               "n_artists": len(topk_rows)}, f, indent=2)
print(f"  Top-1: {n_top1}/{len(topk_rows)}, Top-5: {n_top5}/{len(topk_rows)}")

# ============================================================
# S7: Prompt Confound Check
# ============================================================
print("S7: Prompt confound check...")
prompt_counts = defaultdict(int)
for row in clap_rows:
    aid = row.get("artist_id", "")
    if aid: prompt_counts[aid] += 1

vuln_map = {}
for row in vuln_rows:
    aid = row.get("artist_id", "")
    v = sf(row.get("vulnerability_score"))
    if aid and v is not None: vuln_map[aid] = v

common = set(prompt_counts.keys()) & set(vuln_map.keys())
if len(common) >= 5:
    rho, p = stats.spearmanr([prompt_counts[a] for a in common], [vuln_map[a] for a in common])
    confound = {"spearman_rho": float(rho), "p": float(p), "n": len(common), "confound_detected": bool(p < 0.05)}
else:
    confound = {"error": "too few common artists", "n": len(common)}

with open(f"{OUT}/prompt_confound_check.json", "w") as f:
    json.dump(confound, f, indent=2)
print(f"  Prompt confound: rho={confound.get('spearman_rho', 'N/A')}, p={confound.get('p', 'N/A')}")

print("\nPhase 2 complete. Files saved to:", OUT)
print("Files:", os.listdir(OUT))
SUPPLEMENTARY_EOF

echo ""
echo "============================================================"
echo "PHASE 3: V1 vs V2 Comparative Analysis (C1-C7)"
echo "Started: $(date)"
echo "============================================================"

python3 << 'COMPARISON_EOF'
import csv, json, os, sys
import numpy as np
from collections import defaultdict
from scipy import stats

V1 = "/scratch/$USER/runs/2026-03-10_full/analysis"
V2 = "/scratch/$USER/runs/2026-03-10_full_v2/analysis"
OUT = "/scratch/$USER/runs/2026-03-10_full_v2/comparison"
os.makedirs(OUT, exist_ok=True)

def load_csv(path):
    rows = []
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return rows
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

# Load vulnerability scores
v1_vuln = {r.get("artist_id", ""): r for r in load_csv(f"{V1}/vulnerability_scores.csv")}
v2_vuln = {r.get("artist_id", ""): r for r in load_csv(f"{V2}/vulnerability_scores.csv")}
common_artists = sorted(set(v1_vuln.keys()) & set(v2_vuln.keys()) - {""})
print(f"Common artists: {len(common_artists)}")

# Load CLAP per-artist
v1_clap = load_csv(f"{V1}/clap_per_artist.csv")
v2_clap = load_csv(f"{V2}/clap_per_artist.csv")

def aggregate_clap(rows):
    matched = defaultdict(list)
    mismatched = defaultdict(list)
    for r in rows:
        aid = r.get("artist_id", "")
        if not aid: continue
        m = sf(r.get("matched_mean_sim"))
        mm = sf(r.get("mismatched_mean_sim"))
        if m is not None: matched[aid].append(m)
        if mm is not None: mismatched[aid].append(mm)
    return matched, mismatched

v1_m, v1_mm = aggregate_clap(v1_clap)
v2_m, v2_mm = aggregate_clap(v2_clap)

# Load FAD
v1_fad = {r.get("artist_id", ""): sf(r.get("matched_fad")) for r in load_csv(f"{V1}/per_artist_fad.csv")}
v2_fad = {r.get("artist_id", ""): sf(r.get("matched_fad")) for r in load_csv(f"{V2}/per_artist_fad.csv")}

# Load n-gram summaries
v1_ngram_json = json.load(open(f"{V1}/memorization_verdict.json")) if os.path.exists(f"{V1}/memorization_verdict.json") else {}
v2_ngram_json = json.load(open(f"{V2}/memorization_verdict.json")) if os.path.exists(f"{V2}/memorization_verdict.json") else {}

# ============================================================
# C1: Aggregate Comparison Table
# ============================================================
print("\n=== C1: Aggregate Comparison ===")
v1_vulns = [sf(v1_vuln[a].get("vulnerability_score")) for a in common_artists if sf(v1_vuln[a].get("vulnerability_score")) is not None]
v2_vulns = [sf(v2_vuln[a].get("vulnerability_score")) for a in common_artists if sf(v2_vuln[a].get("vulnerability_score")) is not None]

v1_gaps = [np.mean(v1_m.get(a, [0])) - np.mean(v1_mm.get(a, [0])) for a in common_artists if a in v1_m and a in v1_mm]
v2_gaps = [np.mean(v2_m.get(a, [0])) - np.mean(v2_mm.get(a, [0])) for a in common_artists if a in v2_m and a in v2_mm]

v1_fads = [v1_fad.get(a) for a in common_artists if v1_fad.get(a) is not None]
v2_fads = [v2_fad.get(a) for a in common_artists if v2_fad.get(a) is not None]

agg = {
    "v1": {"mean_vulnerability": float(np.mean(v1_vulns)) if v1_vulns else None,
           "mean_clap_gap": float(np.mean(v1_gaps)) if v1_gaps else None,
           "mean_fad": float(np.mean(v1_fads)) if v1_fads else None,
           "training_loss": 4.22,
           "n_significant_ngram": v1_ngram_json.get("n_significant", 0)},
    "v2": {"mean_vulnerability": float(np.mean(v2_vulns)) if v2_vulns else None,
           "mean_clap_gap": float(np.mean(v2_gaps)) if v2_gaps else None,
           "mean_fad": float(np.mean(v2_fads)) if v2_fads else None,
           "training_loss": 4.07,
           "n_significant_ngram": v2_ngram_json.get("n_significant", 0)},
    "n_common_artists": len(common_artists),
}

# Paired Wilcoxon on vulnerability
paired_v1 = [sf(v1_vuln[a].get("vulnerability_score")) for a in common_artists]
paired_v2 = [sf(v2_vuln[a].get("vulnerability_score")) for a in common_artists]
valid_pairs = [(a, b) for a, b in zip(paired_v1, paired_v2) if a is not None and b is not None]
if len(valid_pairs) >= 5:
    w_stat, w_p = stats.wilcoxon([a for a, b in valid_pairs], [b for a, b in valid_pairs])
    agg["wilcoxon_vulnerability"] = {"statistic": float(w_stat), "p": float(w_p)}
    print(f"  Wilcoxon vulnerability: W={w_stat:.1f}, p={w_p:.4f}")

for k in ["v1", "v2"]:
    print(f"  {k}: vuln={agg[k]['mean_vulnerability']:.4f}, clap_gap={agg[k]['mean_clap_gap']:.4f}, fad={agg[k]['mean_fad']:.4f}")

with open(f"{OUT}/aggregate_comparison.json", "w") as f:
    json.dump(agg, f, indent=2)

# ============================================================
# C2: Per-Artist Delta
# ============================================================
print("\n=== C2: Per-Artist Delta ===")
delta_rows = []
for a in common_artists:
    v1_v = sf(v1_vuln[a].get("vulnerability_score"))
    v2_v = sf(v2_vuln[a].get("vulnerability_score"))
    n_tracks = sf(v1_vuln[a].get("n_catalog_tracks")) or sf(v2_vuln[a].get("n_catalog_tracks"))
    if v1_v is not None and v2_v is not None:
        delta_rows.append({
            "artist_id": a,
            "artist_name": v1_vuln[a].get("artist_name", ""),
            "genre": v1_vuln[a].get("genre", ""),
            "v1_vulnerability": round(v1_v, 6),
            "v2_vulnerability": round(v2_v, 6),
            "delta": round(v2_v - v1_v, 6),
            "n_catalog_tracks": int(n_tracks) if n_tracks else "",
        })

delta_rows.sort(key=lambda x: x["delta"], reverse=True)
with open(f"{OUT}/per_artist_delta.csv", "w") as f:
    w = csv.DictWriter(f, ["artist_id", "artist_name", "genre", "v1_vulnerability", "v2_vulnerability", "delta", "n_catalog_tracks"])
    w.writeheader()
    w.writerows(delta_rows)

deltas = [r["delta"] for r in delta_rows]
n_tracks_list = [r["n_catalog_tracks"] for r in delta_rows if r["n_catalog_tracks"] != ""]
if deltas and n_tracks_list and len(n_tracks_list) >= 5:
    rho, p = stats.spearmanr(deltas[:len(n_tracks_list)], n_tracks_list)
    print(f"  Delta vs n_tracks: rho={rho:.3f}, p={p:.4f}")

gained = sum(1 for d in deltas if d > 0)
lost = sum(1 for d in deltas if d < 0)
print(f"  Gained vulnerability: {gained}, Lost: {lost}, Mean delta: {np.mean(deltas):.4f}")

# ============================================================
# C3: N-gram Memorization Comparison
# ============================================================
print("\n=== C3: N-gram Comparison ===")
v1_ngram_stats = load_csv(f"{V1}/ngram_statistical_tests.csv") if os.path.exists(f"{V1}/ngram_statistical_tests.csv") else []
v2_ngram_stats = load_csv(f"{V2}/ngram_statistical_tests.csv") if os.path.exists(f"{V2}/ngram_statistical_tests.csv") else []

ngram_comp = {"v1_tests": len(v1_ngram_stats), "v2_tests": len(v2_ngram_stats)}
v1_sig = [r for r in v1_ngram_stats if r.get("significant_fdr", "").lower() == "true"]
v2_sig = [r for r in v2_ngram_stats if r.get("significant_fdr", "").lower() == "true"]
ngram_comp["v1_significant"] = len(v1_sig)
ngram_comp["v2_significant"] = len(v2_sig)
print(f"  V1: {len(v1_sig)}/{len(v1_ngram_stats)} significant, V2: {len(v2_sig)}/{len(v2_ngram_stats)} significant")

with open(f"{OUT}/ngram_comparison.json", "w") as f:
    json.dump(ngram_comp, f, indent=2)

# ============================================================
# C4: CLAP Gap Improvement
# ============================================================
print("\n=== C4: CLAP Gap Improvement ===")
paired_gaps_v1 = []
paired_gaps_v2 = []
for a in common_artists:
    if a in v1_m and a in v1_mm and a in v2_m and a in v2_mm:
        paired_gaps_v1.append(np.mean(v1_m[a]) - np.mean(v1_mm[a]))
        paired_gaps_v2.append(np.mean(v2_m[a]) - np.mean(v2_mm[a]))

if len(paired_gaps_v1) >= 5:
    t, p = stats.ttest_rel(paired_gaps_v2, paired_gaps_v1)
    improvement = np.mean(paired_gaps_v2) - np.mean(paired_gaps_v1)
    clap_comp = {"v1_mean_gap": float(np.mean(paired_gaps_v1)), "v2_mean_gap": float(np.mean(paired_gaps_v2)),
                 "improvement": float(improvement), "paired_t": float(t), "p": float(p), "n": len(paired_gaps_v1)}
    print(f"  V1 gap: {np.mean(paired_gaps_v1):.4f}, V2 gap: {np.mean(paired_gaps_v2):.4f}")
    print(f"  Improvement: {improvement:+.4f}, t={t:.3f}, p={p:.4f}")
else:
    clap_comp = {"error": "too few paired artists"}

with open(f"{OUT}/clap_gap_comparison.json", "w") as f:
    json.dump(clap_comp, f, indent=2)

# ============================================================
# C5: Training Loss vs Vulnerability
# ============================================================
print("\n=== C5: Training Loss vs Vulnerability ===")
loss_comp = {
    "v1_final_loss": 4.22, "v2_final_loss": 4.07,
    "loss_improvement_pct": round((4.22 - 4.07) / 4.22 * 100, 1),
    "v1_mean_vuln": float(np.mean(v1_vulns)) if v1_vulns else None,
    "v2_mean_vuln": float(np.mean(v2_vulns)) if v2_vulns else None,
}
if v1_vulns and v2_vulns:
    loss_comp["vuln_change_pct"] = round((np.mean(v2_vulns) - np.mean(v1_vulns)) / np.mean(v1_vulns) * 100, 1)
print(f"  Loss: {loss_comp['v1_final_loss']} -> {loss_comp['v2_final_loss']} ({loss_comp['loss_improvement_pct']}% better)")
print(f"  Vuln: {loss_comp.get('v1_mean_vuln', 'N/A'):.4f} -> {loss_comp.get('v2_mean_vuln', 'N/A'):.4f}")

with open(f"{OUT}/training_loss_comparison.json", "w") as f:
    json.dump(loss_comp, f, indent=2)

# ============================================================
# C6: Rank Stability Across Fine-tuning Levels
# ============================================================
print("\n=== C6: Rank Stability (V1 vs V2 rankings) ===")
v1_ranked = sorted(common_artists, key=lambda a: sf(v1_vuln[a].get("vulnerability_score")) or 0, reverse=True)
v2_ranked = sorted(common_artists, key=lambda a: sf(v2_vuln[a].get("vulnerability_score")) or 0, reverse=True)
v1_ranks = {a: i+1 for i, a in enumerate(v1_ranked)}
v2_ranks = {a: i+1 for i, a in enumerate(v2_ranked)}

if len(common_artists) >= 5:
    rho, p = stats.spearmanr([v1_ranks[a] for a in common_artists], [v2_ranks[a] for a in common_artists])
    rank_stability = {"spearman_rho": float(rho), "p": float(p), "n": len(common_artists)}
    print(f"  Spearman rho: {rho:.3f}, p={p:.4f}")
    print(f"  V1 top 5: {[v1_vuln[a].get('artist_name', a)[:20] for a in v1_ranked[:5]]}")
    print(f"  V2 top 5: {[v2_vuln[a].get('artist_name', a)[:20] for a in v2_ranked[:5]]}")
else:
    rank_stability = {"error": "too few artists"}

with open(f"{OUT}/rank_stability.json", "w") as f:
    json.dump(rank_stability, f, indent=2)

# ============================================================
# C7: Effect Size Comparison
# ============================================================
print("\n=== C7: Effect Size Comparison ===")
v1_supp = "/scratch/$USER/runs/2026-03-10_full/supplementary"
v2_supp = "/scratch/$USER/runs/2026-03-10_full_v2/supplementary"

v1_effects = []
v2_effects = []
if os.path.exists(f"{v2_supp}/effect_sizes.csv"):
    for r in load_csv(f"{v2_supp}/effect_sizes.csv"):
        d = sf(r.get("cohens_d"))
        if d is not None: v2_effects.append(d)

for aid in sorted(set(v1_m.keys()) & set(v1_mm.keys())):
    m_arr = np.array(v1_m[aid])
    mm_arr = np.array(v1_mm[aid])
    if len(m_arr) >= 2 and len(mm_arr) >= 2:
        pooled = np.sqrt((np.var(m_arr, ddof=1) + np.var(mm_arr, ddof=1)) / 2)
        if pooled > 0:
            v1_effects.append((np.mean(m_arr) - np.mean(mm_arr)) / pooled)

effect_comp = {
    "v1_mean_d": float(np.mean(v1_effects)) if v1_effects else None,
    "v2_mean_d": float(np.mean(v2_effects)) if v2_effects else None,
    "v1_n": len(v1_effects),
    "v2_n": len(v2_effects),
}
if v1_effects and v2_effects and len(v1_effects) == len(v2_effects):
    t, p = stats.ttest_rel(v2_effects, v1_effects)
    effect_comp["paired_t"] = float(t)
    effect_comp["p"] = float(p)
    print(f"  V1 mean d: {np.mean(v1_effects):.4f}, V2 mean d: {np.mean(v2_effects):.4f}")
    print(f"  Paired t: {t:.3f}, p={p:.4f}")
elif v1_effects and v2_effects:
    print(f"  V1 mean d: {np.mean(v1_effects):.4f} (n={len(v1_effects)}), V2 mean d: {np.mean(v2_effects):.4f} (n={len(v2_effects)})")

with open(f"{OUT}/effect_size_comparison.json", "w") as f:
    json.dump(effect_comp, f, indent=2)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)
print(f"Output files: {os.listdir(OUT)}")

with open(f"{OUT}/full_comparison_summary.json", "w") as f:
    json.dump({
        "aggregate": agg,
        "clap_gap": clap_comp,
        "ngram": ngram_comp,
        "loss": loss_comp,
        "rank_stability": rank_stability,
        "effect_sizes": effect_comp,
        "per_artist_delta": {
            "n_gained": gained, "n_lost": lost,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
        },
    }, f, indent=2)

COMPARISON_EOF

echo ""
echo "============================================================"
echo "ALL PHASES COMPLETE: $(date)"
echo "============================================================"
echo "V2 analysis: $V2_DIR/analysis/"
echo "V2 supplementary: $V2_DIR/supplementary/"
echo "V1 vs V2 comparison: $V2_DIR/comparison/"
echo ""
echo "File counts:"
echo "  analysis: $(ls $V2_DIR/analysis/ 2>/dev/null | wc -l)"
echo "  supplementary: $(ls $V2_DIR/supplementary/ 2>/dev/null | wc -l)"
echo "  comparison: $(ls $V2_DIR/comparison/ 2>/dev/null | wc -l)"
