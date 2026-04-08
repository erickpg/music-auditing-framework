"""
Per-genre UMAP maps of artist catalog centroids in MuQ-MuLan embedding space.
One plot per genre (top 4: Rock, Folk, Hip-Hop, Pop).
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import re

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

# --- Load data ---
MUQ_DIR = os.environ.get("MUQ_DIR", "/scratch/$USER/runs/muq_validation")
cat_emb = np.load(os.path.join(MUQ_DIR, "catalog_muq.npy"))
with open(os.path.join(MUQ_DIR, "catalog_muq_ids.json")) as f:
    cat_ids = json.load(f)

# Vulnerability scores (2-signal)
VULN_PATH = os.environ.get("VULN_PATH", "/home/$USER/vuln_2sig.csv")
vuln_df = pd.read_csv(VULN_PATH)
vuln_df["vulnerability_score"] = pd.to_numeric(vuln_df["v1_vuln_2sig"], errors="coerce")
def assign_tier(s):
    if s >= 0.67: return "High"
    elif s <= 0.33: return "Low"
    else: return "Intermediate"
vuln_df["tier"] = vuln_df["vulnerability_score"].apply(assign_tier)
vuln_df["aid_str"] = vuln_df["artist_id"].astype(str)

# Genre mapping
genre_df = pd.read_csv(os.environ.get("GENRE_PATH", "/home/$USER/artist_genres.csv"))
genre_df["aid_str"] = genre_df["artist_id"].astype(str)
vuln_df = vuln_df.merge(genre_df[["aid_str", "genre"]], on="aid_str", how="left", suffixes=("_old", ""))
if "genre_old" in vuln_df.columns:
    vuln_df["genre"] = vuln_df["genre"].fillna(vuln_df["genre_old"])
    vuln_df.drop(columns=["genre_old"], inplace=True)

# Track -> artist mapping
tracks_df = pd.read_csv(os.environ.get("TRACKS_MANIFEST", "/home/$USER/tracks_selected.csv"))
track_to_artist = {}
for _, row in tracks_df.iterrows():
    tid = str(row["track_id"])
    aid = str(row["artist_id"])
    track_to_artist[tid] = aid
    track_to_artist[tid.zfill(6)] = aid

# Group embeddings by artist
id_list = list(cat_ids.keys()) if isinstance(cat_ids, dict) else list(cat_ids)
artist_embeddings = {}
for i, tid in enumerate(id_list):
    tid_clean = str(tid).replace(".wav", "").replace(".npy", "")
    aid = track_to_artist.get(tid_clean) or track_to_artist.get(tid_clean.zfill(6))
    if aid is None:
        nums = re.findall(r'\d+', tid_clean)
        for n in nums:
            aid = track_to_artist.get(n) or track_to_artist.get(n.zfill(6))
            if aid: break
    if aid:
        aid = str(int(aid)) if aid.isdigit() else aid
        if aid not in artist_embeddings:
            artist_embeddings[aid] = []
        artist_embeddings[aid].append(cat_emb[i])

# Centroids
artist_ids_ordered = sorted(artist_embeddings.keys())
centroids = np.array([np.mean(artist_embeddings[a], axis=0) for a in artist_ids_ordered])

# Merge
vuln_subset = vuln_df[["aid_str", "artist_name", "vulnerability_score", "tier", "genre"]].drop_duplicates(subset="aid_str")
centroid_df = pd.DataFrame({"aid_str": artist_ids_ordered})
merged = centroid_df.merge(vuln_subset, on="aid_str", how="left")
merged["tier"] = merged["tier"].fillna("Intermediate")
merged["vulnerability_score"] = merged["vulnerability_score"].fillna(0.5)
merged["artist_name"] = merged["artist_name"].fillna(merged["aid_str"])
merged["genre"] = merged["genre"].fillna("")

# Full UMAP (all 50 together — same projection for all subplots)
from umap import UMAP
reducer = UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
coords = reducer.fit_transform(centroids)
merged["x"] = coords[:, 0]
merged["y"] = coords[:, 1]

print(f"Data ready: {len(merged)} artists")

# --- Plot per genre ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

tier_colors = {"High": "#C62828", "Intermediate": "#F57F17", "Low": "#2E7D32"}
tier_markers = {"High": "o", "Intermediate": "s", "Low": "D"}

top_genres = ["Rock", "Folk", "Hip-Hop", "Pop"]

for genre in top_genres:
    gmask = merged["genre"] == genre
    genre_data = merged[gmask].copy()
    other_data = merged[~gmask].copy()
    n_genre = len(genre_data)

    if n_genre == 0:
        print(f"Skipping {genre}: no artists")
        continue

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    # Plot other genres as faded background
    ax.scatter(
        other_data["x"], other_data["y"],
        s=40, c="#D0D0D0", alpha=0.3,
        edgecolors="#BBBBBB", linewidths=0.5, zorder=1,
    )
    # Light labels for background artists
    for _, row in other_data.iterrows():
        name = str(row["artist_name"])
        if len(name) > 18:
            name = name[:16] + "…"
        ax.annotate(
            name, (row["x"], row["y"]),
            fontsize=5, ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points",
            color="#AAAAAA", zorder=1,
        )

    # Plot genre artists by tier
    for tier in ["High", "Intermediate", "Low"]:
        tmask = genre_data["tier"] == tier
        subset = genre_data[tmask]
        if len(subset) == 0:
            continue
        sizes = 120 + subset["vulnerability_score"] * 400
        ax.scatter(
            subset["x"], subset["y"],
            s=sizes, c=tier_colors[tier], alpha=0.85,
            edgecolors="white", linewidths=2,
            marker=tier_markers[tier],
            label=f"{tier} ({len(subset)})", zorder=4,
        )

    # Bold labels for genre artists
    for _, row in genre_data.iterrows():
        name = str(row["artist_name"])
        score = row["vulnerability_score"]
        if len(name) > 22:
            name = name[:20] + "…"
        label = f"{name}\n({score:.2f})"
        ax.annotate(
            label, (row["x"], row["y"]),
            fontsize=8, fontweight="bold", ha="center", va="bottom",
            xytext=(0, 12), textcoords="offset points",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            zorder=5,
        )

    # Tier distribution summary
    tier_counts = genre_data["tier"].value_counts()
    summary_lines = []
    for t in ["High", "Intermediate", "Low"]:
        if t in tier_counts:
            summary_lines.append(f"{t}: {tier_counts[t]}")
    summary = " · ".join(summary_lines)

    ax.legend(
        title="Vulnerability Tier",
        loc="upper right", frameon=True, framealpha=0.9,
        edgecolor="#CCCCCC", fontsize=10, title_fontsize=11,
    )

    ax.set_xlabel("UMAP Dimension 1", fontsize=11, labelpad=8)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11, labelpad=8)
    ax.set_title(
        f"{genre} Artists in MuQ-MuLan Embedding Space ({n_genre} artists)\n"
        f"{summary}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.text(0.02, 0.02,
        f"Gray dots: other genres · Point size ∝ vulnerability score · "
        f"Score range: {genre_data['vulnerability_score'].min():.3f}–{genre_data['vulnerability_score'].max():.3f}",
        fontsize=7.5, color="#666666", style="italic")

    fname = f"catalog_map_{genre.lower().replace('-', '_')}"
    plt.savefig(os.path.join(OUT_DIR, f"{fname}.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, f"{fname}.pdf"))
    plt.close()
    print(f"[SAVED] {fname}.png ({n_genre} artists)")

print("Done!")
