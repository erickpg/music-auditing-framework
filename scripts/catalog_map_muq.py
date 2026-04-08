"""
2D UMAP map of artist catalog centroids in MuQ-MuLan embedding space.
Color by vulnerability tier, size by vulnerability score.
Uses existing MuQ .npy embeddings from muq_validation run.
"""
import numpy as np
import pandas as pd
import json
import os
import sys

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

# --- Paths ---
MUQ_DIR = os.environ.get("MUQ_DIR", "/scratch/$USER/runs/muq_validation")
V1_ANALYSIS = os.environ.get("V1_ANALYSIS", "/home/$USER")

# --- Load MuQ catalog embeddings + IDs ---
cat_emb = np.load(os.path.join(MUQ_DIR, "catalog_muq.npy"))
with open(os.path.join(MUQ_DIR, "catalog_muq_ids.json")) as f:
    cat_ids = json.load(f)
print(f"Catalog embeddings: {cat_emb.shape}, IDs: {len(cat_ids)}")

# --- Load 2-signal vulnerability scores ---
VULN_PATH = os.environ.get("VULN_PATH", "/home/$USER/vuln_2sig.csv")
vuln_df = pd.read_csv(VULN_PATH)
print(f"Vulnerability scores: {len(vuln_df)} artists")
# Use V1 2-signal scores, apply 3-tier
vuln_df["vulnerability_score"] = pd.to_numeric(vuln_df["v1_vuln_2sig"], errors="coerce")
def assign_tier(s):
    if s >= 0.67: return "High"
    elif s <= 0.33: return "Low"
    else: return "Intermediate"
vuln_df["tier"] = vuln_df["vulnerability_score"].apply(assign_tier)
print(f"Tiers: {vuln_df['tier'].value_counts().to_dict()}")

# --- Map track IDs to artist IDs ---
# cat_ids format might vary — check
print(f"ID samples: {list(cat_ids.items())[:3] if isinstance(cat_ids, dict) else cat_ids[:3]}")

# Load tracks manifest for track_id -> artist_id mapping
tracks_path = os.environ.get("TRACKS_MANIFEST", "/home/$USER/tracks_selected.csv")
track_to_artist = {}
if os.path.exists(tracks_path):
    tracks_df = pd.read_csv(tracks_path)
    print(f"Tracks manifest: {len(tracks_df)} rows, cols: {tracks_df.columns.tolist()}")
    for _, row in tracks_df.iterrows():
        tid = str(row["track_id"])
        aid = str(row["artist_id"])
        track_to_artist[tid] = aid
        track_to_artist[tid.zfill(6)] = aid
else:
    print(f"No tracks manifest at {tracks_path}")

# If cat_ids is a dict {filename: index} or list
if isinstance(cat_ids, dict):
    id_list = list(cat_ids.keys())
elif isinstance(cat_ids, list):
    id_list = cat_ids
else:
    id_list = list(cat_ids)

# Group embeddings by artist
artist_embeddings = {}
unmatched = 0
for i, tid in enumerate(id_list):
    tid_clean = str(tid).replace(".wav", "").replace(".npy", "")
    # Try various matching
    import re
    aid = track_to_artist.get(tid_clean)
    if aid is None:
        aid = track_to_artist.get(tid_clean.zfill(6))
    if aid is None:
        nums = re.findall(r'\d+', tid_clean)
        for n in nums:
            aid = track_to_artist.get(n) or track_to_artist.get(n.zfill(6))
            if aid:
                break
    if aid is None:
        # Maybe the ID itself contains the artist_id
        # Check if it matches any artist_id in vuln_df
        for _, vrow in vuln_df.iterrows():
            va = str(vrow["artist_id"])
            if va in tid_clean or va.zfill(6) in tid_clean:
                aid = va
                break
    if aid:
        aid = str(int(aid)) if aid.isdigit() else aid  # normalize
        if aid not in artist_embeddings:
            artist_embeddings[aid] = []
        artist_embeddings[aid].append(cat_emb[i])
    else:
        unmatched += 1

print(f"Artists with embeddings: {len(artist_embeddings)}, unmatched tracks: {unmatched}")

if len(artist_embeddings) == 0:
    print("ERROR: No artist mappings. Trying direct approach...")
    # Maybe cat_ids maps directly to artist
    # Try: if cat_ids is a list of artist_ids
    unique_aids = vuln_df["artist_id"].astype(str).tolist()
    for i, tid in enumerate(id_list):
        for ua in unique_aids:
            if ua in str(tid):
                if ua not in artist_embeddings:
                    artist_embeddings[ua] = []
                artist_embeddings[ua].append(cat_emb[i])
                break
    print(f"Retry: {len(artist_embeddings)} artists")

# Compute centroids
artist_ids_ordered = sorted(artist_embeddings.keys())
centroids = np.array([np.mean(artist_embeddings[a], axis=0) for a in artist_ids_ordered])
print(f"Centroids: {centroids.shape}")

# Merge with vulnerability
vuln_df["aid_str"] = vuln_df["artist_id"].astype(str)
# Load genre mapping
genre_path = os.environ.get("GENRE_PATH", "/home/$USER/artist_genres.csv")
if os.path.exists(genre_path):
    genre_df = pd.read_csv(genre_path)
    genre_df["aid_str"] = genre_df["artist_id"].astype(str)
    vuln_df = vuln_df.merge(genre_df[["aid_str", "genre"]], on="aid_str", how="left", suffixes=("_old", ""))
    if "genre_old" in vuln_df.columns:
        vuln_df["genre"] = vuln_df["genre"].fillna(vuln_df["genre_old"])
        vuln_df.drop(columns=["genre_old"], inplace=True)
    print(f"Genres loaded: {vuln_df['genre'].value_counts().to_dict()}")

keep_cols = ["aid_str", "artist_name", "vulnerability_score", "tier"]
if "genre" in vuln_df.columns:
    keep_cols.append("genre")
vuln_subset = vuln_df[keep_cols].drop_duplicates(subset="aid_str")
centroid_df = pd.DataFrame({"aid_str": artist_ids_ordered})
merged = centroid_df.merge(vuln_subset, on="aid_str", how="left")
merged["tier"] = merged["tier"].fillna("Intermediate")
merged["vulnerability_score"] = merged["vulnerability_score"].fillna(0.5)
merged["artist_name"] = merged["artist_name"].fillna(merged["aid_str"])

print(f"Merged: {len(merged)}, tiers: {merged['tier'].value_counts().to_dict()}")

# --- UMAP ---
try:
    from umap import UMAP
    reducer = UMAP(n_neighbors=min(15, len(centroids)-1), min_dist=0.3, metric="cosine", random_state=42)
    coords = reducer.fit_transform(centroids)
    method = "UMAP"
except ImportError:
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2, perplexity=min(15, len(centroids)-1), metric="cosine", random_state=42)
    coords = reducer.fit_transform(centroids)
    method = "t-SNE"

merged["x"] = coords[:, 0]
merged["y"] = coords[:, 1]

# Save data
merged.to_csv(os.path.join(OUT_DIR, "catalog_map_muq_data.csv"), index=False)
print(f"[SAVED] catalog_map_muq_data.csv")

# --- Plot ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

tier_colors = {"High": "#C62828", "Intermediate": "#F57F17", "Low": "#2E7D32"}
tier_order = ["High", "Intermediate", "Low"]

# Genre colors for background hulls
genre_colors = {
    "Rock": "#4FC3F7",
    "Folk": "#81C784",
    "Hip-Hop": "#CE93D8",
    "Pop": "#FFB74D",
    "Classical": "#A1887F",
    "Jazz": "#90A4AE",
    "Soul-RnB": "#F48FB1",
    "Country": "#FFD54F",
    "Blues": "#7986CB",
    "": "#E0E0E0",
}

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor("#FAFAFA")
fig.patch.set_facecolor("white")

# Draw genre background regions (convex hulls with padding)
if "genre" in merged.columns:
    from scipy.spatial import ConvexHull
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches

    genre_handles = []
    for genre in merged["genre"].dropna().unique():
        if genre == "":
            continue
        gmask = merged["genre"] == genre
        n_genre = gmask.sum()
        if n_genre < 3:
            # For < 3 points, draw circles instead of hull
            gpts = merged.loc[gmask, ["x", "y"]].values
            color = genre_colors.get(genre, "#E0E0E0")
            for pt in gpts:
                circle = plt.Circle(pt, radius=1.5, color=color, alpha=0.12, zorder=1)
                ax.add_patch(circle)
            genre_handles.append(mpatches.Patch(color=color, alpha=0.25, label=f"{genre} ({n_genre})"))
            continue

        gpts = merged.loc[gmask, ["x", "y"]].values
        color = genre_colors.get(genre, "#E0E0E0")
        try:
            hull = ConvexHull(gpts)
            hull_vertices = hull.vertices
            hull_pts = gpts[np.append(hull_vertices, hull_vertices[0])]

            # Expand hull slightly for padding
            centroid = gpts.mean(axis=0)
            expanded = centroid + (hull_pts - centroid) * 1.15

            ax.fill(expanded[:, 0], expanded[:, 1], color=color, alpha=0.12, zorder=1)
            ax.plot(expanded[:, 0], expanded[:, 1], color=color, alpha=0.35,
                    linewidth=1.5, linestyle="--", zorder=1)
            genre_handles.append(mpatches.Patch(color=color, alpha=0.25, label=f"{genre} ({n_genre})"))
        except Exception as e:
            print(f"Hull failed for {genre}: {e}")
            genre_handles.append(mpatches.Patch(color=color, alpha=0.25, label=f"{genre} ({n_genre})"))

    # Add genre centroid labels
    for genre in merged["genre"].dropna().unique():
        if genre == "":
            continue
        gmask = merged["genre"] == genre
        if gmask.sum() >= 3:
            gpts = merged.loc[gmask, ["x", "y"]].values
            cx, cy = gpts.mean(axis=0)
            color = genre_colors.get(genre, "#999999")
            ax.text(cx, cy, genre, fontsize=9, fontweight="bold", color=color,
                    alpha=0.6, ha="center", va="center", zorder=2,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# Plot each tier
for tier in tier_order:
    mask = merged["tier"] == tier
    subset = merged[mask]
    if len(subset) == 0:
        continue
    sizes = 80 + subset["vulnerability_score"] * 320
    ax.scatter(
        subset["x"], subset["y"],
        s=sizes, c=tier_colors[tier], alpha=0.8,
        edgecolors="white", linewidths=1.5,
        label=f"{tier} ({len(subset)})", zorder=3,
    )

# Labels
for _, row in merged.iterrows():
    name = str(row.get("artist_name", row["aid_str"]))
    if pd.isna(name) or name == "nan":
        name = row["aid_str"]
    if len(name) > 22:
        name = name[:20] + "…"
    ax.annotate(
        name, (row["x"], row["y"]),
        fontsize=6.5, ha="center", va="bottom",
        xytext=(0, 8), textcoords="offset points",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        zorder=4,
    )

# Tier legend (upper right)
tier_legend = ax.legend(
    title="Vulnerability Tier",
    loc="upper right", frameon=True, framealpha=0.9,
    edgecolor="#CCCCCC", fontsize=9, title_fontsize=10,
)
ax.add_artist(tier_legend)

# Genre legend (lower left)
if "genre" in merged.columns and len(genre_handles) > 0:
    genre_legend = ax.legend(
        handles=genre_handles, title="Genre (FMA)",
        loc="lower left", frameon=True, framealpha=0.9,
        edgecolor="#CCCCCC", fontsize=8, title_fontsize=9,
    )

ax.set_xlabel(f"{method} Dimension 1", fontsize=11, labelpad=8)
ax.set_ylabel(f"{method} Dimension 2", fontsize=11, labelpad=8)
ax.set_title(
    "Artist Catalog Centroids in MuQ-MuLan Embedding Space\n"
    "Point size ∝ CLAP vulnerability score · Color by 3-tier (≥0.67 High, ≤0.33 Low)",
    fontsize=12, fontweight="bold", pad=12,
)
ax.tick_params(axis="both", which="both", length=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

fig.text(0.02, 0.02,
    f"{len(merged)} artists · {method} projection (cosine) · Tiers from CLAP 2-signal composite",
    fontsize=7.5, color="#666666", style="italic")

plt.savefig(os.path.join(OUT_DIR, "catalog_map_muq.png"), dpi=300)
plt.savefig(os.path.join(OUT_DIR, "catalog_map_muq.pdf"))
print(f"[SAVED] catalog_map_muq.png")
print(f"[SAVED] catalog_map_muq.pdf")
