"""
2D UMAP map of artist catalog centroids in CLAP embedding space.
Color by vulnerability tier, size by vulnerability score.
"""
import numpy as np
import pandas as pd
import json
import os
import sys

# Accept output dir as argument
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

# --- Paths (cluster) ---
V1_ANALYSIS = os.environ.get("V1_ANALYSIS", "/scratch/$USER/runs/2026-03-10_full/analysis")
V1_EMBEDDINGS = os.environ.get("V1_EMBEDDINGS", "/scratch/$USER/runs/2026-03-10_full/embeddings")
MANIFEST = os.environ.get("MANIFEST", "/scratch/$USER/runs/2026-03-10_full/manifests/segment_manifest.csv")

# --- Load vulnerability scores for tier coloring ---
vuln_df = pd.read_csv(os.path.join(V1_ANALYSIS, "vulnerability_scores.csv"))
print(f"Vulnerability scores: {len(vuln_df)} artists")

# Assign 3-tier
def assign_tier(score):
    if score >= 0.67:
        return "High"
    elif score <= 0.33:
        return "Low"
    else:
        return "Intermediate"

vuln_df["tier"] = vuln_df["vulnerability_score"].apply(assign_tier)

# --- Load catalog CLAP embeddings ---
cat_emb_path = os.path.join(V1_EMBEDDINGS, "clap_catalog.npy")
cat_ids_path = os.path.join(V1_EMBEDDINGS, "clap_catalog_ids.npy")

if os.path.exists(cat_emb_path) and os.path.exists(cat_ids_path):
    embeddings = np.load(cat_emb_path)
    track_ids = np.load(cat_ids_path, allow_pickle=True)
    print(f"Catalog embeddings: {embeddings.shape}")
    print(f"Track IDs: {len(track_ids)}, first 5: {track_ids[:5]}")
else:
    print(f"Missing embeddings at {cat_emb_path}")
    # Try alternative: per-artist CLAP CSV
    clap_df = pd.read_csv(os.path.join(V1_ANALYSIS, "clap_per_artist.csv"))
    print(f"Using clap_per_artist.csv instead: {len(clap_df)} rows")
    print(f"Columns: {clap_df.columns.tolist()}")
    sys.exit(1)

# --- Load segment manifest to map track_id -> artist_id ---
manifest = pd.read_csv(MANIFEST)
print(f"Manifest: {len(manifest)} segments")
print(f"Manifest columns: {manifest.columns.tolist()}")

# Build track_id -> artist_id mapping
# track_ids from embeddings might be segment filenames or track IDs
# Let's check format
print(f"Track ID samples: {track_ids[:5]}")
print(f"Manifest artist_id samples: {manifest['artist_id'].unique()[:5]}")

# Try to extract artist_id from track_id
# Common formats: "003720.wav" or "003720_seg001.wav" or just "003720"
track_to_artist = {}
for _, row in manifest.iterrows():
    tid = str(row.get("track_id", row.get("segment_id", "")))
    aid = str(row["artist_id"]).zfill(6) if "artist_id" in manifest.columns else None
    if aid:
        # Map various ID formats
        track_to_artist[tid] = aid
        track_to_artist[tid.zfill(6)] = aid
        # If segment manifest, also map segment_id
        if "segment_id" in manifest.columns:
            seg_id = str(row["segment_id"])
            track_to_artist[seg_id] = aid
            # Strip extension
            track_to_artist[seg_id.replace(".wav", "")] = aid

print(f"Track-to-artist mappings: {len(track_to_artist)}")

# --- Compute per-artist centroids ---
artist_embeddings = {}
for i, tid in enumerate(track_ids):
    tid_str = str(tid).replace(".wav", "").replace(".npy", "")
    # Try direct match
    aid = track_to_artist.get(tid_str)
    if aid is None:
        # Try zero-padded
        aid = track_to_artist.get(tid_str.zfill(6))
    if aid is None:
        # Try extracting numeric part
        import re
        nums = re.findall(r'\d+', tid_str)
        for n in nums:
            aid = track_to_artist.get(n)
            if aid:
                break
            aid = track_to_artist.get(n.zfill(6))
            if aid:
                break
    if aid:
        if aid not in artist_embeddings:
            artist_embeddings[aid] = []
        artist_embeddings[aid].append(embeddings[i])

print(f"Artists with embeddings: {len(artist_embeddings)}")
if len(artist_embeddings) == 0:
    print("ERROR: No artist mappings found. Dumping samples for debugging.")
    print(f"First 10 track_ids: {track_ids[:10]}")
    print(f"First 10 manifest track_ids: {manifest.iloc[:10][['track_id', 'artist_id']].to_string()}")
    sys.exit(1)

# Compute centroids
artist_ids_ordered = []
centroids = []
for aid in sorted(artist_embeddings.keys()):
    embs = np.array(artist_embeddings[aid])
    centroids.append(embs.mean(axis=0))
    artist_ids_ordered.append(aid)

centroids = np.array(centroids)
print(f"Centroids: {centroids.shape}")

# --- Merge with vulnerability data ---
# Match artist_ids
vuln_df["artist_id_str"] = vuln_df["artist_id"].astype(str).str.zfill(6)
centroid_df = pd.DataFrame({"artist_id_str": artist_ids_ordered})
merged = centroid_df.merge(vuln_df, on="artist_id_str", how="left")
print(f"Merged: {len(merged)} artists, {merged['tier'].value_counts().to_dict()}")

# Fill missing
merged["tier"] = merged["tier"].fillna("Intermediate")
merged["vulnerability_score"] = merged["vulnerability_score"].fillna(0.5)
if "artist_name" not in merged.columns:
    merged["artist_name"] = merged["artist_id_str"]

# --- UMAP projection ---
try:
    from umap import UMAP
    reducer = UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
    coords_2d = reducer.fit_transform(centroids)
    method = "UMAP"
except ImportError:
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2, perplexity=min(15, len(centroids)-1), metric="cosine", random_state=42)
    coords_2d = reducer.fit_transform(centroids)
    method = "t-SNE"

print(f"Projection method: {method}")

merged["x"] = coords_2d[:, 0]
merged["y"] = coords_2d[:, 1]

# --- Save data for plotting ---
merged.to_csv(os.path.join(OUT_DIR, "catalog_map_data.csv"), index=False)
print(f"[SAVED] {os.path.join(OUT_DIR, 'catalog_map_data.csv')}")

# --- Plot ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

tier_colors = {"High": "#D32F2F", "Intermediate": "#F9A825", "Low": "#388E3C"}
tier_order = ["High", "Intermediate", "Low"]

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor("#FAFAFA")
fig.patch.set_facecolor("white")

# Plot each tier
for tier in tier_order:
    mask = merged["tier"] == tier
    subset = merged[mask]
    sizes = 80 + subset["vulnerability_score"] * 320  # 80-400 range
    ax.scatter(
        subset["x"], subset["y"],
        s=sizes,
        c=tier_colors[tier],
        alpha=0.75,
        edgecolors="white",
        linewidths=1.2,
        label=f"{tier} ({len(subset)})",
        zorder=3,
    )

# Label all artists
for _, row in merged.iterrows():
    name = row.get("artist_name", row["artist_id_str"])
    if pd.isna(name) or name == row["artist_id_str"]:
        name = row["artist_id_str"]
    # Truncate long names
    if len(str(name)) > 20:
        name = str(name)[:18] + "…"
    ax.annotate(
        name,
        (row["x"], row["y"]),
        fontsize=6.5,
        ha="center",
        va="bottom",
        xytext=(0, 8),
        textcoords="offset points",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        zorder=4,
    )

# Genre cluster hulls (if genre info available)
if "genre" in merged.columns:
    from scipy.spatial import ConvexHull
    for genre in merged["genre"].dropna().unique():
        gmask = merged["genre"] == genre
        if gmask.sum() >= 3:
            pts = merged.loc[gmask, ["x", "y"]].values
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hull_pts, 0], pts[hull_pts, 1], alpha=0.05, color="gray")
                ax.plot(pts[hull_pts, 0], pts[hull_pts, 1], "--", alpha=0.2, color="gray", linewidth=0.8)
            except:
                pass

# Legend
legend = ax.legend(
    title="Vulnerability Tier",
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    edgecolor="#CCCCCC",
    fontsize=9,
    title_fontsize=10,
)

ax.set_xlabel(f"{method} Dimension 1", fontsize=11, labelpad=8)
ax.set_ylabel(f"{method} Dimension 2", fontsize=11, labelpad=8)
ax.set_title(
    "Artist Catalog Centroids in CLAP Embedding Space\n"
    "Point size proportional to vulnerability score · Color by tier (≥0.67 High, ≤0.33 Low)",
    fontsize=12,
    fontweight="bold",
    pad=12,
)

ax.tick_params(axis="both", which="both", length=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

# Add note
fig.text(
    0.02, 0.02,
    f"50 artists · {method} projection (cosine distance) · 2-signal composite score (CLAP gap + FAD)",
    fontsize=7.5,
    color="#666666",
    style="italic",
)

plt.savefig(os.path.join(OUT_DIR, "catalog_map.png"), dpi=300)
plt.savefig(os.path.join(OUT_DIR, "catalog_map.pdf"))
print(f"[SAVED] {os.path.join(OUT_DIR, 'catalog_map.png')}")
print(f"[SAVED] {os.path.join(OUT_DIR, 'catalog_map.pdf')}")
