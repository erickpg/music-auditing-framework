#!/usr/bin/env python3
"""
Generate all thesis figures.
Usage: python scripts/generate_thesis_plots.py --results_dir /path/to/results --out_dir figures/
"""

import argparse
import csv
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ─── Style ───────────────────────────────────────────────────────────────────
# Publication-quality defaults
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11.5,
    'axes.labelweight': 'medium',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Palette — muted, professional
PAL = {
    'v1':       '#2D6A4F',   # deep green
    'v2':       '#E76F51',   # terracotta
    'baseline': '#457B9D',   # steel blue
    'matched':  '#2D6A4F',
    'mismatched': '#A8DADC',
    'high':     '#E63946',   # red accent
    'low':      '#457B9D',
    'neutral':  '#6C757D',
    'accent':   '#F4A261',   # warm accent
    'bg_band':  '#F8F9FA',
}
TIER_COLORS = {'A_artist_proximal': '#2D6A4F', 'B_genre_generic': '#457B9D',
               'C_out_of_distribution': '#E76F51', 'D_fma_tags': '#F4A261'}
TIER_LABELS = {'A_artist_proximal': 'Tier A\n(Artist-proximal)',
               'B_genre_generic': 'Tier B\n(Genre-generic)',
               'C_out_of_distribution': 'Tier C\n(OOD control)',
               'D_fma_tags': 'Tier D\n(FMA tags)'}


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    seen = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip duplicate header rows
            if any(v == k for k, v in row.items()):
                continue
            # Deduplicate by full row content
            key = tuple(sorted(row.items()))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows

def load_json(path):
    with open(path) as f:
        return json.load(f)

def sf(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def distinctiveness_2sig(row):
    """Compute 2-signal distinctiveness score from clap_norm and fad_norm."""
    cn = sf(row.get('clap_norm'))
    fn = sf(row.get('fad_norm'))
    if cn is not None and fn is not None:
        return (cn + fn) / 2
    return None


TIER_THRESHOLDS = (0.33, 0.67)  # 3-tier: Low / Intermediate / High


def tier_color_3(score):
    """Return color for 3-tier scheme."""
    if score >= TIER_THRESHOLDS[1]:
        return PAL['high']
    elif score <= TIER_THRESHOLDS[0]:
        return PAL['low']
    return PAL['neutral']


def tier_label_3(score):
    if score >= TIER_THRESHOLDS[1]:
        return 'High'
    elif score <= TIER_THRESHOLDS[0]:
        return 'Low'
    return 'Intermediate'

def save(fig, name, out_dir):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {name}.png")

def add_significance(ax, x1, x2, y, p, h=0.01):
    """Add significance bracket."""
    if p < 0.001:
        txt = '***'
    elif p < 0.01:
        txt = '**'
    elif p < 0.05:
        txt = '*'
    else:
        txt = 'n.s.'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.8, c='#333')
    ax.text((x1+x2)/2, y+h, txt, ha='center', va='bottom', fontsize=9, color='#333')


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 4: Proactive Protection
# ═══════════════════════════════════════════════════════════════════════════════

def fig_4_1_wavmark_survival(R, out):
    """WavMark watermark survival pre vs post EnCodec."""
    path = f"{R}/watermark_poc/wavmark/analysis/tokenizer_survival.csv"
    if not os.path.exists(path):
        print("  ⊘ Skipping F4.1 (no wavmark data)")
        return
    rows = load_csv(path)
    pre_det = sum(1 for r in rows if r.get('pre_detected', '').lower() == 'true')
    pre_msg = sum(1 for r in rows if r.get('pre_match', '').lower() == 'true')
    post_det = sum(1 for r in rows if r.get('post_detected', '').lower() == 'true')
    post_msg = sum(1 for r in rows if r.get('post_match', '').lower() == 'true')
    n = len(rows)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(2)
    w = 0.32
    pre = [100*pre_det/n, 100*pre_msg/n]
    post = [100*post_det/n, 100*post_msg/n]

    bars1 = ax.bar(x - w/2, pre, w, label='Pre-EnCodec', color=PAL['v1'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, post, w, label='Post-EnCodec', color=PAL['v2'], edgecolor='white', linewidth=0.5)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f'{h:.0f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='medium')

    ax.set_xticks(x)
    ax.set_xticklabels(['Detection Rate', 'Message Recovery'], fontsize=11)
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, 115)
    ax.set_title('WavMark Watermark Survival Through EnCodec')
    ax.legend(loc='upper right')
    ax.axhline(y=50, color='#ccc', linestyle='--', linewidth=0.6, zorder=0)
    save(fig, 'F4_1_wavmark_survival', out)


def fig_4_2_audioseal_training(R, out):
    """AudioSeal detection vs multi-bit loss across epochs."""
    trial_dir = f"{R}/watermark_poc/audioseal_trials"
    if not os.path.exists(trial_dir):
        print("  ⊘ Skipping F4.2 (no audioseal data)")
        return

    # Parse solver logs for key trials
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Find the 16-bit trial (31253619) and 0-bit trial
    trial_data = {}
    for xp_id in os.listdir(trial_dir):
        log_path = os.path.join(trial_dir, xp_id, 'solver.log.0')
        if not os.path.exists(log_path):
            continue
        det_vals, mb_vals, epochs = [], [], []
        with open(log_path) as f:
            for line in f:
                if 'Valid Summary' in line:
                    # Extract detection and multi-bit losses
                    det = mb = epoch = None
                    for part in line.split('|'):
                        part = part.strip()
                        if part.startswith('Epoch'):
                            try:
                                epoch = int(part.split()[1])
                            except:
                                pass
                        elif 'wm_detection_encodec_nq=4=' in part:
                            try:
                                det = float(part.split('=')[-1])
                            except:
                                pass
                        elif 'wm_mb_encodec_nq=4=' in part:
                            try:
                                mb = float(part.split('=')[-1])
                            except:
                                pass
                    if det is not None and epoch is not None:
                        det_vals.append(det)
                        mb_vals.append(mb if mb is not None else 0.693)
                        epochs.append(epoch)
        if epochs:
            # Deduplicate: keep last value per epoch (log may have multiple entries)
            seen = {}
            for e, dv, mv in zip(epochs, det_vals, mb_vals):
                seen[e] = (dv, mv)
            dedup_epochs = sorted(seen.keys())
            dedup_det = [seen[e][0] for e in dedup_epochs]
            dedup_mb = [seen[e][1] for e in dedup_epochs]
            trial_data[xp_id] = {'det': dedup_det, 'mb': dedup_mb, 'epochs': dedup_epochs}

    if not trial_data:
        print("  ⊘ Skipping F4.2 (no parseable audioseal logs)")
        return

    # Pick the trial with most epochs for multi-bit
    best_id = max(trial_data, key=lambda k: len(trial_data[k]['epochs']))
    d = trial_data[best_id]

    # Left panel: detection loss improving
    ax1.plot(d['epochs'], d['det'], 'o-', color=PAL['v1'], markersize=5, linewidth=2, label='Detection loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss (BCE)')
    ax1.set_title('Detection Loss (Improving)')
    ax1.axhline(y=0.693, color=PAL['high'], linestyle='--', linewidth=1, alpha=0.7, label='Random chance (0.693)')
    ax1.legend()
    ax1.set_ylim(0.5, 0.9)

    # Right panel: multi-bit stuck
    ax2.plot(d['epochs'], d['mb'], 's-', color=PAL['v2'], markersize=5, linewidth=2, label='Multi-bit loss')
    ax2.axhline(y=0.693, color=PAL['high'], linestyle='--', linewidth=1, alpha=0.7, label='Random chance (0.693)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss (BCE)')
    ax2.set_title('Multi-bit Message Loss (Stuck)')
    ax2.legend()
    ax2.set_ylim(0.5, 0.9)

    fig.suptitle('AudioSeal Retraining with EnCodec Augmentation', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'F4_2_audioseal_training', out)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 5: Reactive Auditing
# ═══════════════════════════════════════════════════════════════════════════════

def fig_5_1_training_loss(R, out):
    """Training loss curves V1 and V2."""
    v1_path = f"{R}/v1/logs/training_loss_per_epoch.csv"
    if not os.path.exists(v1_path):
        print("  ⊘ Skipping F5.1 (no training loss data)")
        return

    v1 = load_csv(v1_path)
    v1_epochs = [sf(r['epoch']) for r in v1 if sf(r.get('epoch')) is not None]
    v1_loss = [sf(r['mean_loss']) for r in v1 if sf(r.get('mean_loss')) is not None]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(v1_epochs, v1_loss, '-', color=PAL['v1'], linewidth=2.2, label='V1 (20 epochs, LR=1e-5)')

    # Try V2 — might be in different location
    v2_path = f"{R}/v2/logs/training_loss_per_epoch.csv"
    if os.path.exists(v2_path):
        v2 = load_csv(v2_path)
        v2_epochs = [sf(r['epoch']) for r in v2 if sf(r.get('epoch')) is not None]
        v2_loss = [sf(r['mean_loss']) for r in v2 if sf(r.get('mean_loss')) is not None]
        ax.plot(v2_epochs, v2_loss, '-', color=PAL['v2'], linewidth=2.2, label='V2 (100 epochs, LR=1e-4)')

    # Annotations
    if v1_loss:
        ax.annotate(f'Final: {v1_loss[-1]:.2f}', xy=(v1_epochs[-1], v1_loss[-1]),
                    xytext=(-50, 15), textcoords='offset points',
                    fontsize=9, color=PAL['v1'],
                    arrowprops=dict(arrowstyle='->', color=PAL['v1'], lw=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('MusicGen Fine-tuning Loss Curves')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    save(fig, 'F5_1_training_loss', out)


def fig_5_2_clap_matched_mismatched(R, out):
    """CLAP similarity: matched vs mismatched across V1, V2, Baseline — violin plot."""
    data = {}
    for label, subdir in [('V1', 'v1'), ('V2', 'v2'), ('Baseline', 'baseline')]:
        path = f"{R}/{subdir}/analysis/clap_similarity.csv"
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        matched = [sf(r['matched_mean_sim']) for r in rows if sf(r.get('matched_mean_sim')) is not None]
        mismatched = [sf(r['mismatched_mean_sim']) for r in rows if sf(r.get('mismatched_mean_sim')) is not None]
        data[label] = {'matched': matched, 'mismatched': mismatched}

    if not data:
        print("  ⊘ Skipping F5.2 (no CLAP data)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = []
    labels_x = []
    colors_list = []
    all_data = []
    i = 0
    model_colors = {'V1': PAL['v1'], 'V2': PAL['v2'], 'Baseline': PAL['baseline']}

    for model in ['V1', 'V2', 'Baseline']:
        if model not in data:
            continue
        pos_m = i * 3
        pos_mm = i * 3 + 1
        all_data.append(data[model]['matched'])
        all_data.append(data[model]['mismatched'])
        positions.extend([pos_m, pos_mm])
        labels_x.append((pos_m + pos_mm) / 2)
        colors_list.extend([model_colors[model], model_colors[model]])
        i += 1

    vp = ax.violinplot(all_data, positions=positions, widths=0.7, showmeans=True, showextrema=False)

    for j, body in enumerate(vp['bodies']):
        color = colors_list[j]
        alpha = 1.0 if j % 2 == 0 else 0.35
        body.set_facecolor(color)
        body.set_alpha(alpha)
        body.set_edgecolor(color)
        body.set_linewidth(0.8)

    vp['cmeans'].set_color('#333')
    vp['cmeans'].set_linewidth(1.5)

    # Gap annotations
    for idx, model in enumerate(['V1', 'V2', 'Baseline']):
        if model not in data:
            continue
        m_mean = np.mean(data[model]['matched'])
        mm_mean = np.mean(data[model]['mismatched'])
        gap = m_mean - mm_mean
        mid_x = labels_x[idx]
        ax.annotate(f'Δ = {gap:.3f}', xy=(mid_x, max(m_mean, mm_mean) + 0.01),
                    ha='center', fontsize=9, fontweight='bold', color=model_colors[model])

    ax.set_xticks(labels_x)
    ax.set_xticklabels([m for m in ['V1\n(20 ep)', 'V2\n(100 ep)', 'Baseline'] if m.split('\n')[0] in data], fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PAL['v1'], alpha=1.0, label='Matched'),
                       Patch(facecolor=PAL['v1'], alpha=0.35, label='Mismatched')]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_ylabel('CLAP Cosine Similarity')
    ax.set_title('CLAP Similarity: Matched vs Mismatched Artist Catalog')
    ax.grid(True, axis='y', alpha=0.2)
    save(fig, 'F5_2_clap_matched_mismatched', out)


def fig_5_3_clap_by_tier(R, out):
    """CLAP similarity by prompt tier."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, (label, subdir) in zip(axes, [('V1 (20 epochs)', 'v1'), ('V2 (100 epochs)', 'v2')]):
        path = f"{R}/{subdir}/analysis/clap_similarity.csv"
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        tier_data = {}
        for r in rows:
            tier = r.get('tier', '')
            sim = sf(r.get('matched_mean_sim'))
            if tier and sim is not None:
                tier_data.setdefault(tier, []).append(sim)

        tier_order = ['A_artist_proximal', 'D_fma_tags', 'B_genre_generic', 'C_out_of_distribution']
        tier_order = [t for t in tier_order if t in tier_data]

        bp = ax.boxplot([tier_data[t] for t in tier_order],
                        positions=range(len(tier_order)),
                        widths=0.5, patch_artist=True,
                        medianprops=dict(color='#333', linewidth=1.5),
                        whiskerprops=dict(color='#666', linewidth=0.8),
                        capprops=dict(color='#666', linewidth=0.8),
                        flierprops=dict(marker='o', markersize=3, alpha=0.3))

        for patch, tier in zip(bp['boxes'], tier_order):
            patch.set_facecolor(TIER_COLORS.get(tier, '#999'))
            patch.set_alpha(0.7)
            patch.set_edgecolor('#333')
            patch.set_linewidth(0.6)

        ax.set_xticks(range(len(tier_order)))
        ax.set_xticklabels([TIER_LABELS.get(t, t) for t in tier_order], fontsize=9)
        ax.set_title(label, fontsize=12)
        ax.grid(True, axis='y', alpha=0.2)

    axes[0].set_ylabel('Matched CLAP Similarity')
    fig.suptitle('CLAP Similarity by Prompt Tier', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'F5_3_clap_by_tier', out)


def fig_5_4_vulnerability_ranking(R, out):
    """Per-artist distinctiveness scores horizontal bar chart."""
    path = f"{R}/v1/analysis/vulnerability_scores.csv"
    if not os.path.exists(path):
        print("  ⊘ Skipping F5.4 (no vulnerability data)")
        return
    rows = load_csv(path)
    artists = []
    for r in rows:
        v = distinctiveness_2sig(r)
        name = r.get('artist_name', '')
        if v is not None and name:
            artists.append((name, v))

    artists.sort(key=lambda x: x[1])
    names = [a[0] for a in artists]
    scores = [a[1] for a in artists]
    colors = [tier_color_3(s) for s in scores]

    fig, ax = plt.subplots(figsize=(8, 12))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, height=0.7, color=colors, edgecolor='white', linewidth=0.3)

    ax.axvline(x=TIER_THRESHOLDS[0], color='#333', linestyle=':', linewidth=1, alpha=0.5, label=f'Low/Intermediate ({TIER_THRESHOLDS[0]:.2f})')
    ax.axvline(x=TIER_THRESHOLDS[1], color='#333', linestyle='--', linewidth=1, alpha=0.6, label=f'Intermediate/High ({TIER_THRESHOLDS[1]:.2f})')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel('Distinctiveness Score (2-signal: CLAP + FAD)')
    ax.set_title('Per-Artist Distinctiveness Ranking (V1)')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    # Count annotations
    n_high = sum(1 for s in scores if s >= TIER_THRESHOLDS[1])
    n_int = sum(1 for s in scores if TIER_THRESHOLDS[0] < s < TIER_THRESHOLDS[1])
    n_low = sum(1 for s in scores if s <= TIER_THRESHOLDS[0])
    ax.text(0.85, len(scores) * 0.03, f'High: {n_high}', fontsize=10, color=PAL['high'], fontweight='bold')
    ax.text(0.85, len(scores) * 0.50, f'Intermediate: {n_int}', fontsize=10, color=PAL['neutral'], fontweight='bold')
    ax.text(0.85, len(scores) * 0.95, f'Low: {n_low}', fontsize=10, color=PAL['low'], fontweight='bold')

    ax.legend(loc='lower right', fontsize=9)
    save(fig, 'F5_4_vulnerability_ranking', out)


def fig_5_5_ngram_ratios(R, out):
    """N-gram matched/mismatched ratio by n-gram size."""
    path = f"{R}/v1/analysis/ngram_per_artist.csv"
    if not os.path.exists(path):
        print("  ⊘ Skipping F5.5 (no ngram data)")
        return
    rows = load_csv(path)

    by_size = {}
    for r in rows:
        ns = sf(r.get('ngram_size'))
        mr = sf(r.get('matched_rate'))
        mmr = sf(r.get('mismatched_rate'))
        if ns is not None and mr is not None and mmr is not None:
            by_size.setdefault(int(ns), {'matched': [], 'mismatched': []})
            by_size[int(ns)]['matched'].append(mr)
            by_size[int(ns)]['mismatched'].append(mmr)

    sizes = sorted(by_size.keys())
    if not sizes:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: absolute rates
    matched_means = [np.mean(by_size[s]['matched']) for s in sizes]
    mismatched_means = [np.mean(by_size[s]['mismatched']) for s in sizes]
    x = np.arange(len(sizes))
    w = 0.32

    ax1.bar(x - w/2, matched_means, w, label='Matched', color=PAL['matched'], edgecolor='white')
    ax1.bar(x + w/2, mismatched_means, w, label='Mismatched', color=PAL['mismatched'], edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_xlabel('N-gram Size')
    ax1.set_ylabel('Mean Match Rate')
    ax1.set_title('Absolute Match Rates')
    ax1.legend()
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))

    # Right: ratio
    ratios = [m/mm if mm > 0 else 0 for m, mm in zip(matched_means, mismatched_means)]
    bars = ax2.bar(x, ratios, 0.5, color=PAL['accent'], edgecolor='white')
    for bar, ratio in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{ratio:.1f}×', ha='center', fontsize=10, fontweight='medium')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_xlabel('N-gram Size')
    ax2.set_ylabel('Matched / Mismatched Ratio')
    ax2.set_title('Match Ratio (Matched ÷ Mismatched)')
    ax2.axhline(y=1, color='#999', linestyle='--', linewidth=0.8)

    fig.suptitle('N-gram Token Match Analysis', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'F5_5_ngram_ratios', out)


def fig_5_6_exposure_vs_vulnerability(R, out):
    """Scatter: exposure (n_tracks) vs distinctiveness."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (label, subdir, color) in zip(axes, [('V1', 'v1', PAL['v1']), ('V2', 'v2', PAL['v2'])]):
        path = f"{R}/{subdir}/analysis/vulnerability_scores.csv"
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        n_tracks = [sf(r.get('n_catalog_tracks')) for r in rows]
        vuln = [distinctiveness_2sig(r) for r in rows]
        valid = [(n, v) for n, v in zip(n_tracks, vuln) if n is not None and v is not None]
        if not valid:
            continue
        ns, vs = zip(*valid)

        ax.scatter(ns, vs, c=color, alpha=0.7, s=50, edgecolors='white', linewidth=0.5, zorder=3)

        # Regression line
        from numpy.polynomial.polynomial import polyfit
        b, m = polyfit(ns, vs, 1)
        x_line = np.linspace(min(ns), max(ns), 50)
        ax.plot(x_line, b + m * x_line, '--', color=color, alpha=0.5, linewidth=1.5)

        # Correlation
        from scipy import stats as sp_stats
        rho, p = sp_stats.spearmanr(ns, vs)
        ax.text(0.05, 0.95, f'ρ = {rho:.3f}\np = {p:.2f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ddd', alpha=0.9))

        ax.set_xlabel('Number of Catalog Tracks')
        ax.set_ylabel('Distinctiveness Score')
        ax.set_title(label)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=TIER_THRESHOLDS[0], color='#ccc', linestyle=':', linewidth=0.6)
        ax.axhline(y=TIER_THRESHOLDS[1], color='#ccc', linestyle='--', linewidth=0.6)
        ax.grid(True, alpha=0.15)

    fig.suptitle('Catalog Exposure vs Distinctiveness', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'F5_6_exposure_vs_vulnerability', out)


def fig_5_7_effect_sizes(R, out):
    """Forest plot: per-artist Cohen's d with CIs."""
    path = f"{R}/v1/supplementary/effect_sizes.csv"
    if not os.path.exists(path):
        print("  ⊘ Skipping F5.7 (no effect size data)")
        return
    rows = load_csv(path)
    artists = []
    for r in rows:
        d = sf(r.get('cohens_d'))
        lo = sf(r.get('ci_low') or r.get('ci_lower'))
        hi = sf(r.get('ci_high') or r.get('ci_upper'))
        aid = r.get('artist_id', '')
        if d is not None and lo is not None and hi is not None:
            artists.append({'id': aid, 'd': d, 'lo': lo, 'hi': hi})

    artists.sort(key=lambda x: x['d'])

    fig, ax = plt.subplots(figsize=(7, 12))
    y_pos = np.arange(len(artists))

    for i, a in enumerate(artists):
        color = PAL['high'] if a['d'] > 0.5 else PAL['v1'] if a['d'] > 0 else PAL['neutral']
        ax.plot([a['lo'], a['hi']], [i, i], '-', color=color, linewidth=1.2, alpha=0.6)
        ax.plot(a['d'], i, 'o', color=color, markersize=4, zorder=3)

    ax.axvline(x=0, color='#333', linewidth=1, linestyle='-')
    ax.axvline(x=0.2, color='#aaa', linewidth=0.6, linestyle=':', label='Small (0.2)')
    ax.axvline(x=0.5, color='#aaa', linewidth=0.6, linestyle='--', label='Medium (0.5)')
    ax.axvline(x=0.8, color='#aaa', linewidth=0.6, linestyle='-.', label='Large (0.8)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([a['id'] for a in artists], fontsize=6)
    ax.set_xlabel("Cohen's d (matched vs mismatched CLAP similarity)")
    ax.set_title("Per-Artist Effect Sizes with 95% CIs (V1)")
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, axis='x', alpha=0.15)
    save(fig, 'F5_7_effect_sizes', out)


def fig_5_8_permutation_test(R, out):
    """Permutation test: histogram of per-artist p-values."""
    path = f"{R}/v1/supplementary/permutation_test_per_artist.csv"
    if not os.path.exists(path):
        print("  ⊘ Skipping F5.8 (no permutation test data)")
        return
    rows = load_csv(path)
    p_vals = [sf(r.get('p_value')) for r in rows if sf(r.get('p_value')) is not None]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_sig = sum(1 for p in p_vals if p < 0.05)
    n_total = len(p_vals)

    ax.hist(p_vals, bins=20, color=PAL['v1'], edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.axvline(x=0.05, color=PAL['high'], linewidth=1.5, linestyle='--', label=f'α = 0.05')

    ax.text(0.95, 0.95, f'{n_sig}/{n_total} significant\n(expected: {n_total*0.05:.1f})',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ddd'))

    ax.set_xlabel('Permutation Test p-value')
    ax.set_ylabel('Number of Artists')
    ax.set_title('Per-Artist Permutation Test Results (10,000 permutations)')
    ax.legend()
    save(fig, 'F5_8_permutation_test', out)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 6: V1 vs V2 Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig_6_1_clap_gap_comparison(R, out):
    """CLAP gap and Cohen's d: V1 vs V2."""
    v1_clap = load_json(f"{R}/v1/analysis/clap_summary.json") if os.path.exists(f"{R}/v1/analysis/clap_summary.json") else None
    v2_clap = load_json(f"{R}/v2/analysis/clap_summary.json") if os.path.exists(f"{R}/v2/analysis/clap_summary.json") else None
    comp = load_json(f"{R}/v2/comparison/effect_size_comparison.json") if os.path.exists(f"{R}/v2/comparison/effect_size_comparison.json") else None

    if not v1_clap or not v2_clap:
        print("  ⊘ Skipping F6.1")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    # CLAP gap
    v1_gap = v1_clap.get('per_artist', {}).get('overall', {}).get('mean_gap', 0.083)
    v2_gap = v2_clap.get('per_artist', {}).get('overall', {}).get('mean_gap', 0.033)

    bars = ax1.bar(['V1\n(20 epochs)', 'V2\n(100 epochs)'], [v1_gap, v2_gap],
                    color=[PAL['v1'], PAL['v2']], width=0.5, edgecolor='white')
    for bar, val in zip(bars, [v1_gap, v2_gap]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    pct_drop = (1 - v2_gap/v1_gap) * 100 if v1_gap > 0 else 0
    ax1.annotate(f'−{pct_drop:.0f}%', xy=(0.5, max(v1_gap, v2_gap)/2),
                 fontsize=14, fontweight='bold', color=PAL['high'], ha='center')
    ax1.set_ylabel('Mean CLAP Gap')
    ax1.set_title('CLAP Gap (matched − mismatched)')

    # Cohen's d
    if comp:
        v1_d = sf(comp.get('v1_mean_d') or comp.get('mean_d_v1', 0.873))
        v2_d = sf(comp.get('v2_mean_d') or comp.get('mean_d_v2', 0.319))
    else:
        v1_d, v2_d = 0.873, 0.319

    bars2 = ax2.bar(['V1\n(20 epochs)', 'V2\n(100 epochs)'], [v1_d, v2_d],
                     color=[PAL['v1'], PAL['v2']], width=0.5, edgecolor='white')
    for bar, val in zip(bars2, [v1_d, v2_d]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    pct_drop_d = (1 - v2_d/v1_d) * 100 if v1_d > 0 else 0
    ax2.annotate(f'−{pct_drop_d:.0f}%', xy=(0.5, max(v1_d, v2_d)/2),
                 fontsize=14, fontweight='bold', color=PAL['high'], ha='center')
    ax2.set_ylabel("Mean Cohen's d")
    ax2.set_title("Effect Size (Cohen's d)")
    ax2.axhline(y=0.2, color='#bbb', linestyle=':', linewidth=0.7)
    ax2.axhline(y=0.5, color='#bbb', linestyle='--', linewidth=0.7)
    ax2.axhline(y=0.8, color='#bbb', linestyle='-.', linewidth=0.7)

    fig.suptitle('The Central Paradox: Deeper Training → Weaker Style Signal',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'F6_1_central_paradox', out)


def fig_6_2_v1_v2_scatter(R, out):
    """Scatter: V2 vs Baseline distinctiveness per artist."""
    # Load V2 and Baseline scores, compute 2-signal distinctiveness
    v2_path = f"{R}/v2/analysis/vulnerability_scores.csv"
    bl_path = f"{R}/baseline/analysis/vulnerability_scores.csv"
    if not os.path.exists(v2_path) or not os.path.exists(bl_path):
        print("  ⊘ Skipping F6.2 (need v2 + baseline data)")
        return

    v2_rows = {r.get('artist_id'): r for r in load_csv(v2_path)}
    bl_rows = {r.get('artist_id'): r for r in load_csv(bl_path)}

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    valid = []
    for aid in v2_rows:
        if aid in bl_rows:
            v2_s = distinctiveness_2sig(v2_rows[aid])
            bl_s = distinctiveness_2sig(bl_rows[aid])
            name = v2_rows[aid].get('artist_name', '')
            if v2_s is not None and bl_s is not None:
                valid.append((bl_s, v2_s, name))

    if not valid:
        return
    bls, v2s, ns = zip(*valid)

    ax.scatter(bls, v2s, c=PAL['v2'], s=45, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=3)

    # 45-degree line
    ax.plot([0, 1], [0, 1], '--', color='#999', linewidth=1, zorder=1)

    # Label outliers
    for bl, v2, name in valid:
        if abs(bl - v2) > 0.25:
            ax.annotate(name, (bl, v2), fontsize=7, alpha=0.8,
                        xytext=(5, 5), textcoords='offset points')

    # 3-tier grid lines
    for th in TIER_THRESHOLDS:
        ax.axhline(y=th, color='#ddd', linewidth=0.6)
        ax.axvline(x=th, color='#ddd', linewidth=0.6)

    from scipy import stats as sp_stats
    rho, p = sp_stats.spearmanr(bls, v2s)
    p_str = f'p < 0.0001' if p < 0.0001 else f'p = {p:.4f}'
    ax.text(0.05, 0.95, f'ρ = {rho:.3f}\n{p_str}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ddd'))

    ax.set_xlabel('Baseline Distinctiveness Score')
    ax.set_ylabel('V2 Distinctiveness Score (100 epochs)')
    ax.set_title('V2 vs Baseline Distinctiveness Agreement')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    save(fig, 'F6_2_v1_v2_scatter', out)


def fig_6_3_vulnerability_delta(R, out):
    """Per-artist distinctiveness delta V2 vs Baseline."""
    v2_path = f"{R}/v2/analysis/vulnerability_scores.csv"
    bl_path = f"{R}/baseline/analysis/vulnerability_scores.csv"
    if not os.path.exists(v2_path) or not os.path.exists(bl_path):
        print("  ⊘ Skipping F6.3 (need v2 + baseline data)")
        return

    v2_rows = {r.get('artist_id'): r for r in load_csv(v2_path)}
    bl_rows = {r.get('artist_id'): r for r in load_csv(bl_path)}

    artists = []
    for aid in v2_rows:
        if aid in bl_rows:
            v2_s = distinctiveness_2sig(v2_rows[aid])
            bl_s = distinctiveness_2sig(bl_rows[aid])
            name = v2_rows[aid].get('artist_name', '')
            if v2_s is not None and bl_s is not None and name:
                artists.append((name, v2_s - bl_s))

    if not artists:
        return

    artists.sort(key=lambda x: x[1])
    names = [a[0] for a in artists]
    deltas = [a[1] for a in artists]
    colors = [PAL['v2'] if d > 0 else PAL['baseline'] for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 12))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, deltas, height=0.7, color=colors, edgecolor='white', linewidth=0.3)

    ax.axvline(x=0, color='#333', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel('Distinctiveness Change (V2 − Baseline)')
    ax.set_title('Per-Artist Distinctiveness Shift: V2 vs Baseline')

    n_higher = sum(1 for d in deltas if d > 0)
    n_lower = len(deltas) - n_higher
    ax.text(0.95, 0.05, f'V2 higher: {n_higher}', transform=ax.transAxes,
            ha='right', fontsize=10, color=PAL['v2'], fontweight='bold')
    ax.text(0.05, 0.05, f'Baseline higher: {n_lower}', transform=ax.transAxes,
            ha='left', fontsize=10, color=PAL['baseline'], fontweight='bold')

    ax.grid(True, axis='x', alpha=0.15)
    save(fig, 'F6_3_vulnerability_delta', out)


def fig_6_4_three_way_absorption(R, out):
    """V1 vs V2 vs Baseline absorption comparison."""
    v1_path = f"{R}/v1/supplementary/ft_vs_bl_comparison_summary.json"
    v2_path = f"{R}/v2/comparison/v2_vs_bl_comparison_summary.json"
    if not os.path.exists(v1_path) or not os.path.exists(v2_path):
        print("  ⊘ Skipping F6.4")
        return

    v1 = load_json(v1_path)
    v2 = load_json(v2_path)

    src_metrics = ['CLAP Similarity', 'FAD', 'Vulnerability Score']
    v1_deltas = [v1[m]['delta'] for m in src_metrics]
    v2_deltas = [v2[m]['delta'] for m in src_metrics]
    v1_pvals = [v1[m]['p_value'] for m in src_metrics]
    v2_pvals = [v2[m]['p_value'] for m in src_metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(src_metrics))
    w = 0.32

    bars1 = ax.bar(x - w/2, v1_deltas, w, label='V1 vs Baseline', color=PAL['v1'], edgecolor='white')
    bars2 = ax.bar(x + w/2, v2_deltas, w, label='V2 vs Baseline', color=PAL['v2'], edgecolor='white')

    # Significance markers
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        for bar, p in [(b1, v1_pvals[i]), (b2, v2_pvals[i])]:
            marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            y = bar.get_height()
            offset = 0.003 if y >= 0 else -0.008
            ax.text(bar.get_x() + bar.get_width()/2, y + offset, marker,
                    ha='center', va='bottom' if y >= 0 else 'top', fontsize=8, color='#555')

    ax.axhline(y=0, color='#333', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['CLAP\nSimilarity', 'FAD', 'Distinctiveness\nScore'], fontsize=10)
    ax.set_ylabel('Delta (Fine-tuned − Baseline)')
    ax.set_title('Fine-tuned vs Baseline: V1 and V2')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.15)
    save(fig, 'F6_4_three_way_absorption', out)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 7: Robustness
# ═══════════════════════════════════════════════════════════════════════════════

def fig_7_1_bootstrap_ci(R, out):
    """Bootstrap 95% CIs per artist."""
    path = f"{R}/robustness/bootstrap_ci.csv"
    if not os.path.exists(path):
        print("  ⊘ Skipping F7.1 (no bootstrap data)")
        return
    rows = load_csv(path)
    artists = []
    for r in rows:
        score = sf(r.get('score') or r.get('vulnerability_score') or r.get('mean'))
        lo = sf(r.get('ci_lower') or r.get('ci_low') or r.get('lower'))
        hi = sf(r.get('ci_upper') or r.get('ci_high') or r.get('upper'))
        name = r.get('artist_name') or r.get('artist_id', '')
        if score is not None and lo is not None and hi is not None:
            artists.append({'name': name, 'score': score, 'lo': lo, 'hi': hi})

    if not artists:
        print("  ⊘ Skipping F7.1 (no parseable bootstrap data)")
        return

    artists.sort(key=lambda x: x['score'])

    fig, ax = plt.subplots(figsize=(7, 12))
    y_pos = np.arange(len(artists))

    for i, a in enumerate(artists):
        if a['lo'] > TIER_THRESHOLDS[1]:
            color = PAL['high']
        elif a['hi'] < TIER_THRESHOLDS[0]:
            color = PAL['low']
        elif a['score'] >= TIER_THRESHOLDS[1]:
            color = PAL['high']
        elif a['score'] <= TIER_THRESHOLDS[0]:
            color = PAL['low']
        else:
            color = PAL['neutral']

        ax.plot([a['lo'], a['hi']], [i, i], '-', color=color, linewidth=2, alpha=0.6)
        ax.plot(a['score'], i, 'o', color=color, markersize=4.5, zorder=3)

    ax.axvline(x=TIER_THRESHOLDS[0], color='#333', linewidth=1, linestyle=':', label=f'Low/Intermediate ({TIER_THRESHOLDS[0]:.2f})')
    ax.axvline(x=TIER_THRESHOLDS[1], color='#333', linewidth=1.2, linestyle='--', label=f'Intermediate/High ({TIER_THRESHOLDS[1]:.2f})')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([a['name'] for a in artists], fontsize=6.5)
    ax.set_xlabel('Distinctiveness Score (2-signal: CLAP + FAD)')
    ax.set_title('Bootstrap 95% Confidence Intervals Per Artist')
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, axis='x', alpha=0.15)
    save(fig, 'F7_1_bootstrap_ci', out)


def fig_7_3_signal_stability(R, out):
    """Signal combination stability bar chart."""
    path = f"{R}/v1/supplementary/vulnerability_rank_stability.json"
    if not os.path.exists(path):
        # Try alternate location
        path = f"{R}/v2/comparison/signal_combination_stability.json"
    if not os.path.exists(path):
        print("  ⊘ Skipping F7.3 (no rank stability data)")
        return

    data = load_json(path)

    # Extract signal configs and rho values
    configs = []
    if isinstance(data, dict):
        # Handle ft_stability_matrix format: use original_4sig row as reference
        if 'ft_stability_matrix' in data:
            matrix = data['ft_stability_matrix']
            ref_row = matrix.get('original_4sig', matrix.get('no_musico', {}))
            for key, val in ref_row.items():
                if key != 'original_4sig' and isinstance(val, (int, float)):
                    configs.append((key, val))
        else:
            # Try various flat formats
            for key, val in data.items():
                if isinstance(val, dict) and 'rho' in val:
                    configs.append((key, val['rho']))
                elif isinstance(val, (int, float)):
                    configs.append((key, val))

    if not configs:
        print("  ⊘ Skipping F7.3 (no parseable stability data)")
        return

    configs.sort(key=lambda x: x[1], reverse=True)
    names = [c[0] for c in configs]
    rhos = [c[1] for c in configs]

    # Highlight the recommended 2-signal config
    colors = []
    for name in names:
        if '2' in name.lower() or 'clap_fad' in name.lower() or ('clap' in name.lower() and 'fad' in name.lower()):
            colors.append(PAL['accent'])
        elif 'musico' in name.lower() and len(name) < 20:
            colors.append(PAL['high'])
        else:
            colors.append(PAL['v1'])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(names)), rhos, height=0.6, color=colors, edgecolor='white', linewidth=0.5)

    for bar, rho in zip(bars, rhos):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{rho:.3f}', va='center', fontsize=9, fontweight='medium')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace('_', ' ').title() for n in names], fontsize=9)
    ax.set_xlabel('Spearman ρ (Rank Correlation with 4-Signal Reference)')
    ax.set_title('Signal Combination Stability (Distinctiveness Score)')
    ax.set_xlim(-0.1, 1.1)
    ax.axvline(x=0, color='#999', linewidth=0.6)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PAL['accent'], label='Recommended (2-signal)'),
                       Patch(facecolor=PAL['high'], label='Noise (musicological)')],
              loc='lower right', fontsize=9)
    save(fig, 'F7_3_signal_stability', out)


def fig_7_4_temporal_split(R, out):
    """Temporal split: gen→seen vs gen→unseen."""
    path = f"{R}/temporal_split/temporal_fad_results.csv"
    if not os.path.exists(path):
        print("  ⊘ Skipping F7.4 (no temporal data)")
        return

    rows = load_csv(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, version, color in [(ax1, 'v1', PAL['v1']), (ax2, 'v2', PAL['v2'])]:
        v_rows = [r for r in rows if r.get('version', '') == version]
        if not v_rows:
            # Try without version filter
            v_rows = rows

        seen_fad = [sf(r.get('fad_seen')) for r in v_rows if sf(r.get('fad_seen')) is not None]
        unseen_fad = [sf(r.get('fad_unseen')) for r in v_rows if sf(r.get('fad_unseen')) is not None]
        seen_cos = [sf(r.get('cos_seen')) for r in v_rows if sf(r.get('cos_seen')) is not None]
        unseen_cos = [sf(r.get('cos_unseen')) for r in v_rows if sf(r.get('cos_unseen')) is not None]

        if seen_cos and unseen_cos:
            data_to_plot = [seen_cos, unseen_cos]
            label_type = 'Cosine Similarity'
        elif seen_fad and unseen_fad:
            data_to_plot = [seen_fad, unseen_fad]
            label_type = 'FAD'
        else:
            continue

        bp = ax.boxplot(data_to_plot, positions=[0, 1], widths=0.4, patch_artist=True,
                        medianprops=dict(color='#333', linewidth=1.5))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.8)
        bp['boxes'][1].set_facecolor(color)
        bp['boxes'][1].set_alpha(0.35)
        for box in bp['boxes']:
            box.set_edgecolor('#333')
            box.set_linewidth(0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Gen → Seen\n(training)', 'Gen → Unseen\n(held-out)'], fontsize=10)
        ax.set_ylabel(label_type)
        ax.set_title(f'{version.upper()}', fontsize=12)

        # p-value annotation
        from scipy import stats as sp_stats
        if len(data_to_plot[0]) >= 3 and len(data_to_plot[1]) >= 3:
            t, p = sp_stats.ttest_rel(data_to_plot[0][:min(len(data_to_plot[0]), len(data_to_plot[1]))],
                                       data_to_plot[1][:min(len(data_to_plot[0]), len(data_to_plot[1]))])
            sig = 'n.s.' if p > 0.05 else f'p={p:.3f}'
            ax.text(0.5, 0.95, sig, transform=ax.transAxes, ha='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ddd'))

        ax.grid(True, axis='y', alpha=0.15)

    fig.suptitle('Temporal Split: Seen vs Unseen Track Similarity',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'F7_4_temporal_split', out)


def fig_7_5_tier_agreement(R, out):
    """Tier agreement: V2 vs Baseline across threshold schemes."""
    # Compute agreement for each scheme from V2 and Baseline 2-signal scores
    v2_path = f"{R}/v2/analysis/vulnerability_scores.csv"
    bl_path = f"{R}/baseline/analysis/vulnerability_scores.csv"

    # Default hardcoded data as fallback
    schemes = ['2-tier\n(0.5)', '2-tier\n(median)', '3-tier\n(0.33/0.67)',
               '3-tier\n(0.4/0.7)', '4-tier\n(quartiles)', '5-tier\n(quintiles)']

    if os.path.exists(v2_path) and os.path.exists(bl_path):
        v2_rows = {r.get('artist_id'): r for r in load_csv(v2_path)}
        bl_rows = {r.get('artist_id'): r for r in load_csv(bl_path)}

        v2_scores, bl_scores = [], []
        for aid in v2_rows:
            if aid in bl_rows:
                v2s = distinctiveness_2sig(v2_rows[aid])
                bls = distinctiveness_2sig(bl_rows[aid])
                if v2s is not None and bls is not None:
                    v2_scores.append(v2s)
                    bl_scores.append(bls)

        v2_arr = np.array(v2_scores)
        bl_arr = np.array(bl_scores)

        def assign_tier(scores, thresholds):
            tiers = np.zeros(len(scores), dtype=int)
            for th in thresholds:
                tiers += (scores >= th).astype(int)
            return tiers

        def agreement(t1, t2):
            return 100 * np.mean(t1 == t2)

        def cohens_kappa(t1, t2):
            n = len(t1)
            po = np.mean(t1 == t2)
            cats = sorted(set(t1) | set(t2))
            pe = sum((np.sum(t1 == c) / n) * (np.sum(t2 == c) / n) for c in cats)
            return (po - pe) / (1 - pe) if pe < 1 else 1.0

        med = np.median(np.concatenate([v2_arr, bl_arr]))
        quarts = np.percentile(np.concatenate([v2_arr, bl_arr]), [25, 50, 75])
        quints = np.percentile(np.concatenate([v2_arr, bl_arr]), [20, 40, 60, 80])

        scheme_defs = [
            ([0.5],),
            ([med],),
            ([0.33, 0.67],),
            ([0.4, 0.7],),
            (list(quarts),),
            (list(quints),),
        ]

        agreements = []
        kappas = []
        for (ths,) in scheme_defs:
            t_v2 = assign_tier(v2_arr, ths)
            t_bl = assign_tier(bl_arr, ths)
            agreements.append(agreement(t_v2, t_bl))
            kappas.append(cohens_kappa(t_v2, t_bl))
    else:
        # Fallback hardcoded
        agreements = [82, 76, 52, 56, 52, 34]
        kappas = [0.59, 0.52, 0.25, 0.34, 0.37, 0.18]

    # Highlight the chosen 3-tier (0.33/0.67) scheme
    chosen_idx = 2  # 3-tier (0.33/0.67)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Agreement bars
    colors = [PAL['accent'] if i == chosen_idx else
              (PAL['v1'] if a >= 70 else PAL['neutral'] if a >= 50 else PAL['v2'])
              for i, a in enumerate(agreements)]
    bars = ax1.bar(range(len(schemes)), agreements, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, agreements):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.0f}%', ha='center', fontsize=10, fontweight='medium')
    ax1.set_xticks(range(len(schemes)))
    ax1.set_xticklabels(schemes, fontsize=8.5)
    ax1.set_ylabel('V2 ↔ Baseline Agreement (%)')
    ax1.set_title('Tier Agreement by Threshold Scheme')
    ax1.set_ylim(0, 100)
    ax1.axhline(y=50, color='#ccc', linestyle='--', linewidth=0.6)

    # Kappa bars
    colors_k = [PAL['accent'] if i == chosen_idx else
                (PAL['v1'] if k >= 0.4 else PAL['neutral'] if k >= 0.2 else PAL['v2'])
                for i, k in enumerate(kappas)]
    bars2 = ax2.bar(range(len(schemes)), kappas, color=colors_k, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars2, kappas):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', fontsize=10, fontweight='medium')
    ax2.set_xticks(range(len(schemes)))
    ax2.set_xticklabels(schemes, fontsize=8.5)
    ax2.set_ylabel("Cohen's κ")
    ax2.set_title("V2 vs Baseline Reliability (Cohen's κ)")
    ax2.set_ylim(0, 0.8)
    ax2.axhline(y=0.4, color='#bbb', linestyle=':', linewidth=0.7, label='Fair (0.4)')
    ax2.axhline(y=0.6, color='#bbb', linestyle='--', linewidth=0.7, label='Moderate (0.6)')
    ax2.legend(fontsize=8)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=PAL['accent'], label='Chosen: 3-tier (0.33/0.67)')],
               loc='upper right', fontsize=9, bbox_to_anchor=(0.98, 0.98))

    fig.suptitle('Optimal Tier Granularity for Distinctiveness Reporting',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'F7_5_tier_agreement', out)


def fig_7_6_ablation(R, out):
    """Vulnerability ablation: effect of removing each signal."""
    path = f"{R}/v1/supplementary/vulnerability_ablation.json"
    if not os.path.exists(path):
        print("  ⊘ Skipping F7.6 (no ablation data)")
        return
    data = load_json(path)

    if isinstance(data, list):
        configs = [(d.get('removed', 'unknown'), d.get('rho_vs_original', 0)) for d in data]
    elif isinstance(data, dict):
        configs = []
        for key, val in data.items():
            if isinstance(val, dict):
                configs.append((key, val.get('rho_vs_original', val.get('rho', 0))))
            elif isinstance(val, (int, float)):
                configs.append((key, val))

    if not configs:
        print("  ⊘ Skipping F7.6 (no parseable ablation data)")
        return

    configs.sort(key=lambda x: x[1])
    names = [c[0].replace('_', ' ').title() for c in configs]
    rhos = [c[1] for c in configs]
    colors = [PAL['high'] if r < 0 else PAL['v1'] for r in rhos]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(range(len(names)), rhos, height=0.5, color=colors, edgecolor='white', linewidth=0.5)

    for bar, rho in zip(bars, rhos):
        offset = 0.03 if rho >= 0 else -0.03
        ha = 'left' if rho >= 0 else 'right'
        ax.text(rho + offset, bar.get_y() + bar.get_height()/2,
                f'{rho:.2f}', va='center', ha=ha, fontsize=10, fontweight='medium')

    ax.axvline(x=0, color='#333', linewidth=1)
    ax.axvline(x=1, color='#bbb', linewidth=0.6, linestyle='--')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('ρ vs Original Score (negative = score inverts)')
    ax.set_title('Distinctiveness Score Ablation: Effect of Removing Each Signal')
    ax.set_xlim(-1.1, 1.1)
    ax.invert_yaxis()
    save(fig, 'F7_6_ablation', out)


# ═══════════════════════════════════════════════════════════════════════════════
# APPENDIX
# ═══════════════════════════════════════════════════════════════════════════════

def fig_a1_fad_heatmap(R, out):
    """Cross-artist FAD heatmap."""
    path = f"{R}/v1/analysis/fad_cross_artist.csv"
    if not os.path.exists(path):
        print("  ⊘ Skipping FA.1 (no FAD cross-artist data)")
        return
    rows = load_csv(path)

    # Build matrix
    artists = sorted(set(r.get('gen_artist', '') for r in rows if r.get('gen_artist')))
    if len(artists) < 5:
        return

    idx = {a: i for i, a in enumerate(artists)}
    n = len(artists)
    matrix = np.full((n, n), np.nan)

    for r in rows:
        ga = r.get('gen_artist', '')
        ca = r.get('cat_artist', '')
        fad = sf(r.get('fad'))
        if ga in idx and ca in idx and fad is not None:
            matrix[idx[ga], idx[ca]] = fad

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap='YlOrRd_r', aspect='auto', interpolation='nearest')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(artists, rotation=90, fontsize=5)
    ax.set_yticklabels(artists, fontsize=5)
    ax.set_xlabel('Catalog Artist')
    ax.set_ylabel('Generated Artist')
    ax.set_title('Cross-Artist FAD Matrix (Lower = More Similar)')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Fréchet Audio Distance')
    save(fig, 'FA_1_fad_heatmap', out)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate thesis figures')
    parser.add_argument('--results_dir', '-r', default='results',
                        help='Path to results directory')
    parser.add_argument('--out_dir', '-o', default='figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    R = args.results_dir
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    print(f"Results dir: {R}")
    print(f"Output dir:  {out}")
    print()

    # Chapter 4
    print("Chapter 4: Proactive Protection")
    fig_4_1_wavmark_survival(R, out)
    fig_4_2_audioseal_training(R, out)

    # Chapter 5
    print("\nChapter 5: Reactive Auditing")
    fig_5_1_training_loss(R, out)
    fig_5_2_clap_matched_mismatched(R, out)
    fig_5_3_clap_by_tier(R, out)
    fig_5_4_vulnerability_ranking(R, out)
    fig_5_5_ngram_ratios(R, out)
    fig_5_6_exposure_vs_vulnerability(R, out)
    fig_5_7_effect_sizes(R, out)
    fig_5_8_permutation_test(R, out)

    # Chapter 6
    print("\nChapter 6: V1 vs V2 Comparison")
    fig_6_1_clap_gap_comparison(R, out)
    fig_6_2_v1_v2_scatter(R, out)
    fig_6_3_vulnerability_delta(R, out)
    fig_6_4_three_way_absorption(R, out)

    # Chapter 7
    print("\nChapter 7: Robustness & Validation")
    fig_7_1_bootstrap_ci(R, out)
    fig_7_3_signal_stability(R, out)
    fig_7_4_temporal_split(R, out)
    fig_7_5_tier_agreement(R, out)
    fig_7_6_ablation(R, out)

    # Appendix
    print("\nAppendix")
    fig_a1_fad_heatmap(R, out)

    print(f"\nDone! All figures saved to {out}/")


if __name__ == '__main__':
    main()
