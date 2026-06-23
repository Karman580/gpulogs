"""
Extended GPU Power Analysis
============================
Builds on existing preprocessing & clustering pipeline.
Adds: cluster compactness, scatter/PCA plots, peak-time clustering,
repeating pattern detection, and workload distribution analysis.

Input : reduced_gpu_power.csv  (already preprocessed at 5-min resolution)
Output: extended_analysis/      (all plots, tables, reports)
"""

# ── Disable GUI backend ──────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import os
import warnings
import textwrap

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ── Settings ─────────────────────────────────────────────────────────
INPUT_FILE   = "reduced_gpu_power.csv"
OUTPUT_DIR   = "extended_analysis"
NUM_CLUSTERS = 3
PEAK_QUANTILE = 0.80          # top 20% power = peak
PEAK_HOURS    = (18, 23)      # alternative: evening peak window
RANDOM_STATE  = 42

# ── Helpers ──────────────────────────────────────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def add_time_features(df):
    """Add derived time columns used across analyses."""
    df['hour']    = df['timestamp'].dt.hour
    df['day_num'] = (df['timestamp'] - df['timestamp'].min()).dt.days + 1
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['date']    = df['timestamp'].dt.date
    return df

def run_kmeans(X_scaled, n_clusters=NUM_CLUSTERS):
    """Fit KMeans and return the model."""
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    return km

# =====================================================================
# 1. CLUSTER RADIUS / SPREAD ANALYSIS
# =====================================================================

def cluster_compactness(df, features, label="full_dataset"):
    """
    Compute inertia and per-cluster average distance from centroid.
    Returns a summary DataFrame and the fitted model + scaler.
    """
    print(f"\n{'='*60}")
    print(f"  1. CLUSTER COMPACTNESS  —  {label}")
    print(f"{'='*60}")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    km = run_kmeans(X)
    labels = km.labels_
    centroids = km.cluster_centers_

    # Overall inertia
    print(f"  Inertia (WCSS): {km.inertia_:.2f}")

    # Per-cluster metrics
    rows = []
    for c in range(NUM_CLUSTERS):
        mask = labels == c
        pts  = X[mask]
        cent = centroids[c].reshape(1, -1)
        dists = cdist(pts, cent, metric='euclidean').flatten()
        rows.append({
            'cluster':        c,
            'size':           int(mask.sum()),
            'avg_dist':       dists.mean(),
            'max_dist':       dists.max(),
            'std_dist':       dists.std(),
            'mean_power_W':   df.loc[mask, 'power_watts'].mean(),
        })

    radius_df = pd.DataFrame(rows)
    radius_df['tightness'] = radius_df['avg_dist'].apply(
        lambda d: 'Tight' if d < radius_df['avg_dist'].median() else 'Spread'
    )

    # Silhouette
    sil = silhouette_score(X, labels)
    print(f"  Silhouette Score: {sil:.4f}")
    print(f"\n  Per-cluster radius metrics:")
    print(radius_df.to_string(index=False))

    # Save table
    csv_path = os.path.join(OUTPUT_DIR, f"cluster_radius_{label}.csv")
    radius_df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved → {csv_path}")

    return km, scaler, labels, radius_df, sil

# =====================================================================
# 2. CLUSTERING VISUALISATIONS
# =====================================================================

def plot_scatter_clusters(df, labels, label="full_dataset"):
    """Scatter: X=hour, Y=power_watts, color=cluster."""
    print(f"\n{'='*60}")
    print(f"  2A. SCATTER PLOT  —  {label}")
    print(f"{'='*60}")

    palette = sns.color_palette("bright", NUM_CLUSTERS)

    fig, ax = plt.subplots(figsize=(14, 6))
    for c in range(NUM_CLUSTERS):
        mask = labels == c
        ax.scatter(
            df.loc[mask, 'hour'],
            df.loc[mask, 'power_watts'],
            c=[palette[c]], label=f'Cluster {c}',
            alpha=0.35, s=8, edgecolors='none'
        )
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title(f'KMeans Clusters — {label}', fontsize=14, fontweight='bold')
    ax.legend(title='Cluster', markerscale=3, fontsize=10)
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"scatter_clusters_{label}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  ✅ Saved → {path}")


def plot_pca_clusters(X_scaled, labels, label="full_dataset"):
    """PCA 2-D projection coloured by cluster label."""
    print(f"\n{'='*60}")
    print(f"  2B. PCA PLOT  —  {label}")
    print(f"{'='*60}")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2d = pca.fit_transform(X_scaled)

    palette = sns.color_palette("bright", NUM_CLUSTERS)

    fig, ax = plt.subplots(figsize=(10, 7))
    for c in range(NUM_CLUSTERS):
        mask = labels == c
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   c=[palette[c]], label=f'Cluster {c}',
                   alpha=0.35, s=8, edgecolors='none')

    ax.set_xlabel(f'PC1  ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2  ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(f'PCA of Clusters — {label}', fontsize=14, fontweight='bold')
    ax.legend(title='Cluster', markerscale=3, fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"pca_clusters_{label}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  ✅ Saved → {path}")

# =====================================================================
# 3. PEAK-TIME CLUSTERING
# =====================================================================

def peak_time_clustering(df, features, full_km, full_sil, full_radius):
    """
    Filter to peak power periods, re-cluster, and compare
    structure / tightness with the full-dataset clustering.
    """
    print(f"\n{'='*60}")
    print(f"  3. PEAK-TIME CLUSTERING")
    print(f"{'='*60}")

    # Method A: top quantile
    threshold = df['power_watts'].quantile(PEAK_QUANTILE)
    peak_df_q = df[df['power_watts'] >= threshold].copy().reset_index(drop=True)
    print(f"  Quantile threshold ({PEAK_QUANTILE*100:.0f}%): {threshold:.2f} W  →  {len(peak_df_q)} rows")

    # Method B: evening hours
    peak_df_h = df[df['hour'].between(*PEAK_HOURS)].copy().reset_index(drop=True)
    print(f"  Peak hours ({PEAK_HOURS[0]}:00–{PEAK_HOURS[1]}:00): {len(peak_df_h)} rows")

    # Use whichever has more rows for richer analysis
    if len(peak_df_q) >= len(peak_df_h):
        peak_df = peak_df_q
        peak_label = f"peak_top{int((1-PEAK_QUANTILE)*100)}pct"
    else:
        peak_df = peak_df_h
        peak_label = f"peak_{PEAK_HOURS[0]}h_{PEAK_HOURS[1]}h"

    if len(peak_df) < NUM_CLUSTERS:
        print("  ⚠ Not enough peak data to cluster. Skipping.")
        return

    # Cluster peak data
    km_peak, sc_peak, lbl_peak, rad_peak, sil_peak = cluster_compactness(
        peak_df, features, label=peak_label
    )
    X_peak = sc_peak.transform(peak_df[features])

    # Visualisations for peak
    plot_scatter_clusters(peak_df, lbl_peak, label=peak_label)
    plot_pca_clusters(X_peak, lbl_peak, label=peak_label)

    # Comparison table
    print(f"\n  ── Full vs Peak comparison ──")
    comp = pd.DataFrame({
        'metric': ['inertia', 'silhouette', 'avg_cluster_dist', 'n_rows'],
        'full_dataset': [
            full_km.inertia_,
            full_sil,
            full_radius['avg_dist'].mean(),
            len(df)
        ],
        'peak_subset': [
            km_peak.inertia_,
            sil_peak,
            rad_peak['avg_dist'].mean(),
            len(peak_df)
        ]
    })
    print(comp.to_string(index=False))
    comp.to_csv(os.path.join(OUTPUT_DIR, "full_vs_peak_comparison.csv"), index=False)
    print(f"  ✅ Saved comparison table.")

# =====================================================================
# 4. REPEATING PATTERN DETECTION
# =====================================================================

def repeating_patterns(df, labels):
    """
    Identify recurring spike hours and cluster-dominant time ranges.
    """
    print(f"\n{'='*60}")
    print(f"  4. REPEATING PATTERN DETECTION")
    print(f"{'='*60}")

    df = df.copy()
    df['cluster'] = labels

    insights = []

    # A) Spike-hour frequency across all days
    # Use top-quartile power as "spike"
    spike_thresh = df['power_watts'].quantile(0.75)
    spikes = df[df['power_watts'] >= spike_thresh]

    hour_freq = spikes.groupby('hour').size()
    top_spike_hours = hour_freq.nlargest(5)

    insights.append("Top 5 spike hours (across all days):")
    for h, cnt in top_spike_hours.items():
        insights.append(f"  • Hour {h:02d}:00  —  {cnt} spike records")

    # B) Daily spike pattern
    daily_spike = spikes.groupby(['date', 'hour']).size().reset_index(name='count')
    daily_pivot = daily_spike.pivot_table(index='hour', columns='date',
                                          values='count', fill_value=0)

    # Hours that spike on >50% of days
    n_days = df['date'].nunique()
    recurring = (daily_pivot > 0).sum(axis=1)
    recurring_hours = recurring[recurring >= n_days * 0.5].index.tolist()
    insights.append(f"\nHours with spikes on ≥50% of days: {recurring_hours if recurring_hours else 'None detected'}")

    # C) Cluster dominance by time-of-day
    insights.append("\nCluster dominance by time-of-day band:")
    bands = {'Night (0-6)': (0,5), 'Morning (6-12)': (6,11),
             'Afternoon (12-18)': (12,17), 'Evening (18-24)': (18,23)}
    for band_name, (lo, hi) in bands.items():
        band_data = df[df['hour'].between(lo, hi)]
        if band_data.empty:
            continue
        dom_cluster = band_data['cluster'].mode()
        dom_cluster = dom_cluster.iloc[0] if len(dom_cluster) > 0 else 'N/A'
        mean_pw = band_data['power_watts'].mean()
        insights.append(f"  {band_name:20s}  →  Dominant cluster: {dom_cluster}  |  Mean power: {mean_pw:.1f} W")

    # D) Day-over-day consistency
    insights.append("\nDay-over-day cluster consistency (dominant cluster per day):")
    for d in sorted(df['date'].unique()):
        day_data = df[df['date'] == d]
        dom = day_data['cluster'].mode()
        dom = dom.iloc[0] if len(dom) > 0 else 'N/A'
        mean_pw = day_data['power_watts'].mean()
        insights.append(f"  {d}  →  Cluster {dom}  (mean {mean_pw:.1f} W)")

    # Heatmap: hour vs day_num spike frequency
    fig, ax = plt.subplots(figsize=(12, 5))
    heatmap_data = spikes.groupby(['day_num', 'hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot_table(index='hour', columns='day_num',
                                              values='count', fill_value=0)
    sns.heatmap(heatmap_pivot, cmap='YlOrRd', ax=ax, linewidths=0.3,
                cbar_kws={'label': 'Spike count'})
    ax.set_title('Spike Frequency Heatmap (Hour vs Day)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Day Number')
    ax.set_ylabel('Hour of Day')
    plt.tight_layout()
    hm_path = os.path.join(OUTPUT_DIR, "spike_heatmap.png")
    fig.savefig(hm_path, dpi=200)
    plt.close(fig)
    insights.append(f"\n  ✅ Spike heatmap saved → {hm_path}")

    # Print & save
    report = "\n".join(insights)
    print(report)
    with open(os.path.join(OUTPUT_DIR, "repeating_patterns.txt"), 'w') as f:
        f.write(report)
    print(f"  ✅ Saved → repeating_patterns.txt")

# =====================================================================
# 5. WORKLOAD DISTRIBUTION ANALYSIS
# =====================================================================

def workload_distribution(df, labels):
    """
    Analyse idle vs active periods, burstiness, and usage distribution.
    """
    print(f"\n{'='*60}")
    print(f"  5. WORKLOAD DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")

    df = df.copy()
    df['cluster'] = labels

    insights = []

    # Define idle threshold: bottom 25% power
    idle_thresh = df['power_watts'].quantile(0.25)
    active_thresh = df['power_watts'].quantile(0.75)

    df['state'] = pd.cut(df['power_watts'],
                         bins=[-np.inf, idle_thresh, active_thresh, np.inf],
                         labels=['Idle', 'Moderate', 'Active'])

    state_counts = df['state'].value_counts(normalize=True) * 100
    insights.append("Overall workload state distribution:")
    for st, pct in state_counts.items():
        insights.append(f"  {st:10s}  {pct:5.1f}%")

    # Per-hour state breakdown
    hourly_state = df.groupby(['hour', 'state']).size().unstack(fill_value=0)
    hourly_state_pct = hourly_state.div(hourly_state.sum(axis=1), axis=0) * 100

    # Burstiness index: coefficient of variation of power per hour
    hourly_cv = df.groupby('hour')['power_watts'].agg(
        lambda x: x.std() / x.mean() if x.mean() > 0 else 0
    )
    bursty_hours = hourly_cv[hourly_cv > hourly_cv.median()].index.tolist()
    balanced_hours = hourly_cv[hourly_cv <= hourly_cv.median()].index.tolist()

    insights.append(f"\nBursty hours (high CoV): {bursty_hours}")
    insights.append(f"Balanced hours (low CoV): {balanced_hours}")

    overall_cv = df['power_watts'].std() / df['power_watts'].mean()
    if overall_cv > 0.5:
        insights.append(f"\nOverall usage pattern: BURSTY  (CoV = {overall_cv:.3f})")
    else:
        insights.append(f"\nOverall usage pattern: BALANCED  (CoV = {overall_cv:.3f})")

    # Per-host idle ratio
    insights.append("\nPer-host idle ratio:")
    for host in sorted(df['hostname'].unique()):
        h = df[df['hostname'] == host]
        idle_pct = (h['state'] == 'Idle').mean() * 100
        active_pct = (h['state'] == 'Active').mean() * 100
        insights.append(f"  {host:18s}  Idle: {idle_pct:5.1f}%   Active: {active_pct:5.1f}%")

    # ── Plots ──

    # A) Stacked area: state proportion over hours
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly_state_pct.plot.area(ax=ax, alpha=0.7,
                               color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Workload State Distribution by Hour', fontsize=14, fontweight='bold')
    ax.legend(title='State', fontsize=10)
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "workload_state_by_hour.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    insights.append(f"\n  ✅ State-by-hour plot → {path}")

    # B) Burstiness bar chart
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ['#e74c3c' if h in bursty_hours else '#2ecc71' for h in hourly_cv.index]
    ax.bar(hourly_cv.index, hourly_cv.values, color=colors, edgecolor='white')
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Coefficient of Variation', fontsize=12)
    ax.set_title('Burstiness Index by Hour', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "burstiness_by_hour.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    insights.append(f"  ✅ Burstiness plot → {path}")

    # C) Power distribution histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['power_watts'], bins=80, color='steelblue',
            edgecolor='white', alpha=0.85)
    ax.axvline(idle_thresh, color='green', ls='--', label=f'Idle thresh ({idle_thresh:.0f} W)')
    ax.axvline(active_thresh, color='red', ls='--', label=f'Active thresh ({active_thresh:.0f} W)')
    ax.set_xlabel('Power (W)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Power Distribution with State Thresholds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "power_distribution.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    insights.append(f"  ✅ Power distribution plot → {path}")

    report = "\n".join(insights)
    print(report)
    with open(os.path.join(OUTPUT_DIR, "workload_distribution.txt"), 'w') as f:
        f.write(report)
    print(f"  ✅ Saved → workload_distribution.txt")

# =====================================================================
# MAIN
# =====================================================================

def main():
    ensure_dir(OUTPUT_DIR)

    # ── Load data ────────────────────────────────────────────────────
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, parse_dates=['timestamp'])
    df.dropna(subset=['timestamp', 'power_watts'], inplace=True)
    df = add_time_features(df)
    print(f"Loaded {len(df):,} rows  |  {df['hostname'].nunique()} hosts  |  "
          f"{df['pci_bus_id'].nunique()} GPUs  |  Days {df['day_num'].min()}–{df['day_num'].max()}")

    # ── Feature set for clustering ───────────────────────────────────
    features = ['power_watts', 'hour', 'day_num']

    # ── 1. Cluster compactness (full dataset) ────────────────────────
    km_full, sc_full, lbl_full, rad_full, sil_full = cluster_compactness(
        df, features, label="full_dataset"
    )
    X_full = sc_full.transform(df[features])

    # ── 2. Visualisations ────────────────────────────────────────────
    plot_scatter_clusters(df, lbl_full, label="full_dataset")
    plot_pca_clusters(X_full, lbl_full, label="full_dataset")

    # ── 3. Peak-time clustering & comparison ─────────────────────────
    peak_time_clustering(df, features, km_full, sil_full, rad_full)

    # ── 4. Repeating pattern detection ───────────────────────────────
    repeating_patterns(df, lbl_full)

    # ── 5. Workload distribution analysis ────────────────────────────
    workload_distribution(df, lbl_full)

    # ── Done ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🎉  EXTENDED ANALYSIS COMPLETE")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
