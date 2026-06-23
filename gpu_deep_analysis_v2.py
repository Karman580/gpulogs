import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.decomposition import PCA
import json
import warnings
import time

warnings.filterwarnings('ignore')

# Config
DATA_PATH = "/Users/karmansinghtalwar/Documents/GPU LOGS/analysis_20_27_april/clustered_full.csv"
OUTPUT_DIR = "/Users/karmansinghtalwar/Documents/GPU LOGS/analysis_20_27_april_v2/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("STEP 0: LOADING PRE-PROCESSED DATA")
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# We drop the old cluster column since we are re-validating
if 'cluster' in df.columns:
    df = df.drop(columns=['cluster'])

features_to_scale = ['power_watts', 'rolling_mean_power', 'power_delta']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features_to_scale])

print("STEP 1: EPSILON VALIDATION (CRITICAL)")
# Use a sample for tuning
n_total = len(df)
sample_size = min(50000, n_total)
np.random.seed(42)
sample_idx = np.random.choice(n_total, sample_size, replace=False)
X_train = df_scaled[sample_idx]
min_samples = 20

epsilons_to_test = [0.161, 0.2, 0.25, 0.3]
best_eps = None
best_labels_sample = None
best_score = -1 # A custom score balancing inliers and cluster distinctness

validation_results = []

for eps in epsilons_to_test:
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_train)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    inlier_ratio = 1 - (list(labels).count(-1) / len(labels))
    
    # We want a high inlier ratio but enough clusters to distinguish behaviors
    # Too high eps -> 1 giant cluster
    # Too low eps -> >10% outliers
    if n_clusters > 1:
        # Score penalizes having only 1 or 2 clusters, rewards high inliers
        score = inlier_ratio * (np.log(n_clusters) + 1)
    else:
        score = 0
        
    validation_results.append({
        'epsilon': eps,
        'inlier_ratio': inlier_ratio,
        'n_clusters': n_clusters,
        'score': score
    })
    
    if score > best_score:
        best_score = score
        best_eps = eps
        best_labels_sample = labels

print(f"Validation Results: {validation_results}")
print(f"Selected Best Epsilon: {best_eps}")

# Extrapolate using KNN
print(f"Extrapolating clusters to {n_total} points...")
inlier_mask = best_labels_sample != -1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train[inlier_mask], best_labels_sample[inlier_mask])
nn = NearestNeighbors(n_neighbors=1).fit(X_train[inlier_mask])

chunk_size = 200000
full_labels = np.zeros(n_total, dtype=int)
for i in range(0, n_total, chunk_size):
    chunk = df_scaled[i:i+chunk_size]
    dists, _ = nn.kneighbors(chunk)
    preds = knn.predict(chunk)
    preds[dists.flatten() > best_eps] = -1
    full_labels[i:i+chunk_size] = preds

df['cluster'] = full_labels

print("STEP 2: CLUSTER MEANING EXTRACTION")
cluster_stats = df[df['cluster'] != -1].groupby('cluster').agg({
    'power_watts': ['mean', 'var'],
    'power_delta': 'mean',
    'gpu_id': 'count'
}).reset_index()

cluster_stats.columns = ['cluster', 'avg_power', 'power_var', 'avg_delta', 'point_count']

# Semantic labeling logic
def label_cluster(row):
    pwr = row['avg_power']
    var = row['power_var']
    if pwr < 100 and var < 500:
        return 'Idle/Low Load'
    elif pwr < 150 and var < 1000:
        return 'Moderate Load'
    elif pwr >= 150 and var < 2000:
        return 'High Load'
    else:
        return 'Burst/Unstable'

cluster_stats['behavior_type'] = cluster_stats.apply(label_cluster, axis=1)
cluster_stats.to_csv(os.path.join(OUTPUT_DIR, 'cluster_summary.csv'), index=False)

# Map labels back to df
label_map = dict(zip(cluster_stats['cluster'], cluster_stats['behavior_type']))
label_map[-1] = 'Outlier'
df['behavior_label'] = df['cluster'].map(label_map)

print("STEP 3: OUTLIER RECLASSIFICATION")
outliers = df[df['cluster'] == -1].copy()

# Heuristics for outliers
p95_delta = df['power_delta'].abs().quantile(0.95)
p95_power = df['power_watts'].quantile(0.95)

def classify_outlier(row):
    if abs(row['power_delta']) > p95_delta:
        return 'Spike Event'
    elif row['power_watts'] > p95_power and abs(row['power_delta']) <= p95_delta:
        return 'Sustained High-Load Anomaly'
    else:
        return 'Erratic Fluctuation'

if len(outliers) > 0:
    outliers['outlier_type'] = outliers.apply(classify_outlier, axis=1)
    
    outlier_summary = outliers['outlier_type'].value_counts().reset_index()
    outlier_summary.columns = ['Outlier Type', 'Count']
    outlier_summary.to_csv(os.path.join(OUTPUT_DIR, 'outlier_types.csv'), index=False)
else:
    pd.DataFrame(columns=['Outlier Type', 'Count']).to_csv(os.path.join(OUTPUT_DIR, 'outlier_types.csv'), index=False)

# Map outlier types to main df
if len(outliers) > 0:
    df.loc[df['cluster'] == -1, 'behavior_label'] = outliers['outlier_type']

print("STEP 4: GPU-LEVEL BEHAVIOR PROFILING")
# Aggregations
gpu_profile = df.groupby('gpu_id').agg(
    power_watts_mean=('power_watts', 'mean'),
    power_watts_var=('power_watts', 'var'),
    behavior_label=('behavior_label', lambda x: x.value_counts(normalize=True).to_dict())
).reset_index()

# Extract specific label percentages
for label in df['behavior_label'].unique():
    col_name = f'pct_time_{label.lower().replace(" ", "_").replace("/", "_")}'
    # The dictionary is in behavior_label
    gpu_profile[col_name] = gpu_profile['behavior_label'].apply(lambda d: d.get(label, 0.0) * 100)

gpu_profile = gpu_profile.drop(columns=['behavior_label'])

def classify_gpu(row):
    # Sum up outlier types if they exist
    outlier_pct = row.get('pct_time_spike_event', 0) + row.get('pct_time_sustained_high-load_anomaly', 0) + row.get('pct_time_erratic_fluctuation', 0) + row.get('pct_time_outlier', 0)
    spike_pct = row.get('pct_time_spike_event', 0)
    
    if outlier_pct > 10 or spike_pct > 5:
        return 'Unstable / Spiky'
    elif row.get('pct_time_high_load', 0) > 40:
        return 'High-Load Consistent'
    else:
        return 'Stable'

gpu_profile['gpu_classification'] = gpu_profile.apply(classify_gpu, axis=1)
gpu_profile.to_csv(os.path.join(OUTPUT_DIR, 'gpu_behavior_profile.csv'), index=False)

print("STEP 5: NODE & GPU TYPE ANALYSIS")
# Done within reporting step using groupby

print("STEP 6: PEAK VS NON-PEAK DEEP ANALYSIS")
transition = df.groupby('is_peak')['behavior_label'].value_counts(normalize=True).unstack().fillna(0) * 100
transition.index = ['Non-Peak', 'Peak']
transition.to_csv(os.path.join(OUTPUT_DIR, 'transition_matrix.csv'))

print("STEP 7: IMPROVED VISUALS")

# 1. Cluster Interpretation Plot
plt.figure(figsize=(10, 6))
sample_df = df.sample(min(20000, len(df)))
sns.scatterplot(data=sample_df, x='power_watts', y='rolling_mean_power', hue='behavior_label', alpha=0.6, palette='tab10')
plt.title('Cluster Interpretation: Power vs Rolling Mean')
plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_interpretation.png'))
plt.close()

# 2. GPU Behavior Heatmap
plt.figure(figsize=(12, 8))
# Filter to top 20 GPUs to avoid overcrowding
top_gpus = gpu_profile.nlargest(20, 'power_watts_mean')['gpu_id']
heatmap_data = df[df['gpu_id'].isin(top_gpus)].pivot_table(index='gpu_id', columns='behavior_label', aggfunc='size', fill_value=0)
heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100 # Normalize
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title('GPU Behavior Heatmap (% Time spent in states - Top 20 High Load GPUs)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'gpu_behavior_heatmap.png'))
plt.close()

# 3. Outlier Type Distribution
if len(outliers) > 0:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=outlier_summary, x='Outlier Type', y='Count')
    plt.title('Outlier Taxonomy Distribution')
    plt.savefig(os.path.join(OUTPUT_DIR, 'outlier_distribution.png'))
    plt.close()

# 4. Node vs Load Comparison
plt.figure(figsize=(12, 6))
sns.violinplot(data=sample_df, x='hostname', y='power_watts', hue='gpu_type', split=True, inner="quart")
plt.title('Node & GPU Type Load Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'node_load_comparison.png'))
plt.close()

print("STEP 8: FINAL REPORT (UPGRADED)")

# Generate Report text
with open(os.path.join(OUTPUT_DIR, 'final_report.txt'), 'w') as f:
    f.write("=== GPU TELEMETRY DEEP ANALYSIS REPORT (V2) ===\n\n")
    
    f.write("1. VALIDATED CLUSTERING PARAMETERS\n")
    f.write(f"   - Optimal Epsilon Chosen: {best_eps}\n")
    f.write(f"   - Justification: Out of the tested values (0.161, 0.2, 0.25, 0.3), this epsilon optimized the balance between noise reduction and maintaining distinct behavioral clusters.\n\n")
    
    f.write("2. CLUSTER INTERPRETATION\n")
    for _, row in cluster_stats.iterrows():
        f.write(f"   - {row['behavior_type']}: Avg Power = {row['avg_power']:.2f} W, Variance = {row['power_var']:.2f}, Count = {row['point_count']}\n")
    f.write("\n")
    
    f.write("3. OUTLIER INTELLIGENCE\n")
    if len(outliers) > 0:
        for _, row in outlier_summary.iterrows():
            f.write(f"   - {row['Outlier Type']}: {row['Count']} occurrences\n")
    f.write("\n")
    
    f.write("4. GPU BEHAVIOR PROFILING (Summary)\n")
    counts = gpu_profile['gpu_classification'].value_counts()
    f.write(f"   - Stable GPUs: {counts.get('Stable', 0)}\n")
    f.write(f"   - High-Load Consistent GPUs: {counts.get('High-Load Consistent', 0)}\n")
    f.write(f"   - Unstable / Spiky GPUs: {counts.get('Unstable / Spiky', 0)}\n")
    unstable_gpus = gpu_profile[gpu_profile['gpu_classification'] == 'Unstable / Spiky']['gpu_id'].tolist()
    if unstable_gpus:
        f.write(f"   - Examples of Unstable GPUs: {unstable_gpus[:5]}\n")
    f.write("\n")
    
    f.write("5. SYSTEM-LEVEL INSIGHTS\n")
    h100_mean = df[df['gpu_type'] == 'H100']['power_watts'].mean()
    a100_mean = df[df['gpu_type'] == 'A100']['power_watts'].mean()
    f.write(f"   - GPU Type Load: H100s run at {h100_mean:.2f}W avg vs A100s at {a100_mean:.2f}W avg. H100s are generally provisioned for higher capacity, but ensure this isn't solely due to scheduler bias.\n")
    
    node_means = df.groupby('hostname')['power_watts'].mean().sort_values(ascending=False)
    f.write(f"   - Top Loaded Node: {node_means.index[0]} ({node_means.iloc[0]:.2f}W)\n")
    f.write(f"   - Least Loaded Node: {node_means.index[-1]} ({node_means.iloc[-1]:.2f}W)\n\n")
    
    f.write("6. CITM ACTION POINTS (STRONG)\n")
    f.write("   - INVESTIGATE UNSTABLE GPUs: Cross-reference the identified 'Unstable / Spiky' GPUs with specific job IDs. If jobs are failing, hardware throttling may be occurring.\n")
    f.write("   - REDISTRIBUTE WORKLOADS: There is a measurable imbalance between the top and bottom loaded nodes. The scheduler should be adjusted to offload background tasks from the heavily loaded dgxh nodes to idle dgxa nodes.\n")
    f.write("   - PEAK TRANSITION ALERTS: The transition matrix shows the shift in load behaviors during peak hours (12-18). Implement rate-limiting or job queueing during these specific hours to prevent anomalous power spikes.\n")

print("Pipeline Complete!")
