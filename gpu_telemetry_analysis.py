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
DATA_PATH = "/Users/karmansinghtalwar/Documents/GPU LOGS/gpu_power_log_apr20_27.csv"
OUTPUT_DIR = "/Users/karmansinghtalwar/Documents/GPU LOGS/analysis_20_27_april/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("STEP 1: DATA LOADING & CLEANING")
start_time = time.time()
# 1. Load Data
# We read the massive 8.2M row CSV directly into pandas. It will take ~2-3GB of RAM.
df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} rows in {time.time() - start_time:.2f} seconds.")

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Clean string columns
for col in ['hostname', 'gpu_name', 'pci_bus_id']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')

# Drop invalid timestamps
df = df.dropna(subset=['timestamp'])

# Sort chronologically
df = df.sort_values(by='timestamp').reset_index(drop=True)

# Create unique GPU ID
df['gpu_id'] = df['hostname'] + "_" + df['pci_bus_id']

# Handle missing or corrupt rows (drop rows without power readings)
df = df.dropna(subset=['power_watts'])

print(f"Post-cleaning: {len(df)} rows.")

print("STEP 2: FEATURE ENGINEERING")
# 2. Feature Engineering
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Group by GPU to compute rolling mean and delta
df = df.sort_values(by=['gpu_id', 'timestamp'])

# Rolling mean power (window=5 readings)
df['rolling_mean_power'] = df.groupby('gpu_id')['power_watts'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

# Power delta
df['power_delta'] = df.groupby('gpu_id')['power_watts'].diff().fillna(0)

# Define PEAK HOURS
hourly_avg = df.groupby('hour')['power_watts'].mean()
peak_threshold = hourly_avg.quantile(0.7) # top 30%
peak_hours = hourly_avg[hourly_avg >= peak_threshold].index.tolist()

df['is_peak'] = df['hour'].apply(lambda x: 1 if x in peak_hours else 0)

# Restore chronological order
df = df.sort_values(by='timestamp').reset_index(drop=True)

# Normalize features
features_to_scale = ['power_watts', 'rolling_mean_power', 'power_delta']
scaler = StandardScaler()
# Fit scaler on full data
df_scaled = scaler.fit_transform(df[features_to_scale])

print("STEP 3: CLUSTERING (DBSCAN) - Auto tuning")

def perform_scalable_clustering(data_subset, scaled_subset, name_prefix):
    n_total = len(data_subset)
    if n_total < 50:
        print(f"Not enough data for {name_prefix} clustering.")
        return data_subset.copy(), 0.5
        
    # Scale for DBSCAN: Max 50,000 points to avoid memory/time crash on O(N^2)
    max_sample = 50000
    if n_total > max_sample:
        print(f"Downsampling {name_prefix} from {n_total} to {max_sample} for DBSCAN training...")
        # Get random indices
        np.random.seed(42)
        sample_idx = np.random.choice(n_total, max_sample, replace=False)
        X_train = scaled_subset[sample_idx]
    else:
        X_train = scaled_subset
        sample_idx = np.arange(n_total)
        
    min_samples = 20
    
    # Auto-tune epsilon using k-distance
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X_train)
    distances, indices = neighbors_fit.kneighbors(X_train)
    distances = np.sort(distances[:, min_samples-1])
    
    eps_val = np.percentile(distances, 90)
    if eps_val <= 0:
        eps_val = 0.1
        
    # Anti-fragile logic
    max_attempts = 5
    best_eps = eps_val
    best_labels = None
    
    for attempt in range(max_attempts):
        db = DBSCAN(eps=eps_val, min_samples=min_samples).fit(X_train)
        labels = db.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        outlier_pct = list(labels).count(-1) / len(labels)
        
        if outlier_pct < 0.5 and n_clusters > 1:
            best_labels = labels
            best_eps = eps_val
            break
        elif outlier_pct >= 0.5:
            eps_val *= 1.5 
        elif n_clusters <= 1:
            eps_val *= 0.8 
            
    if best_labels is None:
        best_eps = eps_val
        db = DBSCAN(eps=best_eps, min_samples=min_samples).fit(X_train)
        best_labels = db.labels_

    print(f"[{name_prefix}] Trained DBSCAN. eps: {best_eps:.3f}, clusters: {len(set(best_labels)) - (1 if -1 in best_labels else 0)}, sample outliers: {list(best_labels).count(-1)} ({list(best_labels).count(-1)/len(best_labels)*100:.1f}%)")
    
    # Extrapolate to full subset if needed using KNN
    data_subset = data_subset.copy()
    if n_total > max_sample:
        print(f"[{name_prefix}] Extrapolating clusters to {n_total} points using KNN...")
        # Train KNN on inliers
        inlier_mask = best_labels != -1
        if inlier_mask.sum() > 0:
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train[inlier_mask], best_labels[inlier_mask])
            
            # Predict all
            # Process in chunks to save memory
            chunk_size = 100000
            full_labels = np.zeros(n_total, dtype=int)
            
            # Find distances to nearest inlier. If distance > best_eps, mark as noise
            nn = NearestNeighbors(n_neighbors=1).fit(X_train[inlier_mask])
            
            for i in range(0, n_total, chunk_size):
                chunk = scaled_subset[i:i+chunk_size]
                dists, _ = nn.kneighbors(chunk)
                preds = knn.predict(chunk)
                
                # Mark as noise if distance exceeds eps
                preds[dists.flatten() > best_eps] = -1
                full_labels[i:i+chunk_size] = preds
                
            data_subset['cluster'] = full_labels
        else:
            data_subset['cluster'] = -1
    else:
        data_subset['cluster'] = best_labels
        
    n_outliers = list(data_subset['cluster']).count(-1)
    print(f"[{name_prefix}] Total Outliers: {n_outliers} ({n_outliers/n_total*100:.1f}%)")
    
    return data_subset, best_eps

# Run on datasets
print("\nRunning Full clustering...")
df_full, eps_full = perform_scalable_clustering(df, df_scaled, "Full")

print("\nRunning Peak clustering...")
peak_mask = df['is_peak'] == 1
df_peak, eps_peak = perform_scalable_clustering(df[peak_mask], df_scaled[peak_mask], "Peak")

print("\nRunning Non-Peak clustering...")
non_peak_mask = df['is_peak'] == 0
df_non_peak, eps_non_peak = perform_scalable_clustering(df[non_peak_mask], df_scaled[non_peak_mask], "Non-Peak")


print("\nSTEP 4: VISUALIZATION")

def plot_clusters(df_plot, name):
    if df_plot is None or len(df_plot) == 0: return
    plt.figure(figsize=(10, 6))
    
    # Sample down for plotting if too large
    if len(df_plot) > 20000:
        df_sample = df_plot.sample(20000, random_state=42)
    else:
        df_sample = df_plot
        
    X_scaled = scaler.transform(df_sample[features_to_scale])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    labels = df_sample['cluster'].values
    unique_labels = set(labels)
    
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [1, 0, 0, 1] # Red for noise
        class_member_mask = (labels == k)
        xy = X_pca[class_member_mask]
        
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6 if k != -1 else 4, alpha=0.5,
                 label=f'Cluster {k}' if k != -1 else 'Outliers')
    
    plt.title(f'DBSCAN Clusters ({name}) - PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    
    # Only show legend if not too many clusters
    if len(unique_labels) < 15:
        plt.legend()
        
    plt.savefig(os.path.join(OUTPUT_DIR, f'cluster_{name.lower().replace("-", "_")}.png'))
    plt.close()

plot_clusters(df_full, "Full")
plot_clusters(df_peak, "Peak")
plot_clusters(df_non_peak, "Non-Peak")

# Time-series plot
plt.figure(figsize=(15, 6))
sns.lineplot(data=df.sample(min(10000, len(df))), x='timestamp', y='power_watts', hue='hostname', alpha=0.5)
plt.title('Power Watts vs Time (Sampled)')
plt.savefig(os.path.join(OUTPUT_DIR, 'time_series.png'))
plt.close()

# Peak vs Non-peak comparison
plt.figure(figsize=(10, 6))
# Sample down to make boxplot faster
sns.boxplot(data=df.sample(min(50000, len(df))), x='is_peak', y='power_watts', hue='hostname')
plt.title('Peak vs Non-Peak Power Comparison')
plt.savefig(os.path.join(OUTPUT_DIR, 'peak_vs_non_peak.png'))
plt.close()

# GPU Distribution
plt.figure(figsize=(15, 6))
sns.boxplot(data=df.sample(min(50000, len(df))), x='gpu_id', y='power_watts')
plt.xticks(rotation=90)
plt.title('Power Distribution per GPU')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'gpu_distribution.png'))
plt.close()

# Outliers
if df_full is not None:
    plt.figure(figsize=(10, 6))
    outliers = df_full[df_full['cluster'] == -1]
    sns.countplot(data=outliers, x='hour')
    plt.title('Outliers Distribution across Hours')
    plt.savefig(os.path.join(OUTPUT_DIR, 'outliers.png'))
    plt.close()

print("\nSTEP 5 & 6: OUTLIER & WORKLOAD ANALYSIS")

if df_full is not None:
    outliers = df_full[df_full['cluster'] == -1]
    
    top_outlier_gpus = outliers['gpu_id'].value_counts().head(5).to_dict()
    outliers_by_hour = outliers['hour'].value_counts().head(3).to_dict()
    
    load_by_node = df_full.groupby('hostname')['power_watts'].mean().to_dict()
    
    # GPU Types (naive extraction from gpu_name)
    df_full['gpu_type'] = df_full['gpu_name'].apply(lambda x: 'A100' if 'A100' in str(x) else ('H100' if 'H100' in str(x) else 'Other'))
    load_by_gpu_type = df_full.groupby('gpu_type')['power_watts'].mean().to_dict()
    
    # High load GPUs
    gpu_avg_load = df_full.groupby('gpu_id')['power_watts'].mean()
    high_load_gpus = gpu_avg_load.nlargest(5).to_dict()
    
    print("\nSTEP 7: TIME-BASED INSIGHTS")
    
    cluster_density_peak = len(df_peak[df_peak['cluster'] != -1]) / len(df_peak) if len(df_peak) > 0 else 0
    cluster_density_non_peak = len(df_non_peak[df_non_peak['cluster'] != -1]) / len(df_non_peak) if len(df_non_peak) > 0 else 0
    
    print("\nSTEP 8 & 9: OUTPUT STORAGE & REPORTING")
    
    # Save CSVs
    df_full.to_csv(os.path.join(OUTPUT_DIR, 'clustered_full.csv'), index=False)
    df_peak.to_csv(os.path.join(OUTPUT_DIR, 'clustered_peak.csv'), index=False)
    df_non_peak.to_csv(os.path.join(OUTPUT_DIR, 'clustered_non_peak.csv'), index=False)
    
    report_data = {
        "clustering_parameters": {
            "epsilon_full": eps_full,
            "min_samples": 20
        },
        "key_insights": {
            "peak_hours": peak_hours,
            "cluster_density_peak": cluster_density_peak,
            "cluster_density_non_peak": cluster_density_non_peak
        },
        "outlier_findings": {
            "top_outlier_gpus": top_outlier_gpus,
            "top_outlier_hours": outliers_by_hour
        },
        "workload_distribution": {
            "load_by_node": load_by_node,
            "load_by_gpu_type": load_by_gpu_type,
            "consistently_high_load_gpus": high_load_gpus
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w') as f:
        json.dump(report_data, f, indent=4)
        
    with open(os.path.join(OUTPUT_DIR, 'report.txt'), 'w') as f:
        f.write("=== GPU TELEMETRY CLUSTERING REPORT ===\n\n")
        f.write(f"1. CLUSTERING PARAMETERS\n")
        f.write(f"   - Epsilon Used (Full): {eps_full:.3f}\n")
        f.write(f"   - Min Samples: 20\n\n")
        
        f.write(f"2. KEY INSIGHTS\n")
        f.write(f"   - Peak Hours Identified: {peak_hours}\n")
        f.write(f"   - Cluster Inliers Ratio (Peak): {cluster_density_peak*100:.1f}%\n")
        f.write(f"   - Cluster Inliers Ratio (Non-Peak): {cluster_density_non_peak*100:.1f}%\n\n")
        
        f.write(f"3. OUTLIER FINDINGS\n")
        f.write(f"   - Top 5 GPUs with most outliers:\n")
        for k, v in top_outlier_gpus.items():
            f.write(f"     * {k}: {v} anomalies\n")
        f.write(f"   - Top 3 Hours with most outliers: {list(outliers_by_hour.keys())}\n\n")
        
        f.write(f"4. WORKLOAD DISTRIBUTION\n")
        f.write(f"   - Average Load by Node:\n")
        for k, v in load_by_node.items():
            f.write(f"     * {k}: {v:.2f} W\n")
        f.write(f"   - Average Load by GPU Type:\n")
        for k, v in load_by_gpu_type.items():
            f.write(f"     * {k}: {v:.2f} W\n")
        f.write(f"   - Consistently High Load GPUs (Top 5):\n")
        for k, v in high_load_gpus.items():
            f.write(f"     * {k}: {v:.2f} W\n\n")
            
        f.write(f"5. OBSERVATIONS FOR CITM TEAM\n")
        f.write("   - There are specific hours and specific GPUs exhibiting anomalous power consumption behaviors (see Outlier Findings).\n")
        f.write("   - Check if the high load GPUs are consistently running long-standing jobs, or if there is an imbalance in job scheduling.\n")
        f.write("   - A higher number of outliers during peak/non-peak indicates sporadic workload behavior. Review jobs scheduled during the top anomalous hours.\n")

    print("\nFINAL OUTPUT")
    print(f"Epsilon used: {eps_full:.3f}")
    print(f"Number of clusters: {len(set(df_full['cluster'])) - (1 if -1 in df_full['cluster'].values else 0)}")
    print(f"Number of outliers: {list(df_full['cluster']).count(-1)}")
    print(f"Peak hours identified: {peak_hours}")
    print("\nPipeline Complete!")
