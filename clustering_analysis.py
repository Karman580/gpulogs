import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import os
import warnings

# Suppress sklearn warnings about feature names or KMeans memory leaks
warnings.filterwarnings('ignore')

INPUT_DATA = "reduced_gpu_power.csv"
OUTPUT_DIR = "clustering"

def detect_outliers_iqr(df):
    """Detects outliers in a DataFrame using the IQR method on power_watts."""
    if df.empty:
        return df
    Q1 = df['power_watts'].quantile(0.25)
    Q3 = df['power_watts'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df['power_watts'] < lower_bound) | (df['power_watts'] > upper_bound)].copy()

def perform_clustering_labeled(df, features, cluster_col_name, num_clusters=3):
    """
    Applies StandardScaler and KMeans to the provided dataframe context.
    Labels the clusters based on their mean power_watts using domain logic.
    """
    if len(df) < num_clusters:
        return df  # Not enough data to cluster
        
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df[cluster_col_name] = kmeans.fit_predict(X_scaled)
    
    # Analyze and label clusters (Low, Medium, High based on mean power)
    cluster_means = df.groupby(cluster_col_name)['power_watts'].mean().sort_values()
    labels = ['Low usage / idle cluster', 'Medium usage cluster', 'High spike cluster']
    
    # Map the labels if K=3
    mapping = {}
    for i, cluster_idx in enumerate(cluster_means.index):
        if len(cluster_means) == 3:
            mapping[cluster_idx] = labels[i]
        else:
            mapping[cluster_idx] = f"Cluster Rank {i+1}"
            
    df[f'{cluster_col_name}_label'] = df[cluster_col_name].map(mapping)
    return df

def analyze_subset(df, subset_name):
    """
    Main analysis block for a single subset. Modifies and outputs insights.
    """
    print(f"\n{'='*40}")
    print(f"Processing {subset_name} - {len(df)} records")
    print(f"{'='*40}")
    
    if df.empty:
        print("Empty subset. Skipping.")
        return None
        
    # 2. Box Plot Analysis (Per Subset)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='day_num', y='power_watts', color='lightskyblue', fliersize=2)
    plt.title(f'Daily Power Distribution - {subset_name}', fontsize=16)
    plt.xlabel('Day Number', fontsize=12)
    plt.ylabel('Power (Watts)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{subset_name}_boxplot.png'), dpi=200)
    plt.close()
    
    # 3. Outlier Detection
    outliers_df = detect_outliers_iqr(df)
    
    if outliers_df.empty:
        print("No outliers detected in this subset.")
        return {
            'subset': subset_name,
            'variance': df['power_watts'].var(),
            'spikes': 0,
            'mean_power': df['power_watts'].mean()
        }
        
    print(f"Detected {len(outliers_df)} outliers (spikes/drops).")
    
    # Extract and save
    outlier_export = outliers_df[['hostname', 'pci_bus_id', 'timestamp', 'power_watts']]
    outliers_path = os.path.join(OUTPUT_DIR, f'{subset_name}_outliers.csv')
    outlier_export.to_csv(outliers_path, index=False)
    
    # 4. Feature Engineering
    outliers_df['hour'] = outliers_df['timestamp'].dt.hour
    le = LabelEncoder()
    outliers_df['encoded_pci_bus_id'] = le.fit_transform(outliers_df['pci_bus_id'])
    
    clustering_features = ['power_watts', 'hour', 'day_num', 'encoded_pci_bus_id']
    print(f"Features prepared for clustering: {clustering_features}")
    
    # 6. Clustering (Unsupervised K-Means) - 3 Levels
    
    # Level A: Global (Across all GPUs in the subset)
    print("-> Clustering Across GPUs (Global)")
    outliers_df = perform_clustering_labeled(outliers_df, clustering_features, 'global_cluster', num_clusters=3)
    
    # Level B: Per Node (hostname)
    print("-> Clustering Per Node")
    outliers_df['node_cluster'] = -1
    outliers_df['node_cluster_label'] = 'None'
    
    for node in outliers_df['hostname'].unique():
        idx = outliers_df['hostname'] == node
        node_data = outliers_df.loc[idx].copy()
        if len(node_data) >= 3:
            node_data = perform_clustering_labeled(node_data, clustering_features, 'node_cluster', num_clusters=3)
            outliers_df.loc[idx, 'node_cluster'] = node_data['node_cluster']
            outliers_df.loc[idx, 'node_cluster_label'] = node_data['node_cluster_label']
            
    # Level C: Per GPU (pci_bus_id)
    print("-> Clustering Per GPU")
    outliers_df['gpu_cluster'] = -1
    outliers_df['gpu_cluster_label'] = 'None'
    
    for gpu in outliers_df['pci_bus_id'].unique():
        idx = outliers_df['pci_bus_id'] == gpu
        gpu_data = outliers_df.loc[idx].copy()
        if len(gpu_data) >= 3:
            gpu_data = perform_clustering_labeled(gpu_data, clustering_features, 'gpu_cluster', num_clusters=3)
            outliers_df.loc[idx, 'gpu_cluster'] = gpu_data['gpu_cluster']
            outliers_df.loc[idx, 'gpu_cluster_label'] = gpu_data['gpu_cluster_label']

    # 7. Cluster Analysis (Output summary of the global clusters)
    print("\n--- Global Outlier Cluster Analysis ---")
    if 'global_cluster_label' in outliers_df.columns:
        cluster_summary = outliers_df.groupby('global_cluster_label').agg({
            'power_watts': ['mean', 'min', 'max', 'std'],
            'hour': lambda x: x.mode()[0] if not x.empty else np.nan,  # most common hour
            'timestamp': 'count' # frequency
        }).rename(columns={'timestamp': 'frequency'})
        
        print(cluster_summary)
        
        # Save cluster summary
        cluster_summary.to_csv(os.path.join(OUTPUT_DIR, f'{subset_name}_cluster_summary.csv'))
    
    # Save clustered datasets
    clustered_path = os.path.join(OUTPUT_DIR, f'{subset_name}_clustered_outliers.csv')
    outliers_df.to_csv(clustered_path, index=False)
    
    return {
        'subset': subset_name,
        'variance': df['power_watts'].var(),
        'spikes': len(outliers_df),
        'mean_power': df['power_watts'].mean()
    }

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
        
    print(f"Loading data from {INPUT_DATA}...")
    if not os.path.exists(INPUT_DATA):
        print(f"Error: {INPUT_DATA} not found.")
        return
        
    df = pd.read_csv(INPUT_DATA, parse_dates=['timestamp'])
    df.dropna(subset=['timestamp', 'power_watts'], inplace=True)
    
    # 1. Time Segmentation
    min_timestamp = df['timestamp'].min()
    df['day_num'] = (df['timestamp'] - min_timestamp).dt.days + 1
    
    print(f"Total days found in data: {df['day_num'].min()} to {df['day_num'].max()}")
    
    # Split into subsets
    subsets = {
        'subset_1': df[(df['day_num'] >= 1) & (df['day_num'] <= 3)],
        'subset_2': df[(df['day_num'] >= 4) & (df['day_num'] <= 6)],
        'subset_3': df[(df['day_num'] >= 7) & (df['day_num'] <= 9)]
    }
    
    subset_metrics = []
    for name, data in subsets.items():
        metrics = analyze_subset(data, name)
        if metrics:
            subset_metrics.append(metrics)
            
    # 8. Subset Comparison
    print("\n" + "="*50)
    print("SUBSET COMPARISON & OVERALL INSIGHTS")
    print("="*50)
    
    if len(subset_metrics) > 0:
        comp_df = pd.DataFrame(subset_metrics)
        print(comp_df[['subset', 'mean_power', 'variance', 'spikes']].to_string(index=False))
        
        # Identify characteristics
        highest_var = comp_df.loc[comp_df['variance'].idxmax(), 'subset']
        most_spikes = comp_df.loc[comp_df['spikes'].idxmax(), 'subset']
        most_stable = comp_df.loc[comp_df['variance'].idxmin(), 'subset']
        
        print("\nInterpretations:")
        print(f"-> Highest Variability: {highest_var}")
        print(f"-> Most Spikes (Outliers): {most_spikes}")
        print(f"-> Most Stable (Lowest Variance): {most_stable}")
        
        # Save comparison report
        with open(os.path.join(OUTPUT_DIR, 'subset_comparison_report.txt'), 'w') as f:
            f.write("Subset Comparison & Overall Insights\n")
            f.write("="*40 + "\n\n")
            f.write(comp_df[['subset', 'mean_power', 'variance', 'spikes']].to_string(index=False))
            f.write("\n\nInterpretations:\n")
            f.write(f"-> Highest Variability: {highest_var}\n")
            f.write(f"-> Most Spikes (Outliers): {most_spikes}\n")
            f.write(f"-> Most Stable (Lowest Variance): {most_stable}\n")
            
    print("\nAnalysis complete. All requested artifacts saved to the 'clustering' directory.")

if __name__ == "__main__":
    main()
