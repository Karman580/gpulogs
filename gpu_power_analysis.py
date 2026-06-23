# Disable GUI backend (prevents freezing)
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# ---------------- SETTINGS ----------------
INPUT_FILE = "gpu_power_log.csv"
CHUNK_SIZE = 500000
REDUCED_NODE_FILE = "reduced_node_power.csv"
REDUCED_GPU_FILE = "reduced_gpu_power.csv"
OUTLIERS_FILE = "outliers.csv"
STATS_FILE = "stats.csv"
# ------------------------------------------

def prepare_data():
    """
    Checks if reduced datasets exist. If not, processes the raw large CSV in chunks
    and creates the aggregated datasets to prevent full memory load.
    """
    if os.path.exists(REDUCED_NODE_FILE) and os.path.exists(REDUCED_GPU_FILE):
        print("Loading pre-aggregated datasets...")
        node_df = pd.read_csv(REDUCED_NODE_FILE, parse_dates=['timestamp'])
        gpu_df = pd.read_csv(REDUCED_GPU_FILE, parse_dates=['timestamp'])
        return node_df, gpu_df

    print(f"Pre-aggregated datasets not found. Processing {INPUT_FILE} in chunks...")
    
    # Count rows for tqdm
    with open(INPUT_FILE) as f:
        total_rows = sum(1 for _ in f) - 1

    print(f"Total rows in raw data: {total_rows}")
    
    node_chunks = []
    gpu_chunks = []
    
    # Read in chunks to avoid memory errors
    reader = pd.read_csv(
        INPUT_FILE,
        usecols=['hostname', 'timestamp', 'pci_bus_id', 'power_watts'],
        chunksize=CHUNK_SIZE
    )
    
    for chunk in tqdm(reader, total=total_rows // CHUNK_SIZE):
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        
        # Downsample timestamps to 5-minute intervals
        chunk['timestamp'] = chunk['timestamp'].dt.floor('5min')
        
        # Node-level aggregation (Sum of all GPUs for each host & timestamp)
        node_agg = chunk.groupby(['hostname', 'timestamp'])['power_watts'].sum().reset_index()
        node_chunks.append(node_agg)
        
        # GPU-level aggregation (Mean power per GPU over the 5min period)
        gpu_agg = chunk.groupby(['hostname', 'pci_bus_id', 'timestamp'])['power_watts'].mean().reset_index()
        gpu_chunks.append(gpu_agg)
        
    print("\nCombining results and saving reduced datasets...")
    
    # Re-aggregate grouped chunks
    node_df = pd.concat(node_chunks)
    node_df = node_df.groupby(['hostname', 'timestamp'])['power_watts'].sum().reset_index()
    node_df.to_csv(REDUCED_NODE_FILE, index=False)
    
    gpu_df = pd.concat(gpu_chunks)
    gpu_df = gpu_df.groupby(['hostname', 'pci_bus_id', 'timestamp'])['power_watts'].mean().reset_index()
    gpu_df.to_csv(REDUCED_GPU_FILE, index=False)
    
    return node_df, gpu_df


def detect_outliers(df, group_cols, value_col):
    """
    Detects outliers using the IQR method.
    Outliers = values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
    Returns a DataFrame containing only the outliers.
    """
    outliers_list = []
    
    for name, group in df.groupby(group_cols):
        Q1 = group[value_col].quantile(0.25)
        Q3 = group[value_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        mask = (group[value_col] < lower_bound) | (group[value_col] > upper_bound)
        outliers_list.append(group[mask])
        
    if outliers_list:
        return pd.concat(outliers_list)
    return pd.DataFrame(columns=df.columns)

def main():
    # 1. Prepare / Load Data efficiently
    node_df, gpu_df = prepare_data()
    
    print(f"Node-level rows: {len(node_df)}")
    print(f"GPU-level rows: {len(gpu_df)}")
    
    # 2. Box Plot Generation
    print("\nGenerating Box Plots...")
    hosts = node_df['hostname'].unique()
    sns.set_theme(style="whitegrid") # Use clean seaborn theme for readability
    
    for host in hosts:
        # A) Node-level box plot
        host_node_df = node_df[node_df['hostname'] == host]
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=host_node_df['power_watts'], color='skyblue')
        plt.title(f"{host} - Node Level Total Power Box Plot")
        plt.ylabel("Total Power (W)")
        plt.tight_layout()
        filename_node = f"{host}_node_boxplot.png"
        plt.savefig(filename_node, dpi=200)
        plt.close()
        print(f"Saved: {filename_node}")
        
        # B) GPU-level box plot (all 8 GPUs per node)
        host_gpu_df = gpu_df[gpu_df['hostname'] == host]
        
        plt.figure(figsize=(14, 6))
        # Create a boxplot for each GPU in the host
        sns.boxplot(x='pci_bus_id', y='power_watts', data=host_gpu_df, palette="Set2", legend=False)
        plt.title(f"{host} - GPU Level Power Box Plot")
        plt.xlabel("PCI Bus ID")
        plt.ylabel("Power (W)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        filename_gpu = f"{host}_gpu_boxplot.png"
        plt.savefig(filename_gpu, dpi=200)
        plt.close()
        print(f"Saved: {filename_gpu}")
        
    print("\n✅ Box plots generated and saved.")
    
    # 3. Outlier Detection
    print("\nDetecting Outliers using IQR method...")
    
    # Node-level outliers
    node_outliers = detect_outliers(node_df, ['hostname'], 'power_watts')
    node_outliers['pci_bus_id'] = 'NODE_TOTAL'  # Distinguish from individual GPUs
    
    # GPU-level outliers
    gpu_outliers = detect_outliers(gpu_df, ['hostname', 'pci_bus_id'], 'power_watts')
    
    # Combine and save
    all_outliers = pd.concat([node_outliers, gpu_outliers])
    
    # Rearrange and select final columns
    all_outliers = all_outliers[['hostname', 'pci_bus_id', 'timestamp', 'power_watts']]
    all_outliers.to_csv(OUTLIERS_FILE, index=False)
    print(f"✅ Saved outlier timestamps to {OUTLIERS_FILE} ({len(all_outliers)} records).")
    
    # 4. Statistical Analysis
    print("\nComputing Statistical Analysis (Mean, Variance, Std Dev)...")
    
    # Node-level statistics
    node_stats = node_df.groupby('hostname')['power_watts'].agg(['mean', 'var', 'std']).reset_index()
    node_stats['pci_bus_id'] = 'NODE_TOTAL'
    node_stats = node_stats.rename(columns={'var': 'variance', 'std': 'std_dev'})
    
    # GPU-level statistics
    gpu_stats = gpu_df.groupby(['hostname', 'pci_bus_id'])['power_watts'].agg(['mean', 'var', 'std']).reset_index()
    gpu_stats = gpu_stats.rename(columns={'var': 'variance', 'std': 'std_dev'})
    
    # Combine and save
    all_stats = pd.concat([node_stats, gpu_stats])
    all_stats = all_stats[['hostname', 'pci_bus_id', 'mean', 'variance', 'std_dev']]
    all_stats.to_csv(STATS_FILE, index=False)
    print(f"✅ Saved statistics to {STATS_FILE}.")
    
    print("\n🎉 All tasks completed successfully.")

if __name__ == "__main__":
    main()
