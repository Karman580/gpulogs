import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import gc

def generate_daily_boxplots(data_path, output_dir='.', use_day_numbers=True):
    """
    Reads preprocessed GPU power logs and generates daily box plots per node,
    and optionally a global daily box plot.
    """
    print(f"Loading data from {data_path}...")
    
    # Load required columns to minimize memory usage
    columns_to_load = ['hostname', 'timestamp', 'power_watts']
    try:
        # We only read the necessary columns for plotting to save memory
        df = pd.read_csv(data_path, usecols=columns_to_load)
    except ValueError:
        # Fallback if columns are missing or differ slightly
        df = pd.read_csv(data_path)
        
    print("Data loaded. Applying datetime transformations...")
    
    # Ensure timestamp is datetime type
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Drop any NaNs in the required columns
    df.dropna(subset=['timestamp', 'power_watts', 'hostname'], inplace=True)
    
    # Compute day formatting
    if use_day_numbers:
        # Calculate days since start (Day 1, Day 2, etc.)
        min_timestamp = df['timestamp'].min()
        # Create an integer offset
        df['day_number'] = (df['timestamp'] - min_timestamp).dt.days + 1
        df['Day_Label'] = "Day " + df['day_number'].astype(str)
        # Sort by day_number just in case, ensuring correct plotting order
        df.sort_values('day_number', inplace=True)
        x_col = 'Day_Label'
    else:
        # Just use the date string
        df['Date'] = df['timestamp'].dt.date.astype(str)
        df.sort_values('Date', inplace=True)
        x_col = 'Date'

    # Set overall seaborn aesthetic qualities
    sns.set_theme(style="whitegrid")

    # Optional: Global Plot (aggregating across all nodes)
    print("Generating global daily boxplot...")
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df, x=x_col, y='power_watts', color='skyblue', fliersize=2)
    
    plt.title('Global GPU Power Consumption By Day (All Nodes)', fontsize=16)
    plt.xlabel('Day', fontsize=14)
    plt.ylabel('Power (W)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    global_out = os.path.join(output_dir, 'global_daily_boxplot.png')
    plt.savefig(global_out, dpi=300)
    plt.close()

    # Node-wise Plots
    unique_nodes = df['hostname'].unique()
    print(f"Found {len(unique_nodes)} unique nodes. Generating individual plots...")
    
    for node in unique_nodes:
        print(f"  -> Processing node {node}...")
        # isolate data for the node
        node_df = df[df['hostname'] == node]
        
        plt.figure(figsize=(14, 7))
        # Use a distinct color for individual nodes, adjust fliersize to avoid clutter for large datasets
        sns.boxplot(data=node_df, x=x_col, y='power_watts', color='lightgreen', fliersize=2)
        
        plt.title(f'Daily GPU Power Consumption - Node: {node}', fontsize=16)
        plt.xlabel('Day', fontsize=14)
        plt.ylabel('Power (W)', fontsize=14)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        node_safe_name = str(node).replace('/', '_').replace('\\', '_')
        out_path = os.path.join(output_dir, f"{node_safe_name}_daily_boxplot.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        
        # Free up memory explicitly for large datasets
        del node_df
        gc.collect()

    print("All plots generated successfully!")

if __name__ == "__main__":
    CSV_PATH = 'reduced_gpu_power.csv' 
    OUTPUT_DIR = 'boxplots day'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if os.path.exists(CSV_PATH):
        generate_daily_boxplots(CSV_PATH, output_dir=OUTPUT_DIR, use_day_numbers=True)
    else:
        print(f"Dataset not found at {CSV_PATH}. Make sure the path is correct.")
