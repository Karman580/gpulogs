# Disable GUI backend (prevents freezing)
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- SETTINGS ----------------
input_file = "gpu_power_log.csv"
chunk_size = 500000
# ------------------------------------------

print("Counting total rows...")

with open(input_file) as f:
    total_rows = sum(1 for _ in f) - 1

print("Total rows:", total_rows)
print("Processing file...\n")

node_chunks = []
gpu_chunks = []

reader = pd.read_csv(
    input_file,
    usecols=['hostname', 'timestamp', 'pci_bus_id', 'power_watts'],
    chunksize=chunk_size
)

for chunk in tqdm(reader, total=total_rows // chunk_size):
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])

    # Downsample to 5-minute resolution
    chunk['timestamp'] = chunk['timestamp'].dt.floor('5min')

    # Node-level aggregation (sum of 8 GPUs)
    node_agg = chunk.groupby(['hostname', 'timestamp'])['power_watts'].sum().reset_index()
    node_chunks.append(node_agg)

    # GPU-level aggregation (per MAC / pci_bus_id)
    gpu_agg = chunk.groupby(['hostname', 'pci_bus_id', 'timestamp'])['power_watts'].mean().reset_index()
    gpu_chunks.append(gpu_agg)

print("\nCombining results...")

node_df = pd.concat(node_chunks)
node_df = node_df.groupby(['hostname', 'timestamp'])['power_watts'].sum().reset_index()

gpu_df = pd.concat(gpu_chunks)
gpu_df = gpu_df.groupby(['hostname', 'pci_bus_id', 'timestamp'])['power_watts'].mean().reset_index()

print("Node-level rows:", len(node_df))
print("GPU-level rows:", len(gpu_df))

# ----------------------------------------------------
# 1️⃣ NODE-LEVEL CONSUMPTION GRAPHS (One per DGX)
# ----------------------------------------------------

print("\nGenerating per-node total power graphs...")

for host in sorted(node_df['hostname'].unique()):

    host_df = node_df[node_df['hostname'] == host]

    plt.figure(figsize=(12,4))
    plt.plot(
        host_df['timestamp'],
        host_df['power_watts'],
        color='red',
        linewidth=1
    )

    plt.title(f"{host} - Total Power Consumption")
    plt.xlabel("Time")
    plt.ylabel("Total Power (W)")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"{host}_total_power.png"
    plt.savefig(filename, dpi=200)
    plt.close()

    print(f"Saved: {filename}")

# ----------------------------------------------------
# 2️⃣ PER-DGX 8-GPU DIFFERENTIATED GRAPHS
# ----------------------------------------------------

print("\nGenerating per-DGX 8-GPU differentiated graphs...")

for host in sorted(gpu_df['hostname'].unique()):

    host_gpu_df = gpu_df[gpu_df['hostname'] == host]

    plt.figure(figsize=(12,5))

    for gpu_id in host_gpu_df['pci_bus_id'].unique():
        gpu_data = host_gpu_df[host_gpu_df['pci_bus_id'] == gpu_id]

        plt.plot(
            gpu_data['timestamp'],
            gpu_data['power_watts'],
            linewidth=1,
            label=gpu_id
        )

    plt.title(f"{host} - 8 GPU Consumption History")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend(fontsize=6)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"{host}_8gpu_power.png"
    plt.savefig(filename, dpi=200)
    plt.close()

    print(f"Saved: {filename}")

print("\n✅ All graphs generated successfully.")