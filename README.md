# GPU Power Telemetry & Workload Clustering Analysis

This repository contains tools for processing, statistical profiling, and clustering of high-frequency GPU power telemetry logs across multi-GPU compute nodes. The analysis pipeline identifies workload anomalies (outliers), determines peak workload windows, and groups execution behaviors using density-based spatial clustering (DBSCAN).

## 🚀 Overview

Modern GPU clusters (composed of NVIDIA A100s, H100s, etc.) experience complex, bursty workload behaviors. This project processes raw power consumption logs (gigabytes of high-frequency telemetry) to extract actionable operational insights:
- **Aggregated Telemetry**: Handles multi-gigabyte log files efficiently using chunked loading and time-window downsampling (e.g., 5-minute aggregation).
- **Outlier Detection**: Employs IQR-based statistical profiling to flag anomalous power consumption events at both individual GPU and complete node levels.
- **Density-based Clustering**: Utilizes DBSCAN with auto-tuned search radii ($\epsilon$) and scaling to segment workloads across Peak and Non-Peak hours.
- **Rich Visualization**: Generates host-level boxplots, PCA cluster visualizations, and anomaly count distributions.

---

## 📁 Repository Structure

```
gpulogs/
├── .gitignore                      # Excludes large raw logs & redundant archives
├── README.md                       # Documentation
├── clustering_analysis.py          # Script for clustering analysis
├── daily_boxplots.py               # Generates daily node box plots
├── extended_analysis.py            # Advanced temporal/pattern telemetry analysis
├── gpu_deep_analysis_v2.py         # Advanced multi-variable telemetry pipeline
├── gpu_final_analysis.py           # Finalized statistical and visual reporting
├── gpu_power_analysis.py           # Core IQR outlier detector & data downsampler
├── gpu_telemetry_analysis.py       # DBSCAN clustering & outlier analysis pipeline
├── organize.py                     # Utility to structure outputs
├── plt.py                          # Visual plotting helpers
├── BoxPlots/                       # Generated node & GPU box plots (PNGs)
│   ├── gpu_boxplots/
│   └── node_boxplots/
├── Results/                        # Outlier CSVs and boxplots per subset
├── Stats/                          # Combined outliers and statistics
└── extended_analysis/              # Temporal pattern and workload distribution plots
```

---

## 🛠️ Getting Started

### 1. Installation
Install the necessary python dependencies using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
```

### 2. Dataset Setup
This analysis is designed for GPU power telemetry logs (`.csv`). By default, the core pipelines look for:
- `gpu_power_log.csv`: The full raw telemetry logs dataset.
- `gpu_power_log_apr20_27.csv`: The target weekly logs dataset.

> [!NOTE]
> Due to GitHub's file size limits, large raw CSV logs (>100MB) and compressed ZIP bundles are ignored by git. Place your raw telemetry CSV files in the root folder locally before running the scripts.

---

## 📊 Core Workflows

### Phase 1: Downsampling & Statistical Outlier Detection
Run `gpu_power_analysis.py` to downsample the raw data into manageable 5-minute averages and output standard box plots and IQR outliers:
```bash
python gpu_power_analysis.py
```
This produces:
- `reduced_gpu_power.csv` & `reduced_node_power.csv` (used by downstream scripts)
- `outliers.csv` containing anomalous timestamps
- `stats.csv` with mean, variance, and standard deviation for each GPU and node

Organize these results into folders using:
```bash
python organize.py
```

### Phase 2: Workload Clustering (DBSCAN)
Run `gpu_telemetry_analysis.py` to analyze patterns during peak vs. non-peak hours:
```bash
python gpu_telemetry_analysis.py
```
This script:
1. Performs rolling window calculations of mean power and delta change.
2. Identifies peak usage periods automatically.
3. Scales features and fits DBSCAN to classify regular cluster patterns and anomalies.
4. Outputs detailed clustering reports under `analysis_20_27_april/` along with 2D Principal Component Analysis (PCA) projection plots.

### Phase 3: Advanced Temporal & Pattern Analysis
Run `extended_analysis.py` to check for diurnal pattern repetitions, workload burstiness, and temporal state transitions:
```bash
python extended_analysis.py
```

---

## 📈 Key Visualizations & Outputs
The scripts produce rich charts to help compute infrastructure teams audit cluster power efficiency:
* **Host Power Boxplots**: Boxplots indicating standard operating boundaries and outlier points for individual GPUs (`pci_bus_id`) and total hosts.
* **PCA Cluster Projections**: Visualizes the density groupings of GPU workloads during peak and non-peak workloads.
* **Anomalous Hour Count**: Categorizes the exact time of day when power telemetry spikes or drops significantly.
