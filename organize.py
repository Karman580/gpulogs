import os
import shutil
from pathlib import Path

# ---------------- SETTINGS ----------------
base_dir = Path.cwd()

node_box_dir = base_dir / "BoxPlots" / "node_boxplots"
gpu_box_dir = base_dir / "BoxPlots" / "gpu_boxplots"
stats_dir = base_dir / "Stats"
# ------------------------------------------

print("Creating folder structure...")

node_box_dir.mkdir(parents=True, exist_ok=True)
gpu_box_dir.mkdir(parents=True, exist_ok=True)
stats_dir.mkdir(parents=True, exist_ok=True)

# ---------------- MOVE FILES ----------------

print("Organizing files...")

for file in base_dir.iterdir():

    if file.is_file():

        # Node box plots
        if file.name.endswith("_node_boxplot.png"):
            shutil.move(str(file), node_box_dir / file.name)

        # GPU box plots
        elif file.name.endswith("_gpu_boxplot.png"):
            shutil.move(str(file), gpu_box_dir / file.name)

        # Stats CSV
        elif file.name == "stats.csv":
            shutil.move(str(file), stats_dir / file.name)

        # Outliers CSV
        elif file.name == "outliers.csv":
            shutil.move(str(file), stats_dir / file.name)

print("Files organized successfully!")

# ---------------- CREATE ZIP ----------------

print("Creating ZIP file...")

zip_name = "GPU_Analysis_Results"

shutil.make_archive(zip_name, 'zip', base_dir)

print(f"ZIP created: {zip_name}.zip")

print("\n✅ Ready to send to sir!")