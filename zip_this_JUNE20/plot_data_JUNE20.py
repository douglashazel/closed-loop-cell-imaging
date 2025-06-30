import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot timeseries data.")
parser.add_argument(
    "--root_dir",
    required=True,
    help="Root directory containing the 'analysis_results' folder."
)
args = parser.parse_args()

# Path to analysis_results
data_path = os.path.join(args.root_dir, "analysis_results")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"'analysis_results' folder not found in {args.root_dir}")

cells = [c for c in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, c))]
cmap = plt.cm.ocean

fig, axes = plt.subplots(4, 1, dpi=300, figsize=(16, 20))
# With background
axes[0].set_title(f"Intensity: {len(cells)} Cells")
axes[0].set_xlabel("Frames")
axes[0].set_ylabel("Intensity")
axes[0].set_ylim(-25, 225)

# With background derivative
axes[2].set_title(f"Derivative of Intensity: {len(cells)} Cells")
axes[2].set_xlabel("Frames")
axes[2].set_ylabel("ΔIntensity")

# Without background
axes[1].set_title(f"Intensity - Background: {len(cells)} Cells")
axes[1].set_xlabel("Frames")
axes[1].set_ylabel("Intensity")
axes[1].set_ylim(-25, 225)

# Without background derivative
axes[3].set_title(f"Derivative of Intensity - Background: {len(cells)} Cells")
axes[3].set_xlabel("Frames")
axes[3].set_ylabel("ΔIntensity")

for idx, cell in enumerate(cells):
    if cell == 'Cell_0':
        pixels = np.load(f"{data_path}/{cell}/pixels.npy")
        axes[0].plot(pixels, linewidth=2, color='red', zorder=6, label='background')
        axes[2].plot(np.gradient(pixels), linewidth=2, color='red', zorder=6, label='background')
    else:
        pixels = np.load(f"{data_path}/{cell}/pixels.npy")
        pixels_no_bground = np.load(f"{data_path}/{cell}/pixels_no_bground.npy")
        color = cmap(idx / len(cells))
        axes[0].plot(pixels, linewidth=1, alpha=0.5, color=color)
        axes[2].plot(np.gradient(pixels), linewidth=1, alpha=0.5, color=color)
        axes[1].plot(pixels_no_bground, linewidth=1, alpha=0.5, color=color)
        axes[3].plot(np.gradient(pixels_no_bground), linewidth=1, alpha=0.5, color=color)

axes[0].legend()
axes[2].legend()
plt.tight_layout()
plt.savefig(f"{data_path}/luminosity_timeseries.png", dpi=300)
plt.show()