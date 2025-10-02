import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_number(filename):
    match = re.search(r'timepoint_(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_image(path):
    return np.array(Image.open(path)) / 4095.0

def main():
    """
    Compute per-cell luminosity normalized by background across frames.

    Steps:
    1. Load all frame images and corresponding segmentation masks.
    2. Compute the average background luminosity (mask == 0) per frame.
    3. Load the existing luminosity.csv (per-cell average intensities).
    4. Subtract the frame-specific background value from each cell's
       luminosity values.
    5. Drop cell0 (the background itself).
    6. Save the new normalized values as luminosity_no_bground.csv.
    7. Plot normalized traces for all cells using twilight_shifted colormap
       and save as average_luminosity_no_bground.png.
    """

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png','.jpg'))], key=extract_number)
    # compute background trace across frames
    bg_trace = []
    for fname in tqdm(images, desc="Background"):
        img = load_image(os.path.join(image_dir, fname))
        if img.ndim == 3:
            img = img.mean(axis=-1)
        mask = np.load(os.path.join(mask_dir, os.path.splitext(fname)[0] + ".npy"))
        bg_trace.append(img[mask == 0].mean())
    bg_trace = np.array(bg_trace)

    # load existing luminosity CSV
    lum_df = pd.read_csv(f"{analysis_dir}/luminosity.csv")

    # normalize each cell by subtracting per-frame background
    norm_df = lum_df.copy()
    for i, frame in enumerate(images):
        col = f"f{i}"
        if col in norm_df.columns:
            norm_df[col] = norm_df[col] - bg_trace[i]

    # drop cell0 row
    norm_df = norm_df[norm_df["CellID"] != 0]

    # save new CSV
    norm_csv_path = f"{analysis_dir}/luminosity_no_bground.csv"
    norm_df.to_csv(norm_csv_path, index=False)

    # plot normalized luminosities
    cmap = plt.get_cmap("twilight_shifted")
    colors = cmap(np.linspace(0,1,len(norm_df)))

    plt.figure(dpi=300)
    for (cell_id, row), color in zip(norm_df.iterrows(), colors):
        non_nan = row.dropna()
        frames = [int(str(x).lstrip("f")) for x in non_nan.index if x.startswith("f")]
        vals = non_nan.values[1:]  # skip CellID
        plt.plot(frames, vals, alpha=0.7, color=color)

    plt.xlabel("Frame")
    plt.ylabel("Average luminosity (cell - background)")
    plt.title("Cell luminosity over time")

    # save figure
    plot_dir = os.path.join(analysis_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    fig_path = os.path.join(plot_dir, "average_luminosity_no_bground.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

exp = "c2c12_carbachol_1" # modify experiment here!
image_dir = f"{exp}/frames"
mask_dir = f"{exp}/masks"
analysis_dir = f"{exp}/analysis"

main()