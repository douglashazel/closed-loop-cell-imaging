import os
import gc
import re
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)

def extract_number(filename):
    match = re.search(r'(\d{3})', filename)
    return int(match.group(1)) if match else -1

def get_latest_frame(directory, ext):
    files = sorted([f for f in os.listdir(directory) if f.endswith(ext)], key=extract_number)
    return files[2] if files else None

def extract_frame_id(filename):
    return extract_number(filename)

def load_image(image_path):
    return np.array(Image.open(image_path)) / 4095.0  # Normalize

def load_segmentation(seg_path):
    return np.load(seg_path, allow_pickle=True).item()['masks']

def compute_luminosity(x, y, segmentation, image):
    try:
        x, y = int(x), int(y)
        mask_id = segmentation[y, x]
        if mask_id == 0:
            return np.nan  # No cell
        mask = (segmentation == mask_id)
        pixel_values = np.ma.masked_array(image, mask=~mask)
        return float(np.mean(pixel_values))
    except Exception:
        return np.nan

def update_luminosity_csv(traj_df, frame_id, image, segmentation, save_path):
    lum_path = os.path.join(save_path, "luminosity.csv")
    existing = os.path.exists(lum_path)

    if existing:
        lum_df = pd.read_csv(lum_path)
    else:
        lum_df = pd.DataFrame(columns=["CellID"])
    
    new_col = f"f{frame_id}"
    if new_col in lum_df.columns:
        print(f"Frame {frame_id} already processed in luminosity.csv.")
        return

    new_data = []
    for _, row in tqdm(traj_df.iterrows(), desc='Updating luminosity.csv'):
        cell_id = row['CellID']
        x_col = f"x{frame_id}"
        y_col = f"y{frame_id}"

        if x_col not in row or y_col not in row:
            continue
        x, y = row[x_col], row[y_col]
        if pd.isna(x) or pd.isna(y):
            luminosity = np.nan
        else:
            luminosity = compute_luminosity(x, y, segmentation, image)

        new_data.append((cell_id, luminosity))

    new_frame_df = pd.DataFrame(new_data, columns=["CellID", new_col])

    lum_df = pd.merge(lum_df, new_frame_df, on="CellID", how="outer")
    lum_df.sort_values("CellID", key=lambda x: x.astype(int), inplace=True)
    lum_df.to_csv(lum_path, index=False)
    print(f"Updated luminosity.csv with frame {frame_id}")

def main():
    exp_path = "frames"
    save_path = "analysis"

    image_file = get_latest_frame(exp_path, ".png")
    seg_file = get_latest_frame(exp_path, ".npy")

    if not image_file or not seg_file:
        print("Missing new image or segmentation.")
        return

    frame_id = extract_frame_id(image_file)
    print(f"Processing frame {frame_id}...")

    image = load_image(os.path.join(exp_path, image_file))
    segmentation = load_segmentation(os.path.join(exp_path, seg_file))

    traj_path = os.path.join(save_path, "trajectories.csv")
    if not os.path.exists(traj_path):
        print("Missing trajectories.csv.")
        return

    traj_df = pd.read_csv(traj_path)
    update_luminosity_csv(traj_df, frame_id, image, segmentation, save_path)
    gc.collect()

if __name__ == "__main__":
    main()