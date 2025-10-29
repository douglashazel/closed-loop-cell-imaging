import os
import re
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from collections import defaultdict

def extract_number(filename):
    match = re.search(r'timepoint_(\d+)', filename)
    return int(match.group(1)) if match else -1

# Load cells of interest from text file
def load_cells_of_interest(txt_path):
    cells = defaultdict(list)
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            movie, cell = line.split("/")
            cells[movie].append(cell)
    return dict(cells)

# ----- CHANGE HERE ----- #
exp = "pc3_carbachol_1"
FIXED_CROP_SIZE = 10000
save_path = "gifs"

tif_path = f"{exp}/frames"
mask_dir = f"{exp}/masks"
save_cell_path = f"{exp}/analysis"

traj_path = f"{save_cell_path}/trajectories.csv"
# cells_of_interest = load_cells_of_interest(f"{save_cell_path}/selected_cells.txt")
cells_of_interest = {exp: ['Cell17']}

for movie, cell_list in cells_of_interest.items():

    if not os.path.exists(traj_path):
        print(f"Missing trajectories file: {traj_path}")
        continue
    if not os.path.exists(tif_path):
        print(f"Missing movie frames: {tif_path}")
        continue

    png_files = sorted([f for f in os.listdir(tif_path) if f.endswith('.png')], key=extract_number)
    df = pd.read_csv(traj_path)

    for cell_name in cell_list:
        cell_id = int(cell_name.split("Cell")[-1])
        if cell_id not in df["CellID"].values:
            print(f"{cell_id} not found in {traj_path}")
            continue

        output_dir = f"{save_path}/temp_images_{cell_id}"
        os.makedirs(output_dir, exist_ok=True)
        image_files = []

        cell_df = df[df["CellID"] == cell_id]

        # pick fixed anchor location = first valid coordinate
        coords = cell_df.iloc[0, 1:].values
        anchor_x, anchor_y = None, None
        for idx in range(0, len(coords), 2):
            if not np.isnan(coords[idx]) and not np.isnan(coords[idx+1]):
                anchor_x = int(coords[idx])
                anchor_y = int(coords[idx+1])
                break
        if anchor_x is None or anchor_y is None:
            print(f"No valid coordinates for Cell {cell_id}")
            shutil.rmtree(output_dir)
            continue

        for frame_idx, png_file in tqdm(enumerate(png_files)):
            tif_image = plt.imread(os.path.join(tif_path, png_file))
            base_name = os.path.splitext(png_file)[0]
            mask_path = os.path.join(mask_dir, base_name + ".npy")

            contours = []
            if os.path.exists(mask_path):
                mask_all = np.load(mask_path, allow_pickle=True)
                if 0 <= anchor_y < mask_all.shape[0] and 0 <= anchor_x < mask_all.shape[1]:
                    mask_id = mask_all[anchor_y, anchor_x]
                    if mask_id > 0:
                        mask = (mask_all == mask_id).astype(np.uint8)
                        contours = find_contours(mask, level=0.5)

            # fixed crop around anchor
            half_width = FIXED_CROP_SIZE // 2
            img_height, img_width = tif_image.shape[:2]
            crop_min_y = max(0, anchor_y - half_width)
            crop_max_y = min(img_height, anchor_y + half_width)
            crop_min_x = max(0, anchor_x - half_width)
            crop_max_x = min(img_width, anchor_x + half_width)

            cropped_image = tif_image[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

            # pad if needed
            if cropped_image.shape[0] < FIXED_CROP_SIZE or cropped_image.shape[1] < FIXED_CROP_SIZE:
                padded_image = np.zeros((FIXED_CROP_SIZE, FIXED_CROP_SIZE), dtype=cropped_image.dtype)
                y_offset = (FIXED_CROP_SIZE - cropped_image.shape[0]) // 2
                x_offset = (FIXED_CROP_SIZE - cropped_image.shape[1]) // 2
                padded_image[y_offset:y_offset + cropped_image.shape[0],
                             x_offset:x_offset + cropped_image.shape[1]] = cropped_image
                cropped_image = padded_image

            # draw
            plt.figure(dpi=300)
            plt.title(f"Frame{frame_idx}")
            plt.imshow(cropped_image, cmap='gray', alpha=1)
            for contour in contours:
                cropped_contour_y = contour[:, 0] - crop_min_y
                cropped_contour_x = contour[:, 1] - crop_min_x
                plt.plot(cropped_contour_x, cropped_contour_y, color='red', linewidth=.5)

            plt.axis('off')
            output_file = f"{output_dir}/{movie}_{cell_id}_frame_{frame_idx:03d}.png"
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
            plt.close()
            image_files.append(output_file)

        gif_path = f"{save_path}/{movie}_{cell_id}.gif"
        with imageio.get_writer(gif_path, mode='I', duration=1, loop=0) as writer:
            for image_file in image_files:
                image = imageio.imread(image_file)
                writer.append_data(image)

        print(f"GIF saved at: {gif_path}")
        shutil.rmtree(output_dir)