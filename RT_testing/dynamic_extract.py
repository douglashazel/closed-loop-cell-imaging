import os
import gc
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)

def load_images(filenames):
    images = []
    for filename in tqdm(filenames, desc="Loading images"):
        images.append(np.array(Image.open(filename)))
    return np.stack(images)

def load_segmentations(filenames):
    segmentations = []
    for filename in tqdm(filenames, desc="Loading segmentations"):
        seg = np.load(filename, allow_pickle=True).item()['masks']
        segmentations.append(seg)
    return np.stack(segmentations)

def get_movie_frame_filenames(exp_path, suffix='.png'):
    all_files = os.listdir(exp_path)
    filenames = sorted([f'{exp_path}/{file}' for file in all_files if suffix in file])
    return filenames

def cellpose_pixels(cell_data, segmentations, images, background):
    pixels = []
    pixels_no_bground = []
    
    columns = list(cell_data.columns)[1:] # skip first column (CellID)
    xy_pairs = [(columns[i], columns[i+1]) for i in range(0, len(columns), 2)]

    for iteration, ((x_col, y_col), bground) in enumerate(zip(xy_pairs, background)):
        try:

            center_x = cell_data[x_col]
            center_y = cell_data[y_col]
            
            center_x, center_y = int(center_x), int(center_y)
            mask_id = segmentations[iteration, center_y, center_x]
            mask = (segmentations[iteration] == mask_id)
            frame_pixels = np.ma.masked_array(images[iteration], mask=~mask)
            mean_luminosity = np.mean(frame_pixels)

            remove_background = mean_luminosity - bground
            if remove_background < 0:
                remove_background = 0

            pixels.append(mean_luminosity)
            pixels_no_bground.append(remove_background)

        except Exception as e:
            pixels.append(0)
            pixels_no_bground.append(0)

    return pixels, pixels_no_bground

def DOUG(cell_data, segmentations, images, save_path, background):
    pixels, pixels_no_bground = cellpose_pixels(cell_data, segmentations, images, background)
    np.save(f"{save_path}/pixels.npy", pixels)
    np.save(f"{save_path}/pixels_no_bground.npy", pixels_no_bground)

def process_movie():
    global_path = "analysis"
    exp_path = "frames"

    image_files = sorted([f'{exp_path}/{f}' for f in os.listdir(exp_path) if f.endswith('.png')])
    images = load_images(image_files) / 4095  # Normalize between 0 and 1

    cellpose_files = sorted([f'{exp_path}/{f}' for f in os.listdir(exp_path) if f.endswith('.npy')])
    segmentations = load_segmentations(cellpose_files)[:2]

    background = [0 for _ in range(len(segmentations))]

    traj_df = pd.read_csv(f"{global_path}/trajectories.csv")
    all_cell_ids = traj_df['CellID'].unique()

    for cell_id in tqdm(all_cell_ids, desc="Processing cells"):
        cell_data = traj_df[traj_df['CellID'] == cell_id]

        save_path = f"{global_path}/extracted_cells/Cell{cell_id}"
        os.makedirs(save_path, exist_ok=True)

        DOUG(cell_data, segmentations, images, save_path, background)
        gc.collect()

if __name__ == "__main__":
    process_movie()