import os
import gc
import re
import sys
import numpy as np
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

def cellpose_pixels(cell_centers, segmentations, images, background):

    pixels = []
    pixels_no_bground = []

    for iteration, (center, bground) in enumerate(zip(cell_centers, background)):
        try:
            # Get the mask ID at the center coordinate for this frame
            center_x, center_y = map(int, center)
            mask_id = segmentations[iteration, center_y, center_x]
            
            # Create binary mask for this frame
            mask = (segmentations[iteration] == mask_id)
            
            # Extract pixel values using masked array
            frame_pixels = np.ma.masked_array(images[iteration], mask=~mask)
            
            # Apply threshold (brightest 10%)
            # thresh = np.percentile(frame_pixels.compressed(), 90)
            # frame_pixels = frame_pixels[frame_pixels <= thresh]
            
            # Compute mean luminosity
            mean_luminosity = np.mean(frame_pixels)
            
            # Subtract background
            remove_background = mean_luminosity - bground
            if remove_background < 0:
                remove_background = 0
                
            pixels.append(mean_luminosity)
            pixels_no_bground.append(remove_background)
            
        except IndexError:
            print(f"Frame {iteration} does not exist.")
            pixels.append(0)
            pixels_no_bground.append(0)

    return pixels, pixels_no_bground

def DOUG(cell_centers, segmentations, images, save_path, background):
    pixels, pixels_no_bground = cellpose_pixels(cell_centers, segmentations, images, background)
    np.save(f"{save_path}/pixels.npy", pixels)
    np.save(f"{save_path}/pixels_no_bground.npy", pixels_no_bground)

def process_movie():
    global_path = "Patrick_temp_dir_v4"
    exp_path = "/mnt/data/pc3_naoh_channel1_30JAN25"
    directory_path = f"{global_path}/processed_cells"

    image_files = sorted([f'{exp_path}/{f}' for f in os.listdir(exp_path) if f.endswith('.png')])
    images = load_images(image_files) / 4095 # Normalize between 0 and 1

    cellpose_files = sorted([f'{exp_path}/{f}' for f in os.listdir(exp_path) if f.endswith('.npy')])
    segmentations = load_segmentations(cellpose_files)

    try:
        background = [0 for _ in range(len(cellpose_files))]
        all_cells = [f for f in os.listdir(directory_path)]
        for cell in tqdm(all_cells, desc="Processing cells"):
            cell_num = cell.split("_")[0]
            if cell_num != "Cell0":
                cell_centers = np.load(f"{directory_path}/{cell}", allow_pickle=True)['cell_mask_struct']
                save_path = f"{global_path}/extracted_cells/{cell_num}"
                os.makedirs(save_path, exist_ok=True)

                DOUG(cell_centers, segmentations, images, save_path, background)
                gc.collect()

    except FileNotFoundError:
        print(f"No background found!")
        return

if __name__ == "__main__":
    process_movie()