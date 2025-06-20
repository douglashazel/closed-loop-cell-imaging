import os
import gc
import re
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)

def extract_number(filename: str) -> int:
    match = re.search(r'(\d{3})', filename)
    return int(match.group(1)) if match else -1

def load_images(filenames):
        images = []
        for filename in filenames:
            images.append(np.array(Image.open(filename)))
        return np.stack(images)

def sort_key(path):
        # The regular expression pattern for the frame number
        pattern = re.compile(r'R_p(\d+)')
        
        # Find the frame number in the path
        match = pattern.search(path)
        if match:
            # Convert the frame number to an integer and return it
            return int(match.group(1))
        else:
            # No frame number found, return infinity to put this path at the end
            return float('inf')

def get_movie_frame_filenames(exp_path, suffix='.png'):
        all_files = os.listdir(exp_path)
        filenames = [f'{exp_path}/{file}' for file in all_files if suffix in file]
        filenames.sort(key=sort_key) # Remove the lexicographic sort!
        return filenames

def get_movie_frames(start_frame, end_frame, exp_path):
        filenames = get_movie_frame_filenames(exp_path)
        # Set movie frames as well as their dimensions (height and width)
        movie_frames = load_images(filenames[start_frame:end_frame])
        return movie_frames

def cellpose_pixels(cell_centers, cellpose_files, exp_path, background):
    pixels = []
    pixels_no_bground = []

    for iteration, (center, cellpose_file, bground) in tqdm(enumerate(zip(cell_centers, cellpose_files, background))):
        segmentations = np.load(f"{exp_path}/{cellpose_file}", allow_pickle=True).item()['masks']
        
        center_x, center_y = center
        mask_id = segmentations[center_y, center_x]
        mask = (segmentations == mask_id).astype(np.uint8)
        indices = np.where(mask)
        row_indices, column_indices = indices

        try:
            desired_frames = get_movie_frames(
                start_frame=iteration,
                end_frame=iteration+1,
                exp_path=exp_path)

            this_frame = []
            for index, val in enumerate(column_indices):
                val = desired_frames[0][row_indices[index]][column_indices[index]]
                norm_val = val / 4095
                this_frame.append(norm_val)
            
            thresh = np.percentile(this_frame, 90)
            this_frame = [val for val in this_frame if val <= thresh]
            remove_background = np.mean(this_frame) - bground
            if remove_background < 0:
                remove_background = 0
            
            pixels.append(np.mean(this_frame))
            pixels_no_bground.append(remove_background)
            del this_frame, val, norm_val, desired_frames

        except IndexError:
            print(f"Frame {iteration} does not exist.")

    return pixels, pixels_no_bground

def DOUG(cell_centers, cellpose_files, save_path, exp_path, background):
    pixels, pixels_no_bground = cellpose_pixels(cell_centers, cellpose_files, exp_path, background)
    np.save(f"{save_path}/pixels.npy", pixels)
    np.save(f"{save_path}/pixels_no_bground.npy", pixels_no_bground)

######################################################################

def process_movie():
    global_path = "Patrick_temp_dir_v4"
    exp_path = "/mnt/data/pc3_naoh_channel1_30JAN25"
    directory_path = f"{global_path}/processed_cells"
    cellpose_files = sorted([f for f in os.listdir(exp_path) if f.endswith('.npy')], key=extract_number)

    try:
        # background = np.load(f"{global_path}/bground_luminosity.npy", allow_pickle=True)
        background = [0 for _ in range(len(cellpose_files))]

        all_cells = [f for f in os.listdir(directory_path)]
        for cell in tqdm(all_cells):
            cell_num = cell.split("_")[0]
            if cell_num != "Cell0": # don't process bground again
                cell_centers = np.load(f"{directory_path}/{cell}", allow_pickle=True)['cell_mask_struct']
                save_path = f"{global_path}/extracted_cells/{cell_num}"
                os.makedirs(save_path, exist_ok=True)

                DOUG(cell_centers, cellpose_files, save_path, exp_path, background)
                gc.collect()

    except FileNotFoundError:
        print(f"No background found!")
        return

if __name__ == "__main__":
    process_movie()