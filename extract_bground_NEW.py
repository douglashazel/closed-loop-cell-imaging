import os
import re
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)

def load_images(filenames):
    images = [np.array(Image.open(filename)) for filename in filenames]
    return np.stack(images)

def sort_key(path):
    pattern = re.compile(r'R_p(\d+)')
    match = pattern.search(path)
    return int(match.group(1)) if match else float('inf')

def get_movie_frame_filenames(exp_path):
    all_files = os.listdir(exp_path)
    filenames = [f'{exp_path}/{file}' for file in all_files if '.png' in file]
    filenames.sort(key=sort_key)
    return filenames

def get_movie_frames(start_frame, end_frame, exp_path):
    filenames = get_movie_frame_filenames(exp_path)
    return load_images(filenames[start_frame:end_frame])

def cellpose_pixels(mask, image_files, exp_path):
    pixels = []
    indices = np.where(mask)
    row_indices, column_indices = indices
    for frame in tqdm(range(len(image_files))):
        try:
            desired_frames = get_movie_frames(frame, frame+1, exp_path)
            this_frame = [desired_frames[0][row_indices[index]][column_indices[index]] for index in range(len(column_indices))]
            pixels.append(np.mean(this_frame))
        except IndexError:
            print(f"Frame {frame} does not exist.")
    return pixels

def DOUG(mask, image_files, save_path, exp_path):
    pixels = cellpose_pixels(mask, image_files, exp_path)
    luminosity_vals = [(val - min(pixels)) / min(pixels) for val in pixels]
    os.makedirs(save_path, exist_ok=True)
    np.save(f'{save_path}/pixels.npy', pixels)
    np.save(f'{save_path}/luminosity_vals.npy', luminosity_vals)

def process(exp_dir):
    exp_path = os.path.abspath(exp_dir)
    image_files = sorted([f'{exp_path}/{f}' for f in os.listdir(exp_path) if f.endswith('.png')])

    cellpose_files = sorted([f'{exp_path}/{f}' for f in os.listdir(exp_path) if f.endswith('.npy')])
    segmentation_choice = np.load(cellpose_files[2], allow_pickle=True).item()['masks']

    cell_num = 0 # background

    try:
        mask = (segmentation_choice == cell_num)
        # save_path = os.path.join(exp_path, f"analysis_results_v2/Cell_{cell_num}")
        save_path = f"analysis_results/Cell_{cell_num}"
        DOUG(mask, image_files, save_path, exp_path)
    except IndexError:
        print(f"Failed to extract data for background")

if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)
    if len(args) == 1:
        print("Usage: python extract_bground_NEW.py [exp_dir]")
        process(args[0])
