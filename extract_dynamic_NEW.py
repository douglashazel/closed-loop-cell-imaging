import os
import re
import gc
import sys
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.measure import find_contours

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

def get_center(mask):
    contours = find_contours(mask, level=0.5)

    all_y, all_x = [], []
    for contour in contours:
        all_y.extend(contour[:, 0])
        all_x.extend(contour[:, 1])
    
    min_y, max_y = int(min(all_y)), int(max(all_y))
    min_x, max_x = int(min(all_x)), int(max(all_x))
    
    center_y = (min_y + max_y) // 2
    center_x = (min_x + max_x) // 2
    return (center_x, center_y)

def cellpose_pixels(mask, bground, image_files, exp_path):
    pixels = []
    pixels_no_bground = []
    indices = np.where(mask)
    row_indices, column_indices = indices
    for frame in range(len(image_files)):
        try:
            desired_frames = get_movie_frames(frame, frame+1, exp_path)
            this_frame = [desired_frames[0][row_indices[index]][column_indices[index]] for index in range(len(column_indices))]
            pixels.append(np.mean(this_frame))
            pixels_no_bground.append(np.mean(this_frame) - np.mean(bground[frame]))
        except IndexError:
            print(f"Frame {frame} does not exist.")
    return pixels, pixels_no_bground

def DOUG(mask, bground, image_files, save_path, exp_path):
    pixels, pixels_no_bground = cellpose_pixels(mask, bground, image_files, exp_path)
    luminosity_vals = [(val - min(pixels)) / min(pixels) for val in pixels]
    os.makedirs(save_path, exist_ok=True)
    np.save(f'{save_path}/pixels.npy', pixels)
    np.save(f'{save_path}/luminosity_vals.npy', luminosity_vals)

    np.save(f'{save_path}/pixels_no_bground.npy', pixels_no_bground)

    cell_center = get_center(mask)
    np.save(f'{save_path}/cell_center_xy.npy', cell_center)

def process(part, partition, exp_dir):
    exp_path = os.path.abspath(exp_dir)
    bground = np.load(f"analysis_results/Cell_0/pixels.npy", allow_pickle=True)

    image_files = sorted([f'{exp_path}/{f}' for f in os.listdir(exp_path) if f.endswith('.png')])

    cellpose_files = sorted([f'{exp_path}/{f}' for f in os.listdir(exp_path) if f.endswith('.npy')])
    segmentation_choice = np.load(cellpose_files[2], allow_pickle=True).item()['masks']
    num_cells = np.max(segmentation_choice)
    cell_length = num_cells // partition
    start = cell_length * part
    if start == 0: # don't process background again
        start = 1
    end = cell_length * (part + 1)
    if end > num_cells:
        end = num_cells
    for cell_num in tqdm(range(start, end), position=part, desc=f'Part {part}'):
        try:
            mask = (segmentation_choice == cell_num)
            # save_path = os.path.join(exp_path, f"analysis_results_v2/Cell_{cell_num}")
            save_path = f"analysis_results/Cell_{cell_num}"
            DOUG(mask, bground, image_files, save_path, exp_path)
        except IndexError:
            print(f"Failed to extract data for Cell{cell_num}")
            continue
        finally:
            del mask
            gc.collect()

def spawn(partitions, exp_dir):
    for part in range(partitions):
        command = f"python3 {sys.argv[0]} {part} {partitions} {exp_dir}"
        subprocess.Popen(command, shell=True)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Usage: python extract_dynamic.py [partitions] [exp_dir]")
    elif len(args) == 1:
        partitions = int(args[0])
        exp_dir = input("Enter the experiment directory: ").strip()
        spawn(partitions, exp_dir)
    elif len(args) == 2:
        part, partitions = map(int, args[:2])
        exp_dir = input("Enter the experiment directory: ").strip()
        process(part, partitions, exp_dir)
    elif len(args) == 3:
        part, partitions, exp_dir = args
        process(int(part), int(partitions), exp_dir)
