import numpy as np
from tqdm import tqdm
import sys
import os
import re
import gc
import subprocess
import pickle
import gzip
import argparse

np.set_printoptions(threshold=sys.maxsize)

def binarize(array, cell_num):
    ''' Efficiently binarizes the data '''
    return array > 0.5 if cell_num == 0.5 else array == cell_num


def compare_all(binarize_mask_frame1, raw_frame2):
    ''' Compares ROIs across frames to identify matching cells '''
    array_3d = (np.arange(raw_frame2.max() + 1)[:, np.newaxis, np.newaxis] == raw_frame2).astype(bool)
    expanded_2d = np.expand_dims(binarize_mask_frame1, axis=0)
    matches_per_slice = np.sum(array_3d == expanded_2d, axis=(1, 2))
    max_matches_index = np.argmax(matches_per_slice)
    return array_3d[max_matches_index, :, :] > 0


def run_all(file2, global_matrix, cell_num, directory_path):
    ''' Processes all ROIs in the experiment '''
    file_dict2 = np.load(os.path.join(directory_path, file2), allow_pickle=True).item()
    raw_cell_masks2 = file_dict2['masks']
    del file_dict2

    frame2_mask = compare_all(global_matrix[f"Cell{cell_num}"][1][-1], raw_cell_masks2)
    key = f'Cell{cell_num}'
    cell_key = re.search(r'(\d{3})', file2).group()
    total_pixels = np.sum(frame2_mask)

    if key in global_matrix:
        if cell_key not in global_matrix[key][0] and total_pixels >= 600: # number is threshold size in pixels, lower not tracked across images
            global_matrix[key][0].append(cell_key)
            global_matrix[key][1].append(frame2_mask)
    else:
        global_matrix[key] = [[cell_key], [frame2_mask]]

    del raw_cell_masks2, cell_key, key, frame2_mask, total_pixels
    gc.collect()


def extract_number(filename):
    ''' Correctly orders the frames (chronologically) '''
    match = re.search(r'(\d{3})', filename)
    return int(match.group(1)) if match else -1


def process(part, partition, directory_path, save_path):
    ''' Parent function handling subprocesses '''
    full_movie = ''
    movie = full_movie.split(".")[0]

    compressed_dir = os.path.join(save_path, f"{movie}static_compressed")
    os.makedirs(compressed_dir, exist_ok=True)

    file_names_list = sorted([f for f in os.listdir(directory_path) if f.endswith('.npy')], key=extract_number)

    for file1 in file_names_list:
        if "mask_000" in file1:
            file_dict1 = np.load(os.path.join(directory_path, file1), allow_pickle=True).item()
            raw_cell_masks1 = file_dict1['masks']
            num_ROI = np.max(raw_cell_masks1)
            del file_dict1, raw_cell_masks1

    ROI_length = num_ROI // partition
    start = ROI_length * part
    end = ROI_length * (part + 1) if ROI_length * (part + 1) <= num_ROI else num_ROI

    for cell_num in range(start, end):
        this_matrix = {}
        for file1 in file_names_list:
            if "mask_000" in file1:
                file_dict1 = np.load(os.path.join(directory_path, file1), allow_pickle=True).item()
                raw_cell_masks1 = file_dict1['masks']
                frame1_mask = binarize(raw_cell_masks1, cell_num)
                key = f'Cell{cell_num}'
                if key not in this_matrix:
                    this_matrix[key] = [[re.search(r'(\d{3})', file1).group()], [frame1_mask]]
                del frame1_mask, raw_cell_masks1

        try:
            for ii, file1 in enumerate(tqdm(file_names_list, desc='Files:')):
                if ii + 1 < len(file_names_list):
                    file2 = file_names_list[ii + 1]
                    run_all(file2, this_matrix, cell_num, directory_path)

            with gzip.open(os.path.join(compressed_dir, f"Cell{cell_num}_matrix.npy.gz"), 'wb') as f:
                pickle.dump(this_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
            del this_matrix

        except IndexError:
            with gzip.open(os.path.join(compressed_dir, f"Cell{cell_num}_matrix.npy.gz"), 'wb') as f:
                pickle.dump(this_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
            del this_matrix


def spawn(args=None):
    ''' Spawns multiple processes '''
    args = sys.argv[1:]
    directory_path = args[2]
    save_path = args[3]
    part = int(args[0])
    partition = int(args[1])

    for ii in range(0, part):
        command = f"python3 infiltrate_dynamic.py {ii} {partition} {directory_path} {save_path}"
        subprocess.Popen(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process cell segmentation data.")
    parser.add_argument("part", type=int, help="Part of the data to process.")
    parser.add_argument("partition", type=int, help="Total number of partitions.")
    parser.add_argument("--directory_path", type=str, required=True, help="Path to the input directory.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to the save directory.")
    args = parser.parse_args()

    if args.part == -1:
        spawn([args.part, args.partition, args.directory_path, args.save_path])
    else:
        process(args.part, args.partition, args.directory_path, args.save_path)
