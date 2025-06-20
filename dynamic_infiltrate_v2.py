import os
import gc
import re
import sys
import gzip
import torch
import pickle
import argparse
import subprocess
import numpy as np
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)

def binarize(array, cell_num):
    ''' Efficiently binarizes the data '''
    return array > 0.5 if cell_num == 0.5 else array == cell_num

def compare_all(binarize_mask_frame1, raw_frame2, threshold=0.6):
    ''' Compares ROIs across frames to identify matching cells '''
    unique_values = np.unique(raw_frame2)
    match_ratios = [(np.sum(binarize_mask_frame1 == (raw_frame2 == val)) / binarize_mask_frame1.size, val) for val in unique_values]
    max_ratio, best_match = max(match_ratios, key=lambda x: x[0])
    if max_ratio < threshold:
        return np.zeros_like(raw_frame2, dtype=bool)
    return raw_frame2 == best_match

def run_all(file2, global_matrix, cell_num, directory_path, minimum_cell_size=600):
    ''' Processes all ROIs in the experiment '''
    file_dict2 = np.load(directory_path + '/' + file2, allow_pickle=True).item()
    raw_cell_masks2 = file_dict2['masks']
    del file_dict2
    frame2_mask = compare_all(global_matrix[f"Cell{cell_num}"][1][-1], raw_cell_masks2)

    key = f'Cell{cell_num}'
    cell_key = re.search(r'(\d{3})', file2).group()
    total_pixels = np.sum(frame2_mask)
    status = total_pixels >= minimum_cell_size

    if key in global_matrix:
        if cell_key not in global_matrix[key][0] and status:
            global_matrix[key][0].append(cell_key)
            global_matrix[key][1].append(frame2_mask)
    else:
        global_matrix[key] = [[cell_key], [frame2_mask]]

    del raw_cell_masks2, cell_key, key, frame2_mask, total_pixels
    gc.collect()

    return status

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

    file_dict1 = np.load(os.path.join(directory_path, file_names_list[0]), allow_pickle=True).item()
    raw_cell_masks1 = file_dict1['masks']
    num_ROI = np.max(raw_cell_masks1)
    del file_dict1, raw_cell_masks1

    ROI_length = num_ROI // partition
    start = ROI_length * part
    end = ROI_length * (part + 1)
    if end > num_ROI:
        end = num_ROI

    max_gap = 6

    for cell_num in range(start, end):
        this_matrix = {}

        try:
            status = False
            for ii, file1 in enumerate(tqdm(file_names_list, desc='Files:')):
                if ii == 0:
                    tqdm.write(f"Initialize {file1} for Cell{cell_num}")
                    file_dict1 = np.load(os.path.join(directory_path, file1), allow_pickle=True).item()
                    raw_cell_masks1 = file_dict1['masks']

                    frame1_mask = binarize(raw_cell_masks1, cell_num)

                    key = f'Cell{cell_num}'
                    if key not in this_matrix:
                        this_matrix[key] = [[re.search(r'(\d{3})', file1).group()], [frame1_mask]]
                    del frame1_mask, raw_cell_masks1

                    matrix_name = f'Cell{cell_num}_matrix.npy'

                for gap in range(max_gap):
                    if ii + 1 + gap < len(file_names_list):
                        file2 = file_names_list[ii + 1 + gap]
                        status = run_all(file2, this_matrix, cell_num, directory_path)
                    if status:
                        break

                if not status:
                    break

            os.makedirs(save_path, exist_ok=True)
            with gzip.open(f"{save_path}/{matrix_name}.gz", 'wb') as f:
                pickle.dump(this_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

            del this_matrix, file2

        except IndexError:
            print(f"Error for Cell{cell_num}")
            os.makedirs(save_path, exist_ok=True)
            with gzip.open(f"{save_path}/{matrix_name}.gz", 'wb') as f:
                pickle.dump(this_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

            del this_matrix, file2

def spawn(partition, directory_path, save_path):
    ''' Spawns multiple processes '''
    for ii in range(partition):
        command = [
            'python3',
            'dynamic_infiltrate_v2.py',
            str(ii),
            str(partition),
            '--directory_path', directory_path,
            '--save_path', save_path
        ]
        # print(f"Spawning process: {command}")
        subprocess.Popen(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', type=int, help='Index of the part to process, or -1 to spawn all parts')
    parser.add_argument('partition', type=int, help='Total number of partitions')
    parser.add_argument('--directory_path', type=str, required=True, help='Path to the input directory')
    parser.add_argument('--save_path', type=str, required=True, help='Path to the save directory')
    args = parser.parse_args()

    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)

    if args.part == -1:
        spawn(args.partition, args.directory_path, args.save_path)
    else:
        process(args.part, args.partition, args.directory_path, args.save_path)


########################
# python3 dynamic_infiltrate_v2.py -1 2 --directory_path "/mnt/data/pc3_naoh_channel1_30JAN25" --save_path "/mnt/exDisk1/douglashazel/DHcode/Clofilium_Pipeline/Patrick_temp_dir"