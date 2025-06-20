import os
import gc
import re
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from scipy.spatial import distance

def run_all(curr_centers, next_centers, cell_num: int, max_distance: float = 40) -> tuple[bool, int | None]:
    distances = distance.cdist(curr_centers, next_centers, "euclidean")
    this_distance = np.min(distances[cell_num])
    status = this_distance <= max_distance
    cell_match = np.argmin(distances[cell_num]) if status else None
    return status, cell_match

def extract_number(filename: str) -> int:
    match = re.search(r'(\d{3})', filename)
    return int(match.group(1)) if match else -1

def process(part: int, partition: int, directory_path: str, save_path: str) -> None:
    # Load and sort Cellpose files
    cellpose_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.npy')], key=extract_number)
    center_files = sorted([f for f in os.listdir(f"{save_path}/cellpose_centers") if f.endswith('.npy')])
    initial_segmentation = np.load(f"{directory_path}/{cellpose_files[0]}", allow_pickle=True).item()['masks']
    num_ROI = np.max(initial_segmentation)

    # Partition ROIs for parallel processing
    ROI_length = num_ROI // partition
    start = ROI_length * part
    end = min(ROI_length * (part + 1), num_ROI)

    max_gap = 6

    for cell_num in tqdm(range(start, end)):
        try:
            matrix_name = f"Cell{cell_num}_centers.npz"
            matrix_path = f"{save_path}/processed_cells/{matrix_name}"
            if os.path.exists(matrix_path):
                continue
            cell_mask_struct = []
            current_cell_num = cell_num
            frame_idx = 0
            last_known_center = None
            while frame_idx < len(center_files):
                curr_centers = np.load(f"{save_path}/cellpose_centers/{center_files[frame_idx]}", allow_pickle=True)

                if frame_idx == 0:
                    if current_cell_num >= len(curr_centers):
                        print(f"Skipping cell_num={current_cell_num}, curr_centers={len(curr_centers)}, frame={center_files[frame_idx]}")
                        break
                    last_known_center = curr_centers[current_cell_num]
                    cell_mask_struct.append(last_known_center)

                if frame_idx + 1 < len(center_files):
                    # Try matching to the next frame
                    next_centers = np.load(f"{save_path}/cellpose_centers/{center_files[frame_idx + 1]}", allow_pickle=True)
                    status, cell_match = run_all(curr_centers, next_centers, current_cell_num)
                    if status and cell_match is not None:
                        current_cell_num = cell_match
                        last_known_center = next_centers[cell_match]
                        cell_mask_struct.append(last_known_center)
                        frame_idx += 1
                        continue

                    # Gap bridging: try matching to frames up to max_gap ahead
                    matched = False
                    for gap in range(1, max_gap + 1):
                        next_idx = frame_idx + gap
                        if next_idx >= len(center_files):
                            break
                        next_centers = np.load(f"{save_path}/cellpose_centers/{center_files[next_idx]}", allow_pickle=True)
                        status, cell_match = run_all(curr_centers, next_centers, current_cell_num)
                        if status and cell_match is not None:
                            # Fill in all intermediate frames with last_known_center
                            for _ in range(1, gap):  # Only fill intermediate frames
                                cell_mask_struct.append(last_known_center)
                            # Append newly matched center
                            last_known_center = next_centers[cell_match]
                            cell_mask_struct.append(last_known_center)
                            current_cell_num = cell_match
                            frame_idx = next_idx  # Skip to the matched frame
                            matched = True
                            break
                    if not matched:
                        # No match found: fill the next frame and stop tracking
                        if frame_idx + 1 < len(center_files):
                            cell_mask_struct.append(last_known_center)
                        break  # Stop tracking
                else:
                    break  # No more frames to process

            os.makedirs(f"{save_path}/processed_cells", exist_ok=True)
            np.savez_compressed(matrix_path, cell_mask_struct=np.array(cell_mask_struct, dtype=np.int32))
        except Exception as e:
            print(f"Error processing cell_num={cell_num}, frame={center_files[frame_idx]}: {str(e)}")
            os.makedirs(f"{save_path}/processed_cells_catch", exist_ok=True)
            np.savez_compressed(f"{save_path}/processed_cells_catch/{matrix_name}", cell_mask_struct=np.array(cell_mask_struct, dtype=np.int32))
        del current_cell_num, cell_mask_struct
        gc.collect()

def spawn(partition, directory_path, save_path) -> None:
    with Pool(processes=partition) as pool:
        pool.starmap(process, [(ii, partition, directory_path, save_path) for ii in range(partition)])

def process_frame(args: tuple[str, str, str]) -> None:
    """Process a single frame to compute cell centers."""
    file, directory_path, save_path = args
    try:
        frame_centers = []
        curr_segmentation = np.load(f"{directory_path}/{file}", allow_pickle=True).item()['masks']
        num_masks = np.max(curr_segmentation)
        for cellID in tqdm(range(1, num_masks + 1)):
            mask = curr_segmentation == cellID
            y, x = np.where(mask)
            if len(x) > 0 and len(y) > 0:
                frame_centers.append((int(np.mean(x)), int(np.mean(y))))
        os.makedirs(f"{save_path}/cellpose_centers", exist_ok=True)
        np.save(f"{save_path}/cellpose_centers/{file.split('.')[0]}_centers.npy", frame_centers)
    except Exception as e:
        print(f"Error processing frame {file}: {str(e)}")
    finally:
        gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Track cells across frames in segmentation data.")
    parser.add_argument('part', type=int, help='Index of the part to process, or -1 to spawn all parts')
    parser.add_argument('partition', type=int, help='Total number of partitions for parallel processing')
    parser.add_argument('--directory_path', type=str, required=True, help='Path to the input directory containing .npy files')
    parser.add_argument('--save_path', type=str, required=True, help='Path to the directory for saving output files')
    parser.add_argument('--get_centers', action='store_true', help='If set, process cell centers for each frame')
    args = parser.parse_args()

    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    if args.get_centers:
        cellpose_files = sorted([f for f in os.listdir(args.directory_path) if f.endswith('.npy')], key=extract_number)
        num_processes = max(1, min(args.partition, len(cellpose_files)))

        print(f"Processing {len(cellpose_files)} frames with {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            pool.map(process_frame, [(file, args.directory_path, args.save_path) for file in cellpose_files])

    if args.part == -1:
        spawn(args.partition, args.directory_path, args.save_path)
    else:
        process(args.part, args.partition, args.directory_path, args.save_path)

# python3 dynamic_infiltrate_v4.py -1 10 --directory_path "/mnt/data/pc3_naoh_channel1_30JAN25" --save_path "/mnt/exDisk1/douglashazel/DHcode/Clofilium_Pipeline/Patrick_temp_dir_v4" --get_centers

# python3 dynamic_infiltrate_v4.py -1 5 --directory_path "/mnt/exDisk1/CalciumSignaling/Colo_GCAMP/Clofilium_ctl_1.2024-03-26-22-42-25" --save_path "/mnt/exDisk1/douglashazel/DHcode/Clofilium_Pipeline/Clo_ctl1_temp_masks" --get_centers