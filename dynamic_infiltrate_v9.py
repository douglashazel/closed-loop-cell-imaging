import os
import gc
import re
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from scipy.spatial import distance

def run_all(curr_center, next_centers, max_distance: float = 40) -> tuple[bool, int | None]:
    distances = distance.cdist(next_centers, [curr_center], metric="euclidean").flatten()
    status = np.min(distances) <= max_distance
    cell_match = np.argmin(distances) if status else None
    return status, cell_match

def extract_number(filename: str) -> int:
    match = re.search(r'(\d{3})', filename)
    return int(match.group(1)) if match else -1

def process(pair_idx: int, partition: int, save_path: str, center_files: list, total_frames: int, frame_pairs: list) -> None:
    """Process a pair of consecutive frames and save cell trajectories."""
    start_frame = frame_pairs[pair_idx][0]
    end_frame = frame_pairs[pair_idx][1]
    print(f"Partition {pair_idx}: frame pair {start_frame}-{end_frame}")
    
    if start_frame >= len(center_files) or end_frame >= len(center_files):
        print(f"Skipping pair {start_frame}-{end_frame}: start_frame={start_frame}, end_frame={end_frame}, len(center_files)={len(center_files)}")
        return

    temp_dir = f"{save_path}/frame{start_frame}-{end_frame}_dir"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        curr_centers = np.load(f"{save_path}/cellpose_centers/{center_files[start_frame]}", allow_pickle=True)
        next_centers = np.load(f"{save_path}/cellpose_centers/{center_files[end_frame]}", allow_pickle=True)

        cell_trajectories = {}
        for cell_num in tqdm(range(len(curr_centers)), desc="Tracking..."):
            trajectory = {ii: (0, 0) for ii in range(total_frames)}
            trajectory[start_frame] = curr_centers[cell_num]
            status, cell_match = run_all(curr_centers[cell_num], next_centers)
            if status and cell_match is not None:
                trajectory[end_frame] = next_centers[cell_match]
            cell_trajectories[f"Cell{cell_num}"] = trajectory

        for cell_id, trajectory in cell_trajectories.items():
            np.savez_compressed(f"{temp_dir}/{cell_id}.npz", trajectory=trajectory)

    except Exception as e:
        print(f"Error processing frame pair {start_frame}-{end_frame}: {str(e)}")
    finally:
        gc.collect()

def stitch_frame_pairs(partition: int, save_path: str, center_files: list, total_frames: int, frame_pairs: list) -> None:
    """Stitch trajectories from frame0-1_dir to frame2-3_dir into processed_cells."""
    output_dir = f"{save_path}/processed_cells"
    os.makedirs(output_dir, exist_ok=True)

    curr_dir = f"{save_path}/frame{frame_pairs[0][0]}-{frame_pairs[0][1]}_dir"
    next_dir = f"{save_path}/frame{frame_pairs[1][0]}-{frame_pairs[1][1]}_dir"
    if not (os.path.exists(curr_dir) and os.path.exists(next_dir)):
        print(f"Skipping stitching: {curr_dir} or {next_dir} does not exist")
        return

    curr_centers = [c for c in os.listdir(curr_dir)]
    next_centers = np.load(f"{save_path}/cellpose_centers/{center_files[2]}", allow_pickle=True)
    next_trajectories = [c for c in os.listdir(f"{save_path}/frame{frame_pairs[1][0]}-{frame_pairs[1][1]}_dir")]

    curr_trajectories = {}
    for file in curr_centers:
        if file.endswith('.npz'):
            cell_id = file.split('.')[0]
            curr_trajectories[cell_id] = np.load(f"{curr_dir}/{file}", allow_pickle=True)['trajectory'].item()

    used_cell_ids = set()
    for cell_num in tqdm(curr_trajectories.keys(), desc="Stitching..."):
        status, cell_match = run_all(curr_trajectories[cell_num][frame_pairs[0][1]], next_centers)
        if status and cell_match is not None:
            used_cell_ids.add(str(cell_match))
            next_cell = np.load(f"{save_path}/frame{frame_pairs[1][0]}-{frame_pairs[1][1]}_dir/Cell{cell_match}.npz", allow_pickle=True)['trajectory'].item()
            for frame in curr_trajectories[cell_num].keys():
                if np.all(np.array(next_cell[frame]) == 0):
                    continue
                curr_trajectories[cell_num][frame] = next_cell[frame]

    for curr_cell_id, curr_trajectory in curr_trajectories.items():
        np.savez_compressed(f"{output_dir}/{curr_cell_id}.npz", trajectory=curr_trajectory)

    # Get existing cell IDs in processed_cells
    existing_cell_ids = set()
    for file in os.listdir(output_dir):
        if file.endswith('.npz'):
            match = re.search(r'Cell(\d+)', file)
            if match:
                existing_cell_ids.add(int(match.group(1)))

    # Find the largest cell ID to start assigning new IDs
    max_cell_id = max(existing_cell_ids, default=-1)

    # Move unused trajectories from next_trajectories with new cell IDs
    for cell in next_trajectories:
        if not cell.endswith('.npz'):
            continue
        number = re.search(r'Cell(\d+)', cell)
        if not number:
            continue
        number = number.group(1)
        if number not in used_cell_ids:
            to_move = np.load(f"{save_path}/frame{frame_pairs[1][0]}-{frame_pairs[1][1]}_dir/{cell}", allow_pickle=True)['trajectory'].item()
            # Assign new cell ID
            max_cell_id += 1
            new_cell_id = f"Cell{max_cell_id}"
            np.savez_compressed(f"{output_dir}/{new_cell_id}.npz", trajectory=to_move)

def spawn(partition: int, save_path: str, center_files: list) -> None:
    """Spawn processes to handle frame pairs."""
    total_frames = len(center_files)
    print(f"Total frames: {total_frames}, center files: {center_files}")
    frames_per_process = max(1, total_frames // partition)
    frame_pairs = [(ii * frames_per_process, min(ii * frames_per_process + 1, total_frames - 1)) for ii in range(partition)]
    print(f"Frame pairs: {frame_pairs}")

    with Pool(processes=partition) as pool:
        pool.starmap(process, [(part, partition, save_path, center_files, total_frames, frame_pairs) for part in range(partition)])

def get_cell_centers(args: tuple[str, str, str]) -> None:
    """Compute cell centers frame by frame."""
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

def main():
    parser = argparse.ArgumentParser(description="Track cells across frame pairs and stitch trajectories.")
    parser.add_argument('partition', type=int, help='Number of partitions (processes) for parallel processing')
    parser.add_argument('--directory_path', type=str, required=True, help='Path to input directory with .npy files')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save output files')
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
            pool.map(get_cell_centers, [(file, args.directory_path, args.save_path) for file in cellpose_files])

    center_files = sorted([f for f in os.listdir(f"{args.save_path}/cellpose_centers") if f.endswith('.npy')], key=extract_number)[:6]
    total_frames = len(center_files)

    spawn(args.partition, args.save_path, center_files)
    
    frame_pairs = [(ii * max(1, total_frames // args.partition), min(ii * max(1, total_frames // args.partition)+1, total_frames)) for ii in range(args.partition)]
    stitch_frame_pairs(args.partition, args.save_path, center_files, total_frames, frame_pairs)

if __name__ == '__main__':
    main()

# python3 dynamic_infiltrate_v9.py 2 --directory_path "/mnt/data/pc3_naoh_channel1_30JAN25" --save_path "/mnt/exDisk1/douglashazel/DHcode/Clofilium_Pipeline/Patrick_temp_dir_v9"