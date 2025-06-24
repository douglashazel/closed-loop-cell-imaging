import os
import gc
import re
import argparse
import numpy as np
import pandas as pd
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
    """Process a pair of consecutive frames and save cell trajectories to a CSV."""
    start_frame = frame_pairs[pair_idx][0]
    end_frame = frame_pairs[pair_idx][1]
    print(f"Partition {pair_idx}: frame pair {start_frame}-{end_frame}")
    
    if start_frame >= len(center_files) or end_frame >= len(center_files):
        print(f"Skipping pair {start_frame}-{end_frame}: start_frame={start_frame}, end_frame={end_frame}, len(center_files)={len(center_files)}")
        return

    try:
        curr_centers = np.load(f"{save_path}/cellpose_centers/{center_files[start_frame]}", allow_pickle=True)
        next_centers = np.load(f"{save_path}/cellpose_centers/{center_files[end_frame]}", allow_pickle=True)

        # Initialize DataFrame for this frame pair
        columns = ['CellID'] + [f'x{start_frame}', f'y{start_frame}', f'x{end_frame}', f'y{end_frame}']
        data = []
        
        for cell_num in tqdm(range(len(curr_centers)), desc="Tracking..."):
            row = {'CellID': str(cell_num)}
            # Set centers for start frame
            row[f'x{start_frame}'] = curr_centers[cell_num][0]
            row[f'y{start_frame}'] = curr_centers[cell_num][1]
            # Set centers for end frame if matched
            status, cell_match = run_all(curr_centers[cell_num], next_centers)
            if status and cell_match is not None:
                row[f'x{end_frame}'] = next_centers[cell_match][0]
                row[f'y{end_frame}'] = next_centers[cell_match][1]
            else:
                row[f'x{end_frame}'] = np.nan
                row[f'y{end_frame}'] = np.nan
            data.append(row)

        # Save to CSV
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(f"{save_path}/frame{start_frame}-{end_frame}.csv", index=False)

    except Exception as e:
        print(f"Error processing frame pair {start_frame}-{end_frame}: {str(e)}")
    finally:
        gc.collect()

def stitch_frame_pairs(partition: int, save_path: str, center_files: list, total_frames: int, frame_pairs: list) -> None:
    """Stitch trajectories across all frame pairs into a single CSV."""

    # Initialize with the first frame pair's trajectories
    curr_dir = f"{save_path}/frame{frame_pairs[0][0]}-{frame_pairs[0][1]}.csv"
    if not os.path.exists(curr_dir):
        print(f"Skipping stitching: {curr_dir} does not exist")
        return

    curr_df = pd.read_csv(curr_dir)
    
    # Initialize the final DataFrame columns: CellID + x,y for each frame
    final_columns = ['CellID'] + [f'{coord}{i}' for i in range(total_frames) for coord in ['x', 'y']]
    final_data = curr_df.to_dict('records')

    # Iterate through remaining frame pairs
    for pair_idx in range(1, len(frame_pairs)):
        next_dir = f"{save_path}/frame{frame_pairs[pair_idx][0]}-{frame_pairs[pair_idx][1]}.csv"
        if not os.path.exists(next_dir):
            print(f"Skipping stitching for pair {frame_pairs[pair_idx][0]}-{frame_pairs[pair_idx][1]}: {next_dir} does not exist")
            continue

        next_df = pd.read_csv(next_dir)
        next_centers = np.load(f"{save_path}/cellpose_centers/{center_files[frame_pairs[pair_idx][0]]}", allow_pickle=True)
        
        used_cell_ids = set()
        # Match each cell from curr_df to next_df
        for idx, row in tqdm(curr_df.iterrows(), total=len(curr_df), desc=f"Stitching pair {frame_pairs[pair_idx][0]}-{frame_pairs[pair_idx][1]}..."):
            # Find the last non-NaN frame in curr_df
            last_frame = frame_pairs[pair_idx - 1][1]  # Last frame of previous pair
            last_center = [row[f'x{last_frame}'], row[f'y{last_frame}']]
            if np.isnan(last_center).any():
                continue  # Skip if no valid center

            status, cell_match = run_all(last_center, next_centers)
            if status and cell_match is not None:
                used_cell_ids.add(str(cell_match))
                # Copy centers from next_df to curr_df for this cell
                next_df['CellID'] = next_df['CellID'].astype(str)
                next_row = next_df[next_df['CellID'] == str(cell_match)]
                if next_row.empty:
                    continue  # skip if match not found
                next_row = next_row.iloc[0]
                for frame in [frame_pairs[pair_idx][0], frame_pairs[pair_idx][1]]:
                    final_data[idx][f'x{frame}'] = next_row[f'x{frame}']
                    final_data[idx][f'y{frame}'] = next_row[f'y{frame}']

        # Add unmatched cells from next_df with new CellIDs
        max_cell_id = max([int(row['CellID']) for row in final_data], default=-1)
        for _, row in next_df.iterrows():
            if row['CellID'] not in used_cell_ids:
                max_cell_id += 1
                new_row = {'CellID': str(max_cell_id)}
                # Initialize all frames with NaN
                for frame in range(total_frames):
                    new_row[f'x{frame}'] = np.nan
                    new_row[f'y{frame}'] = np.nan
                # Copy centers for the current frame pair
                for frame in [frame_pairs[pair_idx][0], frame_pairs[pair_idx][1]]:
                    new_row[f'x{frame}'] = row[f'x{frame}']
                    new_row[f'y{frame}'] = row[f'y{frame}']
                final_data.append(new_row)

        # Update curr_df for the next iteration
        curr_df = pd.DataFrame(final_data, columns=final_columns)

    # Save final trajectories to CSV
    final_df = pd.DataFrame(final_data, columns=final_columns)
    final_df.to_csv(f"{save_path}/trajectories.csv", index=False)

def spawn(partition: int, save_path: str, center_files: list) -> None:
    """Spawn processes to handle frame pairs."""
    total_frames = len(center_files)
    print(f"Total frames: {total_frames}, center files: {center_files}")
    frame_pairs = [(i, min(i + 1, total_frames - 1)) for i in range(0, total_frames, 2)]
    print(f"Frame pairs: {frame_pairs}")

    with Pool(processes=partition) as pool:
        pool.starmap(process, [(part, partition, save_path, center_files, total_frames, frame_pairs) for part in range(len(frame_pairs))])
    return frame_pairs

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

    center_files = sorted([f for f in os.listdir(f"{args.save_path}/cellpose_centers") if f.endswith('.npy')], key=extract_number)#[:18]
    total_frames = len(center_files)

    frame_pairs = spawn(args.partition, args.save_path, center_files)
    stitch_frame_pairs(args.partition, args.save_path, center_files, total_frames, frame_pairs)

if __name__ == '__main__':
    main()

# python3 dynamic_infiltrate_v11.py 9 --directory_path "/mnt/data/pc3_naoh_channel1_30JAN25" --save_path "/mnt/exDisk1/douglashazel/DHcode/PE_Pipeline/infiltrate_testing/Patrick_temp_dir_v11"