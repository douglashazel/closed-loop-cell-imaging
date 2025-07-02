import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance

def extract_number(filename: str) -> int:
    match = re.search(r'(\d{3})', filename)
    return int(match.group(1)) if match else -1

def run_all(curr_center, next_centers, max_distance: float = 40) -> tuple[bool, int | None]:
    distances = distance.cdist(next_centers, [curr_center], metric="euclidean").flatten()
    status = np.min(distances) <= max_distance
    cell_match = np.argmin(distances) if status else None
    return status, cell_match

def get_and_save_cell_centers(seg_path: str, center_save_path: str) -> list[tuple[int, int]]:
    try:
        frame_centers = []
        seg = np.load(seg_path, allow_pickle=True).item()['masks']
        num_masks = np.max(seg)
        for cellID in tqdm(range(1, num_masks + 1), desc='Calculating cell centers...'):
            mask = seg == cellID
            y, x = np.where(mask)
            if len(x) > 0 and len(y) > 0:
                frame_centers.append((int(np.mean(x)), int(np.mean(y))))
        os.makedirs(center_save_path, exist_ok=True)
        center_file = os.path.join(center_save_path, os.path.basename(seg_path).replace('.npy', '_centers.npy'))
        np.save(center_file, frame_centers)
        return frame_centers
    except Exception as e:
        print(f"Error extracting centers from {seg_path}: {e}")
        return []

def update_trajectories(new_frame_id: int, new_centers: list[tuple[int, int]], save_path: str):
    traj_path = os.path.join(save_path, "trajectories.csv")
    existing = os.path.exists(traj_path)

    if existing:
        traj_df = pd.read_csv(traj_path)
        prev_frame_id = new_frame_id - 1
        prev_col_x = f'x{prev_frame_id}'
        prev_col_y = f'y{prev_frame_id}'
        new_assignments = [None] * len(new_centers)

        for i, row in tqdm(traj_df.iterrows(), desc='Stitching...'):
            if prev_col_x not in row or prev_col_y not in row:
                continue
            last_x, last_y = row[prev_col_x], row[prev_col_y]
            if np.isnan(last_x) or np.isnan(last_y):
                traj_df.loc[i, f'x{new_frame_id}'] = np.nan
                traj_df.loc[i, f'y{new_frame_id}'] = np.nan
                continue
            status, match_idx = run_all([last_x, last_y], new_centers)
            if status:
                traj_df.loc[i, f'x{new_frame_id}'] = new_centers[match_idx][0]
                traj_df.loc[i, f'y{new_frame_id}'] = new_centers[match_idx][1]
                new_assignments[match_idx] = i
            else:
                traj_df.loc[i, f'x{new_frame_id}'] = np.nan
                traj_df.loc[i, f'y{new_frame_id}'] = np.nan

        max_id = traj_df['CellID'].astype(int).max()
        for idx, center in tqdm(enumerate(new_centers)):
            if new_assignments[idx] is None:
                max_id += 1
                new_row = {'CellID': str(max_id)}
                for col in traj_df.columns:
                    if col != 'CellID':
                        new_row[col] = np.nan
                new_row[f'x{new_frame_id}'] = center[0]
                new_row[f'y{new_frame_id}'] = center[1]
                traj_df = pd.concat([traj_df, pd.DataFrame([new_row])], ignore_index=True)

    else:
        data = []
        for i, center in tqdm(enumerate(new_centers), 'Creating initial .csv file...'):
            data.append({
                'CellID': str(i),
                f'x{new_frame_id}': center[0],
                f'y{new_frame_id}': center[1]
            })
        traj_df = pd.DataFrame(data)

    traj_df.to_csv(traj_path, index=False)
    print(f"Saved updated trajectories to {traj_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory_path", type=str, required=True, help="Path with .npy segmentations")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save cell centers and trajectories")
    args = parser.parse_args()

    seg_files = sorted([f for f in os.listdir(args.directory_path) if f.endswith('.npy')], key=extract_number)
    latest_file = seg_files[2]
    latest_frame_id = extract_number(latest_file)
    seg_path = os.path.join(args.directory_path, latest_file)

    print(f"Processing new segmentation: {latest_file} (Frame {latest_frame_id})")

    center_save_path = os.path.join(args.save_path, "cellpose_centers")
    new_centers = get_and_save_cell_centers(seg_path, center_save_path)

    update_trajectories(latest_frame_id, new_centers, args.save_path)

if __name__ == "__main__":
    main()

# python3 dynamic_infiltrate.py --directory_path "frames" --save_path "analysis"