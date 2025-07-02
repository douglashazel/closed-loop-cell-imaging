import os
import gc
import re
import sys
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from scipy.spatial import distance

np.set_printoptions(threshold=sys.maxsize)

def extract_number(filename):
    match = re.search(r'(\d{3})', filename)
    return int(match.group(1)) if match else -1

def get_latest_file(directory, ext):
    files = sorted([f for f in os.listdir(directory) if f.endswith(ext)], key=extract_number)
    return files[9] if files else None  # or use -1 for truly latest

def load_image(path):
    return np.array(Image.open(path)) / 4095.0

def load_segmentation(path):
    return np.load(path, allow_pickle=True).item()['masks']

def run_all(curr_center, next_centers, max_distance: float = 40):
    distances = distance.cdist(next_centers, [curr_center], metric="euclidean").flatten()
    status = np.min(distances) <= max_distance
    match = np.argmin(distances) if status else None
    return status, match

def parallel_extract_centers(args):
    seg, cell_ids, temp_path, worker_id = args
    partial_centers = []
    for cellID in tqdm(cell_ids, desc=f'Calculating cell centers (worker {worker_id})'):
        mask = seg == cellID
        y, x = np.where(mask)
        if len(x) > 0 and len(y) > 0:
            partial_centers.append((int(np.mean(x)), int(np.mean(y))))
        else:
            partial_centers.append(None)
    output_path = os.path.join(temp_path, f'partial_centers_worker{worker_id}.npy')
    np.save(output_path, partial_centers)
    return output_path

def get_and_save_cell_centers(seg_path, center_save_path, num_workers=10):
    seg = load_segmentation(seg_path)
    num_masks = np.max(seg)
    all_ids = list(range(1, num_masks + 1))
    chunks = [all_ids[i::num_workers] for i in range(num_workers)]

    temp_path = os.path.join(center_save_path, 'temp_centers')
    os.makedirs(temp_path, exist_ok=True)

    args = [(seg, chunk, temp_path, i) for i, chunk in enumerate(chunks)]
    with Pool(processes=num_workers) as pool:
        result_files = pool.map(parallel_extract_centers, args)

    frame_centers = []
    for file in result_files:
        partial = np.load(file, allow_pickle=True)
        frame_centers.extend([c for c in partial if c is not None])
        os.remove(file)

    os.rmdir(temp_path)
    os.makedirs(center_save_path, exist_ok=True)
    np.save(
        os.path.join(center_save_path, os.path.basename(seg_path).replace('.npy', '_centers.npy')),
        frame_centers
    )
    return frame_centers

def update_trajectories(new_frame_id, new_centers, save_path):
    traj_path = os.path.join(save_path, "trajectories.csv")
    existing = os.path.exists(traj_path)

    if existing:
        traj_df = pd.read_csv(traj_path)
        prev_id = new_frame_id - 1
        new_assignments = [None] * len(new_centers)

        for i, row in tqdm(traj_df.iterrows(), desc='Stitching'):
            prev_x, prev_y = row.get(f'x{prev_id}'), row.get(f'y{prev_id}')
            if pd.isna(prev_x) or pd.isna(prev_y):
                traj_df.loc[i, f'x{new_frame_id}'] = np.nan
                traj_df.loc[i, f'y{new_frame_id}'] = np.nan
                continue
            status, match_idx = run_all([prev_x, prev_y], new_centers)
            if status:
                traj_df.loc[i, f'x{new_frame_id}'] = new_centers[match_idx][0]
                traj_df.loc[i, f'y{new_frame_id}'] = new_centers[match_idx][1]
                new_assignments[match_idx] = i
            else:
                traj_df.loc[i, f'x{new_frame_id}'] = np.nan
                traj_df.loc[i, f'y{new_frame_id}'] = np.nan

        max_id = traj_df['CellID'].astype(int).max()
        for idx, center in enumerate(new_centers):
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
        traj_df = pd.DataFrame([{
            'CellID': str(i),
            f'x{new_frame_id}': center[0],
            f'y{new_frame_id}': center[1]
        } for i, center in enumerate(new_centers)])

    traj_df.to_csv(traj_path, index=False)
    print(f"Saved updated trajectories to {traj_path}")
    return traj_df

def compute_luminosity(x, y, segmentation, image):
    try:
        x, y = int(x), int(y)
        mask_id = segmentation[y, x]
        if mask_id == 0:
            return np.nan
        mask = segmentation == mask_id
        pixel_values = np.ma.masked_array(image, mask=~mask)
        return float(np.mean(pixel_values))
    except Exception:
        return np.nan

def parallel_extract_luminosity(args):
    traj_subset, frame_id, segmentation, image, temp_path, worker_id = args
    new_col = f"f{frame_id}"
    partial_lums = []
    for _, row in tqdm(traj_subset.iterrows(), desc=f'Calculating luminosity (worker {worker_id})'):
        cell_id = row['CellID']
        x, y = row.get(f"x{frame_id}"), row.get(f"y{frame_id}")
        lum = compute_luminosity(x, y, segmentation, image) if not pd.isna(x) and not pd.isna(y) else np.nan
        partial_lums.append((cell_id, lum))
    output_path = os.path.join(temp_path, f'partial_lum_worker{worker_id}.npy')
    np.save(output_path, partial_lums)
    return output_path

def update_luminosity_csv(traj_df, frame_id, image, segmentation, save_path, num_workers=10):
    from multiprocessing import Pool

    lum_path = os.path.join(save_path, "luminosity.csv")
    if os.path.exists(lum_path):
        lum_df = pd.read_csv(lum_path)
    else:
        lum_df = pd.DataFrame(columns=["CellID"])

    new_col = f"f{frame_id}"
    if new_col in lum_df.columns:
        print(f"Frame {frame_id} already in luminosity.csv")
        return

    temp_path = os.path.join(save_path, 'temp_luminosity')
    os.makedirs(temp_path, exist_ok=True)

    chunks = []
    chunk_size = (len(traj_df) + num_workers - 1) // num_workers
    for i in range(num_workers):
        start = i * chunk_size
        end = min(start + chunk_size, len(traj_df))
        chunks.append(traj_df.iloc[start:end])

    args = [(chunk, frame_id, segmentation, image, temp_path, i) for i, chunk in enumerate(chunks)]

    with Pool(processes=num_workers) as pool:
        result_files = pool.map(parallel_extract_luminosity, args)

    all_lums = []
    for file in result_files:
        partial = np.load(file, allow_pickle=True)
        all_lums.extend(partial)
        os.remove(file)
    os.rmdir(temp_path)

    new_frame_df = pd.DataFrame(all_lums, columns=["CellID", new_col])

    # Normalize CellID in both DataFrames before merging
    lum_df["CellID"] = lum_df["CellID"].astype(str).apply(lambda x: str(int(float(x))) if x != 'nan' else x)
    new_frame_df["CellID"] = new_frame_df["CellID"].astype(str).apply(lambda x: str(int(float(x))) if x != 'nan' else x)

    lum_df = pd.merge(lum_df, new_frame_df, on="CellID", how="outer")
    lum_df.sort_values("CellID", key=lambda x: x.astype(int), inplace=True)
    lum_df.to_csv(lum_path, index=False)
    print(f"Updated luminosity.csv with frame {frame_id}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory_path", required=True, help="Directory with .npy and .png")
    parser.add_argument("--save_path", required=True, help="Where to save .csv and centers")
    args = parser.parse_args()

    seg_file = get_latest_file(args.directory_path, ".npy")
    img_file = get_latest_file(args.directory_path, ".png")
    if not seg_file or not img_file:
        print("Missing segmentation or image file.")
        return

    frame_id = extract_number(seg_file)
    print(f"Processing Frame {frame_id}...")

    seg_path = os.path.join(args.directory_path, seg_file)
    img_path = os.path.join(args.directory_path, img_file)

    image = load_image(img_path)
    segmentation = load_segmentation(seg_path)

    center_path = os.path.join(args.save_path, "cellpose_centers")
    centers = get_and_save_cell_centers(seg_path, center_path, num_workers=10)

    traj_df = update_trajectories(frame_id, centers, args.save_path)
    update_luminosity_csv(traj_df, frame_id, image, segmentation, args.save_path, num_workers=10)
    gc.collect()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total script runtime: {elapsed:.2f} seconds")

# python3 dynamic_run.py --directory_path "frames" --save_path "analysis"