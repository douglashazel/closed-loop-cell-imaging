import os
import gc
import re
import sys
import time
import numba
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.optimize import linear_sum_assignment

# ---------------- helpers from tracking code ---------------- #
def extract_number(filename):
    match = re.search(r'timepoint_(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_image(path):
    return np.array(Image.open(path)) / 4095.0

def load_segmentation(path):
    seg = np.load(path, allow_pickle=True)
    if isinstance(seg, dict):
        return seg['masks']
    try:
        return seg.item()['masks']
    except Exception:
        return seg
    
def calculate_circle_mask(image, radius):
    h, w = image.shape[:2]  # works whether image is 2D or 3D
    cy, cx = h / 2 + 70, w / 2
    yy, xx = np.ogrid[:h, :w]
    circle_mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2
    return circle_mask, (cx, cy)

def parallel_extract_centers(args):
    seg, cell_ids, temp_path, worker_id = args
    partial_centers = []
    for cellID in cell_ids:
        mask = seg == cellID
        y, x = np.nonzero(mask)
        if len(x) > 0 and len(y) > 0:
            partial_centers.append((int(np.mean(x)), int(np.mean(y))))
        else:
            partial_centers.append(None)
    output_path = os.path.join(temp_path, f'partial_centers_worker{worker_id}.npy')
    np.save(output_path, partial_centers)
    return output_path

def get_and_save_cell_centers(seg_path, center_save_path, num_workers=20):
    center_file = os.path.join(
        center_save_path,
        os.path.basename(seg_path).replace('.npy', '_centers.npy')
    )

    if os.path.exists(center_file):
        return np.load(center_file, allow_pickle=True)
    
    seg = load_segmentation(seg_path)
    num_masks = np.max(seg)
    all_ids = list(range(1, num_masks + 1))
    chunks = [all_ids[i::num_workers] for i in range(num_workers)]

    temp_path = os.path.join(center_save_path, 'temp_centers')
    os.makedirs(temp_path, exist_ok=True)

    args = [(seg, chunk, temp_path, i) for i, chunk in enumerate(chunks)]
    # print("Computing cell centers...")
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

def update_trajectories(new_frame_id, new_centers, save_path, grace_period=3):
    traj_path = os.path.join(save_path, "trajectories.csv")
    existing = os.path.exists(traj_path)
    frame_shift = {shift_frame: (shift_dx, shift_dy)}

    if existing:
        traj_df = pd.read_csv(traj_path)
        valid_centers = [c for c in new_centers if c is not None]
        new_centers_np = np.array(valid_centers, dtype=np.float64)

        # Find each cell's last known position within the grace window
        # Returns: dict of {df_index: (last_x, last_y, last_frame_seen)}
        cell_last_seen = {}
        for look_back in range(1, grace_period + 2):  # check prev frame first, then older
            check_frame = new_frame_id - look_back
            if check_frame < 0:
                break
            xcol, ycol = f'x{check_frame}', f'y{check_frame}'
            if xcol not in traj_df.columns:
                continue
            for i, row in traj_df.iterrows():
                if i in cell_last_seen:
                    continue  # already found a more recent position
                if pd.notna(row[xcol]) and pd.notna(row[ycol]):
                    cell_last_seen[i] = (row[xcol], row[ycol], check_frame)

        # Only consider cells last seen within grace_period frames
        active_cells = {i: v for i, v in cell_last_seen.items()
                        if new_frame_id - v[2] <= grace_period}

        # print(f"[DEBUG frame {new_frame_id}] active(+grace)={len(active_cells)}, new_centers={len(valid_centers)}")

        if len(active_cells) == 0 or len(valid_centers) == 0:
            traj_df[f'x{new_frame_id}'] = np.nan
            traj_df[f'y{new_frame_id}'] = np.nan
            traj_df.to_csv(traj_path, index=False)
            return traj_df

        active_indices = list(active_cells.keys())
        prev_centers_np = np.array([[v[0], v[1]] for v in active_cells.values()], dtype=np.float64)

        # Apply shift
        for idx_pos, df_idx in enumerate(active_indices):
            last_frame = active_cells[df_idx][2]
            dx, dy = 0, 0
            for f, (sdx, sdy) in frame_shift.items():
                if last_frame < f <= new_frame_id:
                    dx += sdx
                    dy += sdy
            prev_centers_np[idx_pos, 0] += dx
            prev_centers_np[idx_pos, 1] += dy

        # Cost matrix + Hungarian
        diff = prev_centers_np[:, np.newaxis, :] - new_centers_np[np.newaxis, :, :]
        cost_matrix = np.sqrt((diff ** 2).sum(axis=2))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        max_distance = 120.0
        new_assignments = [None] * len(valid_centers)
        new_x = {}
        new_y = {}
        matched_count = 0
        unmatched_count = 0

        for r, c in zip(row_ind, col_ind):
            df_idx = active_indices[r]
            if cost_matrix[r, c] <= max_distance:
                new_x[df_idx] = valid_centers[c][0]
                new_y[df_idx] = valid_centers[c][1]
                new_assignments[c] = df_idx
                matched_count += 1
            else:
                unmatched_count += 1

        newly_spawned = sum(1 for a in new_assignments if a is None)
        # print(f"[DEBUG frame {new_frame_id}] matched={matched_count} lost={unmatched_count} new_spawned={newly_spawned}")

        traj_df[f'x{new_frame_id}'] = pd.Series(new_x)
        traj_df[f'y{new_frame_id}'] = pd.Series(new_y)

        # Append genuinely new cells
        max_id = traj_df['CellID'].astype(int).max()
        for idx, center in enumerate(valid_centers):
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
        } for i, center in enumerate(new_centers) if center is not None])

    traj_df.to_csv(traj_path, index=False)
    return traj_df

@numba.jit(nopython=True)
def precompute_averages(segmentation, image):
    h, w = segmentation.shape
    max_id = segmentation.max()
    sum_vals = np.zeros(max_id + 1, dtype=np.float64)
    counts = np.zeros(max_id + 1, dtype=np.int64)
    for i in range(h):
        for j in range(w):
            mid = segmentation[i, j]
            if mid > 0:
                sum_vals[mid] += image[i, j]
                counts[mid] += 1
    averages = np.zeros(max_id + 1, dtype=np.float64)
    for mid in range(1, max_id + 1):
        if counts[mid] > 0:
            averages[mid] = sum_vals[mid] / counts[mid]
        else:
            averages[mid] = np.nan
    return averages

@numba.jit(nopython=True)
def compute_luminosity(x, y, segmentation, averages):
    x = int(x)
    y = int(y)
    if y < 0 or y >= segmentation.shape[0] or x < 0 or x >= segmentation.shape[1]:
        return np.nan
    mask_id = segmentation[y, x]
    if mask_id == 0:
        return np.nan
    return averages[mask_id]

def parallel_extract_luminosity(args):
    traj_subset, frame_id, segmentation, averages, temp_path, worker_id = args
    partial_lums = []
    for _, row in traj_subset.iterrows():
        cell_id = row['CellID']
        x, y = row.get(f"x{frame_id}"), row.get(f"y{frame_id}")
        lum = compute_luminosity(x, y, segmentation, averages) if not pd.isna(x) and not pd.isna(y) else np.nan
        partial_lums.append((cell_id, lum))
    output_path = os.path.join(temp_path, f'partial_lum_worker{worker_id}.npy')
    np.save(output_path, partial_lums)
    return output_path

def update_luminosity_csv(traj_df, frame_id, image, segmentation, save_path, num_workers=1):
    lum_path = os.path.join(save_path, "luminosity.csv")
    if os.path.exists(lum_path):
        lum_df = pd.read_csv(lum_path)
    else:
        lum_df = pd.DataFrame(columns=["CellID"])

    new_col = f"f{frame_id}"
    if new_col in lum_df.columns:
        return

    averages = precompute_averages(segmentation, image)
    temp_path = os.path.join(save_path, 'temp_luminosity')
    os.makedirs(temp_path, exist_ok=True)

    chunks = []
    chunk_size = (len(traj_df) + num_workers - 1) // num_workers
    # print(f'Extracting luminosity for frame {frame_id}...')
    for i in range(num_workers):
        start = i * chunk_size
        end = min(start + chunk_size, len(traj_df))
        chunks.append(traj_df.iloc[start:end])

    args = [(chunk, frame_id, segmentation, averages, temp_path, i) for i, chunk in enumerate(chunks)]

    with Pool(processes=num_workers) as pool:
        result_files = pool.map(parallel_extract_luminosity, args)

    all_lums = []
    for file in result_files:
        partial = np.load(file, allow_pickle=True)
        all_lums.extend(partial)
        os.remove(file)
    os.rmdir(temp_path)

    new_frame_df = pd.DataFrame(all_lums, columns=["CellID", new_col])
    lum_df["CellID"] = lum_df["CellID"].astype(str).apply(lambda x: str(int(float(x))) if x != 'nan' else x)
    new_frame_df["CellID"] = new_frame_df["CellID"].astype(str).apply(lambda x: str(int(float(x))) if x != 'nan' else x)

    lum_df = pd.merge(lum_df, new_frame_df, on="CellID", how="outer")
    lum_df.sort_values("CellID", key=lambda x: x.astype(int), inplace=True)
    lum_df.to_csv(lum_path, index=False)

# ---------------- segmentation + live processing ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", required=True)
parser.add_argument("--mask_dir", required=True)
parser.add_argument("--save_path", required=True)
parser.add_argument("--shift_frame", type=int, default=5, help="Frame where shift occurs")
parser.add_argument("--shift_xy", type=float, nargs=2, default=[-260, 10], help="Shift dx dy for frame")
args = parser.parse_args()
radius = 320 # set to 0 to disable circle filtering
circle_mask = None
valid_ids_set = None

image_dir = args.image_dir
mask_dir = args.mask_dir
save_path = args.save_path
shift_frame = args.shift_frame
shift_dx, shift_dy = args.shift_xy

os.makedirs(mask_dir, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

processed_frames = set()
exit_loop = False
start_time = time.time()

initial_count = None
while not exit_loop:
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))], key=extract_number)

    all_masked = all(os.path.exists(os.path.join(mask_dir, os.path.splitext(f)[0] + ".npy")) for f in images)
    if all_masked and len(images) > 0:
        exit_loop = True

    prev_cell_count = None
    pbar = tqdm(images, desc="Processing images...", unit="image", colour="green")
    for f in pbar:
        frame_id = extract_number(f)
        mask_path = os.path.join(mask_dir, os.path.splitext(f)[0] + ".npy")

        if frame_id not in processed_frames and os.path.exists(mask_path):
            image = load_image(os.path.join(image_dir, f))
            segmentation = load_segmentation(mask_path)
            # Compute circle mask once from first frame
            if circle_mask is None:
                circle_mask, dimensions = calculate_circle_mask(image, radius)  # pass image not image.shape
                cx, cy = dimensions
                cell_ids = np.unique(segmentation)
                
                if radius == 0:
                    valid_ids_set = set(cell_ids)  # all cells
                    print(f"{len(valid_ids_set)} ROIs (no circle filter)")

                else:
                    valid_ids = []
                    for cell_id in cell_ids:
                        ys, xs = np.where(segmentation == cell_id)
                        centroid_y, centroid_x = ys.mean(), xs.mean()
                        if (centroid_x - cx)**2 + (centroid_y - cy)**2 <= radius**2:
                            valid_ids.append(cell_id)
                    valid_ids_set = set(valid_ids)
                    print(f"{len(valid_ids)} ROIs within circle (out of {len(cell_ids)} total)")

            # Filter segmentation to only keep valid cells
            filtered_segmentation = segmentation.copy()
            filtered_segmentation[~np.isin(segmentation, list(valid_ids_set))] = 0

            center_path = os.path.join(save_path, "cellpose_centers")
            os.makedirs(center_path, exist_ok=True)

            # Use filtered segmentation for centers and luminosity
            centers = get_and_save_cell_centers(mask_path, center_path, num_workers=20)
            if radius == 0:
                filtered_centers = [tuple(c) for c in centers if c is not None]
            else:
                filtered_centers = [tuple(c) for c in centers if c is not None and
                                    (float(c[0]) - cx)**2 + (float(c[1]) - cy)**2 <= radius**2]

            # if frame_id <= 2:  # only check early frames
            #     print(f"[DEBUG] Sample filtered_centers (first 5): {filtered_centers[:5]}")
            #     print(f"[DEBUG] Total filtered: {len(filtered_centers)}, circle center: ({cx:.1f}, {cy:.1f}), r={radius}")

            curr_cell_count = len(filtered_centers)
            lost = (prev_cell_count - curr_cell_count) if prev_cell_count is not None else 0
            prev_cell_count = curr_cell_count

            traj_df = update_trajectories(frame_id, filtered_centers, save_path, grace_period=3)
            update_luminosity_csv(traj_df, frame_id, image, filtered_segmentation, save_path, num_workers=1)

            if initial_count is None:
                initial_count = curr_cell_count
            pbar.set_postfix({
                "frame": frame_id,
                "cells": curr_cell_count,
                "change": lost,
                "total_lost": initial_count - curr_cell_count
            })

            processed_frames.add(frame_id)
            gc.collect()

            start_time = time.time()

    # Print elapsed time in seconds on the same line
    elapsed = int(time.time() - start_time)
    sys.stdout.write(f"\rWaiting for new masks... {elapsed} sec elapsed")
    sys.stdout.flush()

    time.sleep(3)

# optional: print newline when loop finishes
print("\nAll frames processed.")

# ---------------- plot luminosities ---------------- #
def plot_luminosities_from_csv(traj_csv, save_path, tag, cmap_name="twilight_shifted"):
    df = pd.read_csv(traj_csv, index_col=0)  # CellID as index
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(df)))

    plt.figure(dpi=300)
    for (cell_id, row), color in tqdm(zip(df.iterrows(), colors), total=len(df), desc='Plotting cells...'):
        non_nan = row.dropna()
        frames = [int(str(x).lstrip("f")) for x in non_nan.index]  # clean column labels
        vals = non_nan.values
        plt.plot(frames, vals, alpha=0.7, color=color)

    plt.xlabel("Frame")
    plt.ylabel("Average luminosity")
    plt.title("Cell luminosity over time")
    plt.tight_layout()

    plot_dir = os.path.join(save_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_name = f'average_luminosity{tag}.png'
    plt.savefig(os.path.join(plot_dir, plot_name), dpi=300)
    plt.close()
    return plot_name

plot_name = plot_luminosities_from_csv(f"{save_path}/luminosity.csv", save_path, "", cmap_name="twilight_shifted")
print(f"Figure saved to {save_path}/{plot_name} frames processed.")

# ---------------- save first frame cells ---------------- #
def filter_first_frame_cells(traj_path):
    df = pd.read_csv(traj_path)

    coord_cols = [c for c in df.columns if c != "CellID"]

    # Identify columns for the first frame (first x and y)
    first_frame_cols = coord_cols[:2]

    # Keep only rows where the first frame coordinates are not NaN
    filtered = df.dropna(subset=first_frame_cols, how="any")

    base, ext = os.path.splitext(traj_path)
    output_path = base + "_firstframe" + ext
    filtered.to_csv(output_path, index=False)
    return output_path

traj_path = os.path.join(save_path, "trajectories.csv")
out_path = filter_first_frame_cells(traj_path)
print("Saved:", out_path)

# ---------------- save complete cells ---------------- #
def filter_complete_cells(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)
    complete_df = df.dropna()
    
    # If no output path provided, auto-generate one
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = base + "_complete" + ext
    
    complete_df.to_csv(output_csv, index=False)
    return output_csv

traj_path = os.path.join(save_path, "trajectories.csv")
out_path = filter_complete_cells(traj_path)
print("Saved:", out_path)

# ---------------- plot complete cells ---------------- #
def process_movies(data_path, traj_suffixes=["complete"]):

        for suffix in traj_suffixes:
            trajectories_csv = f"{data_path}/trajectories_{suffix}.csv"
            luminosity_csv = f"{data_path}/luminosity.csv"
            output_lum_csv = f"{data_path}/luminosity_{suffix}.csv"

            traj_df = pd.read_csv(trajectories_csv)
            lum_df = pd.read_csv(luminosity_csv)

            # Ensure consistent types
            traj_df['CellID'] = traj_df['CellID'].astype(str)
            lum_df['CellID'] = lum_df['CellID'].astype(str)

            # Keep only CellIDs present in trajectories
            lum_df = lum_df[lum_df['CellID'].isin(traj_df['CellID'].unique())]

            # Collapse duplicates
            lum_merged = lum_df.groupby('CellID', as_index=False).mean()

            lum_merged.to_csv(output_lum_csv, index=False)
            print(f"Merged luminosity saved to {output_lum_csv}")

            plot_name = plot_luminosities_from_csv(output_lum_csv, f"{data_path}", "_complete", cmap_name="twilight_shifted")
            print(f"Figure saved to {data_path}/{plot_name}")

process_movies(save_path)