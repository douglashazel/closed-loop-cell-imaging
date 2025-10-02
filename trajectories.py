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

# ---------------- helpers from tracking code ---------------- #
def extract_number(filename):
    match = re.search(r'timepoint_(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_image(path):
    return np.array(Image.open(path)) / 4095.0

def load_segmentation(path):
    return np.load(path, allow_pickle=True)

@numba.jit(nopython=True)
def run_all(curr_center, next_centers, max_distance=40.0):
    min_distance = np.inf
    min_idx = -1
    for i in range(next_centers.shape[0]):
        dx = next_centers[i, 0] - curr_center[0]
        dy = next_centers[i, 1] - curr_center[1]
        distance = np.sqrt(dx * dx + dy * dy)
        if distance < min_distance:
            min_distance = distance
            min_idx = i
    status = min_distance <= max_distance
    match = min_idx if status else -1
    return status, match

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
    seg = load_segmentation(seg_path)
    num_masks = np.max(seg)
    all_ids = list(range(1, num_masks + 1))
    chunks = [all_ids[i::num_workers] for i in range(num_workers)]

    temp_path = os.path.join(center_save_path, 'temp_centers')
    os.makedirs(temp_path, exist_ok=True)

    args = [(seg, chunk, temp_path, i) for i, chunk in enumerate(chunks)]
    print("Computing cell centers...")
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
        valid_centers = [c for c in new_centers if c is not None]
        new_centers_np = np.array(valid_centers, dtype=np.float64)

        print('Updating trajectories...')
        for i, row in traj_df.iterrows():
            prev_x, prev_y = row.get(f'x{prev_id}'), row.get(f'y{prev_id}')
            if pd.isna(prev_x) or pd.isna(prev_y):
                traj_df.loc[i, f'x{new_frame_id}'] = np.nan
                traj_df.loc[i, f'y{new_frame_id}'] = np.nan
                continue
            curr_center = np.array([prev_x, prev_y], dtype=np.float64)
            status, match_idx = run_all(curr_center, new_centers_np)
            if status and match_idx != -1:
                traj_df.loc[i, f'x{new_frame_id}'] = valid_centers[match_idx][0]
                traj_df.loc[i, f'y{new_frame_id}'] = valid_centers[match_idx][1]
                new_assignments[match_idx] = i
            else:
                traj_df.loc[i, f'x{new_frame_id}'] = np.nan
                traj_df.loc[i, f'y{new_frame_id}'] = np.nan

        max_id = traj_df['CellID'].astype(int).max()
        for idx, center in enumerate(new_centers):
            if new_assignments[idx] is None and center is not None:
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
    print('Extracting luminosity...')
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
args = parser.parse_args()

image_dir = args.image_dir
mask_dir = args.mask_dir
save_path = args.save_path

os.makedirs(mask_dir, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

processed_frames = set()
exit_loop = False
start_time = time.time()

while not exit_loop:
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))], key=extract_number)

    all_masked = all(os.path.exists(os.path.join(mask_dir, os.path.splitext(f)[0] + ".npy")) for f in images)
    if all_masked and len(images) > 0:
        exit_loop = True

    for f in images:
        frame_id = extract_number(f)
        mask_path = os.path.join(mask_dir, os.path.splitext(f)[0] + ".npy")

        if frame_id not in processed_frames and os.path.exists(mask_path):
            image = load_image(os.path.join(image_dir, f))
            segmentation = load_segmentation(mask_path)

            center_path = os.path.join(save_path, "cellpose_centers")
            centers = get_and_save_cell_centers(mask_path, center_path, num_workers=20)
            traj_df = update_trajectories(frame_id, centers, save_path)
            update_luminosity_csv(traj_df, frame_id, image, segmentation, save_path, num_workers=1)

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
def plot_luminosities_from_csv(traj_csv, save_path, cmap_name="twilight_shifted"):
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
    plot_name = 'average_luminosity.png'
    plt.savefig(f"{save_path}/{plot_name}", dpi=300)
    plt.close()
    return plot_name

plot_name = plot_luminosities_from_csv(f"{save_path}/luminosity.csv", save_path, cmap_name="twilight_shifted")
print(f"Figure saved to {save_path}/{plot_name} frames processed.")