import os
import gc
import re
import sys
import time
import numba
import msgpack
import argparse
import numpy as np
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
    seg = np.load(path, allow_pickle=True)
    if isinstance(seg, dict):
        return seg['masks']
    try:
        return seg.item()['masks']
    except Exception:
        return seg

def calculate_circle_mask(image, radius, y_shift=0, x_shift=0):
    h, w = image.shape[:2]
    cy, cx = h / 2 + y_shift, w / 2 + x_shift
    yy, xx = np.ogrid[:h, :w]
    circle_mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2
    return circle_mask, (cx, cy)

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
    seg, cell_ids = args
    partial_centers = []
    for cellID in cell_ids:
        mask = seg == cellID
        y, x = np.nonzero(mask)
        if len(x) > 0 and len(y) > 0:
            cx, cy = np.mean(x), np.mean(y)
            dists = (x - cx)**2 + (y - cy)**2
            best = np.argmin(dists)
            partial_centers.append((int(x[best]), int(y[best])))
        else:
            partial_centers.append(None)
    return partial_centers

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

    args = [(seg, chunk) for chunk in chunks]
    with Pool(processes=num_workers) as pool:
        results = pool.map(parallel_extract_centers, args)

    frame_centers = [c for partial in results for c in partial if c is not None]
    os.makedirs(center_save_path, exist_ok=True)
    np.save(
        os.path.join(center_save_path, os.path.basename(seg_path).replace('.npy', '_centers.npy')),
        frame_centers
    )
    return frame_centers

# ---------------- in-memory trajectory tracking ---------------- #
def update_trajectories_inplace(traj_dict, new_frame_id, new_centers, grace_period=3, max_distance=40.0):
    """
    Mutates traj_dict in place. No disk I/O.
    traj_dict: {cell_id_str: {"x5": 100.0, "y5": 200.0, ...}}
    """
    frame_shift = {shift_frame: (shift_dx, shift_dy)}

    valid_centers = [c for c in new_centers if c is not None]
    new_centers_np = np.array(valid_centers, dtype=np.float64) if valid_centers else np.empty((0, 2))
    new_assignments = [None] * len(new_centers)

    xkey = f"x{new_frame_id}"
    ykey = f"y{new_frame_id}"

    if traj_dict:
        for cell_id, coords in traj_dict.items():
            last_seen = None
            for look_back in range(1, grace_period + 2):
                check_frame = new_frame_id - look_back
                if check_frame < 0:
                    break
                cx_key = f"x{check_frame}"
                cy_key = f"y{check_frame}"
                if cx_key in coords and cy_key in coords:
                    px = coords[cx_key]
                    py = coords[cy_key]
                    if px is not None and py is not None:
                        last_seen = (px, py, check_frame)
                        break

            if last_seen is None or (new_frame_id - last_seen[2]) > grace_period:
                continue

            prev_x, prev_y, last_frame = last_seen

            dx, dy = 0.0, 0.0
            for f, (sdx, sdy) in frame_shift.items():
                if last_frame < f <= new_frame_id:
                    dx += sdx
                    dy += sdy

            curr_center = np.array([prev_x + dx, prev_y + dy], dtype=np.float64)

            if len(new_centers_np) == 0:
                continue

            status, match_idx = run_all(curr_center, new_centers_np, max_distance=max_distance)
            if status and match_idx != -1:
                coords[xkey] = float(valid_centers[match_idx][0])
                coords[ykey] = float(valid_centers[match_idx][1])
                new_assignments[match_idx] = cell_id

        max_id = max((int(k) for k in traj_dict.keys()), default=-1)
        for idx, center in enumerate(new_centers):
            if new_assignments[idx] is None and center is not None:
                max_id += 1
                traj_dict[str(max_id)] = {
                    xkey: float(center[0]),
                    ykey: float(center[1])
                }
    else:
        for i, center in enumerate(new_centers):
            if center is not None:
                traj_dict[str(i)] = {
                    xkey: float(center[0]),
                    ykey: float(center[1])
                }

# ---------------- in-memory luminosity extraction ---------------- #
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

def update_luminosity_inplace(lum_dict, traj_dict, frame_id, image, segmentation):
    """
    Mutates lum_dict in place. No disk I/O.
    lum_dict: {cell_id_str: {"f0": 0.42, "f1": 0.44, ...}}
    """
    new_col = f"f{frame_id}"
    xkey, ykey = f"x{frame_id}", f"y{frame_id}"

    averages = precompute_averages(segmentation, image)

    for cell_id, coords in traj_dict.items():
        x = coords.get(xkey)
        y = coords.get(ykey)
        if x is not None and y is not None:
            lum = compute_luminosity(x, y, segmentation, averages)
            lum_val = None if np.isnan(lum) else float(lum)
        else:
            lum_val = None

        if cell_id not in lum_dict:
            lum_dict[cell_id] = {}
        lum_dict[cell_id][new_col] = lum_val

# ---------------- disk save/load ---------------- #
def save_json(data, path):
    with open(path, 'wb') as f:
        msgpack.pack(data, f)

def load_json(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return msgpack.unpack(f, raw=False)
    return {}

# ---------------- segmentation + live processing ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", required=True)
parser.add_argument("--mask_dir", required=True)
parser.add_argument("--save_path", required=True)
parser.add_argument("--max_distance", type=float, default=40.0, help="Max distance for trajectory linking")
parser.add_argument("--grace_period", type=int, default=3, help="Number of frames to look back for linking")
parser.add_argument("--radius", type=int, default=0, help="Radius for circular mask (0 to disable)")
parser.add_argument("--y_shift", type=int, default=0, help="Y shift for circular mask")
parser.add_argument("--x_shift", type=int, default=0, help="X shift for circular mask")
parser.add_argument("--shift_frame", type=int, default=5, help="Frame where shift occurs")
parser.add_argument("--shift_xy", type=float, nargs=2, default=[0, 0], help="Shift dx dy for frame")
parser.add_argument("--save_interval", type=int, default=10, help="Save to disk every N frames")
args = parser.parse_args()

image_dir = args.image_dir
mask_dir = args.mask_dir
save_path = args.save_path
max_distance = args.max_distance
grace_period = args.grace_period

shift_frame = args.shift_frame
shift_dx, shift_dy = args.shift_xy
save_interval = args.save_interval

radius = args.radius
y_shift = args.y_shift
x_shift = args.x_shift
circle_mask = None

os.makedirs(mask_dir, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

traj_json_path = os.path.join(save_path, "trajectories.json")
lum_json_path  = os.path.join(save_path, "luminosity.json")

# Load from disk if resuming
traj_dict = load_json(traj_json_path)
lum_dict  = load_json(lum_json_path)

processed_frames = set()
exit_loop = False
start_time = time.time()
frames_since_save = 0

while not exit_loop:
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))], key=extract_number)

    all_masked = all(os.path.exists(os.path.join(mask_dir, os.path.splitext(f)[0] + ".npy")) for f in images)
    if all_masked and len(images) > 0:
        exit_loop = True

    for f in tqdm(images, desc="Processing images...", unit="image", colour="green"):
        frame_id = extract_number(f)
        mask_path = os.path.join(mask_dir, os.path.splitext(f)[0] + ".npy")

        if frame_id not in processed_frames and os.path.exists(mask_path):

            image = load_image(os.path.join(image_dir, f))
            segmentation = load_segmentation(mask_path)

            if circle_mask is None:
                circle_mask, dimensions = calculate_circle_mask(image, radius, y_shift, x_shift)
                cx, cy = dimensions
                cell_ids = np.unique(segmentation)

            if radius == 0:
                filtered_segmentation = segmentation
            else:
                filtered_segmentation = segmentation.copy()
                filtered_segmentation[~circle_mask] = 0

            center_path = os.path.join(save_path, "cellpose_centers")
            centers = get_and_save_cell_centers(mask_path, center_path, num_workers=20)
            if radius == 0:
                filtered_centers = [tuple(c) for c in centers if c is not None]
            else:
                filtered_centers = [tuple(c) for c in centers if c is not None and
                                    (float(c[0]) - cx)**2 + (float(c[1]) - cy)**2 <= radius**2]
            os.makedirs(center_path, exist_ok=True)

            # Pure in-memory updates — no disk I/O
            update_trajectories_inplace(traj_dict, frame_id, filtered_centers, grace_period=grace_period, max_distance=max_distance)
            update_luminosity_inplace(lum_dict, traj_dict, frame_id, image, filtered_segmentation)

            processed_frames.add(frame_id)
            frames_since_save += 1
            gc.collect()
            start_time = time.time()

            # Periodic disk save
            if frames_since_save >= save_interval:
                save_json(traj_dict, traj_json_path)
                save_json(lum_dict, lum_json_path)
                frames_since_save = 0

    elapsed = int(time.time() - start_time)
    sys.stdout.write(f"\rWaiting for new masks... {elapsed} sec elapsed")
    sys.stdout.flush()

    time.sleep(3)

# Final save when all frames done
save_json(traj_dict, traj_json_path)
save_json(lum_dict, lum_json_path)
print("\nAll frames processed.")


# ---------------- plot luminosities ---------------- #
def plot_luminosities_from_dict(lum_dict, save_path, tag, cmap_name="twilight_shifted"):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(lum_dict)))

    plt.figure(dpi=300)
    for (cell_id, frame_lums), color in tqdm(zip(lum_dict.items(), colors), total=len(lum_dict), desc='Plotting cells...'):
        frames_vals = [(int(k.lstrip('f')), v) for k, v in frame_lums.items() if v is not None]
        if not frames_vals:
            continue
        frames_vals.sort()
        frames, vals = zip(*frames_vals)
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

plot_name = plot_luminosities_from_dict(lum_dict, save_path, "", cmap_name="twilight_shifted")
print(f"Figure saved to {save_path}/{plot_name}")

# ---------------- filter first frame cells ---------------- #
def filter_first_frame_cells(traj_dict):
    all_frames = set()
    for coords in traj_dict.values():
        for key in coords:
            if key.startswith('x'):
                try:
                    all_frames.add(int(key[1:]))
                except ValueError:
                    pass
    if not all_frames:
        return {}
    first_frame = min(all_frames)
    xkey, ykey = f"x{first_frame}", f"y{first_frame}"
    return {cid: coords for cid, coords in traj_dict.items()
            if xkey in coords and ykey in coords
            and coords[xkey] is not None and coords[ykey] is not None}

firstframe_traj = filter_first_frame_cells(traj_dict)
out_path = os.path.join(save_path, "trajectories_firstframe.json")
save_json(firstframe_traj, out_path)
print("Saved:", out_path)

# ---------------- filter complete cells ---------------- #
def filter_complete_cells(traj_dict, lum_dict):
    all_frames = set()
    for coords in traj_dict.values():
        for key in coords:
            if key.startswith('x'):
                try:
                    all_frames.add(int(key[1:]))
                except ValueError:
                    pass

    complete_traj = {}
    for cid, coords in traj_dict.items():
        cell_frames = set()
        for key in coords:
            if key.startswith('x') and coords[key] is not None:
                try:
                    cell_frames.add(int(key[1:]))
                except ValueError:
                    pass
        if cell_frames == all_frames:
            complete_traj[cid] = coords

    complete_ids = set(complete_traj.keys())
    lum_complete = {cid: v for cid, v in lum_dict.items() if cid in complete_ids}
    return complete_traj, lum_complete

complete_traj, lum_complete = filter_complete_cells(traj_dict, lum_dict)

traj_out = os.path.join(save_path, "trajectories_complete.json")
lum_out  = os.path.join(save_path, "luminosity_complete.json")
save_json(complete_traj, traj_out)
save_json(lum_complete, lum_out)
print("Saved:", traj_out)
print("Saved:", lum_out)

# ---------------- plot complete cells ---------------- #
plot_name = plot_luminosities_from_dict(lum_complete, save_path, "_complete", cmap_name="twilight_shifted")
print(f"Figure saved to {save_path}/{plot_name}")