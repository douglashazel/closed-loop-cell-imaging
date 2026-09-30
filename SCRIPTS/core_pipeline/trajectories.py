import os
import gc
import re
import sys
import csv
import json
import time
import numba
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

from io_utils import load_msgpack, save_msgpack

# Tracking parameters recorded in <save_path>/run_params.json. A resumed run
# must use the same values, or trajectories.json would mix two settings.
TRACKING_PARAMS = ("max_distance", "grace_period", "radius", "radius_y",
                   "radius_x", "shift_frame", "shift_xy")

# ---------------- helpers from tracking code ---------------- #
def extract_number(filename):
    match = re.search(r'timepoint_(\d+)', filename)
    if match is None:
        raise ValueError(f"No 'timepoint_NNNNN' token in frame filename {filename!r}")
    return int(match.group(1))

def load_image(path):
    img = np.array(Image.open(path))
    if img.ndim == 3:
        # Same grayscale conversion as PostAnalysis.load_image_float.
        img = img.astype(np.float32).mean(axis=-1)
    return img

def load_segmentation(path):
    return np.load(path)

def calculate_circle_mask(image, radius, y_shift=0, x_shift=0):
    h, w = image.shape[:2]
    cy, cx = h / 2 + y_shift, w / 2 + x_shift
    yy, xx = np.ogrid[:h, :w]
    circle_mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2
    return circle_mask, (cx, cy)

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

def get_and_save_cell_centers(seg_path, center_save_path, pool, num_workers):
    center_file = os.path.join(
        center_save_path,
        os.path.basename(seg_path).replace('.npy', '_centers.npy')
    )

    # The cache is keyed by mask filename only, so reuse it only when it is
    # newer than the mask it was computed from.
    if os.path.exists(center_file):
        if os.path.getmtime(center_file) >= os.path.getmtime(seg_path):
            try:
                return np.load(center_file)
            except (EOFError, ValueError, OSError):
                pass
        os.remove(center_file)

    seg = load_segmentation(seg_path)
    num_masks = np.max(seg)
    all_ids = list(range(1, num_masks + 1))
    chunks = [all_ids[i::num_workers] for i in range(num_workers)]

    args = [(seg, chunk) for chunk in chunks]
    results = pool.map(parallel_extract_centers, args)

    frame_centers = [c for partial in results for c in partial if c is not None]
    os.makedirs(center_save_path, exist_ok=True)
    try:
        arr = np.asarray(frame_centers, dtype=np.int64).reshape(-1, 2)
    except Exception as e:
        weird = [(i, type(c).__name__, c) for i, c in enumerate(frame_centers)
                if not (isinstance(c, tuple) and len(c) == 2
                        and all(isinstance(v, (int, np.integer)) for v in c))]
        print(f"\n[centers FAIL] {seg_path}")
        print(f"  error: {e}")
        print(f"  n={len(frame_centers)}  weird entries (first 5): {weird[:5]}")
        raise
    np.save(center_file, arr)
    return frame_centers


# ---------------- in-memory trajectory tracking ---------------- #
def update_trajectories_inplace(traj_dict, new_frame_id, new_centers, frame_shift,
                                grace_period=3, max_distance=40.0):
    """
    Mutates traj_dict in place. No disk I/O.
    traj_dict: {cell_id_str: {"x5": 100.0, "y5": 200.0, ...}}
    frame_shift: {frame_id: (dx, dy)} stage shift applied to tracks that cross
    that frame.

    Each detection goes to at most one track: every (track, detection) pair
    within max_distance is a candidate, and candidates are taken nearest first,
    skipping tracks and detections that are already matched. A track left
    without a detection misses this frame (the grace period covers it); a
    detection left without a track starts a new one. When no two tracks want
    the same detection, every track gets its nearest detection.
    """
    valid_centers = [c for c in new_centers if c is not None]
    new_centers_np = np.array(valid_centers, dtype=np.float64) if valid_centers else np.empty((0, 2))
    new_assignments = [None] * len(new_centers)

    xkey = f"x{new_frame_id}"
    ykey = f"y{new_frame_id}"

    if traj_dict:
        track_ids = []
        cand_dist, cand_track, cand_det = [], [], []
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

            if len(new_centers_np) == 0:
                continue

            ddx = new_centers_np[:, 0] - (prev_x + dx)
            ddy = new_centers_np[:, 1] - (prev_y + dy)
            dist = np.sqrt(ddx * ddx + ddy * ddy)
            near = np.nonzero(dist <= max_distance)[0]
            if near.size:
                cand_dist.append(dist[near])
                cand_track.append(np.full(near.size, len(track_ids)))
                cand_det.append(near)
            track_ids.append(cell_id)

        if cand_dist:
            cand_dist = np.concatenate(cand_dist)
            cand_track = np.concatenate(cand_track)
            cand_det = np.concatenate(cand_det)
            # Nearest first; ties go to the earlier track, then the lower detection index.
            track_matched = set()
            for k in np.lexsort((cand_det, cand_track, cand_dist)):
                t, match_idx = int(cand_track[k]), int(cand_det[k])
                if t in track_matched or new_assignments[match_idx] is not None:
                    continue
                cell_id = track_ids[t]
                traj_dict[cell_id][xkey] = float(valid_centers[match_idx][0])
                traj_dict[cell_id][ykey] = float(valid_centers[match_idx][1])
                new_assignments[match_idx] = cell_id
                track_matched.add(t)

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

# CSV exports for quick inspection
def save_dict_as_csv(data, path):
    if not data:
        return
    all_keys = sorted({k for row in data.values() for k in row.keys()},
                      key=lambda k: (int(k[1:]) if k[1:].isdigit() else 0, k[0]))
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cell_id'] + all_keys)
        for cell_id, row in data.items():
            writer.writerow([cell_id] + [row.get(k) for k in all_keys])

# ---------------- run parameters ---------------- #
def check_run_params(args, save_path, resuming):
    """Record the tracking parameters; refuse to resume a run made with different ones."""
    params = {k: getattr(args, k) for k in TRACKING_PARAMS}
    params_path = os.path.join(save_path, "run_params.json")
    if resuming:
        if os.path.exists(params_path):
            with open(params_path) as f:
                previous = json.load(f)
            changed = {k: (previous.get(k), v) for k, v in params.items() if previous.get(k) != v}
            if changed and not args.force_resume:
                sys.exit(
                    f"{save_path} holds a run made with different tracking parameters "
                    f"(previous, now): {changed}. Delete that analysis/ directory to start "
                    f"over, or pass --force_resume to continue anyway."
                )
        else:
            print(f"Warning: resuming {save_path} without run_params.json; "
                  f"cannot check that the tracking parameters match the earlier run.")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

# ---------------- segmentation + live processing ---------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--max_distance", type=float, default=40.0, help="Max distance for trajectory linking")
    parser.add_argument("--grace_period", type=int, default=3, help="Number of frames to look back for linking")
    parser.add_argument("--radius", type=int, default=0, help="Radius for circular mask (0 to disable)")
    parser.add_argument("--radius_y", type=int, default=0, help="Y shift for circular mask")
    parser.add_argument("--radius_x", type=int, default=0, help="X shift for circular mask")
    parser.add_argument("--shift_frame", type=int, default=5,
                        help="timepoint_NNNNN token of the frame where the shift occurs")
    parser.add_argument("--shift_xy", type=float, nargs=2, default=[0, 0], help="Shift dx dy for frame")
    parser.add_argument("--save_interval", type=int, default=10, help="Save to disk every N frames")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Worker processes for cell-centre extraction (default: half the CPUs)")
    parser.add_argument("--mask_timeout_sec", type=float, default=3600,
                        help="Exit with an error when no new mask appears for this long")
    parser.add_argument("--force_resume", action="store_true",
                        help="Resume even if run_params.json records different tracking parameters")
    args = parser.parse_args()

    image_dir = args.image_dir
    mask_dir = args.mask_dir
    save_path = args.save_path
    max_distance = args.max_distance
    grace_period = args.grace_period

    frame_shift = {args.shift_frame: tuple(args.shift_xy)}
    save_interval = args.save_interval

    radius = args.radius
    y_shift = args.radius_y
    x_shift = args.radius_x
    circle_mask = None

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    traj_json_path = os.path.join(save_path, "trajectories.json")
    lum_json_path  = os.path.join(save_path, "luminosity.json")

    # Load from disk if resuming
    resuming = os.path.exists(traj_json_path)
    check_run_params(args, save_path, resuming)
    traj_dict = load_msgpack(traj_json_path) if resuming else {}
    lum_dict  = load_msgpack(lum_json_path) if os.path.exists(lum_json_path) else {}

    processed_frames = {int(k[1:]) for coords in traj_dict.values()
                        for k in coords if k.startswith('x') and k[1:].isdigit()}
    exit_loop = False
    start_time = time.time()
    last_progress = time.time()
    frames_since_save = 0
    center_path = os.path.join(save_path, "cellpose_centers")

    with Pool(processes=args.workers) as pool:
        while not exit_loop:
            images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))], key=extract_number)

            all_masked = all(os.path.exists(os.path.join(mask_dir, os.path.splitext(f)[0] + ".npy")) for f in images)
            if all_masked and len(images) > 0:
                exit_loop = True

            for f in tqdm(images, desc="Processing images...", unit="image", colour="green"):
                frame_id = extract_number(f)
                mask_path = os.path.join(mask_dir, os.path.splitext(f)[0] + ".npy")

                if frame_id not in processed_frames and os.path.exists(mask_path):

                    try:
                        segmentation = load_segmentation(mask_path)
                    except (ValueError, OSError, EOFError) as e:
                        print(f"\nCould not read {mask_path} ({e}); retrying on the next pass")
                        exit_loop = False
                        continue
                    image = load_image(os.path.join(image_dir, f))

                    if circle_mask is None:
                        circle_mask, dimensions = calculate_circle_mask(image, radius, y_shift, x_shift)
                        cx, cy = dimensions

                    if radius == 0:
                        filtered_segmentation = segmentation
                    else:
                        filtered_segmentation = segmentation.copy()
                        filtered_segmentation[~circle_mask] = 0

                    centers = get_and_save_cell_centers(mask_path, center_path, pool, args.workers)
                    if radius == 0:
                        filtered_centers = [tuple(c) for c in centers if c is not None]
                    else:
                        filtered_centers = [tuple(c) for c in centers if c is not None and
                                            (float(c[0]) - cx)**2 + (float(c[1]) - cy)**2 <= radius**2]

                    # Pure in-memory updates — no disk I/O
                    update_trajectories_inplace(traj_dict, frame_id, filtered_centers, frame_shift,
                                                grace_period=grace_period, max_distance=max_distance)
                    update_luminosity_inplace(lum_dict, traj_dict, frame_id, image, filtered_segmentation)

                    processed_frames.add(frame_id)
                    frames_since_save += 1
                    gc.collect()
                    start_time = time.time()
                    last_progress = time.time()

                    # Periodic disk save
                    if frames_since_save >= save_interval:
                        save_msgpack(traj_dict, traj_json_path)
                        save_msgpack(lum_dict, lum_json_path)
                        frames_since_save = 0

            if exit_loop:
                break

            if time.time() - last_progress > args.mask_timeout_sec:
                save_msgpack(traj_dict, traj_json_path)
                save_msgpack(lum_dict, lum_json_path)
                n_missing = len(images) - len(processed_frames)
                sys.exit(
                    f"\nNo new mask in {args.mask_timeout_sec:.0f} s and {n_missing} frame(s) "
                    f"still unprocessed; giving up. Progress saved to {save_path}; "
                    f"rerun to resume once the masks exist."
                )

            elapsed = int(time.time() - start_time)
            sys.stdout.write(f"\rWaiting for new masks... {elapsed} sec elapsed")
            sys.stdout.flush()

            time.sleep(3)

    # Final save when all frames done
    save_msgpack(traj_dict, traj_json_path)
    save_msgpack(lum_dict, lum_json_path)
    print("\nAll frames processed.")

    firstframe_traj = filter_first_frame_cells(traj_dict)
    out_path = os.path.join(save_path, "trajectories_firstframe.json")
    save_msgpack(firstframe_traj, out_path)
    print("Saved:", out_path)

    complete_traj, lum_complete = filter_complete_cells(traj_dict, lum_dict)

    traj_out = os.path.join(save_path, "trajectories_complete.json")
    lum_out  = os.path.join(save_path, "luminosity_complete.json")
    save_msgpack(complete_traj, traj_out)
    save_msgpack(lum_complete, lum_out)
    print("Saved:", traj_out)
    print("Saved:", lum_out)

    traj_csv = os.path.join(save_path, "trajectories_complete.csv")
    lum_csv  = os.path.join(save_path, "luminosity_complete.csv")
    save_dict_as_csv(complete_traj, traj_csv)
    save_dict_as_csv(lum_complete, lum_csv)
    print("Saved:", traj_csv)
    print("Saved:", lum_csv)

    print("Done.")


if __name__ == "__main__":
    main()
