"""Pure-numpy helpers from preprocess_gui.py.

Kept backend-only (no Qt, no napari) so the Flask endpoints can reuse them
directly.
"""

import io
import os
import re

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass
from scipy.spatial import Delaunay


# ── filename + load helpers ────────────────────────────────────────────────
def timepoint_sort_key(fname: str) -> int:
    m = re.search(r"timepoint_(\d+)", fname)
    return int(m.group(1)) if m else 10**9


def timepoint_token(fname: str) -> int:
    """The frame number Stage 1 keys on (trajectories.py extract_number):
    the timepoint_NNNNN token, or -1 if the name has none."""
    m = re.search(r"timepoint_(\d+)", fname)
    return int(m.group(1)) if m else -1


def load_segmentation(path: str) -> np.ndarray:
    seg = np.load(path, allow_pickle=True)
    if isinstance(seg, dict):
        return seg["masks"]
    try:
        return seg.item()["masks"]
    except Exception:
        return seg


# ── image → PNG helpers ────────────────────────────────────────────────────
def normalize_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


def array_to_png_bytes(arr: np.ndarray, mode: str = "L") -> io.BytesIO:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    buf.seek(0)
    return buf


def colorize_labels(mask: np.ndarray, alpha: int = 140) -> np.ndarray:
    """Random-hue LUT per cell ID → RGBA uint8."""
    if mask is None:
        return None
    mask = mask.astype(np.int32)
    h, w = mask.shape[:2]
    max_id = int(mask.max()) if mask.size else 0
    if max_id <= 0:
        return np.zeros((h, w, 4), dtype=np.uint8)
    rng = np.random.default_rng(42)
    lut = np.zeros((max_id + 1, 4), dtype=np.uint8)
    lut[1:, :3] = rng.integers(64, 255, size=(max_id, 3), dtype=np.uint8)
    lut[1:, 3] = alpha
    return lut[mask]


def downsample_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    if target_w <= 0 or img.shape[1] <= target_w:
        return img
    scale = target_w / img.shape[1]
    new_h = max(1, int(round(img.shape[0] * scale)))
    pil = Image.fromarray(img)
    pil = pil.resize((target_w, new_h), Image.BILINEAR)
    return np.array(pil)


# ── Delaunay (same as preprocess_gui.py) ───────────────────────────────────
def get_delaunay_neighbors(cell_id: int, centroids_by_id: dict) -> list:
    ids = np.array(list(centroids_by_id.keys()))
    coords = np.array([centroids_by_id[i] for i in ids])
    chosen_pos = int(np.where(ids == cell_id)[0][0])
    tri = Delaunay(coords)
    neighbors = set()
    for simplex in tri.simplices:
        if chosen_pos in simplex:
            neighbors.update(simplex)
    neighbors.discard(chosen_pos)
    return [int(ids[i]) for i in neighbors]


def centroids_from_seg(seg: np.ndarray) -> dict:
    cell_ids = np.arange(1, int(seg.max()) + 1)
    if len(cell_ids) == 0:
        return {}
    raw = center_of_mass(seg > 0, labels=seg, index=cell_ids)
    return {int(cid): rc for cid, rc in zip(cell_ids, raw) if not np.isnan(rc[0])}


def all_mean_neighbor_distances(centroids_by_id: dict) -> dict:
    """For every cell, compute mean distance to its Delaunay neighbours.
    Returns {cell_id: mean_distance}."""
    if len(centroids_by_id) < 4:
        return {}
    ids = np.array(list(centroids_by_id.keys()))
    coords = np.array([centroids_by_id[i] for i in ids])
    tri = Delaunay(coords)
    neighbour_sets: dict[int, set] = {int(i): set() for i in ids}
    for simplex in tri.simplices:
        for a in simplex:
            for b in simplex:
                if a != b:
                    neighbour_sets[int(ids[a])].add(int(ids[b]))
    out = {}
    for cid, nbrs in neighbour_sets.items():
        if not nbrs:
            continue
        c = np.asarray(centroids_by_id[cid])
        d = [float(np.linalg.norm(np.asarray(centroids_by_id[n]) - c)) for n in nbrs]
        out[cid] = float(np.mean(d))
    return out


# ── Shift tab split view ───────────────────────────────────────────────────
def split_frames_png(img_prev: np.ndarray, img_curr: np.ndarray) -> tuple:
    """Return (combined_uint8, left_width) for ShiftTab's split view."""
    p = normalize_gray(img_prev)
    c = normalize_gray(img_curr)
    # Pad if shapes mismatch
    h = max(p.shape[0], c.shape[0])
    def pad(img):
        out = np.zeros((h, img.shape[1]), dtype=np.uint8)
        out[: img.shape[0], :] = img
        return out
    p = pad(p)
    c = pad(c)
    combined = np.concatenate([p, c], axis=1)
    return combined, int(p.shape[1])


# ── Experiment discovery ───────────────────────────────────────────────────
def scan_experiments(experiments_root: str) -> list:
    """Walk EXPERIMENTS/<cell_line>/<experiment>/ and return rows with counts."""
    out: list[dict] = []
    if not os.path.isdir(experiments_root):
        return out
    for cell_line in sorted(os.listdir(experiments_root)):
        cl_dir = os.path.join(experiments_root, cell_line)
        if not os.path.isdir(cl_dir):
            continue
        for exp in sorted(os.listdir(cl_dir)):
            exp_dir = os.path.join(cl_dir, exp)
            if not os.path.isdir(exp_dir):
                continue
            frames_dir = os.path.join(exp_dir, "frames")
            # Some experiments have sub-channels (e.g. "channel 2")
            if not os.path.isdir(frames_dir):
                for sub in sorted(os.listdir(exp_dir)):
                    sub_dir = os.path.join(exp_dir, sub)
                    if os.path.isdir(sub_dir) and \
                       os.path.isdir(os.path.join(sub_dir, "frames")):
                        out.append(_exp_row(experiments_root, sub_dir))
                continue
            out.append(_exp_row(experiments_root, exp_dir))
    return out


def _exp_row(root: str, exp_dir: str) -> dict:
    frames_dir = os.path.join(exp_dir, "frames")
    masks_dir = os.path.join(exp_dir, "masks")
    # pipeline_config.yaml, or the analysis/config.txt older runs wrote
    cfg_paths = (os.path.join(exp_dir, "pipeline_config.yaml"),
                 os.path.join(exp_dir, "analysis", "config.txt"))
    n_frames = sum(
        1 for f in os.listdir(frames_dir)
        if f.endswith((".png", ".jpg"))
    ) if os.path.isdir(frames_dir) else 0
    n_masks = sum(1 for f in os.listdir(masks_dir) if f.endswith(".npy")) \
        if os.path.isdir(masks_dir) else 0
    return {
        "path": exp_dir,
        "rel": os.path.relpath(exp_dir, root),
        "frames": n_frames,
        "masks": n_masks,
        "has_config": any(os.path.isfile(p) for p in cfg_paths),
    }


def list_frames_in_dir(frames_dir: str) -> list:
    if not os.path.isdir(frames_dir):
        return []
    fs = [f for f in os.listdir(frames_dir)
          if f.endswith((".png", ".jpg"))]
    return sorted(fs, key=timepoint_sort_key)


def list_masks_in_dir(masks_dir: str) -> list:
    if not os.path.isdir(masks_dir):
        return []
    fs = [f for f in os.listdir(masks_dir) if f.endswith(".npy")]
    return sorted(fs, key=timepoint_sort_key)
