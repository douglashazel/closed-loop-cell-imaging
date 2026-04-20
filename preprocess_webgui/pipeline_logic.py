"""Pure-numpy helpers lifted from preprocess_gui.py + a phase-correlation
auto-shift that is new for the web GUI.

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


def load_segmentation(path: str) -> np.ndarray:
    seg = np.load(path, allow_pickle=True)
    if isinstance(seg, dict):
        return seg["masks"]
    try:
        return seg.item()["masks"]
    except Exception:
        return seg


# ── image → PNG helpers (lifted from PE_Pipeline/V6) ───────────────────────
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
    out = np.zeros((h, w, 4), dtype=np.uint8)
    ids = np.unique(mask)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return out
    rng = np.random.default_rng(42)
    lut = rng.integers(64, 255, size=(int(ids.max()) + 1, 3), dtype=np.uint8)
    for cid in ids:
        sel = mask == cid
        out[sel, 0:3] = lut[cid]
        out[sel, 3] = alpha
    return out


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


# ── ROI filtering (lifted from ROITab._update_display) ─────────────────────
def roi_filter(seg: np.ndarray, radius: int, y_shift: int, x_shift: int):
    h, w = seg.shape[:2]
    cx = w / 2 + x_shift
    cy = h / 2 + y_shift

    cell_ids = np.unique(seg)
    cell_ids = cell_ids[cell_ids != 0]
    if len(cell_ids) == 0:
        return np.zeros_like(seg), 0, (cx, cy)

    cxs, cys = [], []
    for cid in cell_ids:
        ys, xs = np.where(seg == cid)
        cxs.append(float(xs.mean()))
        cys.append(float(ys.mean()))
    cxs = np.array(cxs)
    cys = np.array(cys)

    inside = (cxs - cx) ** 2 + (cys - cy) ** 2 <= radius**2
    valid = cell_ids[inside]
    filtered = np.where(np.isin(seg, valid), seg, 0)
    return filtered, int(len(valid)), (cx, cy)


# ── Phase correlation (new) ────────────────────────────────────────────────
def phase_correlation_shift(img_a: np.ndarray, img_b: np.ndarray) -> tuple:
    """Return (dx, dy) such that shifting img_b by (-dx, -dy) aligns it with
    img_a. Uses straight-forward FFT cross power spectrum."""
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    if b.ndim == 3:
        b = b.mean(axis=-1)
    # Pad to same shape (they should already match)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]

    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    r = fa * np.conj(fb)
    eps = 1e-12
    r /= np.abs(r) + eps
    corr = np.fft.ifft2(r).real

    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = peak
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return int(dx), int(dy)


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
    cfg_path = os.path.join(exp_dir, "analysis", "config.txt")
    n_frames = sum(
        1 for f in os.listdir(frames_dir)
        if f.endswith((".png", ".jpg", ".tif", ".tiff"))
    ) if os.path.isdir(frames_dir) else 0
    n_masks = sum(1 for f in os.listdir(masks_dir) if f.endswith(".npy")) \
        if os.path.isdir(masks_dir) else 0
    return {
        "path": exp_dir,
        "rel": os.path.relpath(exp_dir, root),
        "frames": n_frames,
        "masks": n_masks,
        "has_config": os.path.isfile(cfg_path),
    }


def list_frames_in_dir(frames_dir: str) -> list:
    if not os.path.isdir(frames_dir):
        return []
    fs = [f for f in os.listdir(frames_dir)
          if f.endswith((".png", ".jpg", ".tif", ".tiff"))]
    return sorted(fs, key=timepoint_sort_key)


def list_masks_in_dir(masks_dir: str) -> list:
    if not os.path.isdir(masks_dir):
        return []
    fs = [f for f in os.listdir(masks_dir) if f.endswith(".npy")]
    return sorted(fs, key=timepoint_sort_key)
