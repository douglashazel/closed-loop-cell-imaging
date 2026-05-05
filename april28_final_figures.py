#!/usr/bin/env python3
"""
April 28 — final figures pipeline (script form of April28_Final_Figures.ipynb).

This script is a 1-to-1 port of the notebook. It produces the same set of figures
and writes them to disk under ``April28_preprint_results/<experiment>/`` instead
of rendering them inline.

Pipeline overview
-----------------
1. Resolve per-channel stimulus frames from monitoring logs and/or perfusion
   timestamps.
2. Build a per-frame, sampled polynomial background surface for every channel of
   each experiment, then subtract that surface at each tracked cell's
   ``(x, y)`` position to produce a corrected luminosity trace. Results are
   cached per-experiment under ``April28_preprint_results/bg_cache/`` so reruns
   are cheap.
2b. Build a per-frame ``minutes`` lookup for each (experiment, channel). For
   C2C12, minutes come from each channel's ``timestamps/...csv``
   (per-channel t=0). For NRK, minutes come from the per-frame timestamps in
   ``monitoring.log`` (per-channel t=0 = first frame log entry, including
   calibration). For NRK we additionally extract the minute at which the
   "real" (post-calibration) setpoint is activated — taken as the second
   ``Setpoint channelN: VALUE`` line in ``monitoring.log`` — and draw it as
   a labeled vertical marker on every time-resolved NRK figure.
3. Generate, for every experiment / channel (x-axis = minutes throughout):
     - a 4-column background-correction diagnostic figure
     - corrected-luminosity time traces (with stimulus markers)
     - clean per-cell corrected traces (no stimulus markers)
     - dF/F0 normalized traces
     - pairwise correlation vs. pairwise distance
     - per-stimulus peak-value violin (asymmetric: notched box on the left,
       half-violin on the right; tick labels = peak time in minutes)
     - per-stimulus response violin (baseline − peak, same layout, tick
       labels = baseline → peak time in minutes)
     - sliding-window Pearson and Spearman correlation traces
4. For the NRK acid feedback experiment only, plot the hardware feedback
   luminosity log with shaded setpoint regions, acidic-pulse markers, and the
   real-setpoint activation marker. All frames are shown — no cropping —
   because the calibration-to-real transition is now made explicit by the
   marker rather than by trimming the axis.

Two experiments are configured (see ``EXPERIMENTS`` below):
    - ``c2c12_dmso_09APR26`` — 3 channels, stim frames from perfusion timestamps
    - ``nrk_acid_13APR26``   — 4 channels, stim frames from monitoring.log

To run: ``python april28_final_figures.py`` from the project root (so that the
relative ``SCRIPTS/`` and ``EXPERIMENTS/`` paths resolve correctly).
"""

# =============================================================================
# Imports
# =============================================================================
# matplotlib backend MUST be set before pyplot is imported. ``Agg`` is a
# non-interactive backend; it never opens a window or attempts to render to
# screen, which is the whole point of running this as a script.
import matplotlib
matplotlib.use("Agg")

import json
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm
import umap

# Project-local helpers. The notebook used ``sys.path.insert(0, 'SCRIPTS')``;
# we preserve that so the script keeps working when invoked from the project
# root.
sys.path.insert(0, "SCRIPTS")
from io_utils import load_msgpack, lum_dict_to_df  # noqa: E402


# =============================================================================
# Output / cache locations
# =============================================================================
OUT_ROOT = "April28_preprint_results"
CACHE_DIR = os.path.join(OUT_ROOT, "bg_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# =============================================================================
# Pipeline toggles
# =============================================================================
# Set to True to force a rebuild of the per-experiment background cache.
# Otherwise, cached pickles in ``CACHE_DIR`` are reused. Cache is also
# invalidated automatically whenever ``BG_FIT`` differs from the cached values.
RECOMPUTE_BG = False

# Peak frame for a given stimulus is ``stim_frame + PEAK_OFFSET``.
PEAK_OFFSET = 2

# Acidic-pulse dedup window. Any acidic-media request within this many frames
# of the previously-kept pulse is treated as a duplicate (the previous pulse
# is still being delivered — pulses last ~30 seconds). Tune for your frame
# rate: at 1 Hz, 30 frames ≈ 30 s.
PULSE_DEDUP_FRAMES = 10

# Marker string in monitoring.log that identifies a stimulus event.
ACID_ACTION = "add acidic media"

# Root path for the acid-feedback PE pipeline outputs (used by the NRK
# experiment to locate monitoring.log + luminosity_log_channelN.json).
PE_PIPELINE = "/mnt/exDisk1/douglashazel/DHcode/PE_Pipeline/V5"


# =============================================================================
# Plot parameter sets
# =============================================================================
# The notebook redefines ``PLOT_PARAMS`` twice (in the sliding-correlation cell
# and the NRK hardware-log cell) because each section uses a different set of
# style keys. In script form we keep them as separate named dicts so each
# plotting function pulls only from the dict relevant to it.

# Used by: bg diagnostic, time traces, dF/F0, corr-vs-dist, violin
PLOT_PARAMS = {
    "figsize": (10, 6),
    "figsize_wide": (18, 10),
    "dpi": 300,
    "title_fontsize": 14,
    "title_fontweight": "bold",
    "axis_label_fontsize": 13,
    "legend_fontsize": 10,
    "colors": ["#e74c3c", "#363fe9", "#e67e22", "#1a9d51"],
    # Muted, low-pop palette for the corr-vs-distance scatter clouds.
    # Earth-tone desaturations: slate, taupe, sage, mauve.
    "corr_scatter_colors": ["#e4776b", "#7fb0d1", "#f0984c", "#2b8a43"],
    "corr_fit_color": "#000000",   # black trend line
    "corr_band_color": "#9a9a9a",  # gray ±3 SEM band
    "cell_color": "#074f79cc",
    "cell_alpha": 0.3,
    "cell_lw": 0.5,
    "mean_color": "#1a1a1a",
    "mean_lw": 1.8,
    "stim_color": "#e74c3c",
    "stim_lw": 1.5,
    "f0_color": "#1a9d51",
    "f0_lw": 1.5,
    "trace_cmap": "twilight_shifted",
    "bg_cmap": "viridis",
    "img_cmap": "gray",
    "roi_color": "red",
    "roi_lw": 2.0,
    "violin_face": "#a0c8f0",
    "violin_edge": "#3782d3",
    "median_color": "#1aa821",
    "mean_marker_color": "#ed0d0d",
    "scatter_color": "#222222",
    "scatter_alpha": 0.5,
    "scatter_size": 12,
    "jitter_strength": 0.08,
    "fit_color": "#363fe9",
}

# Used by: sliding-window correlation (notebook cell 14)
PLOT_PARAMS_SLIDING = {
    "figsize": (18, 7),
    "dpi": 300,
    "title_fontsize": 13,
    "title_fontweight": "bold",
    "suptitle_fontsize": 15,
    "axis_label_fontsize": 13,
    "legend_fontsize": 10,
    "window_size": 30,           # frames per sliding window
    "step": 15,                  # frames between window centers
    "global_corr_cutoff": 0.6,   # exclude pairs with full-series Pearson >= this
    "line_alpha": 0.04,          # individual pair lines
    "line_lw": 0.4,
    "mean_lw": 2.5,
    "sem_alpha": 0.18,
    "sem_n": 6,                  # number of SEMs to shade
    "pearson_color": "#0b95e5",
    "spearman_color": "#dc2846",
    "mean_color_pearson": "#003d6b",
    "mean_color_spearman": "#7a0020",
    "stim_color": "#2a8618",
    "stim_lw": 1.8,
}

# Used by: NRK hardware feedback luminosity log (notebook cell 15)
PLOT_PARAMS_HW_LOG = {
    "figsize": (12, 5),
    "dpi": 300,
    "title_fontsize": 14,
    "title_fontweight": "bold",
    "axis_label_fontsize": 12,
    "legend_fontsize": 9,
    "line_color": "steelblue",
    "line_lw": 1.4,
    "acid_color": "#c0392b",
    "acid_lw": 0.9,
    "setpoint_colors": ["#e67e22", "#1a9d51", "#9b59b6", "#3498db", "#f1c40f"],
    "setpoint_alpha": 0.22,
    "setpoint_lw": 1.2,
}


# =============================================================================
# Background-fit configuration
# =============================================================================
BG_FIT = {
    "clearance_r": 40,        # required cell-free radius around each sample point
    "sample_r": 30,           # radius used for local background sample means
    "grid_n": 25,             # NxN candidate sampling grid
    "search_r": 180,          # max drift from grid point when snapping to clear pixels
    "n_mask_snapshots": 15,   # masks per channel used to build the cell-free map
    "poly_degree": 4,         # degree of smooth 2D polynomial background fit
    "min_grid_n": 160,        # coarse grid used to find the fitted-field minimum
}


# =============================================================================
# Experiment definitions
# =============================================================================
# stim_frames accepts:
#   - list[int]              → applied to every channel
#   - dict[ch, list[int]]    → per-channel override
#
# stim_logs (optional, overrides stim_frames for that channel):
#   dict[ch, (monitoring.log path, log channel number 1|2)]
#   Frames where action == 'add acidic media' are treated as stimuli.
#
# stim_minutes (optional, used with 'timestamps'):
#   list[float] minutes from perfusion start. Each channel's stim frame is the
#   frame whose absolute datetime is nearest to ``perfusion_start + stim_min``.
#   ``perfusion_start`` defaults to the earliest frame-0 datetime across the
#   channels listed in 'timestamps' (override with 'perfusion_start').
EXPERIMENTS = {
    "c2c12_dmso_09APR26": {
        "dir": "EXPERIMENTS/other/c2c12_dmso_pulses_perfusion_09APR26",
        "channels": ["channel 1", "channel 2", "channel 3"],
        "bg_ref": "EXPERIMENTS/other/background images for subtraction method/20APR26/8.png",
        "stim_frames": [],   # auto-filled by stim_minutes/timestamps below
        "f0_frame": 0,
        # 15 min neutral, then 3 blocks of (5x: 2 min DMSO + 8 min neutral) separated
        # by 30 min neutral. The first 15 min is treated as a single neutral block
        # (the priming sub-step is invisible at the cell side).
        "stim_minutes": [
            15, 25, 35, 45, 55,
            95, 105, 115, 125, 135,
            175, 185, 195, 205, 215,
        ],
        "stim_duration_minutes": 2.0,
        "stim_label": "DMSO pulse (2 min)",
        # DMSO drives luminosity *up* — the response is the post-stim maximum.
        "response_direction": "increase",
        "response_window": (1, 8),  # frames after stim: search for extremum in [stim+1, stim+8)
        "timestamps": {
            "channel 1": "timestamps/C2C12 DMSO perfusion 09APR26 channel 1 timestamps.csv",
            "channel 2": "timestamps/C2C12 DMSO perfusion 09APR26 channel 2 timestamps.csv",
            "channel 3": "timestamps/C2C12 DMSO perfusion 09APR26 channel 3 timestamps.csv",
        },
        # Optional: 'perfusion_start': '09-Apr-2026 14:08:16'
        # Default = earliest frame-0 datetime across channels listed in 'timestamps'.
    },
    "nrk_acid_13APR26": {
        "dir": "EXPERIMENTS/other/nrk_acid_feedback_experiment_13APR26",
        "channels": ["channel 1 A", "channel 1 C", "channel 2 B", "channel 2 D"],
        "bg_ref": "EXPERIMENTS/other/background images for subtraction method/20APR26/8.png",
        "stim_frames": [],   # auto-filled by stim_logs below
        "f0_frame": 0,
        # Acid pulse delivery window is ~30 s.
        "stim_duration_minutes": 0.5,
        "stim_label": "Acid pulse (30 s)",
        # Acid drives luminosity *down* — the response is the post-stim minimum.
        "response_direction": "decrease",
        "response_window": (1, 8),
        "stim_logs": {
            "channel 1 A": (f"{PE_PIPELINE}/resultsApril13_exp2_channel1A_channel2B/monitoring.log", 1),
            "channel 2 B": (f"{PE_PIPELINE}/resultsApril13_exp2_channel1A_channel2B/monitoring.log", 2),
            "channel 1 C": (f"{PE_PIPELINE}/resultsApril13_exp3_channel1C_channel2D/monitoring.log", 1),
            "channel 2 D": (f"{PE_PIPELINE}/resultsApril13_exp3_channel1C_channel2D/monitoring.log", 2),
        },
    },
}


# =============================================================================
# Path / I/O utilities
# =============================================================================
def _slug(s):
    """Convert ``s`` into a filesystem-safe slug by collapsing whitespace to ``_``."""
    return re.sub(r"\s+", "_", s.strip())


def fig_path(exp_name, name, ext="png"):
    """Return the save path ``April28_preprint_results/<exp_name>/<name>.<ext>``.

    The per-experiment subdirectory is created on first use, so each
    experiment's figures land in their own folder.
    """
    out_dir = os.path.join(OUT_ROOT, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{_slug(name)}.{ext}")


def load_segmentation(path):
    """Load a Cellpose / mask ``.npy`` file as a 2-D mask array.

    Handles both the bare-array format and the dict-wrapped
    ``{'masks': <array>, ...}`` format (Cellpose's default save layout).
    """
    seg = np.load(path, allow_pickle=True)
    if isinstance(seg, np.ndarray) and seg.dtype == object:
        try:
            seg = seg.item()["masks"]
        except Exception:
            pass
    return np.asarray(seg)


def sorted_image_files(frame_dir):
    """Return the sorted list of ``.png``/``.jpg`` filenames in ``frame_dir``."""
    return sorted([f for f in os.listdir(frame_dir) if f.endswith((".png", ".jpg"))])


# =============================================================================
# Background-fit primitives
# =============================================================================
def poly_design(xn, yn, degree):
    """Build a 2-D polynomial design matrix up to total degree ``degree``.

    Columns are ``x^i * y^j`` for all ``i + j <= degree``. ``xn`` / ``yn`` are
    expected to already be normalised to ``[-1, 1]``.
    """
    xn = np.asarray(xn, dtype=np.float64)
    yn = np.asarray(yn, dtype=np.float64)
    cols = [
        (xn ** i) * (yn ** j)
        for i in range(degree + 1)
        for j in range(degree + 1 - i)
    ]
    return np.stack(cols, axis=-1)


def build_background_sampler(cfg, fit_cfg=BG_FIT):
    """Build a reusable background sampler for one experiment.

    The sampler unions cell masks across channels (sampled at
    ``fit_cfg['n_mask_snapshots']`` time points per channel) to find the pixels
    that are *never* part of a cell, then snaps a coarse grid of candidate
    sample points onto those pixels, requiring at least ``clearance_r`` of
    cell-free space around each one.

    Returns a dict containing:
        shape          - (H, W) of the frame
        sample_points  - Nx2 array of (x, y) sample-point coords
        A_pinv         - pseudoinverse of the design matrix at the sample points
        A_min          - design matrix on a coarse grid for finding the fit minimum
        patch_specs    - precomputed circular-patch slicers for each sample point
        degree         - polynomial degree
        clearance      - HxW distance-to-nearest-cell-pixel map (debug aid)
    """
    first_dir = os.path.join(cfg["dir"], cfg["channels"][0], "frames")
    first_files = sorted_image_files(first_dir)
    H, W = np.array(Image.open(os.path.join(first_dir, first_files[0])).convert("L")).shape

    # Union of all cell masks across channels and a sampled subset of frames.
    union = np.zeros((H, W), dtype=bool)
    for ch in cfg["channels"]:
        mdir = os.path.join(cfg["dir"], ch, "masks")
        mfiles = sorted([f for f in os.listdir(mdir) if f.endswith(".npy")])
        if not mfiles:
            continue
        idxs = np.unique(
            np.linspace(
                0,
                len(mfiles) - 1,
                min(fit_cfg["n_mask_snapshots"], len(mfiles)),
                dtype=int,
            )
        )
        for i in idxs:
            mask = load_segmentation(os.path.join(mdir, mfiles[i])) > 0
            if mask.shape != (H, W):
                raise ValueError(
                    f"{ch} mask shape {mask.shape} does not match frame shape {(H, W)}"
                )
            union |= mask

    # Distance from each pixel to the nearest cell pixel.
    clearance = distance_transform_edt(~union)

    # Snap each grid point onto the most cell-free spot within ``search_r``.
    margin = fit_cfg["sample_r"]
    search_r = fit_cfg["search_r"]
    sample_points = []
    for gy in np.linspace(margin, H - margin - 1, fit_cfg["grid_n"]).astype(int):
        for gx in np.linspace(margin, W - margin - 1, fit_cfg["grid_n"]).astype(int):
            y0, y1 = max(margin, gy - search_r), min(H - margin, gy + search_r + 1)
            x0, x1 = max(margin, gx - search_r), min(W - margin, gx + search_r + 1)
            sub = clearance[y0:y1, x0:x1]
            idx = np.unravel_index(sub.argmax(), sub.shape)
            if sub[idx] >= fit_cfg["clearance_r"]:
                sample_points.append((x0 + idx[1], y0 + idx[0]))
    sample_points = np.asarray(sample_points, dtype=int)

    n_terms = (fit_cfg["poly_degree"] + 1) * (fit_cfg["poly_degree"] + 2) // 2
    if len(sample_points) < n_terms:
        raise ValueError(
            f"Only {len(sample_points)} cell-free background samples found; "
            f"need at least {n_terms}."
        )

    # Pseudoinverse for least-squares fit at the sample points.
    xn_samples = (sample_points[:, 0] - W / 2) / (W / 2)
    yn_samples = (sample_points[:, 1] - H / 2) / (H / 2)
    A_samples = poly_design(xn_samples, yn_samples, fit_cfg["poly_degree"])
    A_pinv = np.linalg.pinv(A_samples)

    # Design matrix on a coarse evaluation grid for finding the fit minimum
    # (used to subtract off a global offset so dF/F0 stays well-conditioned).
    g = fit_cfg["min_grid_n"]
    gx = np.linspace(-1, 1, g)
    gy = np.linspace(-1, 1, g)
    GX, GY = np.meshgrid(gx, gy)
    A_min = poly_design(GX.ravel(), GY.ravel(), fit_cfg["poly_degree"])

    # Precompute the circular-patch slicers for each sample point so we don't
    # rebuild them every frame.
    r = fit_cfg["sample_r"]
    patch_specs = []
    for sx, sy in sample_points:
        y0, y1 = max(0, sy - r), min(H, sy + r + 1)
        x0, x1 = max(0, sx - r), min(W, sx + r + 1)
        py = np.arange(y0, y1)[:, None]
        px = np.arange(x0, x1)[None, :]
        cmask = (py - sy) ** 2 + (px - sx) ** 2 <= r ** 2
        patch_specs.append((y0, y1, x0, x1, cmask))

    return {
        "shape": (H, W),
        "sample_points": sample_points,
        "A_pinv": A_pinv,
        "A_min": A_min,
        "patch_specs": patch_specs,
        "degree": fit_cfg["poly_degree"],
        "clearance": clearance,
    }


def sample_background_means(img, sampler):
    """Return the mean intensity inside each cell-free sample patch on ``img``."""
    means = np.empty(len(sampler["patch_specs"]), dtype=np.float64)
    for pi, (y0, y1, x0, x1, cmask) in enumerate(sampler["patch_specs"]):
        means[pi] = img[y0:y1, x0:x1][cmask].mean()
    return means


def fit_background_from_image(img, sampler):
    """Fit the polynomial background to one frame.

    Returns ``(coefs, bg_min, means)`` where ``bg_min`` is the minimum of the
    fitted surface over a coarse grid (subtracted later so the corrected image
    sits at zero baseline).
    """
    means = sample_background_means(img, sampler)
    coefs = sampler["A_pinv"] @ means
    bg_min = float((sampler["A_min"] @ coefs).min())
    return coefs, bg_min, means


def eval_background_at_points(coefs, bg_min, xs, ys, shape, degree):
    """Evaluate the fitted background surface at arbitrary ``(x, y)`` pixels."""
    H, W = shape
    xn = (np.asarray(xs, dtype=np.float64) - W / 2) / (W / 2)
    yn = (np.asarray(ys, dtype=np.float64) - H / 2) / (H / 2)
    return poly_design(xn, yn, degree) @ coefs - bg_min


def eval_background_image(coefs, bg_min, shape, degree):
    """Render the fitted background as a full HxW image."""
    H, W = shape
    xn = (np.arange(W, dtype=np.float32) - W / 2) / (W / 2)
    yn = (np.arange(H, dtype=np.float32) - H / 2) / (H / 2)
    XF, YF = np.meshgrid(xn, yn)
    bg = np.zeros((H, W), dtype=np.float32)
    k = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            bg += coefs[k] * (XF ** i) * (YF ** j)
            k += 1
    return bg - bg_min


# =============================================================================
# Stim-frame resolution
# =============================================================================
def _dedup_close_frames(frames, min_gap):
    """Drop frames within ``min_gap`` of the previously kept frame.

    Greedy left-to-right pass: keep the first frame, then keep each subsequent
    frame only if it is more than ``min_gap`` frames after the most recently
    kept frame. Used for acidic-pulse events: a pulse takes ~30 seconds to
    deliver, so any later request within the delivery window is the controller
    re-asking for a pulse already in progress, not a new one.

    Examples (with min_gap=30):
        [100, 101, 102, 105]            -> [100]
        [100, 101, 105, 200, 201]       -> [100, 200]
        [100, 130, 161]                 -> [100, 161]   (130 is exactly at the boundary)
    """
    if not frames:
        return []
    sorted_frames = sorted(set(frames))
    result = [sorted_frames[0]]
    for f in sorted_frames[1:]:
        if f - result[-1] > min_gap:
            result.append(f)
    return result


def parse_stim_frames_from_log(log_path, channel_num, action=ACID_ACTION):
    """Return frame indices for ``action`` events on ``channel_num``.

    Looks for monitoring.log lines matching::

        [ts] NNNNN_channelM: value -> <action> (...)

    Pulses within ``PULSE_DEDUP_FRAMES`` of the previous kept pulse are
    dropped (see :func:`_dedup_close_frames`).
    """
    pat = re.compile(
        rf"(\d{{5}})_channel{channel_num}\s*:\s*[0-9.+\-eE]+\s*->\s*{re.escape(action)}"
    )
    frames = set()
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                frames.add(int(m.group(1)))
    return _dedup_close_frames(frames, PULSE_DEDUP_FRAMES)


def parse_monitor_log_frame_times(log_path, channel_num):
    """Return ``[(frame_idx, datetime), ...]`` per frame entry for ``channel_num``.

    Parses ``monitoring.log`` lines of the form::

        [YYYY-MM-DD HH:MM:SS] NNNNN_channelM: <value> -> <decision> ...

    Retry / housekeeping lines (which do not start with ``NNNNN_channel{M}:``)
    are ignored. Output is sorted ascending by frame index.
    """
    pat = re.compile(
        rf"^\[(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}})\]\s+"
        rf"(\d{{5}})_channel{channel_num}\s*:"
    )
    entries = []
    with open(log_path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                ts = pd.to_datetime(m.group(1), format="%Y-%m-%d %H:%M:%S")
                frame = int(m.group(2))
                entries.append((frame, ts))
    entries.sort(key=lambda t: t[0])
    return entries


def parse_monitor_log_setpoint_events(log_path, channel_num):
    """Return ``[(datetime, value), ...]`` for ``Setpoint channelN: VALUE`` lines.

    The first event is the calibration setpoint (deliberately set very high to
    discover the working setpoint). The second event marks the activation of
    the real working setpoint — i.e. the start of the actual experiment.
    Output is sorted ascending by datetime.
    """
    pat = re.compile(
        rf"^\[(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}})\]\s+"
        rf"Setpoint channel{channel_num}\s*:\s*([0-9.+\-eE]+)"
    )
    events = []
    with open(log_path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                ts = pd.to_datetime(m.group(1), format="%Y-%m-%d %H:%M:%S")
                events.append((ts, float(m.group(2))))
    events.sort(key=lambda t: t[0])
    return events


def parse_stim_frames_from_timestamps(ts_path, stim_minutes, perfusion_start_dt):
    """Map a list of minutes-since-perfusion-start to frame indices.

    ``ts_path`` is a 3-column CSV: filename, ``DD-Mon-YYYY HH:MM:SS``,
    minutes_from_frame_0. For each minute in ``stim_minutes`` we return the
    frame whose absolute datetime is nearest to
    ``perfusion_start_dt + stim_min``.
    """
    df = pd.read_csv(
        ts_path, header=None, names=["filename", "datetime", "minutes"]
    )
    dts = pd.to_datetime(df["datetime"].str.strip(), format="%d-%b-%Y %H:%M:%S")
    frames = []
    for sm in stim_minutes:
        target = perfusion_start_dt + pd.Timedelta(minutes=float(sm))
        idx = int((dts - target).abs().idxmin())
        frames.append(idx)
    return frames


def resolve_all_stim_frames(experiments):
    """Mutate ``experiments`` in place so each ``cfg['stim_frames']`` is a dict.

    After this step ``cfg['stim_frames']`` is always
    ``dict[channel_name, list[int]]``.

    Precedence (highest to lowest):
        1. ``stim_logs[ch]`` (parsed monitoring.log)
        2. ``stim_minutes`` resolved via ``timestamps``
        3. dict ``stim_frames[ch]``
        4. list ``stim_frames`` broadcast to all channels
    """
    for exp_name, cfg in experiments.items():
        base = cfg.get("stim_frames", [])
        if isinstance(base, dict):
            resolved = {ch: list(base.get(ch, [])) for ch in cfg["channels"]}
        else:
            resolved = {ch: list(base) for ch in cfg["channels"]}

        # stim_minutes + timestamps → per-channel frame indices (used for
        # ONIX-driven perfusions where there is no monitoring.log).
        if "stim_minutes" in cfg and "timestamps" in cfg:
            ts_paths = {
                ch: os.path.join(cfg["dir"], rel)
                for ch, rel in cfg["timestamps"].items()
            }
            if "perfusion_start" in cfg:
                perfusion_start = pd.to_datetime(
                    cfg["perfusion_start"], format="%d-%b-%Y %H:%M:%S"
                )
            else:
                first_ts = []
                for p in ts_paths.values():
                    first = pd.read_csv(
                        p,
                        header=None,
                        nrows=1,
                        names=["filename", "datetime", "minutes"],
                    )
                    first_ts.append(
                        pd.to_datetime(
                            first["datetime"].iloc[0].strip(),
                            format="%d-%b-%Y %H:%M:%S",
                        )
                    )
                perfusion_start = min(first_ts)
            for ch, p in ts_paths.items():
                if ch not in cfg["channels"]:
                    print(f"  warning: timestamps references unknown channel '{ch}'")
                    continue
                frames = parse_stim_frames_from_timestamps(
                    p, cfg["stim_minutes"], perfusion_start
                )
                resolved[ch] = frames
                print(
                    f"  {exp_name} / {ch}: resolved {len(frames)} stims from "
                    f"timestamps (perfusion_start={perfusion_start}) → {frames}"
                )

        for ch, (log_path, ch_num) in cfg.get("stim_logs", {}).items():
            if ch not in cfg["channels"]:
                print(f"  warning: stim_logs references unknown channel '{ch}'")
                continue
            frames = parse_stim_frames_from_log(log_path, ch_num)
            resolved[ch] = frames
            print(
                f"  {exp_name} / {ch}: parsed {len(frames)} stims from "
                f"{os.path.basename(os.path.dirname(log_path))}/monitoring.log "
                f"(log channel {ch_num})"
            )

        cfg["stim_frames"] = resolved

    print()
    for exp_name, cfg in experiments.items():
        for ch in cfg["channels"]:
            sf = cfg["stim_frames"][ch]
            preview = f"{sf[:3]}...{sf[-3:]}" if len(sf) > 6 else str(sf)
            print(
                f"  {exp_name:25s} / {ch:12s} → {len(sf):3d} stims  {preview}"
            )


# =============================================================================
# Pipeline step 1 — Background correction
# =============================================================================
def compute_background_correction(experiments, recompute=RECOMPUTE_BG):
    """Run per-frame sampled polynomial background correction for every experiment.

    For each experiment / channel:
        * fit the polynomial background to every frame
        * subtract the fitted background at each tracked cell's (x, y)
        * store everything keyed by ``[exp_name][channel]`` in the returned
          state dict

    Caches per-experiment results to ``CACHE_DIR/<exp>.pkl`` and reuses them
    unless ``recompute`` is True or ``BG_FIT`` has changed since the cache
    was written.

    Returns a ``state`` dict with these top-level keys, each one a
    ``{exp_name: {channel: ...}}`` nested dict (or its analog):
        corrected_lum, bg_trace, traj_by_channel, frame_counts, probe_data,
        bg_map_by_ch, bg_coefs_by_ch, bg_min_by_ch, bg_sampler_by_exp,
        bg_sample_points_exp
    """
    state = {
        "corrected_lum": {},
        "bg_trace": {},
        "traj_by_channel": {},
        "frame_counts": {},
        "probe_data": {},
        "bg_map_by_ch": {},
        "bg_sampler_by_exp": {},
        "bg_coefs_by_ch": {},
        "bg_min_by_ch": {},
        "bg_sample_points_exp": {},
    }

    for exp_name, cfg in experiments.items():
        cache_path = os.path.join(CACHE_DIR, f"{exp_name}.pkl")

        # Try to reuse cache.
        if not recompute and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("BG_FIT") != BG_FIT:
                print(
                    f"=== {exp_name} === cached BG_FIT differs from current — recomputing."
                )
            else:
                state["corrected_lum"][exp_name] = blob["corrected_lum"]
                state["bg_trace"][exp_name] = blob["bg_trace"]
                state["traj_by_channel"][exp_name] = blob["traj_by_channel"]
                state["frame_counts"][exp_name] = blob["frame_counts"]
                state["probe_data"][exp_name] = blob["probe_data"]
                state["bg_map_by_ch"][exp_name] = blob["bg_map_by_ch"]
                state["bg_coefs_by_ch"][exp_name] = blob["bg_coefs_by_ch"]
                state["bg_min_by_ch"][exp_name] = blob["bg_min_by_ch"]
                state["bg_sampler_by_exp"][exp_name] = blob["sampler"]
                state["bg_sample_points_exp"][exp_name] = blob["sampler"]["sample_points"]
                print(
                    f"=== {exp_name} === loaded cache from {cache_path} "
                    f"({len(blob['corrected_lum'])} channels)"
                )
                continue

        # No usable cache — run the full computation.
        state["corrected_lum"][exp_name] = {}
        state["bg_trace"][exp_name] = {}
        state["traj_by_channel"][exp_name] = {}
        state["frame_counts"][exp_name] = {}
        state["probe_data"][exp_name] = {}
        state["bg_map_by_ch"][exp_name] = {}
        state["bg_coefs_by_ch"][exp_name] = {}
        state["bg_min_by_ch"][exp_name] = {}

        sampler = build_background_sampler(cfg)
        state["bg_sampler_by_exp"][exp_name] = sampler
        state["bg_sample_points_exp"][exp_name] = sampler["sample_points"]
        H, W = sampler["shape"]
        print()
        print(
            f"=== {exp_name} === {len(sampler['sample_points'])} / "
            f"{BG_FIT['grid_n']**2} grid points snapped to cell-free background"
        )

        for ch in cfg["channels"]:
            fdir = os.path.join(cfg["dir"], ch, "frames")
            adir = os.path.join(cfg["dir"], ch, "analysis")
            ffiles = sorted_image_files(fdir)
            n_frames = len(ffiles)

            coefs_by_frame = np.zeros(
                (n_frames, sampler["A_pinv"].shape[0]), dtype=np.float64
            )
            bg_min_by_frame = np.zeros(n_frames, dtype=np.float64)
            sampled_bg_mean = np.zeros(n_frames, dtype=np.float32)
            sampled_bg_range = np.zeros((n_frames, 2), dtype=np.float32)

            # We snapshot one mid-experiment frame as a "probe" for the
            # diagnostic figure later.
            probe_idx = n_frames // 2
            probe = None
            probe_bg = None
            probe_means = None

            for i, fname in enumerate(
                tqdm(ffiles, desc=f"{exp_name} / {ch} sampled bg", leave=False)
            ):
                img = np.array(
                    Image.open(os.path.join(fdir, fname)).convert("L"),
                    dtype=np.float32,
                )
                if img.shape != (H, W):
                    raise ValueError(
                        f"{exp_name} / {ch} frame {fname} shape {img.shape} "
                        f"does not match sampler shape {(H, W)}"
                    )

                coefs, bg_min, means = fit_background_from_image(img, sampler)
                coefs_by_frame[i] = coefs
                bg_min_by_frame[i] = bg_min
                sampled_bg_mean[i] = float(means.mean())
                sampled_bg_range[i] = (float(means.min()), float(means.max()))

                if i == probe_idx:
                    probe = img
                    probe_means = means
                    probe_bg = eval_background_image(
                        coefs, bg_min, (H, W), sampler["degree"]
                    )

            # Per-cell luminosity and trajectory data, written out by the
            # upstream tracking pipeline.
            lum = load_msgpack(os.path.join(adir, "luminosity_complete.json"))
            traj = load_msgpack(os.path.join(adir, "trajectories_complete.json"))

            corr_ch = {}
            for cid, frames in lum.items():
                coords = traj.get(cid, {})
                c_corr = {}
                for fk, v in frames.items():
                    if v is None or not fk.startswith("f"):
                        continue
                    idx = int(fk[1:])
                    if idx >= n_frames:
                        continue
                    x = coords.get(f"x{idx}")
                    y = coords.get(f"y{idx}")
                    if x is None or y is None:
                        continue
                    try:
                        xi = float(x)
                        yi = float(y)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= yi < H and 0 <= xi < W:
                        bg_val = eval_background_at_points(
                            coefs_by_frame[idx],
                            bg_min_by_frame[idx],
                            [xi],
                            [yi],
                            (H, W),
                            sampler["degree"],
                        )[0]
                        c_corr[fk] = float(v) - float(bg_val)
                if c_corr:
                    corr_ch[cid] = c_corr

            state["corrected_lum"][exp_name][ch] = corr_ch
            state["bg_trace"][exp_name][ch] = sampled_bg_mean
            state["traj_by_channel"][exp_name][ch] = traj
            state["frame_counts"][exp_name][ch] = n_frames
            state["bg_map_by_ch"][exp_name][ch] = probe_bg
            state["bg_coefs_by_ch"][exp_name][ch] = coefs_by_frame
            state["bg_min_by_ch"][exp_name][ch] = bg_min_by_frame

            state["probe_data"][exp_name][ch] = {
                "probe": probe,
                "probe_idx": probe_idx,
                "probe_bg": probe_bg,
                "corrected": probe - probe_bg,
                "probe_sample_mean": float(sampled_bg_mean[probe_idx]),
                "probe_sample_min": float(sampled_bg_range[probe_idx, 0]),
                "probe_sample_max": float(sampled_bg_range[probe_idx, 1]),
                "probe_bg_min": float(np.nanmin(probe_bg)),
                "probe_bg_max": float(np.nanmax(probe_bg)),
                "probe_means": probe_means,
            }
            print(
                f"  {ch}: {n_frames} frames | sample mean range "
                f"[{sampled_bg_mean.min():.1f}, {sampled_bg_mean.max():.1f}] | "
                f"probe bg max={np.nanmax(probe_bg):.2f} | "
                f"{len(corr_ch)} cells corrected"
            )

        # Persist this experiment's state to disk.
        blob = {
            "BG_FIT": BG_FIT.copy(),
            "sampler": sampler,
            "corrected_lum": state["corrected_lum"][exp_name],
            "bg_trace": state["bg_trace"][exp_name],
            "traj_by_channel": state["traj_by_channel"][exp_name],
            "frame_counts": state["frame_counts"][exp_name],
            "probe_data": state["probe_data"][exp_name],
            "bg_map_by_ch": state["bg_map_by_ch"][exp_name],
            "bg_coefs_by_ch": state["bg_coefs_by_ch"][exp_name],
            "bg_min_by_ch": state["bg_min_by_ch"][exp_name],
        }
        with open(cache_path, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  → cached to {cache_path}")

    print()
    print(
        "In-memory state ready: corrected_lum[exp][ch], bg_trace[exp][ch], "
        "traj_by_channel[exp][ch], frame_counts[exp][ch]."
    )
    return state


# =============================================================================
# Pipeline step 1b — Per-frame minutes & NRK real-setpoint anchor
# =============================================================================
def build_frame_to_minutes_lookups(experiments, state):
    """Populate ``state['frame_minutes_src']`` and ``state['real_setpoint_min']``.

    For every (experiment, channel) we store the raw ``(known_frames,
    known_minutes)`` source arrays used to interpolate any frame index to
    minutes-from-t0 via :func:`frames_to_min`. Anchors:

        * C2C12: each channel's own frame 0 (read directly from
          ``timestamps/...csv`` -> ``minutes_from_frame_0`` column).
        * NRK:   each channel's first ``NNNNN_channelN:`` log line in
          ``monitoring.log`` (i.e. monitor-start / experiment-start, including
          the calibration period).

    For NRK channels we additionally compute the minute at which the real
    (post-calibration) setpoint is activated, taken as the *second*
    ``Setpoint channelN: VALUE`` line in ``monitoring.log``. If only one such
    line is found (calibration only, no real setpoint yet), the value is
    ``None`` and downstream plots simply don't draw the marker.

    Storing the raw (frames, minutes) source — rather than a precomputed
    per-frame array — lets ``frames_to_min`` handle arbitrary frame indices
    (e.g. luminosity-JSON frames that extend past the cellpose tracking
    range) via ``np.interp`` instead of array indexing.
    """
    state["frame_minutes_src"] = {}
    state["real_setpoint_min"] = {}

    for exp_name, cfg in experiments.items():
        state["frame_minutes_src"][exp_name] = {}
        state["real_setpoint_min"][exp_name] = {}

        for ch in cfg["channels"]:
            n_frames = state["frame_counts"][exp_name][ch]

            # ---- NRK: monitor.log -----------------------------------------
            if "stim_logs" in cfg and ch in cfg["stim_logs"]:
                log_path, ch_num = cfg["stim_logs"][ch]
                entries = parse_monitor_log_frame_times(log_path, ch_num)
                if not entries:
                    raise RuntimeError(
                        f"No frame log entries for {exp_name}/{ch} "
                        f"(channel {ch_num}) in {log_path}"
                    )
                t0 = entries[0][1]
                known_frames = np.asarray([e[0] for e in entries], dtype=float)
                known_minutes = np.asarray(
                    [(e[1] - t0).total_seconds() / 60.0 for e in entries],
                    dtype=float,
                )
                state["frame_minutes_src"][exp_name][ch] = (
                    known_frames, known_minutes,
                )

                sp_events = parse_monitor_log_setpoint_events(log_path, ch_num)
                if len(sp_events) >= 2:
                    real_sp_ts = sp_events[1][0]
                    state["real_setpoint_min"][exp_name][ch] = (
                        (real_sp_ts - t0).total_seconds() / 60.0
                    )
                    print(
                        f"  {exp_name} / {ch}: real setpoint @ "
                        f"{state['real_setpoint_min'][exp_name][ch]:.2f} min "
                        f"({len(sp_events)} setpoint events in log)"
                    )
                else:
                    state["real_setpoint_min"][exp_name][ch] = None
                    print(
                        f"  {exp_name} / {ch}: only {len(sp_events)} setpoint "
                        f"event(s) in monitor.log — no real-setpoint marker"
                    )

            # ---- C2C12: timestamps.csv ------------------------------------
            elif "timestamps" in cfg and ch in cfg["timestamps"]:
                ts_path = os.path.join(cfg["dir"], cfg["timestamps"][ch])
                df = pd.read_csv(
                    ts_path, header=None,
                    names=["filename", "datetime", "minutes"],
                )
                minutes = df["minutes"].astype(float).values
                known_frames = np.arange(len(minutes), dtype=float)
                state["frame_minutes_src"][exp_name][ch] = (
                    known_frames, minutes,
                )
                state["real_setpoint_min"][exp_name][ch] = None

            # ---- Fallback: identity (frame == "minutes") ------------------
            else:
                print(
                    f"  WARNING: {exp_name}/{ch} has neither stim_logs nor "
                    f"timestamps — using frame index as the minutes axis"
                )
                idx = np.arange(n_frames, dtype=float)
                state["frame_minutes_src"][exp_name][ch] = (idx, idx.copy())
                state["real_setpoint_min"][exp_name][ch] = None

    print()


def frames_to_min(state, exp_name, ch, frames):
    """Look up minutes-from-t0 for an iterable / array of frame indices.

    Uses ``np.interp`` against the raw (known_frames, known_minutes) source
    stored in ``state['frame_minutes_src']``. ``np.interp`` clamps to the
    boundary values for indices outside the known range, which is the right
    behaviour for our use case (interpolation between the first and last log
    entries, flat extrapolation outside).
    """
    known_frames, known_minutes = state["frame_minutes_src"][exp_name][ch]
    fi = np.asarray(frames, dtype=float)
    return np.interp(fi, known_frames, known_minutes)


def stim_spans_min(state, exp_name, ch, cfg):
    """Return ``(spans, label)`` for the channel's stimulus shaded blocks.

    ``spans`` is a list of ``(start_min, end_min)`` tuples, one per stimulus,
    with width set by ``cfg['stim_duration_minutes']`` (defaults to a thin
    1-pixel-equivalent if missing). ``label`` is a legend-ready string from
    ``cfg.get('stim_label')``, or a sensible default.
    """
    stim_frames = cfg["stim_frames"][ch]
    if not stim_frames:
        return [], cfg.get("stim_label", "Stimulus")
    duration = float(cfg.get("stim_duration_minutes", 0.0) or 0.0)
    starts = frames_to_min(state, exp_name, ch, stim_frames)
    spans = [(float(s), float(s) + duration) for s in starts]
    return spans, cfg.get("stim_label", "Stimulus")


def draw_stim_spans(ax, spans, label, color, alpha=0.18):
    """Shade each ``(start, end)`` span on ``ax``; label only the first."""
    for idx, (start_m, end_m) in enumerate(spans):
        ax.axvspan(
            start_m, end_m,
            color=color, alpha=alpha,
            linewidth=0, zorder=0,
            label=label if idx == 0 else None,
        )


def per_cell_response_delta(values_by_col, stim_col, direction, window):
    """Return per-cell ``response_value − baseline`` for one stimulus.

    ``values_by_col`` is a 2-D array shaped ``(n_cells, n_cols)`` where each
    column is one frame in chronological order (caller supplies a contiguous
    frame matrix, e.g. from ``lum_dict_to_df`` sorted by frame number).
    ``stim_col`` is the column index of the stimulus onset frame.
    ``direction`` is ``"increase"`` (DMSO-like) or ``"decrease"`` (acid-like).
    ``window`` is ``(lo, hi)`` — column offsets, half-open. The response
    value is the per-cell max (increase) or min (decrease) over
    ``[stim_col + lo, stim_col + hi)``.

    Returns a 1-D array of deltas (sign preserved: positive for increase
    responses, negative for decrease responses). Cells with no valid samples
    in the window get NaN.
    """
    n_cells, n_cols = values_by_col.shape
    lo, hi = window
    if stim_col < 0 or stim_col >= n_cols:
        return np.full(n_cells, np.nan)
    base = values_by_col[:, stim_col]
    win_lo = max(0, stim_col + lo)
    win_hi = min(n_cols, stim_col + hi)
    if win_lo >= win_hi:
        return np.full(n_cells, np.nan)
    win = values_by_col[:, win_lo:win_hi]
    if direction == "decrease":
        extremum = np.nanmin(win, axis=1)
    else:
        extremum = np.nanmax(win, axis=1)
    return extremum - base


# =============================================================================
# Pipeline step 2 — Background diagnostic figure
# =============================================================================
def plot_bg_diagnostic(experiments, state):
    """Per-experiment diagnostic: rows=channels, cols=4 background-fit views.

    Columns: probe frame + sample points, fitted background, corrected probe,
    sampled-background-mean trace over time.
    """
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]
        sample_points = state["bg_sample_points_exp"][exp_name]

        fig, axes = plt.subplots(
            len(channels), 4,
            figsize=(22, 5 * len(channels)),
            dpi=PLOT_PARAMS["dpi"],
        )
        if len(channels) == 1:
            axes = np.array([axes])

        col_titles = [
            "probe frame + cell-free samples",
            "fitted probe background",
            "corrected = probe - fitted bg",
            "per-frame sampled background mean",
        ]

        for row, ch in enumerate(channels):
            pd_ = state["probe_data"][exp_name][ch]
            bg = state["bg_map_by_ch"][exp_name][ch]
            trace = state["bg_trace"][exp_name][ch]

            # Col 0: probe frame with sample points overlaid.
            ax = axes[row, 0]
            ax.imshow(pd_["probe"], cmap=PLOT_PARAMS["img_cmap"], vmin=0, vmax=255)
            ax.scatter(
                sample_points[:, 0], sample_points[:, 1],
                s=2, c=PLOT_PARAMS["roi_color"], alpha=0.6, linewidths=0,
            )
            ax.annotate(
                f"frame {pd_['probe_idx']}\nsample mean = {pd_['probe_sample_mean']:.1f}",
                xy=(0.02, 0.98), xycoords="axes fraction", va="top", ha="left",
                color=PLOT_PARAMS["roi_color"], fontweight="bold",
            )
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(
                ch,
                fontsize=PLOT_PARAMS["axis_label_fontsize"],
                fontweight="bold",
            )
            if row == 0:
                ax.set_title(
                    col_titles[0],
                    fontsize=PLOT_PARAMS["title_fontsize"],
                    fontweight=PLOT_PARAMS["title_fontweight"],
                )

            # Col 1: fitted background surface.
            ax = axes[row, 1]
            im = ax.imshow(bg, cmap=PLOT_PARAMS["bg_cmap"])
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.annotate(
                f"bg range {pd_['probe_bg_min']:.1f}-{pd_['probe_bg_max']:.1f}",
                xy=(0.02, 0.98), xycoords="axes fraction", va="top", ha="left",
                color="white", fontweight="bold",
            )
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    col_titles[1],
                    fontsize=PLOT_PARAMS["title_fontsize"],
                    fontweight=PLOT_PARAMS["title_fontweight"],
                )

            # Col 2: corrected probe.
            ax = axes[row, 2]
            ax.imshow(pd_["corrected"], cmap=PLOT_PARAMS["img_cmap"], vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    col_titles[2],
                    fontsize=PLOT_PARAMS["title_fontsize"],
                    fontweight=PLOT_PARAMS["title_fontweight"],
                )

            # Col 3: sampled-bg-mean trace.
            ax = axes[row, 3]
            ax.spines[["top", "right"]].set_visible(False)
            mins = frames_to_min(state, exp_name, ch, np.arange(len(trace)))
            ax.plot(
                mins, trace,
                color=PLOT_PARAMS["colors"][1], linewidth=1.2,
                label="sample mean",
            )
            ax.axvline(
                frames_to_min(state, exp_name, ch, [pd_["probe_idx"]])[0],
                color=PLOT_PARAMS["colors"][3], linestyle="--",
                linewidth=PLOT_PARAMS["f0_lw"],
                label=f"probe frame {pd_['probe_idx']}",
            )
            ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax.set_ylabel(
                "Sampled bg mean (px intensity)",
                fontsize=PLOT_PARAMS["axis_label_fontsize"],
            )
            ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
            if row == 0:
                ax.set_title(
                    col_titles[3],
                    fontsize=PLOT_PARAMS["title_fontsize"],
                    fontweight=PLOT_PARAMS["title_fontweight"],
                )

        fig.suptitle(
            f"{exp_name} - per-frame sampled polynomial background correction diagnostic",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, "bg_diagnostic"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# Pipeline step 3 — Corrected luminosity time traces
# =============================================================================
def plot_time_traces(experiments, state):
    """One side-by-side figure per experiment, one subplot per channel.

    Each subplot shows every cell's corrected luminosity over time, coloured
    by a perceptual cmap, with shaded blocks marking each stimulus pulse
    (width = ``cfg['stim_duration_minutes']``).
    """
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]

        fig, axes = plt.subplots(
            1, len(channels),
            figsize=(6 * len(channels), 5),
            dpi=PLOT_PARAMS["dpi"], sharey=True,
        )
        if len(channels) == 1:
            axes = np.array([axes])

        for col, ch in enumerate(channels):
            ax = axes[col]
            ax.spines[["top", "right"]].set_visible(False)

            stim_frames = cfg["stim_frames"][ch]
            corr_ch = state["corrected_lum"][exp_name][ch]
            cmap = plt.get_cmap(PLOT_PARAMS["trace_cmap"])
            colors = cmap(np.linspace(0, 1, max(len(corr_ch), 1)))

            for (cid, frames), color in zip(corr_ch.items(), colors):
                items = sorted(
                    ((int(k[1:]), v) for k, v in frames.items() if v is not None),
                    key=lambda t: t[0],
                )
                if not items:
                    continue
                xs, ys = zip(*items)
                xs_min = frames_to_min(state, exp_name, ch, xs)
                ax.plot(
                    xs_min, ys,
                    alpha=0.7, color=color,
                    linewidth=PLOT_PARAMS["cell_lw"] * 1.2,
                )

            spans, stim_label = stim_spans_min(state, exp_name, ch, cfg)
            draw_stim_spans(
                ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18
            )

            # NRK only: vertical marker at real-setpoint activation.
            rsp = state["real_setpoint_min"][exp_name].get(ch)
            if rsp is not None:
                ax.axvline(
                    rsp,
                    color="#000000",
                    linewidth=2.0, linestyle=":",
                    alpha=0.9, zorder=5,
                    label=f"Real setpoint ({rsp:.1f} min)",
                )

            ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            if col == 0:
                ax.set_ylabel(
                    "Corrected luminosity",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )
            ax.set_title(
                f"{ch}  ({len(corr_ch)} cells, {len(stim_frames)} stims)",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
            if stim_frames or rsp is not None:
                ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")

        fig.suptitle(
            f"{exp_name} — corrected cell luminosity over time",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, "time_traces"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# Pipeline step 3b — Clean corrected per-cell traces (no stim markers)
# =============================================================================
def plot_corrected_traces(experiments, state):
    """One figure per experiment, one subplot per channel — clean per-cell traces.

    Same per-cell corrected luminosity content as :func:`plot_time_traces` but
    *without* stimulus vertical lines, for figures where the stim markers are
    visual noise. Each cell's trace is colored from the ``twilight_shifted``
    colormap. Total cell count appears in the per-channel subplot title.
    """
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]

        fig, axes = plt.subplots(
            1, len(channels),
            figsize=(6 * len(channels), 5),
            dpi=PLOT_PARAMS["dpi"], sharey=True,
        )
        if len(channels) == 1:
            axes = np.array([axes])

        for col, ch in enumerate(channels):
            ax = axes[col]
            ax.spines[["top", "right"]].set_visible(False)

            corr_ch = state["corrected_lum"][exp_name][ch]
            cmap = plt.get_cmap(PLOT_PARAMS["trace_cmap"])
            colors = cmap(np.linspace(0, 1, max(len(corr_ch), 1)))

            for (cid, frames), color in zip(corr_ch.items(), colors):
                items = sorted(
                    ((int(k[1:]), v) for k, v in frames.items() if v is not None),
                    key=lambda t: t[0],
                )
                if not items:
                    continue
                xs, ys = zip(*items)
                xs_min = frames_to_min(state, exp_name, ch, xs)
                ax.plot(
                    xs_min, ys,
                    alpha=0.7, color=color,
                    linewidth=PLOT_PARAMS["cell_lw"] * 1.2,
                )

            rsp = state["real_setpoint_min"][exp_name].get(ch)
            if rsp is not None:
                ax.axvline(
                    rsp,
                    color="#000000",
                    linewidth=2.0, linestyle=":",
                    alpha=0.9, zorder=5,
                    label=f"Real setpoint ({rsp:.1f} min)",
                )
                ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")

            ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            if col == 0:
                ax.set_ylabel(
                    "Corrected luminosity",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )
            ax.set_title(
                f"{ch}  ({len(corr_ch)} cells)",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )

        fig.suptitle(
            f"{exp_name} — corrected cell luminosity over time",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, "corrected_traces"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# Pipeline step 4 — dF/F0 normalization
# =============================================================================
def plot_dff(experiments, state):
    """One figure per (experiment, channel): raw corrected + dF/F0 stacked.

    F0 is taken from ``cfg['f0_frame']``. The mean across cells is overlaid in
    bold. Stimuli are drawn as shaded blocks of width
    ``cfg['stim_duration_minutes']``; the F0 frame is marked with a dashed line.
    """
    for exp_name, cfg in experiments.items():
        f0_frame = cfg["f0_frame"]

        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = np.array([int(str(c).lstrip("f")) for c in frame_cols])
            frame_min = frames_to_min(state, exp_name, ch, frame_nums)
            mat = df[frame_cols].values

            f0_col = f"f{f0_frame}"
            if f0_col not in df.columns:
                print(f"{exp_name} / {ch}: F0 frame {f0_frame} missing — skipping dF/F0")
                continue
            F0 = df[f0_col].values[:, np.newaxis]
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe

            f0_min = frames_to_min(state, exp_name, ch, [f0_frame])[0]
            spans, stim_label = stim_spans_min(state, exp_name, ch, cfg)
            rsp = state["real_setpoint_min"][exp_name].get(ch)

            fig, axes = plt.subplots(
                2, 1,
                figsize=PLOT_PARAMS["figsize_wide"],
                dpi=PLOT_PARAMS["dpi"], sharex=True,
            )

            panels = [
                (mat, "Corrected luminosity", "Corrected luminosity"),
                (dff_mat, "dF/F₀", f"dF/F₀  (F₀ = frame {f0_frame})"),
            ]

            for ax, (data, ylabel, title_suffix) in zip(axes, panels):
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(top=False, right=False)

                mean_trace = np.nanmean(data, axis=0)
                for row in data:
                    ax.plot(
                        frame_min, row,
                        color=PLOT_PARAMS["cell_color"],
                        alpha=PLOT_PARAMS["cell_alpha"],
                        linewidth=PLOT_PARAMS["cell_lw"], zorder=1,
                    )
                ax.plot(
                    frame_min, mean_trace,
                    color=PLOT_PARAMS["mean_color"],
                    linewidth=PLOT_PARAMS["mean_lw"], zorder=3,
                    label="Mean",
                )

                draw_stim_spans(
                    ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18
                )

                ax.axvline(
                    f0_min,
                    color=PLOT_PARAMS["f0_color"],
                    linewidth=PLOT_PARAMS["f0_lw"],
                    linestyle="--", zorder=4,
                    label=f"F0 frame ({f0_frame})",
                )

                if rsp is not None:
                    ax.axvline(
                        rsp,
                        color="#000000",
                        linewidth=2.0, linestyle=":",
                        alpha=0.9, zorder=5,
                        label=f"Real setpoint ({rsp:.1f} min)",
                    )

                ax.set_ylabel(ylabel, fontsize=PLOT_PARAMS["axis_label_fontsize"])
                ax.set_title(
                    title_suffix,
                    fontsize=PLOT_PARAMS["title_fontsize"],
                    fontweight=PLOT_PARAMS["title_fontweight"],
                )
                ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="upper right")

            axes[-1].set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            fig.suptitle(
                f"{exp_name} / {ch} — {mat.shape[0]} cells, {len(stim_frames)} stims",
                fontsize=PLOT_PARAMS["title_fontsize"] + 1,
                fontweight="bold", y=1.01,
            )
            plt.tight_layout()
            fig.savefig(
                fig_path(exp_name, f"{ch}_dff"),
                dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
            )
            plt.close(fig)


def plot_dff_mean_combined(experiments, state):
    """One figure per experiment: mean dF/F0 trace from every channel overlaid.

    Each channel's mean is computed the same way as in :func:`plot_dff` —
    normalize each cell to its own ``f0_frame`` then average across cells —
    so the per-channel and combined views agree by construction.

    Stimuli are shaded per channel using the channel's own minute axis (in
    practice the schedules align across channels of one experiment, so the
    bands overlap visually).
    """
    for exp_name, cfg in experiments.items():
        f0_frame = cfg["f0_frame"]
        channels = cfg["channels"]

        fig, ax = plt.subplots(
            figsize=PLOT_PARAMS["figsize"],
            dpi=PLOT_PARAMS["dpi"],
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)

        cmap = plt.get_cmap("tab10")
        any_drawn = False
        for col, ch in enumerate(channels):
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            if not frame_cols:
                continue
            frame_nums = np.array([int(str(c).lstrip("f")) for c in frame_cols])
            frame_min = frames_to_min(state, exp_name, ch, frame_nums)
            mat = df[frame_cols].values

            f0_col = f"f{f0_frame}"
            if f0_col not in df.columns:
                print(
                    f"{exp_name} / {ch}: F0 frame {f0_frame} missing — "
                    "skipping in combined dF/F0."
                )
                continue
            F0 = df[f0_col].values[:, np.newaxis]
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe
            mean_trace = np.nanmean(dff_mat, axis=0)

            ax.plot(
                frame_min, mean_trace,
                color=cmap(col % 10),
                linewidth=PLOT_PARAMS["mean_lw"],
                label=f"{ch} ({mat.shape[0]} cells)",
                zorder=3,
            )
            any_drawn = True

        if not any_drawn:
            plt.close(fig)
            continue

        # Use the first channel's stim spans as a shared reference (schedules
        # align across channels in practice).
        ref_ch = channels[0]
        spans, stim_label = stim_spans_min(state, exp_name, ref_ch, cfg)
        draw_stim_spans(
            ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18
        )

        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
        ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
        ax.set_ylabel(
            f"Mean dF/F₀  (F₀ = frame {f0_frame})",
            fontsize=PLOT_PARAMS["axis_label_fontsize"],
        )
        ax.set_title(
            f"{exp_name} — mean dF/F₀ per channel",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"],
        )
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, "dff_mean_combined"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


def plot_dff_response_diagnostic(experiments, state, only_experiments=("c2c12_dmso_09APR26",)):
    """Per-channel diagnostic: is the small mean dF/F0 driven by responder
    fraction or by responder magnitude?

    Top panel — same dF/F0 traces as :func:`plot_dff` (light gray) with four
    summary curves overlaid: mean, median, 75th percentile, 90th percentile.
    If most cells are responding at a lower level the median tracks the mean;
    if only a fraction respond the median sits near zero and only the upper
    percentiles spike.

    Bottom panel — histogram of each cell's *signed peak dF/F0 response* taken
    as the direction-aware extremum (max for ``response_direction='increase'``,
    min for ``'decrease'``) over the post-stim ``response_window`` across every
    stimulus, minus that stim's pre-stim baseline. Vertical lines mark a few
    cutoffs so you can read off responder fractions by inspection.

    Saves ``<channel>_dff_response_breakdown.png`` per channel. Only runs for
    the experiments named in ``only_experiments`` (default: c2c12 only — the
    NRK setup has too few stims per channel for percentile readouts to be
    meaningful).
    """
    for exp_name, cfg in experiments.items():
        if exp_name not in only_experiments:
            continue
        f0_frame = cfg["f0_frame"]
        direction = cfg.get("response_direction", "increase")
        window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
        # Direction-aware cutoffs: positive for increase, negative for decrease.
        cutoff_mags = (0.02, 0.05, 0.10)
        sign = -1.0 if direction == "decrease" else 1.0
        cutoffs = tuple(sign * c for c in cutoff_mags)

        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = np.array([int(str(c).lstrip("f")) for c in frame_cols])
            frame_min = frames_to_min(state, exp_name, ch, frame_nums)
            mat = df[frame_cols].values

            f0_col = f"f{f0_frame}"
            if f0_col not in df.columns:
                continue
            F0 = df[f0_col].values[:, np.newaxis]
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe

            # Per-cell peak dF/F0 *response* per stim, then take the largest-
            # magnitude response across stims. Signed: positive for increase
            # responses, negative for decrease responses.
            frame_to_col = {f: i for i, f in enumerate(frame_nums.tolist())}
            n_cells = dff_mat.shape[0]
            per_stim = []  # list of (n_cells,) arrays
            for p in stim_frames:
                if p not in frame_to_col:
                    continue
                col = frame_to_col[p]
                deltas = per_cell_response_delta(dff_mat, col, direction, window)
                per_stim.append(deltas)
            if per_stim:
                stacked = np.vstack(per_stim)  # (n_stims, n_cells)
                if direction == "decrease":
                    per_cell_peak = np.nanmin(stacked, axis=0)
                else:
                    per_cell_peak = np.nanmax(stacked, axis=0)
            else:
                per_cell_peak = np.full(n_cells, np.nan)
            per_cell_peak = per_cell_peak[~np.isnan(per_cell_peak)]

            mean_trace = np.nanmean(dff_mat, axis=0)
            median_trace = np.nanmedian(dff_mat, axis=0)
            p75_trace = np.nanpercentile(dff_mat, 75, axis=0)
            p90_trace = np.nanpercentile(dff_mat, 90, axis=0)

            f0_min = frames_to_min(state, exp_name, ch, [f0_frame])[0]
            spans, stim_label = stim_spans_min(state, exp_name, ch, cfg)

            fig, axes = plt.subplots(
                2, 1,
                figsize=(14, 9),
                dpi=PLOT_PARAMS["dpi"],
                gridspec_kw={"height_ratios": [3, 2]},
            )

            ax_top = axes[0]
            ax_top.spines[["top", "right"]].set_visible(False)
            ax_top.tick_params(top=False, right=False)
            for row in dff_mat:
                ax_top.plot(
                    frame_min, row,
                    color=PLOT_PARAMS["cell_color"],
                    alpha=PLOT_PARAMS["cell_alpha"],
                    linewidth=PLOT_PARAMS["cell_lw"], zorder=1,
                )
            for trace, color, label, lw in [
                (mean_trace,   "#1a1a1a", "Mean",   2.0),
                (median_trace, "#1aa821", "Median", 2.0),
                (p75_trace,    "#e67e22", "75th percentile", 1.6),
                (p90_trace,    "#c0392b", "90th percentile", 1.6),
            ]:
                ax_top.plot(
                    frame_min, trace,
                    color=color, linewidth=lw,
                    label=label, zorder=4,
                )
            draw_stim_spans(
                ax_top, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18
            )
            ax_top.axvline(
                f0_min,
                color=PLOT_PARAMS["f0_color"],
                linewidth=PLOT_PARAMS["f0_lw"],
                linestyle="--", zorder=3,
                label=f"F0 frame ({f0_frame})",
            )
            ax_top.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.6, zorder=2)
            ax_top.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax_top.set_ylabel("dF/F₀", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax_top.set_title(
                f"{exp_name} / {ch} — dF/F₀ summary curves "
                f"(mean / median / 75th / 90th)",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
            ax_top.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="upper right")

            # Bottom: per-cell peak dF/F0 histogram.
            ax_bot = axes[1]
            ax_bot.spines[["top", "right"]].set_visible(False)
            ax_bot.tick_params(top=False, right=False)
            if len(per_cell_peak) > 0:
                ax_bot.hist(
                    per_cell_peak,
                    bins=60,
                    color=PLOT_PARAMS["violin_face"],
                    edgecolor=PLOT_PARAMS["violin_edge"],
                    linewidth=0.6,
                    zorder=2,
                )
            cutoff_colors = ["#1aa821", "#e67e22", "#c0392b"]
            cmp_op = (lambda v, c: v <= c) if direction == "decrease" else (lambda v, c: v >= c)
            cmp_str = "≤" if direction == "decrease" else "≥"
            for cut, ccolor in zip(cutoffs, cutoff_colors):
                frac = float(cmp_op(per_cell_peak, cut).mean()) if len(per_cell_peak) else 0.0
                ax_bot.axvline(
                    cut, color=ccolor, linewidth=1.5, linestyle="--",
                    label=f"{cmp_str} {cut:+g}: {frac * 100:.1f}% of cells",
                    zorder=3,
                )
            extremum_label = "max" if direction == "increase" else "min"
            ax_bot.set_xlabel(
                f"Per-cell peak Δ dF/F₀  (largest-magnitude {extremum_label}−baseline across stims)",
                fontsize=PLOT_PARAMS["axis_label_fontsize"],
            )
            ax_bot.set_ylabel("Cell count", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax_bot.set_title(
                f"Per-cell peak Δ dF/F₀ distribution  (n = {len(per_cell_peak)} of {n_cells} cells)",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
            ax_bot.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="upper right")

            mean_peak = float(np.nanmean(per_cell_peak)) if len(per_cell_peak) else float("nan")
            med_peak = float(np.nanmedian(per_cell_peak)) if len(per_cell_peak) else float("nan")
            fracs = [
                float(cmp_op(per_cell_peak, c).mean()) * 100 if len(per_cell_peak) else 0.0
                for c in cutoffs
            ]
            print(
                f"{exp_name} / {ch}: peak Δ dF/F0 ({direction}) — "
                f"mean={mean_peak:+.4f}, median={med_peak:+.4f}, "
                f"frac {cmp_str} {cutoffs[0]:+g} = {fracs[0]:.1f}%, "
                f"frac {cmp_str} {cutoffs[1]:+g} = {fracs[1]:.1f}%, "
                f"frac {cmp_str} {cutoffs[2]:+g} = {fracs[2]:.1f}%"
            )

            plt.tight_layout()
            fig.savefig(
                fig_path(exp_name, f"{ch}_dff_response_breakdown"),
                dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
            )
            plt.close(fig)


# =============================================================================
# Pipeline step 5 — Pairwise correlation vs distance
# =============================================================================
def mean_cell_positions(traj, n_frames):
    """Return ``{cell_id_str: (mean_x, mean_y)}`` averaged over valid frames.

    A frame is valid for a cell if both ``x{i}`` and ``y{i}`` exist and parse
    as floats.
    """
    positions = {}
    for cid, coords in traj.items():
        xs, ys = [], []
        for i in range(n_frames):
            x = coords.get(f"x{i}")
            y = coords.get(f"y{i}")
            if x is None or y is None:
                continue
            try:
                xs.append(float(x))
                ys.append(float(y))
            except (TypeError, ValueError):
                continue
        if xs:
            positions[cid] = (float(np.mean(xs)), float(np.mean(ys)))
    return positions


def _scatter_corr_vs_dist(ax, dists, corrs, color, title):
    """Helper: scatter pairwise (distance, correlation) with a fitted line + ±3 SEM band.

    Trend line is always black; ±3 SEM band is always gray. The ``color``
    argument controls only the scatter cloud, so each panel can use a
    distinct (muted) hue while the fit overlay reads the same way everywhere.
    """
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter(dists, corrs, color=color, alpha=0.35, s=8, edgecolors="none", zorder=1)

    valid = ~np.isnan(dists) & ~np.isnan(corrs)
    if np.sum(valid) > 2:
        xv, yv = dists[valid], corrs[valid]
        res = linregress(xv, yv)
        x_line = np.linspace(xv.min(), xv.max(), 100)
        y_line = res.slope * x_line + res.intercept

        dof = len(xv) - 2
        rss = np.sum((yv - (res.slope * xv + res.intercept)) ** 2)
        rse = np.sqrt(rss / dof) if dof > 0 else np.nan
        mean_x = np.mean(xv)
        ssx = np.sum((xv - mean_x) ** 2)
        y_err = (
            rse * np.sqrt(1 / len(xv) + (x_line - mean_x) ** 2 / ssx)
            if ssx > 0 else np.zeros_like(x_line)
        )

        ax.plot(
            x_line, y_line,
            color=PLOT_PARAMS["corr_fit_color"],
            linewidth=PLOT_PARAMS["mean_lw"],
            label=(
                f"Linear fit\n"
                f"Pearson r = {res.rvalue:.3f}\n"
                f"R² = {res.rvalue ** 2:.3f}\n"
                f"p (slope=0) = {res.pvalue:.2e}"
            ),
            zorder=3,
        )
        ax.fill_between(
            x_line, y_line - 3 * y_err, y_line + 3 * y_err,
            color=PLOT_PARAMS["corr_band_color"], alpha=0.30, zorder=2,
            label="±3 SEM",
        )

    ax.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")


def plot_correlation_vs_distance(experiments, state):
    """One figure per experiment: pairwise Pearson r vs spatial distance.

    Cell positions are the per-cell mean ``(x, y)`` across time (NaN-robust).
    """
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]
        fig, axes = plt.subplots(
            1, len(channels),
            figsize=(6 * len(channels), 6),
            dpi=PLOT_PARAMS["dpi"], sharey=True,
        )
        if len(channels) == 1:
            axes = np.array([axes])

        per_channel_pairs = []  # collected for the combined figure below

        for col, ch in enumerate(channels):
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            mat = df[frame_cols].values
            cell_ids_int = list(df.index)

            positions = mean_cell_positions(
                state["traj_by_channel"][exp_name][ch],
                state["frame_counts"][exp_name][ch],
            )

            # Match cells from the corrected luminosity table to entries in
            # the trajectory dict (keys may be either str or int).
            keep_rows, pos_xy = [], []
            for r, cid_int in enumerate(cell_ids_int):
                for key in (str(cid_int), cid_int):
                    if key in positions:
                        keep_rows.append(r)
                        pos_xy.append(positions[key])
                        break
            mat_k = mat[keep_rows]
            pos_xy = np.array(pos_xy, dtype=float)

            if len(pos_xy) < 2:
                axes[col].set_title(
                    f"{ch}: insufficient data",
                    fontsize=PLOT_PARAMS["title_fontsize"],
                )
                continue

            corr_mat = pd.DataFrame(mat_k).T.corr(method="pearson").values
            dist_mat = squareform(pdist(pos_xy, metric="euclidean"))
            iu = np.triu_indices(corr_mat.shape[0], k=1)
            pw_corr = corr_mat[iu]
            pw_dist = dist_mat[iu]

            per_channel_pairs.append((ch, pw_dist, pw_corr, len(keep_rows)))

            _scatter_corr_vs_dist(
                axes[col], pw_dist, pw_corr,
                color=PLOT_PARAMS["corr_scatter_colors"][col % len(PLOT_PARAMS["corr_scatter_colors"])],
                title=f"{ch}  ({len(keep_rows)} cells)",
            )
            axes[col].set_xlabel(
                "Pairwise distance (px)",
                fontsize=PLOT_PARAMS["axis_label_fontsize"],
            )
            if col == 0:
                axes[col].set_ylabel(
                    "Pearson r (full corrected time series)",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )

        fig.suptitle(
            f"{exp_name} — pairwise correlation vs pairwise distance",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, "corr_vs_dist"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)

        _plot_corr_vs_dist_combined(exp_name, per_channel_pairs)


def _plot_corr_vs_dist_combined(exp_name, per_channel_pairs):
    """Single-axes scatter pooling pairwise (distance, correlation) across channels.

    Each channel is drawn in its own muted hue (reusing
    ``PLOT_PARAMS['corr_scatter_colors']``) so the pooled cloud still reads as
    per-channel, while a single black linear fit + ±3 SEM band describes the
    pooled trend.
    """
    if not per_channel_pairs:
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=PLOT_PARAMS["dpi"])
    ax.spines[["top", "right"]].set_visible(False)

    all_dist, all_corr = [], []
    for col, (ch, pw_dist, pw_corr, n_cells) in enumerate(per_channel_pairs):
        c = PLOT_PARAMS["corr_scatter_colors"][col % len(PLOT_PARAMS["corr_scatter_colors"])]
        ax.scatter(
            pw_dist, pw_corr,
            color=c, alpha=0.30, s=8, edgecolors="none", zorder=1,
            label=f"{ch} ({n_cells} cells)",
        )
        all_dist.append(np.asarray(pw_dist))
        all_corr.append(np.asarray(pw_corr))

    dists = np.concatenate(all_dist)
    corrs = np.concatenate(all_corr)
    valid = ~np.isnan(dists) & ~np.isnan(corrs)
    if np.sum(valid) > 2:
        xv, yv = dists[valid], corrs[valid]
        res = linregress(xv, yv)
        x_line = np.linspace(xv.min(), xv.max(), 200)
        y_line = res.slope * x_line + res.intercept

        dof = len(xv) - 2
        rss = np.sum((yv - (res.slope * xv + res.intercept)) ** 2)
        rse = np.sqrt(rss / dof) if dof > 0 else np.nan
        mean_x = np.mean(xv)
        ssx = np.sum((xv - mean_x) ** 2)
        y_err = (
            rse * np.sqrt(1 / len(xv) + (x_line - mean_x) ** 2 / ssx)
            if ssx > 0 else np.zeros_like(x_line)
        )

        ax.plot(
            x_line, y_line,
            color=PLOT_PARAMS["corr_fit_color"],
            linewidth=PLOT_PARAMS["mean_lw"],
            label=(
                f"Pooled linear fit\n"
                f"Pearson r = {res.rvalue:.3f}\n"
                f"R² = {res.rvalue ** 2:.3f}\n"
                f"p (slope=0) = {res.pvalue:.2e}"
            ),
            zorder=3,
        )
        ax.fill_between(
            x_line, y_line - 3 * y_err, y_line + 3 * y_err,
            color=PLOT_PARAMS["corr_band_color"], alpha=0.30, zorder=2,
            label="±3 SEM",
        )

    ax.set_xlabel(
        "Pairwise distance (px)",
        fontsize=PLOT_PARAMS["axis_label_fontsize"],
    )
    ax.set_ylabel(
        "Pearson r (full corrected time series)",
        fontsize=PLOT_PARAMS["axis_label_fontsize"],
    )
    ax.set_title(
        f"{exp_name} — pairwise correlation vs distance (all channels combined)",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
    plt.tight_layout()
    fig.savefig(
        fig_path(exp_name, "corr_vs_dist_combined"),
        dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# Pipeline step 5b — Trace clustering (PCA + UMAP + KMeans)
# =============================================================================
def plot_trace_clustering(experiments, state, k_min=2, k_max=8, random_state=0):
    """Cluster cells by trace shape and visualize as a 2x2 figure per channel.

    Pipeline, per (experiment, channel):
      1. Build the (n_cells, n_frames) corrected-luminosity matrix.
      2. Per-cell z-score along the time axis (groups by *response shape*,
         not absolute brightness or amplitude).
      3. PCA -> n_pca = min(20, n_cells, n_frames) components.
      4. UMAP(n_components=2) on the PCA scores for the 2D layout.
      5. KMeans on the PCA scores (UMAP distances are unreliable for clustering),
         with k chosen by silhouette score over k_min..k_max.

    Saves one PNG per (experiment, channel) at
    ``fig_path(exp, f"{ch}_trace_clustering")``. Layout:
       (a) top-left:  PCA scree.
       (b) top-right: UMAP embedding colored by cluster.
       (c) bot-left:  cluster mean traces (corrected luminosity, +/-1 SEM)
                      with stim shading and F0 marker.
       (d) bot-right: cluster size bar chart.

    Requires scikit-learn and umap-learn.
    """
    for exp_name, cfg in experiments.items():
        f0_frame = cfg["f0_frame"]

        for ch in cfg["channels"]:
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = np.array([int(str(c).lstrip("f")) for c in frame_cols])
            frame_min = frames_to_min(state, exp_name, ch, frame_nums)

            X_raw = df[frame_cols].values.astype(float)  # (n_cells, n_frames)
            row_all_nan = np.isnan(X_raw).all(axis=1)
            X_raw = X_raw[~row_all_nan]
            if X_raw.shape[0] < max(k_min + 1, 5):
                print(f"{exp_name} / {ch}: only {X_raw.shape[0]} cells — skipping clustering")
                continue
            row_means = np.nanmean(X_raw, axis=1, keepdims=True)
            X_raw = np.where(np.isnan(X_raw), row_means, X_raw)

            mu = X_raw.mean(axis=1, keepdims=True)
            sd = X_raw.std(axis=1, keepdims=True)
            keep = sd[:, 0] > 0
            X_raw = X_raw[keep]
            mu = mu[keep]
            sd = sd[keep]
            X_z = (X_raw - mu) / sd
            X_z = np.nan_to_num(X_z, nan=0.0, posinf=0.0, neginf=0.0)

            n_cells, n_frames = X_z.shape
            if n_cells < max(k_min + 1, 5):
                print(f"{exp_name} / {ch}: only {n_cells} cells after filter — skipping")
                continue

            n_pca = int(min(20, n_cells, n_frames))
            pca = PCA(n_components=n_pca, random_state=random_state)
            pcs = pca.fit_transform(X_z)
            evr = pca.explained_variance_ratio_

            n_neighbors = int(min(15, max(2, n_cells - 1)))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=random_state,
            )
            embedding = reducer.fit_transform(pcs)

            best_k = k_min
            best_score = -np.inf
            k_upper = min(k_max, n_cells - 1)
            for k in range(k_min, k_upper + 1):
                km_try = KMeans(n_clusters=k, n_init=10, random_state=random_state)
                labs_try = km_try.fit_predict(pcs)
                if len(np.unique(labs_try)) < 2:
                    continue
                try:
                    s = silhouette_score(pcs, labs_try)
                except ValueError:
                    continue
                if s > best_score:
                    best_score = s
                    best_k = k
            km = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
            labels = km.fit_predict(pcs)

            cmap = plt.get_cmap("tab10")
            cluster_colors = [cmap(i % 10) for i in range(best_k)]

            fig, axes = plt.subplots(
                2, 2,
                figsize=(14, 10),
                dpi=PLOT_PARAMS["dpi"],
            )
            for ax in axes.ravel():
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(top=False, right=False)

            ax = axes[0, 0]
            n_show = int(min(10, len(evr)))
            ax.bar(
                np.arange(1, n_show + 1), evr[:n_show],
                color=PLOT_PARAMS["fit_color"], alpha=0.85,
            )
            ax.set_xlabel("Principal component", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax.set_ylabel("Explained variance ratio", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax.set_xticks(np.arange(1, n_show + 1))
            ax.set_title(
                f"PCA scree (top {n_show} of {n_pca}) — cum. {evr[:n_show].sum() * 100:.1f}%",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )

            ax = axes[0, 1]
            for cid in range(best_k):
                mask = labels == cid
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    s=PLOT_PARAMS["scatter_size"] * 1.4,
                    color=cluster_colors[cid],
                    alpha=0.75,
                    edgecolors="none",
                    label=f"Cluster {cid} (n={int(mask.sum())})",
                )
            ax.set_xlabel("UMAP 1", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax.set_ylabel("UMAP 2", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax.set_title(
                f"UMAP embedding — k={best_k} (silhouette={best_score:.3f})",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
            ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")

            ax = axes[1, 0]
            spans, stim_label = stim_spans_min(state, exp_name, ch, cfg)
            f0_min = frames_to_min(state, exp_name, ch, [f0_frame])[0]
            for cid in range(best_k):
                mask = labels == cid
                cluster_traces = X_raw[mask]
                mean_trace = np.nanmean(cluster_traces, axis=0)
                n_in = max(int(mask.sum()), 1)
                sem_trace = np.nanstd(cluster_traces, axis=0) / np.sqrt(n_in)
                ax.fill_between(
                    frame_min, mean_trace - sem_trace, mean_trace + sem_trace,
                    color=cluster_colors[cid], alpha=0.15, linewidth=0,
                )
                ax.plot(
                    frame_min, mean_trace,
                    color=cluster_colors[cid],
                    linewidth=PLOT_PARAMS["mean_lw"],
                    label=f"Cluster {cid} (n={n_in})",
                )
            draw_stim_spans(
                ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18,
            )
            ax.axvline(
                f0_min,
                color=PLOT_PARAMS["f0_color"],
                linewidth=PLOT_PARAMS["f0_lw"],
                linestyle="--",
                label=f"F0 frame ({f0_frame})",
            )
            rsp = state["real_setpoint_min"][exp_name].get(ch)
            if rsp is not None:
                ax.axvline(
                    rsp,
                    color="#000000",
                    linewidth=2.0, linestyle=":", alpha=0.9,
                    label=f"Real setpoint ({rsp:.1f} min)",
                )
            ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax.set_ylabel(
                "Corrected luminosity (cluster mean ±1 SEM)",
                fontsize=PLOT_PARAMS["axis_label_fontsize"],
            )
            ax.set_title(
                "Cluster mean traces",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
            ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")

            ax = axes[1, 1]
            sizes = np.array([int((labels == cid).sum()) for cid in range(best_k)])
            ax.bar(
                np.arange(best_k), sizes,
                color=cluster_colors, edgecolor="#222222", linewidth=0.6,
            )
            for cid, n in enumerate(sizes):
                ax.text(
                    cid, n, str(int(n)),
                    ha="center", va="bottom",
                    fontsize=PLOT_PARAMS["legend_fontsize"],
                )
            ax.set_xticks(np.arange(best_k))
            ax.set_xticklabels([f"C{cid}" for cid in range(best_k)])
            ax.set_xlabel("Cluster", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax.set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax.set_title(
                f"Cluster sizes (total {n_cells})",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )

            fig.suptitle(
                f"{exp_name} / {ch} — trace clustering (k={best_k}, n={n_cells})",
                fontsize=PLOT_PARAMS["title_fontsize"] + 1,
                fontweight="bold",
                y=1.00,
            )
            plt.tight_layout()
            fig.savefig(
                fig_path(exp_name, f"{ch}_trace_clustering"),
                dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
            )
            plt.close(fig)


# =============================================================================
# Pipeline step 6 — Per-stimulus violin plots
# =============================================================================
# Layout constants for the asymmetric violin/box composite below.
# All offsets are in axis-units of the categorical x-position (1 unit = 1 stim).
_VIOLIN_BOX_OFFSET = 0.18   # x-distance of box from violin center
_VIOLIN_BOX_WIDTH = 0.18    # width of the box
_VIOLIN_SCATTER_JITTER = 0.045  # tight jitter on the scatter dots (must stay < BOX_OFFSET)


def _draw_half_violin_with_box(ax, violin_data, x_label, y_label, title, save_path,
                                x_axis_label="Peak frame"):
    """Render the asymmetric violin/box composite for one figure and save it.

    Layout per category column ``i`` (one column per element in ``violin_data``):
        * RIGHT of x = i: half-violin (distribution shape only, vertex-clipped)
                          + horizontal red mean marker at the violin's left edge
        * LEFT of x = i: notched box at (i - offset)
                          - box body         = IQR (Q1..Q3)
                          - center line      = median
                          - notch            = 95% CI of the median (1.57·IQR/√n)
                          - whiskers         = 1.5·IQR Tukey whiskers
        * BEHIND the box (low zorder): tightly-jittered scatter of every value

    ``violin_data`` is a list of 1-D arrays (possibly empty for missing
    categories). The figure is saved to ``save_path`` and closed.
    """
    n_cat = len(violin_data)
    non_empty_idx = [i for i, v in enumerate(violin_data) if len(v) > 0]
    if not non_empty_idx:
        return False

    fig, ax_ = plt.subplots(
        figsize=(max(6, n_cat * 1.2), 10),
        dpi=PLOT_PARAMS["dpi"],
    )
    ax_.spines[["top", "right"]].set_visible(False)
    ax_.tick_params(top=False, right=False)

    box_x = [i - _VIOLIN_BOX_OFFSET for i in non_empty_idx]
    box_data = [violin_data[i] for i in non_empty_idx]

    # ---- 1) Tight-jitter scatter on the LEFT, behind everything ----
    rng = np.random.default_rng(42)
    for col_idx, bx in zip(non_empty_idx, box_x):
        vd = violin_data[col_idx]
        xs = bx + rng.uniform(
            -_VIOLIN_SCATTER_JITTER, _VIOLIN_SCATTER_JITTER, size=len(vd)
        )
        ax_.scatter(
            xs, vd,
            color=PLOT_PARAMS["scatter_color"],
            alpha=PLOT_PARAMS["scatter_alpha"],
            s=PLOT_PARAMS["scatter_size"],
            zorder=2, linewidths=0,
        )

    # ---- 2) Notched box on the LEFT: median + IQR + 95% CI of median + Tukey whiskers ----
    # ``notch=True`` draws the matplotlib notch at median ± 1.57·IQR/√n, which
    # is the canonical 95% CI of the median for a box plot.
    bp = ax_.boxplot(
        box_data,
        positions=box_x,
        widths=_VIOLIN_BOX_WIDTH,
        notch=True,
        bootstrap=None,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(
            color=PLOT_PARAMS["median_color"], linewidth=2.0
        ),
        boxprops=dict(
            facecolor=PLOT_PARAMS["violin_face"],
            edgecolor=PLOT_PARAMS["violin_edge"],
            linewidth=1.2,
        ),
        whiskerprops=dict(color=PLOT_PARAMS["violin_edge"], linewidth=1.0),
        capprops=dict(color=PLOT_PARAMS["violin_edge"], linewidth=1.0),
        zorder=3,
    )
    # Label the median line so it shows up in the legend alongside "Mean".
    median_handles = bp.get("medians", [])
    for i, handle in enumerate(median_handles):
        handle.set_label("Median" if i == 0 else None)

    # ---- 3) Half-violin on the RIGHT (distribution only) ----
    vp = ax_.violinplot(
        box_data,
        positions=non_empty_idx,
        showmedians=False,
        showextrema=False,
        showmeans=False,
    )
    for body, pos in zip(vp["bodies"], non_empty_idx):
        # Each body is a PolyCollection with one path. Snap any vertex whose
        # x falls left of the violin's center onto the center line, leaving
        # only the right half visible.
        for path in body.get_paths():
            verts = path.vertices
            verts[verts[:, 0] < pos, 0] = pos
        body.set_facecolor(PLOT_PARAMS["violin_face"])
        body.set_edgecolor(PLOT_PARAMS["violin_edge"])
        body.set_alpha(0.85)
        body.set_zorder(3)

    # ---- 4) Mean marker on the RIGHT (sits at the violin's left edge) ----
    means = [float(np.mean(v)) for v in box_data]
    ax_.scatter(
        non_empty_idx, means,
        marker="_",
        color=PLOT_PARAMS["mean_marker_color"],
        s=200, linewidths=2.5,
        zorder=6, label="Mean",
    )

    ax_.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="center right")

    ax_.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax_.set_xticks(range(n_cat))
    ax_.set_xticklabels([str(lbl) for lbl in x_label], fontsize=9)
    ax_.set_xlabel(x_axis_label, fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax_.set_ylabel(y_label, fontsize=PLOT_PARAMS["axis_label_fontsize"])
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)
    return True


def plot_per_stimulus_peak_violins(experiments, state):
    """Violin of corrected luminosity at the peak frame for every stimulus.

    Peak frame = ``stim + PEAK_OFFSET``. One figure per (experiment, channel),
    saved as ``<channel>_peak_value_violin.png``. See
    :func:`_draw_half_violin_with_box` for the visual layout.
    """
    for exp_name, cfg in experiments.items():
        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            if not stim_frames:
                print(f"{exp_name} / {ch}: no stim_frames — skipping peak violin.")
                continue

            df_indexed = lum_dict_to_df(
                state["corrected_lum"][exp_name][ch]
            ).set_index("CellID")
            peak_cols = [f"f{p + PEAK_OFFSET}" for p in stim_frames]
            available_cols = [c for c in peak_cols if c in df_indexed.columns]
            if not available_cols:
                print(f"{exp_name} / {ch}: no peak columns available, skipping.")
                continue
            complete_df = df_indexed[available_cols].dropna()
            print(
                f"{exp_name} / {ch}: peak violin — {len(complete_df)} cells "
                f"complete across {len(available_cols)} peaks."
            )

            violin_data = [
                df_indexed[f"f{p + PEAK_OFFSET}"].dropna().values
                if f"f{p + PEAK_OFFSET}" in df_indexed.columns
                else np.array([])
                for p in stim_frames
            ]
            peak_min = frames_to_min(
                state, exp_name, ch, [p + PEAK_OFFSET for p in stim_frames]
            )
            x_labels = [f"{m:.1f}" for m in peak_min]
            ok = _draw_half_violin_with_box(
                ax=None,
                violin_data=violin_data,
                x_label=x_labels,
                y_label="Corrected luminosity at peak",
                title=(
                    f"{exp_name} / {ch} — peak luminosity at each stimulus "
                    f"({len(complete_df)} complete cells)"
                ),
                save_path=fig_path(exp_name, f"{ch}_peak_value_violin"),
                x_axis_label="Peak time (min)",
            )
            if not ok:
                print(f"{exp_name} / {ch}: no non-empty peak data — skipped.")


def plot_per_stimulus_response_violins(experiments, state):
    """Violin of per-cell delta = response_value − baseline, where the response
    value is the post-stim *extremum* (max for ``response_direction='increase'``,
    min for ``'decrease'``) over ``response_window`` frames after the stim.

    Sign convention is preserved: positive deltas for increasing dyes (DMSO),
    negative deltas for decreasing dyes (acid). One figure per
    (experiment, channel), saved as ``<channel>_response_violin.png``.
    """
    for exp_name, cfg in experiments.items():
        direction = cfg.get("response_direction", "increase")
        window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))

        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            if not stim_frames:
                print(f"{exp_name} / {ch}: no stim_frames — skipping response violin.")
                continue

            df_indexed = lum_dict_to_df(
                state["corrected_lum"][exp_name][ch]
            ).set_index("CellID")
            frame_cols = sorted(
                [c for c in df_indexed.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
            mat = df_indexed[frame_cols].values
            # Map frame number → column index for safe indexing.
            frame_to_col = {f: i for i, f in enumerate(frame_nums)}

            violin_data = []
            for p in stim_frames:
                if p not in frame_to_col:
                    violin_data.append(np.array([]))
                    continue
                col = frame_to_col[p]
                deltas = per_cell_response_delta(mat, col, direction, window)
                deltas = deltas[~np.isnan(deltas)]
                violin_data.append(deltas)

            n_complete = sum(len(v) > 0 for v in violin_data)
            print(
                f"{exp_name} / {ch}: response violin — "
                f"{n_complete}/{len(stim_frames)} stimuli have data."
            )

            base_min = frames_to_min(state, exp_name, ch, stim_frames) if stim_frames else []
            x_labels = [f"{bm:.1f}" for bm in base_min]

            extremum_label = "max" if direction == "increase" else "min"
            window_str = f"stim+{window[0]}…stim+{window[1] - 1} frames"
            ok = _draw_half_violin_with_box(
                ax=None,
                violin_data=violin_data,
                x_label=x_labels,
                y_label=f"Δ luminosity  ({extremum_label} − baseline)",
                title=(
                    f"{exp_name} / {ch} — per-stimulus Δ luminosity "
                    f"({extremum_label} over {window_str} − baseline)"
                ),
                save_path=fig_path(exp_name, f"{ch}_response_violin"),
                x_axis_label="Stimulus onset (min)",
            )
            if not ok:
                print(f"{exp_name} / {ch}: no non-empty response data — skipped.")


def compute_responder_thresholds(
    experiments, state,
    alpha=0.01,
    exclusion_pad=10,
    n_pseudo=100,
    rng_seed=42,
):
    """Per-(experiment, channel) Bonferroni-corrected responder threshold
    (dF/F0 magnitude).

    Two steps:

    1. Per-experiment per-stim null distribution (same as before): pool
       every cell × pseudo-stim Δ dF/F0 across all channels of the
       experiment to characterise that experiment's noise floor.

    2. Per-channel Bonferroni cutoff: each cell sees ``N_real`` real stims,
       each an independent chance to clear the bar. To hold the per-cell
       false-positive rate at ``alpha``, the per-stim cutoff must be at
       the ``(1 − alpha / N_real)``-quantile of the null. So channels with
       more stims get stricter thresholds (drawn from the same per-experiment
       null pool) and channels with fewer stims get looser ones — but every
       channel's per-cell FPR matches.

    Returns ``{(exp_name, ch_name): threshold_magnitude}``. Channels with
    zero real stims are omitted.
    """
    rng = np.random.default_rng(rng_seed)
    thresholds = {}

    for exp_name, cfg in experiments.items():
        direction = cfg.get("response_direction", "increase")
        window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
        f0_frame = cfg["f0_frame"]
        win_lo, win_hi = window

        pooled_null = []
        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            df_indexed = lum_dict_to_df(
                state["corrected_lum"][exp_name][ch]
            ).set_index("CellID")
            frame_cols = sorted(
                [c for c in df_indexed.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
            mat = df_indexed[frame_cols].values
            n_cells, n_cols = mat.shape
            frame_to_col = {f: i for i, f in enumerate(frame_nums)}

            f0_col = f"f{f0_frame}"
            if f0_col not in df_indexed.columns:
                continue
            F0 = df_indexed[f0_col].values[:, np.newaxis]
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe

            stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]
            excluded = np.zeros(n_cols, dtype=bool)
            pad = max(exclusion_pad, win_hi)
            for sc in stim_cols:
                lo = max(0, sc - pad)
                hi = min(n_cols, sc + pad + 1)
                excluded[lo:hi] = True
            valid_mask = ~excluded
            valid_mask[: max(0, -win_lo)] = False
            valid_mask[max(0, n_cols - win_hi + 1):] = False
            valid_cols = np.where(valid_mask)[0]
            if len(valid_cols) < 1:
                continue

            replace = len(valid_cols) < n_pseudo
            pseudo_cols = rng.choice(valid_cols, size=n_pseudo, replace=replace)
            for pc in pseudo_cols:
                deltas = per_cell_response_delta(dff_mat, int(pc), direction, window)
                deltas = deltas[~np.isnan(deltas)]
                pooled_null.append(deltas)

        if not pooled_null:
            for ch in cfg["channels"]:
                if cfg["stim_frames"][ch]:
                    thresholds[(exp_name, ch)] = 0.10
            continue
        pooled_abs = np.abs(np.concatenate(pooled_null))

        # Bonferroni: per-channel percentile derived from N_real for that channel.
        for ch in cfg["channels"]:
            n_real = len(cfg["stim_frames"][ch])
            if n_real == 0:
                continue
            per_stim_alpha = alpha / n_real
            pct = 100.0 * (1.0 - per_stim_alpha)
            threshold = float(np.nanpercentile(pooled_abs, pct))
            thresholds[(exp_name, ch)] = threshold
            print(
                f"  responder threshold ({exp_name} / {ch}): "
                f"|Δ dF/F0| ≥ {threshold:.4f}  "
                f"(Bonferroni: {pct:.4f}th pct of per-stim null, "
                f"N_real={n_real}, per-cell α={alpha:g})"
            )

    return thresholds


def plot_per_stimulus_response_violins_responders(experiments, state, thresholds=None):
    """Same as :func:`plot_per_stimulus_response_violins` but restricted to
    cells whose largest-magnitude per-stim Δ dF/F0 (across every stim) crosses
    a per-(experiment, channel) Bonferroni-corrected threshold derived from
    the experiment's per-stim null distribution
    (see :func:`compute_responder_thresholds`).

    ``thresholds`` is ``{(exp_name, ch_name): magnitude}``; if ``None``, it
    is computed on the fly. The threshold is sign-aware (``≥ +T`` for
    ``response_direction='increase'``, ``≤ -T`` for ``'decrease'``).

    Saves ``<channel>_response_violin_responders.png`` per channel.
    """
    if thresholds is None:
        thresholds = compute_responder_thresholds(experiments, state)

    for exp_name, cfg in experiments.items():
        direction = cfg.get("response_direction", "increase")
        window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
        f0_frame = cfg["f0_frame"]
        sign = -1.0 if direction == "decrease" else 1.0

        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            if not stim_frames:
                print(
                    f"{exp_name} / {ch}: no stim_frames — skipping responder violin."
                )
                continue

            dff_threshold = float(thresholds.get((exp_name, ch), 0.10))
            signed_threshold = sign * dff_threshold

            df_indexed = lum_dict_to_df(
                state["corrected_lum"][exp_name][ch]
            ).set_index("CellID")
            frame_cols = sorted(
                [c for c in df_indexed.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
            mat = df_indexed[frame_cols].values
            frame_to_col = {f: i for i, f in enumerate(frame_nums)}

            f0_col = f"f{f0_frame}"
            if f0_col not in df_indexed.columns:
                print(
                    f"{exp_name} / {ch}: F0 frame {f0_frame} missing — "
                    "skipping responder violin."
                )
                continue
            F0 = df_indexed[f0_col].values[:, np.newaxis]
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe

            # Per-stim per-cell Δ dF/F0 → take the strongest (signed) across stims.
            per_stim_dff = []
            per_stim_lum = []  # parallel: corrected-luminosity deltas
            stim_cols_used = []
            for p in stim_frames:
                if p not in frame_to_col:
                    continue
                col = frame_to_col[p]
                stim_cols_used.append((p, col))
                per_stim_dff.append(
                    per_cell_response_delta(dff_mat, col, direction, window)
                )
                per_stim_lum.append(
                    per_cell_response_delta(mat, col, direction, window)
                )
            if not per_stim_dff:
                print(f"{exp_name} / {ch}: no usable stims — skipping responder violin.")
                continue

            stacked_dff = np.vstack(per_stim_dff)        # (n_stims, n_cells)
            stacked_lum = np.vstack(per_stim_lum)
            if direction == "decrease":
                per_cell_peak_dff = np.nanmin(stacked_dff, axis=0)
                responder_mask = per_cell_peak_dff <= signed_threshold
            else:
                per_cell_peak_dff = np.nanmax(stacked_dff, axis=0)
                responder_mask = per_cell_peak_dff >= signed_threshold
            n_total = mat.shape[0]
            n_resp = int(np.sum(responder_mask & ~np.isnan(per_cell_peak_dff)))

            if n_resp == 0:
                print(
                    f"{exp_name} / {ch}: 0 responders at |dF/F0| ≥ {dff_threshold} — "
                    "skipping responder violin."
                )
                continue

            violin_data = []
            for row in stacked_lum:
                vals = row[responder_mask]
                vals = vals[~np.isnan(vals)]
                violin_data.append(vals)

            base_min = frames_to_min(
                state, exp_name, ch, [p for p, _ in stim_cols_used]
            ) if stim_cols_used else []
            x_labels = [f"{bm:.1f}" for bm in base_min]

            extremum_label = "max" if direction == "increase" else "min"
            window_str = f"stim+{window[0]}…stim+{window[1] - 1} frames"
            cmp_str = "≥" if direction == "increase" else "≤"
            ok = _draw_half_violin_with_box(
                ax=None,
                violin_data=violin_data,
                x_label=x_labels,
                y_label=f"Δ luminosity  ({extremum_label} − baseline)",
                title=(
                    f"{exp_name} / {ch} — responders only "
                    f"(peak Δ dF/F₀ {cmp_str} {sign * dff_threshold:+.4f}, "
                    f"Bonferroni per-cell α=0.01, N={len(stim_frames)};  "
                    f"{n_resp} of {n_total} cells)"
                ),
                save_path=fig_path(exp_name, f"{ch}_response_violin_responders"),
                x_axis_label="Stimulus onset (min)",
            )
            if not ok:
                print(
                    f"{exp_name} / {ch}: no non-empty responder data — skipped."
                )
            else:
                print(
                    f"{exp_name} / {ch}: responder violin — "
                    f"{n_resp}/{n_total} cells passed |dF/F0| ≥ {dff_threshold}."
                )


# =============================================================================
# Pipeline step 7 — Sliding-window pairwise correlation
# =============================================================================
def plot_sliding_correlation(experiments, state):
    """Sliding-window Pearson + Spearman pairwise correlation traces.

    For each (experiment, channel):
        * compute the full-series Pearson r per pair, drop pairs with
          ``r >= global_corr_cutoff`` (these are dominated by trend rather
          than dynamics)
        * for the surviving pairs, slide a window of size
          ``PLOT_PARAMS_SLIDING['window_size']`` with stride ``step`` across
          the time series and compute Pearson + Spearman per pair per window
        * plot all per-pair traces in light colour with the across-pair mean
          and ±N·SEM band overlaid

    Output: one stacked Pearson/Spearman figure per (experiment, channel).
    """
    pp = PLOT_PARAMS_SLIDING
    window_size = pp["window_size"]
    step = pp["step"]
    half_w = window_size // 2
    cutoff = pp["global_corr_cutoff"]
    sem_n = pp["sem_n"]

    for exp_name, cfg in experiments.items():
        for ch in cfg["channels"]:
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            lum_matrix = df[frame_cols].values
            n_cells, n_frames = lum_matrix.shape

            if n_cells < 2 or n_frames < window_size:
                print(
                    f"{exp_name} / {ch}: insufficient data "
                    f"(n_cells={n_cells}, n_frames={n_frames}) — skipping."
                )
                continue

            # Full-series Pearson, used to drop already-correlated pairs.
            corr_full = pd.DataFrame(lum_matrix).T.corr(method="pearson").values
            i_idx, j_idx = np.triu_indices(n_cells, k=1)
            pw_pearson_full = corr_full[i_idx, j_idx]

            pair_mask = pw_pearson_full < cutoff
            n_pairs_all = len(pw_pearson_full)
            n_pairs_filtered = int(pair_mask.sum())
            if n_pairs_filtered < 2:
                print(
                    f"{exp_name} / {ch}: only {n_pairs_filtered} pairs below "
                    f"cutoff {cutoff} — skipping."
                )
                continue
            i_idx_f = i_idx[pair_mask]
            j_idx_f = j_idx[pair_mask]
            print(
                f"{exp_name} / {ch}: pairs {n_pairs_filtered:,} / "
                f"{n_pairs_all:,} ({100 * n_pairs_filtered / n_pairs_all:.1f}%) "
                f"below global Pearson {cutoff}"
            )

            # Compute sliding correlations.
            centers = np.arange(half_w, n_frames - half_w, step)
            n_windows = len(centers)
            pearson_over_time = np.full((n_windows, n_pairs_filtered), np.nan)
            spearman_over_time = np.full((n_windows, n_pairs_filtered), np.nan)

            for wi, t in enumerate(
                tqdm(centers, desc=f"{exp_name} / {ch} sliding corr", leave=False)
            ):
                win = lum_matrix[:, t - half_w : t + half_w]
                for k, (i, j) in enumerate(zip(i_idx_f, j_idx_f)):
                    ti, tj = win[i], win[j]
                    if np.std(ti) < 1e-8 or np.std(tj) < 1e-8:
                        continue
                    pearson_over_time[wi, k], _ = pearsonr(ti, tj)
                    spearman_over_time[wi, k], _ = spearmanr(ti, tj)

            stim_frames = cfg["stim_frames"][ch]
            stim_min = frames_to_min(state, exp_name, ch, stim_frames) if stim_frames else []
            centers_min = frames_to_min(state, exp_name, ch, centers)
            rsp = state["real_setpoint_min"][exp_name].get(ch)

            fig, axes = plt.subplots(
                2, 1, figsize=pp["figsize"], dpi=pp["dpi"],
            )

            for ax, corr_mat, pair_color, mean_color, label in [
                (axes[0], pearson_over_time, pp["pearson_color"],
                 pp["mean_color_pearson"], "Pearson"),
                (axes[1], spearman_over_time, pp["spearman_color"],
                 pp["mean_color_spearman"], "Spearman"),
            ]:
                ax.spines[["top", "right"]].set_visible(False)
                ax.plot(
                    centers_min, corr_mat,
                    color=pair_color,
                    alpha=pp["line_alpha"],
                    lw=pp["line_lw"],
                )

                n_valid = np.sum(~np.isnan(corr_mat), axis=1)
                mean_corr = np.nanmean(corr_mat, axis=1)
                std_corr = np.nanstd(corr_mat, axis=1)
                sem_corr = np.where(
                    n_valid > 0, std_corr / np.sqrt(n_valid), np.nan
                )

                ax.fill_between(
                    centers_min,
                    mean_corr - sem_n * sem_corr,
                    mean_corr + sem_n * sem_corr,
                    color=mean_color, alpha=pp["sem_alpha"],
                    label=f"±{sem_n} SEM", zorder=4,
                )
                ax.plot(
                    centers_min, mean_corr,
                    color=mean_color, lw=pp["mean_lw"],
                    label=f"Mean {label}", zorder=5,
                )
                ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

                for idx, p in enumerate(stim_min):
                    ax.axvline(
                        p,
                        color=pp["stim_color"],
                        linewidth=pp["stim_lw"],
                        alpha=1, zorder=0,
                        label="Stimulus" if idx == 0 else None,
                    )

                if rsp is not None:
                    ax.axvline(
                        rsp,
                        color="#000000",
                        linewidth=2.0, linestyle=":",
                        alpha=0.9, zorder=6,
                        label=f"Real setpoint ({rsp:.1f} min)",
                    )

                ax.set_xlabel("Time (min)", fontsize=pp["axis_label_fontsize"])
                ax.set_ylabel(
                    f"{label} correlation",
                    fontsize=pp["axis_label_fontsize"],
                )
                ax.set_title(
                    f"{label} (window = {window_size} frames, step = {step}, "
                    f"global Pearson < {cutoff})",
                    fontsize=pp["title_fontsize"],
                    fontweight=pp["title_fontweight"],
                )
                ax.legend(fontsize=pp["legend_fontsize"])

            fig.suptitle(
                f"{exp_name} / {ch} — sliding-window pairwise correlation "
                f"({n_pairs_filtered:,} of {n_pairs_all:,} pairs)",
                fontsize=pp["suptitle_fontsize"], fontweight="bold",
            )
            plt.tight_layout()
            fig.savefig(
                fig_path(exp_name, f"{ch}_sliding_corr"),
                dpi=pp["dpi"], bbox_inches="tight",
            )
            plt.close(fig)


# =============================================================================
# Pipeline step 8 — NRK hardware-feedback luminosity log
# =============================================================================
def _setpoint_regions_from_log(entries):
    """Group consecutive log entries by setpoint.

    Returns a list of ``(start_frame, end_frame, setpoint)`` tuples.
    """
    regions = []
    if not entries:
        return regions
    start = entries[0]["frame"]
    sp = entries[0]["setpoint"]
    last = start
    for e in entries[1:]:
        if e["setpoint"] != sp:
            regions.append((start, last, sp))
            start = e["frame"]
            sp = e["setpoint"]
        last = e["frame"]
    regions.append((start, last, sp))
    return regions


def plot_nrk_hardware_log(experiments, state, exp_name="nrk_acid_13APR26"):
    """Plot the hardware-feedback luminosity log for the NRK acid experiment.

    For each channel: shaded setpoint regions (all of them, including the
    initial calibration region), the mean-luminosity trace from
    ``luminosity_log_channelN.json`` (sibling of monitoring.log), dashed
    vertical markers at every deduplicated ``add acidic media`` decision, and
    a labeled vertical line marking activation of the real (post-calibration)
    setpoint as parsed from ``monitoring.log``.

    The x-axis is minutes from monitor-start (each channel's own t=0). No
    cropping is applied — all frames present in the JSON are shown so the
    calibration→real-setpoint transition is visible in context.

    Acidic-pulse markers are collapsed to one per pulse: when the controller
    requests acidic media on consecutive frames, only the first counts (the
    others were queued during the prior pulse's 30-second delivery window).
    """
    pp = PLOT_PARAMS_HW_LOG
    cfg = experiments[exp_name]

    for ch in cfg["channels"]:
        log_path, ch_num = cfg["stim_logs"][ch]
        lum_log_path = os.path.join(
            os.path.dirname(log_path),
            f"luminosity_log_channel{ch_num}.json",
        )

        with open(lum_log_path) as f:
            entries = json.load(f)
        entries = [e for e in entries if e.get("channel") == ch_num]
        if not entries:
            print(f"NRK / {ch}: no entries in {lum_log_path} — skipping.")
            continue
        entries.sort(key=lambda e: e["frame"])

        regions = _setpoint_regions_from_log(entries)
        if not regions:
            print(f"NRK / {ch}: no setpoint regions parsed — skipping.")
            continue

        frames = [e["frame"] for e in entries]
        luminosity = [e["mean_luminosity"] for e in entries]

        # Acidic-pulse frames, collapsed across the 30 s delivery window.
        acid_frames_raw = [
            e["frame"] for e in entries if e.get("decision") == "add acidic media"
        ]
        acid_frames = _dedup_close_frames(acid_frames_raw, PULSE_DEDUP_FRAMES)

        # Convert frames -> minutes via the per-channel monitor.log lookup.
        frames_min = frames_to_min(state, exp_name, ch, frames)
        acid_min = frames_to_min(state, exp_name, ch, acid_frames) if acid_frames else []
        rsp = state["real_setpoint_min"][exp_name].get(ch)

        rsp_str = f"{rsp:.2f} min" if rsp is not None else "not detected"
        print(
            f"NRK / {ch}: {len(frames)} frames | "
            f"{len(acid_frames)} acidic pulses (raw {len(acid_frames_raw)}) | "
            f"{len(regions)} setpoint regions | real setpoint @ {rsp_str}"
        )

        fig, ax = plt.subplots(figsize=pp["figsize"], dpi=pp["dpi"])
        ax.spines[["top", "right"]].set_visible(False)

        # Shaded setpoint regions, one colour per region in chronological
        # order. The first region is calibration; the rest are real-setpoint
        # segments. We don't drop calibration here — the real-setpoint
        # vertical line below makes the boundary explicit.
        seen_sp = {}
        for idx, (start_f, end_f, sp) in enumerate(regions):
            if idx == 0:
                continue  # skip the initial calibration region
            color = pp["setpoint_colors"][idx % len(pp["setpoint_colors"])]
            label = f"Setpoint {sp:.2f}" if sp not in seen_sp else None
            seen_sp[sp] = color
            start_m, end_m = frames_to_min(state, exp_name, ch, [start_f, end_f])
            ax.fill_between(
                [start_m, end_m], 0, sp,
                color=color, alpha=pp["setpoint_alpha"],
                label=label, zorder=1,
            )
            ax.hlines(
                sp, xmin=start_m, xmax=end_m,
                colors=color, linewidths=pp["setpoint_lw"], zorder=2,
            )

        # Measured luminosity trace.
        ax.plot(
            frames_min, luminosity,
            color=pp["line_color"], linewidth=pp["line_lw"],
            label="Mean luminosity", zorder=3,
        )

        # Acidic-pulse markers (one per pulse, post-dedup).
        for i, m in enumerate(acid_min):
            ax.axvline(
                m,
                color=pp["acid_color"],
                linewidth=pp["acid_lw"], linestyle="--",
                label="Acidic pulse" if i == 0 else None,
                zorder=4,
            )

        # Real-setpoint activation marker (parsed from monitoring.log).
        if rsp is not None:
            ax.axvline(
                rsp,
                color="#000000",
                linewidth=2.0, linestyle=":",
                alpha=0.9, zorder=5,
                label=f"Real setpoint ({rsp:.1f} min)",
            )

        # Trim view to the first 30 minutes (reviewer request). Y-limits are
        # recomputed over only the in-window samples so the trace fills the panel.
        x_lo = float(np.asarray(frames_min).min())
        x_hi = 30.0
        in_window = [
            lum for lum, m in zip(luminosity, frames_min)
            if x_lo <= m <= x_hi
        ]
        if in_window:
            y_lo = min(in_window) - 2
            y_hi = max(in_window) * 1.05
        else:
            y_lo = min(luminosity) - 2
            y_hi = max(luminosity) * 1.05
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(x_lo, x_hi)

        ax.set_title(
            f"NRK / {ch} — hardware feedback luminosity log",
            fontsize=pp["title_fontsize"], fontweight=pp["title_fontweight"],
        )
        ax.set_xlabel("Time (min)", fontsize=pp["axis_label_fontsize"])
        ax.set_ylabel("Mean luminosity", fontsize=pp["axis_label_fontsize"])
        ax.legend(loc="upper left", fontsize=pp["legend_fontsize"])
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, f"{ch}_hw_lum_log"),
            dpi=pp["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================
def main():
    """Run the full April 28 figure pipeline."""
    # Print experiment summary, mirroring the notebook's first cell.
    for name, cfg in EXPERIMENTS.items():
        extras = " + stim_logs" if "stim_logs" in cfg else ""
        extras += " + stim_minutes" if "stim_minutes" in cfg else ""
        print(
            f"{name:25s} | {len(cfg['channels'])} channels | "
            f"sampled bg: {BG_FIT['grid_n']}x{BG_FIT['grid_n']} grid, "
            f"degree {BG_FIT['poly_degree']}{extras}"
        )

    # Step 0: resolve stim_frames into a {channel: [frame, ...]} dict per experiment.
    resolve_all_stim_frames(EXPERIMENTS)

    # Step 1: per-frame sampled polynomial background correction.
    state = compute_background_correction(EXPERIMENTS, recompute=RECOMPUTE_BG)

    # Step 1b: per-frame minutes lookup + NRK real-setpoint anchor.
    build_frame_to_minutes_lookups(EXPERIMENTS, state)

    # Steps 2–8: figures for every experiment / channel.
    plot_bg_diagnostic(EXPERIMENTS, state)
    plot_time_traces(EXPERIMENTS, state)
    plot_corrected_traces(EXPERIMENTS, state)
    plot_dff(EXPERIMENTS, state)
    plot_dff_mean_combined(EXPERIMENTS, state)
    plot_dff_response_diagnostic(EXPERIMENTS, state)
    plot_correlation_vs_distance(EXPERIMENTS, state)
    plot_trace_clustering(EXPERIMENTS, state)
    # Absolute peak-luminosity violin disabled per reviewer request — only
    # peak−baseline deltas are shown going forward.
    # plot_per_stimulus_peak_violins(EXPERIMENTS, state)
    plot_per_stimulus_response_violins(EXPERIMENTS, state)
    responder_thresholds = compute_responder_thresholds(
        EXPERIMENTS, state, alpha=0.01,
    )
    plot_per_stimulus_response_violins_responders(
        EXPERIMENTS, state, thresholds=responder_thresholds,
    )
    # plot_sliding_correlation(EXPERIMENTS, state)

    # Step 9: NRK-only hardware feedback luminosity log.
    plot_nrk_hardware_log(EXPERIMENTS, state)

    print("\nDone.")


if __name__ == "__main__":
    main()