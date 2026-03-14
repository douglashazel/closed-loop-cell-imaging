import os
import re
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from io_utils import load_msgpack, save_msgpack, lum_dict_to_df, traj_dict_to_df

# ── Style ─────────────────────────────────────────────────────────────────────
PLOT_PARAMS = {
    'figsize': (10, 6),
    'figsize_wide': (18, 10),
    'dpi': 300,
    'title_fontsize': 14,
    'title_fontweight': 'bold',
    'axis_label_fontsize': 13,
    'legend_fontsize': 10,
    'cell_color': "#074f79cc",
    'cell_alpha': 0.3,
    'cell_lw': 0.5,
    'mean_color': '#1a1a1a',
    'mean_lw': 1.8,
    'sem_alpha': 0.18,
    'stim_color': '#e74c3c',
    'stim_lw': 1.8,
    'f0_color': '#1a9d51',
    'f0_lw': 1.8,
    'derivative_cmap': 'twilight_shifted',
    'trace_cmap': 'twilight_shifted',
    'std_color': 'darkred',
    'std_markersize': 3,
}


# ── Spline background correction ─────────────────────────────────────────────
def estimate_background_surface(image, mask, grid_size=6):
    h, w = image.shape
    ys = np.linspace(0, h - 1, grid_size)
    xs = np.linspace(0, w - 1, grid_size)

    gy = ys.astype(int)
    gx = xs.astype(int)

    Z = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i, yy in enumerate(gy):
        for j, xx in enumerate(gx):
            r = 20
            y0, y1 = max(0, yy - r), min(h, yy + r)
            x0, x1 = max(0, xx - r), min(w, xx + r)
            local = image[y0:y1, x0:x1]
            local_mask = mask[y0:y1, x0:x1]
            if np.any(local_mask):
                Z[i, j] = np.mean(local[local_mask])
            else:
                Z[i, j] = np.mean(local)

    spline = RectBivariateSpline(ys, xs, Z, kx=3, ky=3)
    return spline(np.arange(h), np.arange(w))


def illumination_correct(img, grid_size=6, sigma_threshold=1.5, max_iters=20, tol=1e-3):
    corrected = img.copy()
    prev_bg = np.zeros_like(img)

    for _ in range(max_iters):
        mu = np.mean(corrected)
        sd = np.std(corrected)
        mask = corrected < (mu + sigma_threshold * sd)

        bg = estimate_background_surface(img, mask, grid_size)

        diff = np.sqrt(np.mean((bg - prev_bg) ** 2))
        corrected = img - bg

        if diff < tol * np.std(img):
            break

        prev_bg = bg

    return corrected, bg


# ── Image loading ─────────────────────────────────────────────────────────────
def extract_number(filename):
    match = re.search(r'timepoint_(\d+)', filename)
    return int(match.group(1)) if match else -1


def load_image_float(path):
    img = np.array(Image.open(path), dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    return img


# ── Background correction logic ──────────────────────────────────────────────
def compute_per_cell_background(images, image_dir, traj_df, n_frames):
    """For each frame, compute spline bg surface and query at each cell's position."""
    cell_ids = traj_df['CellID'].values
    bg_values = {int(cid): {} for cid in cell_ids}

    for frame_idx in tqdm(range(n_frames), desc="Spline background correction"):
        if frame_idx >= len(images):
            break

        img = load_image_float(os.path.join(image_dir, images[frame_idx]))
        _, bg_surface = illumination_correct(img)

        xcol = f'x{frame_idx}'
        ycol = f'y{frame_idx}'

        if xcol not in traj_df.columns or ycol not in traj_df.columns:
            continue

        for _, row in traj_df.iterrows():
            cid = int(row['CellID'])
            x_val = row.get(xcol)
            y_val = row.get(ycol)

            if x_val is not None and y_val is not None and not (isinstance(x_val, float) and np.isnan(x_val)):
                yi = int(np.clip(round(y_val), 0, bg_surface.shape[0] - 1))
                xi = int(np.clip(round(x_val), 0, bg_surface.shape[1] - 1))
                bg_values[cid][f'f{frame_idx}'] = float(bg_surface[yi, xi])

    return bg_values


def apply_correction(lum_dict, bg_values):
    """Subtract per-cell background from raw luminosity."""
    corrected = {}
    for cell_id_str, frames in lum_dict.items():
        cid = int(cell_id_str)
        if cid == 0:
            continue
        corrected_frames = {}
        for fkey, val in frames.items():
            if val is None:
                corrected_frames[fkey] = None
                continue
            bg_val = bg_values.get(cid, {}).get(fkey)
            if bg_val is not None:
                corrected_frames[fkey] = val - bg_val
            else:
                corrected_frames[fkey] = val
        corrected[cell_id_str] = corrected_frames
    return corrected


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_corrected_traces(df, analysis_dir, tag):
    cmap = plt.get_cmap(PLOT_PARAMS['trace_cmap'])
    colors = cmap(np.linspace(0, 1, len(df)))

    fig, ax = plt.subplots(figsize=PLOT_PARAMS['figsize'], dpi=PLOT_PARAMS['dpi'])
    ax.spines[['top', 'right']].set_visible(False)

    for (_, row), color in zip(df.iterrows(), colors):
        non_nan = row.dropna()
        frame_keys = [c for c in non_nan.index if str(c).startswith('f')]
        frames = [int(str(x).lstrip('f')) for x in frame_keys]
        vals = non_nan[frame_keys].values
        ax.plot(frames, vals, alpha=0.7, color=color)

    ax.set_xlabel("Frame", fontsize=PLOT_PARAMS['axis_label_fontsize'])
    ax.set_ylabel("Luminosity (cell - background)", fontsize=PLOT_PARAMS['axis_label_fontsize'])
    ax.set_title("Cell Luminosity Over Time (Spline Corrected)",
                 fontsize=PLOT_PARAMS['title_fontsize'], fontweight=PLOT_PARAMS['title_fontweight'])

    plot_dir = os.path.join(analysis_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, f'corrected_traces{tag}.png'), dpi=PLOT_PARAMS['dpi'])
    plt.close(fig)


def plot_derivative_and_std(df, analysis_dir):
    df_data = df.set_index('CellID')
    frame_cols = sorted([c for c in df_data.columns if c.startswith('f')], key=lambda x: int(x[1:]))
    df_frames = df_data[frame_cols]

    # Derivative
    derivative_matrix = np.gradient(df_frames.values, axis=1)
    df_derivative = pd.DataFrame(derivative_matrix, index=df_frames.index, columns=df_frames.columns)

    fig, ax = plt.subplots(figsize=PLOT_PARAMS['figsize'], dpi=PLOT_PARAMS['dpi'])
    ax.spines[['top', 'right']].set_visible(False)
    df_derivative.T.plot(legend=False, ax=ax, linewidth=1, cmap=PLOT_PARAMS['derivative_cmap'], alpha=0.7)
    ax.set_title("Derivative of Cell Luminosity Over Time",
                 fontsize=PLOT_PARAMS['title_fontsize'], fontweight=PLOT_PARAMS['title_fontweight'])
    ax.set_xlabel("Frame", fontsize=PLOT_PARAMS['axis_label_fontsize'])
    ax.set_ylabel("Derivative of Luminosity", fontsize=PLOT_PARAMS['axis_label_fontsize'])

    plot_dir = os.path.join(analysis_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, 'luminosity_derivative_complete.png'), dpi=PLOT_PARAMS['dpi'])
    plt.close(fig)

    # STD across cells per frame
    std_per_frame = df_frames.T.std(axis=1)
    frames = [int(str(x).lstrip('f')) for x in std_per_frame.index]
    std_values = std_per_frame.values

    fig, ax = plt.subplots(figsize=PLOT_PARAMS['figsize'], dpi=PLOT_PARAMS['dpi'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.plot(frames, std_values, 'o-', color=PLOT_PARAMS['std_color'], alpha=0.7,
            markersize=PLOT_PARAMS['std_markersize'])
    ax.set_xlabel("Frame", fontsize=PLOT_PARAMS['axis_label_fontsize'])
    ax.set_ylabel("Standard Deviation of Luminosity (Across All Cells)",
                  fontsize=PLOT_PARAMS['axis_label_fontsize'])
    ax.set_title("Luminosity Variability Over Time",
                 fontsize=PLOT_PARAMS['title_fontsize'], fontweight=PLOT_PARAMS['title_fontweight'])

    fig.savefig(os.path.join(plot_dir, 'std_luminosity_complete.png'), dpi=PLOT_PARAMS['dpi'])
    plt.close(fig)


def plot_avg_and_dff(df, analysis_dir, f0_frame, stim_frames):
    df_data = df.set_index('CellID')
    frame_cols = sorted([c for c in df_data.columns if c.startswith('f')], key=lambda c: int(str(c).lstrip('f')))
    frame_nums = np.array([int(str(c).lstrip('f')) for c in frame_cols])

    mat = df_data[frame_cols].values

    f0_col = f'f{f0_frame}'
    if f0_col not in df_data.columns:
        print(f"Warning: F0 frame {f0_frame} not in data, skipping dF/F0 plot")
        return

    F0 = df_data[f0_col].values[:, np.newaxis]
    # Avoid division by zero
    F0_safe = np.where(F0 == 0, np.nan, F0)
    dff_mat = (mat - F0) / F0_safe

    fig, axes = plt.subplots(2, 1, figsize=PLOT_PARAMS['figsize_wide'], dpi=PLOT_PARAMS['dpi'], sharex=True)

    panels = [
        (mat, "Luminosity (Corrected)", "Corrected Luminosity"),
        (dff_mat, "dF/F\u2080", f"dF/F\u2080  (F\u2080 = frame {f0_frame})"),
    ]

    for ax, (data, ylabel, title_suffix) in zip(axes, panels):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(top=False, right=False)

        mean_trace = np.nanmean(data, axis=0)

        for row in data:
            ax.plot(frame_nums, row, color=PLOT_PARAMS['cell_color'],
                    alpha=PLOT_PARAMS['cell_alpha'], linewidth=PLOT_PARAMS['cell_lw'], zorder=1)

        ax.plot(frame_nums, mean_trace, color=PLOT_PARAMS['mean_color'],
                linewidth=PLOT_PARAMS['mean_lw'], zorder=3, label='Mean')

        if stim_frames:
            for idx, p in enumerate(stim_frames):
                ax.axvline(p, color=PLOT_PARAMS['stim_color'], linewidth=PLOT_PARAMS['stim_lw'],
                           alpha=1, zorder=0, label='Stimulus' if idx == 0 else None)

        ax.axvline(f0_frame, color=PLOT_PARAMS['f0_color'], linewidth=PLOT_PARAMS['f0_lw'],
                   linestyle='--', zorder=4, label=f'F0 frame ({f0_frame})')

        ax.set_ylabel(ylabel, fontsize=PLOT_PARAMS['axis_label_fontsize'])
        ax.set_title(title_suffix, fontsize=PLOT_PARAMS['title_fontsize'],
                     fontweight=PLOT_PARAMS['title_fontweight'])
        ax.legend(fontsize=PLOT_PARAMS['legend_fontsize'], loc='upper right');

    axes[-1].set_xlabel("Frame", fontsize=PLOT_PARAMS['axis_label_fontsize'])
    fig.suptitle(f"Luminosity Traces \u2014 All Cells ({mat.shape[0]} cells)",
                 fontsize=PLOT_PARAMS['title_fontsize'], fontweight=PLOT_PARAMS['title_fontweight'], y=1.01)
    plt.tight_layout();

    plot_dir = os.path.join(analysis_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, 'avg_and_dff_complete.png'), dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--f0_frame", type=int, default=None)
    parser.add_argument("--stim_frames", type=str, default=None,
                        help="Comma-separated stimulus frame indices")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    image_dir = args.image_dir

    stim_frames = None
    if args.stim_frames:
        stim_frames = [int(x.strip()) for x in args.stim_frames.split(',')]

    # Load image file list
    images = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))],
        key=extract_number
    )
    n_frames = len(images)
    print(f"Found {n_frames} frames in {image_dir}")

    # Load trajectory data for cell positions
    traj_path = os.path.join(analysis_dir, 'trajectories_complete.json')
    if not os.path.exists(traj_path):
        print(f"Trajectory file not found: {traj_path}")
        exit(1)
    traj_df = traj_dict_to_df(load_msgpack(traj_path))
    print(f"Loaded {len(traj_df)} complete cell trajectories")

    # ── Step 1: Spline background correction ──────────────────────────────
    bg_cache_path = os.path.join(analysis_dir, 'bg_values_cache.npy')

    if os.path.exists(bg_cache_path):
        print(f">>> Loading cached background values from {bg_cache_path}")
        bg_values = np.load(bg_cache_path, allow_pickle=True).item()
    else:
        print(f">>> Computing spline background correction for {args.exp}")
        bg_values = compute_per_cell_background(images, image_dir, traj_df, n_frames)
        np.save(bg_cache_path, bg_values)
        print(f"  Cached background values to {bg_cache_path}")

    for tag in ['_complete', '']:
        lum_path = os.path.join(analysis_dir, f'luminosity{tag}.json')
        if not os.path.exists(lum_path):
            print(f"  Skipping {lum_path} (not found)")
            continue

        lum_dict = load_msgpack(lum_path)
        corrected_dict = apply_correction(lum_dict, bg_values)

        out_path = os.path.join(analysis_dir, f'luminosity_corrected{tag}.json')
        save_msgpack(corrected_dict, out_path)
        print(f"  Saved {out_path}")

        corrected_df = lum_dict_to_df(corrected_dict)
        plot_corrected_traces(corrected_df, analysis_dir, tag)
        print(f"  Saved corrected_traces{tag}.png")

    # ── Step 2: Derivative and STD (complete cells only) ──────────────────
    corrected_complete_path = os.path.join(analysis_dir, 'luminosity_corrected_complete.json')
    if os.path.exists(corrected_complete_path):
        print(f">>> Computing derivative and STD for {args.exp}")
        df_complete = lum_dict_to_df(load_msgpack(corrected_complete_path))
        try:
            plot_derivative_and_std(df_complete, analysis_dir)
            print("  Saved luminosity_derivative_complete.png and std_luminosity_complete.png")
        except TypeError:
            print(f"  Error computing derivative/STD for {args.exp}")

    # ── Step 3: Average luminosity + dF/F0 (complete cells only) ──────────
    if args.f0_frame is not None and os.path.exists(corrected_complete_path):
        print(f">>> Computing average luminosity and dF/F0 for {args.exp}")
        df_complete = lum_dict_to_df(load_msgpack(corrected_complete_path))
        plot_avg_and_dff(df_complete, analysis_dir, args.f0_frame, stim_frames)
        print("  Saved avg_and_dff_complete.png")

    print(f">>> Post-analysis finished for {args.exp}")
