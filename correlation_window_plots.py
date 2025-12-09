import os
import numba
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

# --- Utility Function for Logging ---
def log_message(log_file_path, message, print_to_console=False):
    """Appends a message to the log file and optionally prints it to the console."""
    with open(log_file_path, 'a') as f:
        f.write(message + '\n')
    if print_to_console:
        print(message)

# --- Core Functions (Unchanged) ---
@numba.jit(nopython=True)
def pairwise_distances_unique_numba(centers):
    """Calculates all unique pairwise Euclidean distances using Numba."""
    n = centers.shape[0]
    num_pairs = n * (n - 1) // 2
    pairs = np.empty((num_pairs, 2), dtype=np.int32)
    dists = np.empty(num_pairs, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            pairs[k, 0] = i
            pairs[k, 1] = j
            dists[k] = d
            k += 1
    return pairs, dists

def calculate_pairwise_trace_correlation(lum_df, cell_ids, log_file_path, frame_start=None, frame_stop=None):
    """
    Calculates Pearson correlation (R) for the specified luminosity segment.
    Log messages about excluded cells are handled inside this function.
    """
    if frame_start is None and frame_stop is None:
        lum_cols = [col for col in lum_df.columns if col.startswith('f')]
        segment_info = "Full Trace"
    else:
        lum_cols = [f'f{i}' for i in range(frame_start, frame_stop)]
        segment_info = f"Frames {frame_start} to {frame_stop - 1}"

    if not lum_cols:
        log_message(log_file_path, f"Warning: No frames in segment: {segment_info}. Returning empty DataFrame.")
        return pd.DataFrame({"CellID": [], "TargetCellID": [], "TraceCorrelation_R": []})
        
    missing_cols = [col for col in lum_cols if col not in lum_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required frame columns: {missing_cols}")

    sorted_lum_df = lum_df.set_index("CellID").sort_index()
    traces_df = sorted_lum_df.loc[cell_ids, lum_cols].copy()
    
    # Filter for valid traces
    cells_with_nan = traces_df.index[traces_df.isnull().any(axis=1)].tolist()
    trace_stdev = traces_df.std(axis=1)
    cells_with_zero_variance = trace_stdev.index[trace_stdev == 0].tolist()
    invalid_cell_ids = set(cells_with_nan + cells_with_zero_variance)
    valid_cell_ids = [cid for cid in cell_ids if cid not in invalid_cell_ids]
    
    # LOG exclusion message
    log_message(log_file_path, f"Segment {segment_info}: Excluded {len(invalid_cell_ids)} cells due to invalid trace.")
    
    valid_traces = traces_df.loc[valid_cell_ids].values
    n_valid_cells = len(valid_cell_ids)
    
    correlations = []
    cell_pairs = []

    for i, j in combinations(range(n_valid_cells), 2):
        r, _ = pearsonr(valid_traces[i], valid_traces[j])
        correlations.append(r)
        cell_pairs.append((valid_cell_ids[i], valid_cell_ids[j]))

    return pd.DataFrame({
        "CellID": [p[0] for p in cell_pairs],
        "TargetCellID": [p[1] for p in cell_pairs],
        "TraceCorrelation_R": correlations,
    })

def plot_sliding_window_correlations(dist_df, lum_df, cell_ids, num_frames, data_path, window_size, step_size, log_file_path):
    """
    Calculates and plots the pairwise distance vs trace correlation for sliding windows.
    Generates a single figure with subplots for each window.
    """
    
    # 1. Define all sliding windows
    windows = []
    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        windows.append({"start": start, "stop": end, "name": f"F{start}-F{end-1}"})
    
    if not windows:
        log_message(log_file_path, f"CRITICAL ERROR: No windows of size {window_size} found for {num_frames} frames.")
        return

    # 2. Setup Figure
    num_plots = len(windows)
    ncols = 2
    nrows = int(np.ceil(num_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    
    log_message(log_file_path, f"\nStarting {num_plots} sliding window analyses ({window_size} frames, step {step_size})...")

    # 3. Calculate and Plot for each window
    for i, window in enumerate(windows):
        ax = axes.flatten()[i]
        
        # Calculate functional correlation for the segment (Logging handled inside)
        df_corr = calculate_pairwise_trace_correlation(
            lum_df, cell_ids, 
            frame_start=window["start"], 
            frame_stop=window["stop"], 
            log_file_path=log_file_path
        )
        
        # Merge distance and correlation data
        merged = pd.merge(
            dist_df[["CellID", "TargetCellID", "Distance"]],
            df_corr[["CellID", "TargetCellID", "TraceCorrelation_R"]],
            on=["CellID", "TargetCellID"],
            how="inner"
        )

        x = merged["Distance"].values
        y = merged["TraceCorrelation_R"].values
        
        if len(x) < 2:
            log_message(log_file_path, f"Warning: Insufficient cell pairs for window {window['name']}. Skipping plot.")
            continue

        # Calculate and Log correlations
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        
        log_message(log_file_path, f"--- Window: {window['name']} ---")
        log_message(log_file_path, f"Pearson r={pearson_r:.4f}, p={pearson_p:.4f}")
        log_message(log_file_path, f"Spearman rho={spearman_r:.4f}, p={spearman_p:.4f}")

        # Plotting (using hexbin for density)
        ax.hexbin(x, y, gridsize=20, cmap='Blues', mincnt=1, vmin=0, vmax=200)
        
        # Linear Fit Plot
        m_p, b_p = np.polyfit(x, y, 1)
        ax.plot(np.sort(x), m_p * np.sort(x) + b_p, color="red", linewidth=1, linestyle='--')

        # Add title and stats
        ax.set_title(f"Window {window['name']}")
        ax.text(0.95, 0.05, 
                f"$r$={pearson_r:.2f}, $\\rho$={spearman_r:.2f}",
                transform=ax.transAxes, 
                fontsize=8, 
                verticalalignment='bottom', 
                horizontalalignment='right', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

        ax.set_xlabel("Distance (pixels)")
        ax.set_ylabel("Trace Correlation ($R_{trace}$)")
        ax.set_ylim(-1.05, 1.05)


    # 4. Hide any unused subplots
    for j in range(num_plots, nrows * ncols):
        fig.delaxes(axes.flatten()[j])

    # 5. Finalize and Save Figure
    fig.suptitle(f"Distance vs. Trace Correlation (Sliding Windows: Size {window_size}, Step {step_size})\nDistance Reference: Frame {dist_df['Frame'].iloc[0]}", 
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle

    plots_dir = os.path.join(data_path, "plots")
    out_png = os.path.join(plots_dir, f"Distance_vs_TraceCorrelation_Sliding_Window_W{window_size}_S{step_size}.png")
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    
    log_message(log_file_path, "Sliding window plot complete.")
    log_message(log_file_path, f"Saved plot: {out_png}")
    

# --- Modified Unified Main Execution ---
def run_all_analyses(exp, log_file_path):
    """Loads data, calculates constant distance (F0), and runs all Trace Correlation analyses."""
    
    data_path = f"{exp}/analysis"
    traj_file = f"{data_path}/trajectories_complete.csv"
    lum_file = f"{data_path}/luminosity_no_bground_complete.csv"

    # 1. Load and Clean Data
    try:
        traj_df = pd.read_csv(traj_file)
        lum_df = pd.read_csv(lum_file)
    except FileNotFoundError as e:
        log_message(log_file_path, f"CRITICAL ERROR: Data file not found: {e.filename}", print_to_console=True)
        raise e

    traj_df = traj_df[traj_df["CellID"] != 0]
    lum_df = lum_df[lum_df["CellID"] != 0]
    cell_ids = traj_df["CellID"].values
    coords = traj_df.drop(columns=["CellID"]).values
    num_frames = coords.shape[1] // 2
    
    if num_frames < 3: # Reduced minimum requirement to 3 frames for size 3 window
        log_message(log_file_path, f"CRITICAL ERROR: Need at least 3 frames. Found {num_frames}.", print_to_console=True)
        raise ValueError(f"Need at least 3 frames (f0-f2) for a size 3 window. Found {num_frames}.")

    # 2. Calculate Physical Distance (Frame 0)
    frame_for_distance = 0
    frame_coords = coords[:, (2*frame_for_distance):(2*frame_for_distance+2)]
    pairs, dists = pairwise_distances_unique_numba(frame_coords)

    df_dist_ref = pd.DataFrame({
        "CellID": cell_ids[pairs[:, 0]],
        "TargetCellID": cell_ids[pairs[:, 1]],
        "Distance": dists,
        "Frame": frame_for_distance
    })
    log_message(log_file_path, f"Calculated {len(df_dist_ref)} unique pairwise distances using Frame {frame_for_distance} for all plots.")
    
    # --- Analysis 1: Sliding Window Correlation Plots ---
    WINDOW_SIZE = 3
    STEP_SIZE = 1

    plot_sliding_window_correlations(
        df_dist_ref, 
        lum_df, 
        cell_ids, 
        num_frames, 
        data_path, 
        window_size=WINDOW_SIZE, 
        step_size=STEP_SIZE, 
        log_file_path=log_file_path
    )

# --- Final Execution (Unchanged) ---
exp = 'c2c12_carbachol_1'
data_path = f"{exp}/analysis"
plots_dir = os.path.join(data_path, "plots")
os.makedirs(plots_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(plots_dir, "correlation_window_log.txt")

# Clear the log file or add a header for a fresh run
with open(LOG_FILE_PATH, 'w') as f:
    f.write("--- Correlation Analysis Log ---\n")
    f.write(f"Experiment: {exp}\n")
    f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

log_message(LOG_FILE_PATH, "Starting all analyses...")

try:
    run_all_analyses(exp, LOG_FILE_PATH)
except (FileNotFoundError, ValueError) as e:
    pass 

log_message(LOG_FILE_PATH, "\nAll analyses complete.")
print(f"Analysis complete. Results logged to: {LOG_FILE_PATH}")