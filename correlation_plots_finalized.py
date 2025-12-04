import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
import numba

# --- Core Functions ---
@numba.jit(nopython=True)
def pairwise_distances_unique_numba(centers):
    """Calculates all unique pairwise Euclidean distances using Numba."""
    n = centers.shape[0]
    # Pre-allocate for performance
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

def calculate_pairwise_trace_correlation(lum_df, cell_ids, frame_start=None, frame_stop=None):
    """
    Calculates Pearson correlation (R) for the specified luminosity segment 
    (f[frame_start] up to f[frame_stop-1]) between all unique cell pairs.
    Handles invalid traces (NaNs, zero variance).
    """
    if frame_start is None and frame_stop is None:
        # Full trace correlation
        lum_cols = [col for col in lum_df.columns if col.startswith('f')]
        segment_info = "Full Trace"
    else:
        # Segment correlation
        lum_cols = [f'f{i}' for i in range(frame_start, frame_stop)]
        segment_info = f"Frames {frame_start} to {frame_stop - 1}"

    if not lum_cols:
        print(f"Warning: No frames in segment: {segment_info}. Returning empty DataFrame.")
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
    
    print(f"Segment {segment_info}: Excluded {len(invalid_cell_ids)} cells due to invalid trace.")
    
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

def plot_correlation(dist_df, y_data_df, data_path, analysis_name, title_suffix):
    """
    Plots the pairwise distance vs a chosen Y-variable (luminosity difference or trace correlation).
    """
    # Assuming the Y column name is either 'LuminosityChangeDiff' or 'TraceCorrelation_R'
    y_col = y_data_df.columns[-1]
    
    merged = pd.merge(
        dist_df[["CellID", "TargetCellID", "Distance"]],
        y_data_df[["CellID", "TargetCellID", y_col]],
        on=["CellID", "TargetCellID"],
        how="inner"
    )

    x = merged["Distance"].values
    y = merged[y_col].values
    
    if "LuminosityChangeDiff" in y_col:
        # For the first plot, we use the absolute difference
        y = np.abs(y)
        ylabel = "|Δ(ΔLuminosity)|"
        ylim_val = None
    else: # TraceCorrelation_R
        ylabel = "Luminosity Trace Correlation ($R_{trace}$)"
        ylim_val = (-1.05, 1.05)
    
    if len(x) < 2:
        print(f"Warning: Insufficient cell pairs for {analysis_name}.")
        return

    # Calculate correlations
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    print(f"--- Analysis: {analysis_name} ---")
    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.4f}")
    print(f"Spearman r={spearman_r:.4f}, p={spearman_p:.4f}")

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=15, alpha=0.6, label="Cell Pairs")

    # Linear Fit
    m_p, b_p = np.polyfit(x, y, 1)
    label_text = (
        f"Linear Fit (Pearson $r$={pearson_r:.3f}, $p$={pearson_p:.4f})\n"
        f"Spearman $\\rho$={spearman_r:.3f}, $p$={spearman_p:.4f}"
    )
    plt.plot(np.sort(x), m_p * np.sort(x) + b_p, color="red", linewidth=2, label=label_text)

    plt.xlabel("Pairwise Distance (pixels)")
    plt.ylabel(ylabel)
    plt.title(f"{analysis_name}\n{title_suffix}")
    plt.legend()
    if ylim_val:
        plt.ylim(ylim_val)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    out_png = f"{data_path}/plots/{analysis_name.replace(' ', '_').replace(':', '')}.png"
    plt.savefig(out_png, dpi=300)
    print(f"Plot complete for {analysis_name}")
    print(f"Saved plot: {out_png}")

# --- Unified Main Execution ---
def run_all_analyses(exp):
    """Loads data, calculates constant distance, and runs all four analyses."""
    
    data_path = f"{exp}/analysis"
    traj_file = f"{data_path}/trajectories_complete.csv"
    lum_file = f"{data_path}/luminosity_no_bground_complete.csv"

    # 1. Load and Clean Data
    try:
        traj_df = pd.read_csv(traj_file)
        lum_df = pd.read_csv(lum_file)
    except FileNotFoundError as e:
        # Note: You need to have these files in the correct path for the script to run
        raise FileNotFoundError(f"Error: Data file not found. Please ensure '{e.filename}' is accessible.")

    traj_df = traj_df[traj_df["CellID"] != 0]
    lum_df = lum_df[lum_df["CellID"] != 0]
    cell_ids = traj_df["CellID"].values
    coords = traj_df.drop(columns=["CellID"]).values
    num_frames = coords.shape[1] // 2
    
    # Check for minimum required frames
    if num_frames < 7: # Need at least f0 to f6 for all analyses
        raise ValueError(f"Need at least 7 frames (f0-f6) for all analyses. Found {num_frames}.")

    # 2. Calculate Physical Distance (Constant for 3 of 4 plots)
    # We use Frame 0 as the reference for physical distance.
    frame_for_distance = 0
    frame_coords = coords[:, (2*frame_for_distance):(2*frame_for_distance+2)]
    
    # Use the Numba-accelerated function
    pairs, dists = pairwise_distances_unique_numba(frame_coords)

    # DataFrame for physical distance
    df_dist_ref = pd.DataFrame({
        "CellID": cell_ids[pairs[:, 0]],
        "TargetCellID": cell_ids[pairs[:, 1]],
        "Distance": dists,
        "Frame": frame_for_distance
    })
    print(f"Calculated {len(df_dist_ref)} unique pairwise distances using Frame {frame_for_distance}.")

    # --- Analysis 1: Distance (F5) vs. DeltaDeltaLuminosity (F4->F5 Diff) ---
    analysis_frame = 5
    
    # Get coordinates for distance at Frame 5
    frame_coords_f5 = coords[:, (2*analysis_frame):(2*analysis_frame+2)]
    pairs_f5, dists_f5 = pairwise_distances_unique_numba(frame_coords_f5)
    df_dist_f5 = pd.DataFrame({
        "CellID": cell_ids[pairs_f5[:, 0]],
        "TargetCellID": cell_ids[pairs_f5[:, 1]],
        "Distance": dists_f5,
        "Frame": analysis_frame
    })

    # Compute Δluminosity: f5 - f4
    lum_prev = lum_df[f"f{analysis_frame-1}"].values
    lum_curr = lum_df[f"f{analysis_frame}"].values
    delta_lum = lum_curr - lum_prev
    
    # Get CellID mapping back to the delta_lum array
    id_to_idx = {id: i for i, id in enumerate(lum_df["CellID"].values)}
    cell_indices = np.array([id_to_idx[id] for id in df_dist_f5["CellID"].unique() if id in id_to_idx])

    # Now compute pairwise differences in these Δluminosity values
    n = len(cell_indices)
    pairs_i = np.empty(n * (n - 1) // 2, dtype=np.int32)
    pairs_j = np.empty(n * (n - 1) // 2, dtype=np.int32)
    k = 0
    for i, j in combinations(range(n), 2):
        pairs_i[k] = cell_indices[i]
        pairs_j[k] = cell_indices[j]
        k += 1

    # Only use pairs that correspond to the calculated distances (to match original script intent)
    # The original script calculated ALL pairs, then merged. We'll do the same for consistency.
    n_all_cells = len(delta_lum)
    pairs_i_all, pairs_j_all = np.triu_indices(n_all_cells, k=1)
    
    delta_lum_diff = delta_lum[pairs_i_all] - delta_lum[pairs_j_all]

    df_delta_lum_diff = pd.DataFrame({
        "CellID": lum_df["CellID"].values[pairs_i_all],
        "TargetCellID": lum_df["CellID"].values[pairs_j_all],
        "LuminosityChangeDiff": delta_lum_diff,
        "Frame": analysis_frame
    })

    plot_correlation(
        df_dist_f5, df_delta_lum_diff, data_path, 
        analysis_name="Distance vs. ΔΔLuminosity", 
        title_suffix=f"Frame {analysis_frame} Distance, $\Delta L$ from F{analysis_frame-1} to F{analysis_frame}"
    )

    # --- Analysis 2, 3, 4: Distance (F0) vs. Trace Correlation (Segments) ---
    # Define the correlation segments to analyze
    correlation_segments = [
        {"start": None, "stop": None, "name": "Trace Correlation (Full)"},   # Full Trace
        {"start": 0, "stop": 5, "name": "Trace Correlation (F0-F4)"},       # Segment F0 to F4
        {"start": 6, "stop": num_frames, "name": "Trace Correlation (F6-End)"} # Segment F6 to End
    ]
    
    for segment in correlation_segments:
        # Calculate functional correlation for the segment
        df_corr = calculate_pairwise_trace_correlation(
            lum_df, cell_ids, 
            frame_start=segment["start"], 
            frame_stop=segment["stop"]
        )
        
        if segment["start"] is None:
            title_suffix = f"{segment['name']} (Distance at F{frame_for_distance})"
        else:
            title_suffix = f"F{segment['start']}-F{segment['stop']-1} Segment (Distance at F{frame_for_distance})"
        
        # Plot the physical distance (F0) vs. functional correlation
        plot_correlation(
            df_dist_ref, df_corr, data_path, 
            analysis_name=f"Distance vs. {segment['name']}", 
            title_suffix=title_suffix
        )
        
exp = 'c2c12_carbachol_2'
run_all_analyses()