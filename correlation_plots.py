import os
import numba
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests

# --- Utility Function for Logging ---
def log_message(log_file_path, message, print_to_console=False):
    """Appends a message to the log file and optionally prints it to the console."""
    with open(log_file_path, 'a') as f:
        f.write(message + '\n')
    if print_to_console:
        print(message)

# --- Core Functions ---
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
    log_message(log_file_path, f"Segmenting {segment_info}: Excluded {len(invalid_cell_ids)} cells due to invalid trace.")
    
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

def calculate_pairwise_delta_delta_luminosity(lum_df, frame1, frame2, cell_ids, log_file_path):
    """
    Calculates the absolute difference in the change of luminosity.
    Log messages about excluded cells are handled inside this function.
    """
    frame_cols = [f'f{frame1}', f'f{frame2}']
    
    missing_cols = [col for col in frame_cols if col not in lum_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required frame columns: {missing_cols}")
        
    sorted_lum_df = lum_df.set_index("CellID").sort_index()
    traces_df = sorted_lum_df.loc[cell_ids, frame_cols].copy()
    
    delta_L = traces_df[f'f{frame2}'] - traces_df[f'f{frame1}']
    
    valid_cells_series = delta_L.dropna()
    valid_cell_ids = valid_cells_series.index.tolist()
    valid_delta_L = valid_cells_series.values
    
    # LOG exclusion message
    log_message(log_file_path, f"ΔLuminosity (F{frame1} to F{frame2}): Excluded {len(cell_ids) - len(valid_cell_ids)} cells due to invalid traces.")
    
    n_valid_cells = len(valid_cell_ids)
    diffs = []
    cell_pairs = []

    for i, j in combinations(range(n_valid_cells), 2):
        ddl = np.abs(valid_delta_L[i] - valid_delta_L[j])
        diffs.append(ddl)
        cell_pairs.append((valid_cell_ids[i], valid_cell_ids[j]))

    return pd.DataFrame({
        "CellID": [p[0] for p in cell_pairs],
        "TargetCellID": [p[1] for p in cell_pairs],
        "LuminosityChangeDiff": diffs,
    })

def calculate_pairwise_granger_causality(lum_df, cell_ids, maxlag, log_file_path):
    """
    Calculates Granger Causality F-statistic and p-value for all unique cell pairs.
    
    Returns a DataFrame with the F-statistic and p-value for the direction:
    CellID (A) Granger-causes TargetCellID (B) (A -> B).
    """
    lum_cols = [col for col in lum_df.columns if col.startswith('f')]
    
    sorted_lum_df = lum_df.set_index("CellID").sort_index()
    traces_df = sorted_lum_df.loc[cell_ids, lum_cols].T # Transpose: frames are rows, CellIDs are columns
    
    # Filter for valid traces (no NaN, non-zero variance)
    cells_with_nan = traces_df.columns[traces_df.isnull().any()].tolist()
    trace_stdev = traces_df.std(axis=0)
    cells_with_zero_variance = trace_stdev.index[trace_stdev == 0].tolist()
    invalid_cell_ids = set(cells_with_nan + cells_with_zero_variance)
    valid_cell_ids = [cid for cid in cell_ids if cid not in invalid_cell_ids]

    log_message(log_file_path, f"Granger Causality: Excluded {len(invalid_cell_ids)} cells due to invalid trace.")
    
    # Further check: Granger requires at least (maxlag + 1) observations.
    num_observations = len(lum_cols)
    if num_observations <= maxlag:
        log_message(log_file_path, f"Warning: Not enough observations ({num_observations}) for maxlag={maxlag}. Skipping Granger analysis.")
        return pd.DataFrame({"CellID": [], "TargetCellID": [], "Granger_F_AB": [], "Granger_P_AB": []})


    valid_traces_df = traces_df[valid_cell_ids]
    # n_valid_cells = len(valid_cell_ids)
    
    granger_results = []
    cell_pairs = []

    # Iterate over all unique pairs
    for cid_A, cid_B in combinations(valid_cell_ids, 2):
        data_AB = pd.concat([valid_traces_df[cid_B], valid_traces_df[cid_A]], axis=1)

        try:
            # Test: A does NOT Granger-cause B (Null Hypothesis)
            results = grangercausalitytests(data_AB, maxlag=maxlag, verbose=False)
            
            # Extract results for the F-test at the maximum lag
            # Index 0 is the F-test; Index 1 is the p-value
            f_stat_ab = results[maxlag][0]['ssr_ftest'][0]
            p_val_ab = results[maxlag][0]['ssr_ftest'][1]

            granger_results.append({
                "CellID": cid_A, # Independent variable (Cause)
                "TargetCellID": cid_B, # Dependent variable (Effect)
                "Granger_F_AB": f_stat_ab,
                "Granger_P_AB": p_val_ab,
            })
            cell_pairs.append((cid_A, cid_B))

        except Exception as e:
            log_message(log_file_path, f"Error running Granger test for pair {cid_A}-{cid_B}: {e}")
            continue

    return pd.DataFrame(granger_results)

# --- Modified Plot Function ---
def plot_correlation(dist_df, y_data_df, data_path, analysis_name, title_suffix, log_file_path):
    """
    Plots the pairwise distance vs a chosen Y-variable and logs correlation stats.
    Includes special handling for Granger Causality plotting.
    """
    y_col = y_data_df.columns[-1]
    
    merged = pd.merge(
        dist_df[["CellID", "TargetCellID", "Distance"]],
        y_data_df[["CellID", "TargetCellID", y_col]],
        on=["CellID", "TargetCellID"],
        how="inner"
    )

    x = merged["Distance"].values
    y = merged[y_col].values
    
    # --- Y-Axis Configuration ---
    if "LuminosityChangeDiff" in y_col:
        y = np.abs(y)
        ylabel = "|Δ(ΔLuminosity)|"
        ylim_val = None
        log_metrics = True
    elif "Granger" in y_col:
        ylabel = "Granger Causality F-Statistic (A -> B)" if 'F' in y_col else "Granger Causality P-value (A -> B)"
        ylim_val = (0, 1) if 'P' in y_col else None
        log_metrics = False # Correlation metrics are not relevant here
    else: # Pearson/Spearman correlation
        ylabel = "Luminosity Trace Correlation ($R_{trace}$)"
        ylim_val = (-1.05, 1.05)
        log_metrics = True
    
    if len(x) < 2:
        log_message(log_file_path, f"Warning: Insufficient cell pairs for {analysis_name}.")
        return

    # Calculate and Log correlations (ONLY for non-Granger/non-Diff plots)
    if log_metrics:
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        log_message(log_file_path, f"--- Analysis: {analysis_name} ---")
        log_message(log_file_path, f"Pearson r={pearson_r:.4f}, p={pearson_p:.4f}")
        log_message(log_file_path, f"Spearman rho={spearman_r:.4f}, p={spearman_p:.4f}")

    # Plotting...
    gridsize = 50
    vmax = 160
    if 'Delta-Delta' in analysis_name:
        vmax = 300
    plt.figure(figsize=(8, 8))
    plt.hexbin(x, y, gridsize=gridsize, cmap='Blues', mincnt=1, vmin=0, vmax=vmax)
    plt.colorbar(label='Count in Bin')
    
    if log_metrics:
        m_p, b_p = np.polyfit(x, y, 1)
        label_text = (
            f"Linear Fit (Pearson $r$={pearson_r:.3f}, $p$={pearson_p:.4f})\n"
            f"Spearman $\\rho$={spearman_r:.3f}, $p$={spearman_p:.4f}"
        )
        plt.plot(np.sort(x), m_p * np.sort(x) + b_p, color="red", linewidth=2, linestyle='--', label=label_text)

    plt.xlabel("Pairwise Distance (pixels)")
    plt.ylabel(ylabel)
    plt.title(f"{analysis_name} (Hexbin Plot)\n{title_suffix}")
    if log_metrics:
        plt.legend()
    if ylim_val:
        plt.ylim(ylim_val)
    plt.tight_layout()

    plots_dir = os.path.join(data_path, "plots")
    out_png = os.path.join(plots_dir, f"{analysis_name.replace(' ', '_').replace(':', '')}.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    
    log_message(log_file_path, f"Plot complete for {analysis_name}")
    log_message(log_file_path, f"Saved plot: {out_png}")

# --- Unified Main Execution (Cleaned) ---
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
    
    if num_frames < 7:
        log_message(log_file_path, f"CRITICAL ERROR: Need at least 7 frames. Found {num_frames}.", print_to_console=True)
        raise ValueError(f"Need at least 7 frames (f0-f6) for all analyses. Found {num_frames}.")

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
    log_message(log_file_path, f"Calculated {len(df_dist_ref)} unique pairwise distances using Frame {frame_for_distance} for Trace Correlation plots.")
    
    # --- Analysis 2, 3, 4: Distance (F0) vs. Trace Correlation (Segments) ---
    correlation_segments = [
        {"start": None, "stop": None, "name": "Trace Correlation (Full)"},
        {"start": 0, "stop": 5, "name": "Trace Correlation (F0-F4)"},
        {"start": 6, "stop": num_frames, "name": "Trace Correlation (F6-End)"}
    ]
    
    for segment in correlation_segments:
        # Calculate functional correlation for the segment (Logging handled inside)
        df_corr = calculate_pairwise_trace_correlation(
            lum_df, cell_ids, 
            frame_start=segment["start"], 
            frame_stop=segment["stop"], 
            log_file_path=log_file_path
        )
        
        if segment["start"] is None:
            title_suffix = f"{segment['name']} (Distance at F{frame_for_distance})"
        else:
            title_suffix = f"F{segment['start']}-F{segment['stop']-1} Segment (Distance at F{frame_for_distance})"
        
        plot_correlation(
            df_dist_ref, df_corr, data_path, 
            analysis_name=f"Distance vs {segment['name']}", 
            title_suffix=title_suffix,
            log_file_path=log_file_path
        )
    
    # --- NEW ANALYSIS: Distance (F0) vs. Granger Causality ---
    MAX_LAG = 2
    log_message(log_file_path, f"\n--- Starting Granger Causality Analysis (Max Lag: {MAX_LAG}) ---")
    
    df_granger = calculate_pairwise_granger_causality(
        lum_df, cell_ids, 
        maxlag=MAX_LAG, 
        log_file_path=log_file_path
    )
    
    if not df_granger.empty:
        # Plot Granger P-value vs Distance
        plot_correlation(
            df_dist_ref, df_granger.rename(columns={'Granger_P_AB': 'Granger_P_AB_Value'}), data_path, 
            analysis_name=f"Distance vs Granger pvalue (Lag{MAX_LAG})", 
            title_suffix=f"A -> B Causality P-value (Distance at F{frame_for_distance})",
            log_file_path=log_file_path
        )
    
    log_message(log_file_path, "Granger Causality Analysis Complete.")
    
# --- Delta Delta Luminosity Analysis ---
def run_delta_delta_luminosity_analysis(exp, log_file_path):
    """Executes the specific F5 Distance vs. Delta(F5-F4) Luminosity analysis."""
    
    data_path = f"{exp}/analysis"
    traj_file = f"{data_path}/trajectories_complete.csv"
    lum_file = f"{data_path}/luminosity_no_bground_complete.csv"

    # 1. Load Data
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
    
    # --- Analysis Parameters ---
    analysis_frame_for_distance = 5
    frame1_for_delta_L = 4
    frame2_for_delta_L = 5
    
    # 2. Calculate Physical Distance at F5
    frame_coords_f5 = coords[:, (2*analysis_frame_for_distance):(2*analysis_frame_for_distance+2)]
    pairs_f5, dists_f5 = pairwise_distances_unique_numba(frame_coords_f5)
    df_dist_f5 = pd.DataFrame({
        "CellID": cell_ids[pairs_f5[:, 0]],
        "TargetCellID": cell_ids[pairs_f5[:, 1]],
        "Distance": dists_f5,
        "Frame": analysis_frame_for_distance
    })
    log_message(log_file_path, f"Calculated {len(df_dist_f5)} unique pairwise distances using Frame {analysis_frame_for_distance}.")

    # 3. Calculate Functional Metric: Delta Delta Luminosity (Logging handled inside)
    df_ddl = calculate_pairwise_delta_delta_luminosity(
        lum_df, 
        frame1=frame1_for_delta_L, 
        frame2=frame2_for_delta_L, 
        cell_ids=cell_ids,
        log_file_path=log_file_path
    )

    # 4. Plot the Result
    analysis_name = "Distance (F5) vs Delta-Delta Luminosity (F5-F4)"
    title_suffix = f"Physical Distance at F{analysis_frame_for_distance} vs. |ΔL(F{frame2_for_delta_L}-F{frame1_for_delta_L}) Diff|"
    
    plot_correlation(
        df_dist_f5, 
        df_ddl, 
        data_path, 
        analysis_name=analysis_name, 
        title_suffix=title_suffix,
        log_file_path=log_file_path
    )

# --- Final Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--analysis_dir", required=True)
    args = parser.parse_args()

    exp = args.exp
    data_path = args.analysis_dir
    plots_dir = os.path.join(data_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    LOG_FILE_PATH = os.path.join(plots_dir, "correlation_log.txt")

    # Clear the log file or add a header for a fresh run
    with open(LOG_FILE_PATH, 'w') as f:
        f.write("--- Correlation Analysis Log ---\n")
        f.write(f"Experiment: {exp}\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    log_message(LOG_FILE_PATH, "Starting all analyses...")

    # Run all analyses, passing the log file path
    run_all_analyses(exp, LOG_FILE_PATH)
    run_delta_delta_luminosity_analysis(exp, LOG_FILE_PATH)

    log_message(LOG_FILE_PATH, "\nAll analyses complete.")
    print(f"Analysis complete. Results logged to: {LOG_FILE_PATH}")