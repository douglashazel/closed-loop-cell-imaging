import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

@numba.jit(nopython=True)
def pairwise_distances_unique(centers):
    n = centers.shape[0]
    pairs = []
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            pairs.append((i, j))
            dists.append(d)
    return np.array(pairs), np.array(dists)

def corr_plot(dist_df, lum_change_df, data_path, frame=5):
    merged = pd.merge(
        dist_df[["CellID", "TargetCellID", "Distance"]],
        lum_change_df[["CellID", "TargetCellID", "LuminosityChangeDiff"]],
        on=["CellID", "TargetCellID"],
        how="inner"
    )

    merged = merged[merged["CellID"] != merged["TargetCellID"]]

    x = merged["Distance"].values
    y = np.abs(merged["LuminosityChangeDiff"].values)

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.4f}")
    print(f"Spearman r={spearman_r:.4f}, p={spearman_p:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=15, alpha=0.6, label="Pairs")

    # Pearson regression line (Linear fit to the data)
    m_p, b_p = np.polyfit(x, y, 1)
    
    # Combined label for the linear fit
    label_text = (
        f"Linear Fit (Pearson $r$={pearson_r:.3f}, $p$={pearson_p:.4f})\n"
        f"Spearman $\\rho$={spearman_r:.3f}, $p$={spearman_p:.4f}"
    )
    
    plt.plot(
        np.sort(x),
        m_p * np.sort(x) + b_p,
        color="red",
        linewidth=2,
        label=label_text
    )

    # The Spearman fit line logic has been removed.

    plt.xlabel("Pairwise Distance (pixels)")
    plt.ylabel("|Δ(ΔLuminosity)|")
    plt.title(f"Frame {frame}: Distance vs ΔΔLuminosity")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # The rest of your saving and printing logic follows...
    out_png = f"{data_path}/plots/dist_vs_deltaDeltaLum_frame{frame}.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot: {out_png}")

# -----------------------
# Main
# -----------------------
exp = "c2c12_carbachol_2"
data_path = f"{exp}/analysis"
traj_file = f"{data_path}/trajectories_complete.csv"
lum_file = f"{data_path}/luminosity_no_bground_complete.csv"

traj_df = pd.read_csv(traj_file)
lum_df = pd.read_csv(lum_file)

traj_df = traj_df[traj_df["CellID"] != 0]
lum_df = lum_df[lum_df["CellID"] != 0]

cell_ids = traj_df["CellID"].values
coords = traj_df.drop(columns=["CellID"]).values
num_frames = coords.shape[1] // 2

frame = 5
if frame >= num_frames:
    raise ValueError(f"Requested frame {frame} exceeds available frames ({num_frames}).")

# --- pairwise distances for positions ---
frame_coords = coords[:, (2*frame):(2*frame+2)]
pairs, dists = pairwise_distances_unique(frame_coords)

df_dist = pd.DataFrame({
    "CellID": cell_ids[pairs[:, 0]],
    "TargetCellID": cell_ids[pairs[:, 1]],
    "Distance": dists,
    "Frame": frame
})

out_parquet_dist = f"{data_path}/pairwise_distances_frame{frame}.parquet"
out_csv_dist = f"{data_path}/pairwise_distances_frame{frame}.csv"
df_dist.to_parquet(out_parquet_dist, index=False)
df_dist.to_csv(out_csv_dist, index=False)

# --- compute Δluminosity per cell between frame-1 and frame ---
lum_prev = lum_df[f"f{frame-1}"].values
lum_curr = lum_df[f"f{frame}"].values
delta_lum = lum_curr - lum_prev

# --- now compute pairwise differences in these Δluminosity values ---
n = len(delta_lum)
pairs_i, pairs_j = np.triu_indices(n, k=1)
delta_lum_diff = delta_lum[pairs_i] - delta_lum[pairs_j]

df_lum_change = pd.DataFrame({
    "CellID": cell_ids[pairs_i],
    "TargetCellID": cell_ids[pairs_j],
    "LuminosityChangeDiff": delta_lum_diff,
    "Frame": frame
})

out_parquet_lum_change = f"{data_path}/pairwise_delta_lum_diff_frame{frame}.parquet"
out_csv_lum_change = f"{data_path}/pairwise_delta_lum_diff_frame{frame}.csv"
df_lum_change.to_parquet(out_parquet_lum_change, index=False)
df_lum_change.to_csv(out_csv_lum_change, index=False)

print(f"Saved ΔΔluminosity pairwise results for f{frame}.")

# --- correlation plot ---
dist_df = pd.read_csv(out_csv_dist)
lum_change_df = pd.read_csv(out_csv_lum_change)

corr_plot(dist_df, lum_change_df, data_path, frame=frame)

import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from itertools import combinations

# --- Data Preparation Functions ---
@numba.jit(nopython=True)
def pairwise_distances_unique(centers):
    n = centers.shape[0]
    pairs = []
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            pairs.append((i, j))
            dists.append(d)
    return np.array(pairs), np.array(dists)

def calculate_pairwise_trace_correlation(lum_df, cell_ids):
    """
    Calculates the Pearson correlation coefficient (R) for the full luminosity 
    time series between every unique pair of cells, while handling invalid traces.
    """
    # 1. Prepare Traces and Filter for Validity
    lum_cols = [col for col in lum_df.columns if col.startswith('f')]
    sorted_lum_df = lum_df.set_index("CellID").sort_index()
    
    # Select only the relevant cell traces
    traces_df = sorted_lum_df.loc[cell_ids, lum_cols]
    
    # CRITICAL: Filter out traces that have NaN values or zero variance
    
    # Identify cells with any NaN values in their trace
    cells_with_nan = traces_df.index[traces_df.isnull().any(axis=1)].tolist()
    
    # Identify cells with zero variance (flat lines)
    trace_stdev = traces_df.std(axis=1)
    cells_with_zero_variance = trace_stdev.index[trace_stdev == 0].tolist()
    
    invalid_cell_ids = set(cells_with_nan + cells_with_zero_variance)
    valid_cell_ids = [cid for cid in cell_ids if cid not in invalid_cell_ids]
    
    print(f"Excluded {len(invalid_cell_ids)} cells due to flat-line trace (zero variance) or NaN values.")
    
    # Filter the traces array to include only valid cells
    valid_traces_df = traces_df.loc[valid_cell_ids]
    valid_traces = valid_traces_df.values
    
    n_valid_cells = len(valid_cell_ids)
    
    correlations = []
    cell_pairs = []

    # 2. Calculate Correlation only for valid pairs
    # Use indices corresponding to the valid_traces array
    for i, j in combinations(range(n_valid_cells), 2):
        cell_id_1 = valid_cell_ids[i]
        cell_id_2 = valid_cell_ids[j]
        
        trace_a = valid_traces[i]
        trace_b = valid_traces[j]
        
        # Calculate Pearson's r between the two time traces
        # We know traces are clean and have variance > 0, so this should succeed.
        r, _ = pearsonr(trace_a, trace_b)
        
        correlations.append(r)
        cell_pairs.append((cell_id_1, cell_id_2))

    df_corr = pd.DataFrame({
        "CellID": [p[0] for p in cell_pairs],
        "TargetCellID": [p[1] for p in cell_pairs],
        "TraceCorrelation_R": correlations,
    })
    
    return df_corr

# --- Plotting and Analysis Function ---
def corr_plot(dist_df, corr_df, data_path, analysis_frame):
    """
    Correlates Pairwise Distance (X) with Time Trace Correlation (Y).
    """
    merged = pd.merge(
        dist_df[["CellID", "TargetCellID", "Distance"]],
        corr_df[["CellID", "TargetCellID", "TraceCorrelation_R"]],
        on=["CellID", "TargetCellID"],
        how="inner"
    )

    merged = merged[merged["CellID"] != merged["TargetCellID"]]

    # X-axis: Distance
    x = merged["Distance"].values
    # Y-axis: Trace Correlation (R value)
    y = merged["TraceCorrelation_R"].values

    # 1. Correlate X (Distance) with Y (Trace Correlation)
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    print("--- Final Analysis: Distance vs. Trace Correlation ---")
    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.4f}")
    print(f"Spearman r={spearman_r:.4f}, p={spearman_p:.4f}")

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=15, alpha=0.6, label="Cell Pairs (Distance vs. R)")

    # Pearson regression line (Linear fit to the data)
    m_p, b_p = np.polyfit(x, y, 1)
    
    # Combined label for the linear fit
    label_text = (
        f"Linear Fit (Pearson $r$={pearson_r:.3f}, $p$={pearson_p:.4f})\n"
        f"Spearman $\\rho$={spearman_r:.3f}, $p$={spearman_p:.4f}"
    )
    
    # Plot the linear regression line for Pearson's r
    plt.plot(
        np.sort(x),
        m_p * np.sort(x) + b_p,
        color="red",
        linewidth=2,
        label=label_text
    )

    plt.xlabel("Pairwise Distance (pixels)")
    plt.ylabel("Luminosity Trace Correlation ($R_{trace}$)")
    plt.title(f"Distance vs. Functional Connectivity (Frame {analysis_frame} reference)")
    plt.legend()
    plt.ylim(-1.05, 1.05) # Correlation values range from -1 to 1
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # out_png = f"{data_path}/plots/dist_vs_traceCorrelation_frame{analysis_frame}.png"
    # plt.savefig(out_png, dpi=300)
    # print(f"Saved plot: {out_png}")

# -----------------------
# Main Script Execution
# -----------------------
exp = "c2c12_carbachol_2"
data_path = f"{exp}/analysis"
traj_file = f"{data_path}/trajectories_complete.csv"
lum_file = f"{data_path}/luminosity_no_bground_complete.csv"

# --- Load Data ---
traj_df = pd.read_csv(traj_file)
lum_df = pd.read_csv(lum_file)

# Clean up data (assuming CellID 0 is background/junk)
traj_df = traj_df[traj_df["CellID"] != 0]
lum_df = lum_df[lum_df["CellID"] != 0]

cell_ids = traj_df["CellID"].values
coords = traj_df.drop(columns=["CellID"]).values
num_frames = coords.shape[1] // 2

# We will use frame 0 just as a reference point for calculating physical distance,
# even though the correlation uses the full trace.
frame_for_distance = 0
if frame_for_distance >= num_frames:
    raise ValueError(f"Requested distance reference frame {frame_for_distance} exceeds available frames ({num_frames}).")

# --- 1. Calculate Pairwise Distances (Physical Space) ---
print(f"Calculating pairwise distances based on coordinates in Frame {frame_for_distance}...")
frame_coords = coords[:, (2*frame_for_distance):(2*frame_for_distance+2)]
pairs, dists = pairwise_distances_unique(frame_coords)

df_dist = pd.DataFrame({
    "CellID": cell_ids[pairs[:, 0]],
    "TargetCellID": cell_ids[pairs[:, 1]],
    "Distance": dists,
    "Frame": frame_for_distance
})

# Save results (optional, but good practice)
# out_csv_dist = f"{data_path}/pairwise_distances_frame{frame_for_distance}.csv"
# df_dist.to_csv(out_csv_dist, index=False)
# print(f"Saved pairwise distance results for f{frame_for_distance}.")

# --- 2. Calculate Pairwise Trace Correlation (Functional Space) ---
print("Calculating pairwise correlation for the full luminosity time traces...")
df_corr = calculate_pairwise_trace_correlation(lum_df, cell_ids)

# Save results (optional, but good practice)
# out_csv_corr = f"{data_path}/pairwise_trace_correlation.csv"
# df_corr.to_csv(out_csv_corr, index=False)
# print("Saved pairwise trace correlation results.")

# --- 3. Correlation Plot (Distance vs. Correlation) ---
corr_plot(df_dist, df_corr, data_path, analysis_frame=frame_for_distance)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from itertools import combinations

# --- Data Preparation Functions ---
# Rewritten to remove numba dependency
def pairwise_distances_unique(centers):
    n = centers.shape[0]
    pairs = []
    dists = []
    
    # Iterate through all unique pairs of indices (i, j) where i < j
    for i, j in combinations(range(n), 2):
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        d = np.sqrt(dx * dx + dy * dy)
        pairs.append((i, j))
        dists.append(d)
        
    return np.array(pairs), np.array(dists)

# MODIFIED: Takes 'frame_stop' to calculate correlation on a segment
def calculate_pairwise_trace_correlation_segment(lum_df, cell_ids, frame_stop):
    """
    Calculates the Pearson correlation coefficient (R) for the luminosity 
    time series segment (f0 up to f(frame_stop-1)) between every unique 
    pair of cells, while handling invalid traces.
    """
    # 1. Prepare Traces and Filter for Validity
    # Define the columns to use: f0, f1, ..., f(frame_stop-1)
    lum_cols = [f'f{i}' for i in range(frame_stop)]

    # CRITICAL: Check if the required columns exist
    missing_cols = [col for col in lum_cols if col not in lum_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required frame columns: {missing_cols}")

    sorted_lum_df = lum_df.set_index("CellID").sort_index()
    
    # Select only the relevant cell traces and the segment columns
    traces_df = sorted_lum_df.loc[cell_ids, lum_cols].copy()
    
    # CRITICAL: Filter out traces that have NaN values or zero variance
    
    # Identify cells with any NaN values in their trace *in the segment*
    cells_with_nan = traces_df.index[traces_df.isnull().any(axis=1)].tolist()
    
    # Identify cells with zero variance (flat lines) *in the segment*
    trace_stdev = traces_df.std(axis=1)
    cells_with_zero_variance = trace_stdev.index[trace_stdev == 0].tolist()
    
    invalid_cell_ids = set(cells_with_nan + cells_with_zero_variance)
    valid_cell_ids = [cid for cid in cell_ids if cid not in invalid_cell_ids]
    
    print(f"Excluded {len(invalid_cell_ids)} cells due to flat-line trace (zero variance) or NaN values in the segment (f0 to f{frame_stop-1}).")
    
    # Filter the traces array to include only valid cells
    valid_traces_df = traces_df.loc[valid_cell_ids]
    valid_traces = valid_traces_df.values
    
    n_valid_cells = len(valid_cell_ids)
    
    correlations = []
    cell_pairs = []

    # 2. Calculate Correlation only for valid pairs
    for i, j in combinations(range(n_valid_cells), 2):
        cell_id_1 = valid_cell_ids[i]
        cell_id_2 = valid_cell_ids[j]
        
        trace_a = valid_traces[i]
        trace_b = valid_traces[j]
        
        # Calculate Pearson's r between the two time traces
        r, _ = pearsonr(trace_a, trace_b)
        
        correlations.append(r)
        cell_pairs.append((cell_id_1, cell_id_2))

    df_corr = pd.DataFrame({
        "CellID": [p[0] for p in cell_pairs],
        "TargetCellID": [p[1] for p in cell_pairs],
        "TraceCorrelation_R": correlations,
    })
    
    return df_corr

# MODIFIED: Takes 'frame_stop' to update the plot title and file name
def corr_plot(dist_df, corr_df, data_path, analysis_frame, frame_stop):
    """
    Correlates Pairwise Distance (X) with Time Trace Correlation (Y).
    """
    merged = pd.merge(
        dist_df[["CellID", "TargetCellID", "Distance"]],
        corr_df[["CellID", "TargetCellID", "TraceCorrelation_R"]],
        on=["CellID", "TargetCellID"],
        how="inner"
    )

    merged = merged[merged["CellID"] != merged["TargetCellID"]]

    # X-axis: Distance
    x = merged["Distance"].values
    # Y-axis: Trace Correlation (R value)
    y = merged["TraceCorrelation_R"].values

    # 1. Correlate X (Distance) with Y (Trace Correlation)
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    print("--- Final Analysis: Distance vs. Trace Correlation ---")
    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.4f}")
    print(f"Spearman r={spearman_r:.4f}, p={spearman_p:.4f}")

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=15, alpha=0.6, label="Cell Pairs (Distance vs. R)")

    # Pearson regression line (Linear fit to the data)
    m_p, b_p = np.polyfit(x, y, 1)
    
    # Combined label for the linear fit
    label_text = (
        f"Linear Fit (Pearson $r$={pearson_r:.3f}, $p$={pearson_p:.4f})\n"
        f"Spearman $\\rho$={spearman_r:.3f}, $p$={spearman_p:.4f}"
    )
    
    # Plot the linear regression line for Pearson's r
    plt.plot(
        np.sort(x),
        m_p * np.sort(x) + b_p,
        color="red",
        linewidth=2,
        label=label_text
    )

    plt.xlabel("Pairwise Distance (pixels)")
    plt.ylabel("Luminosity Trace Correlation ($R_{trace}$)")
    
    # Update the title and use frame_stop information
    correlation_segment_info = f"Frames 0 to {frame_stop - 1}"
    title = f"Distance vs. Functional Connectivity ({correlation_segment_info} Correlation, Frame {analysis_frame} Distance)"
    plt.title(title)

    plt.legend()
    plt.ylim(-1.05, 1.05) # Correlation values range from -1 to 1
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Dynamic filename to reflect the segment used
    # out_png = f"dist_vs_traceCorrelation_segment_f0_to_f{frame_stop-1}.png" 
    # plt.savefig(out_png, dpi=300)
    # print(f"Saved plot: {out_png}")

# -----------------------
# Main Script Execution for Segment before Frame 5
# -----------------------
exp = "c2c12_carbachol_2"
data_path = f"{exp}/analysis"
traj_file = f"{data_path}/trajectories_complete.csv"
lum_file = f"{data_path}/luminosity_no_bground_complete.csv"

# --- Load Data ---
try:
    traj_df = pd.read_csv(traj_file)
    lum_df = pd.read_csv(lum_file)
except FileNotFoundError as e:
    # If files are not available, this will raise an error, allowing you to provide them.
    raise FileNotFoundError(f"Error: Required file not found. Please ensure '{e.filename}' is accessible.")

# Clean up data (assuming CellID 0 is background/junk)
traj_df = traj_df[traj_df["CellID"] != 0]
lum_df = lum_df[lum_df["CellID"] != 0]

cell_ids = traj_df["CellID"].values
coords = traj_df.drop(columns=["CellID"]).values
num_frames = coords.shape[1] // 2

# We will use frame 0 as a reference point for calculating physical distance.
frame_for_distance = 0
if frame_for_distance >= num_frames:
    raise ValueError(f"Requested distance reference frame {frame_for_distance} exceeds available frames ({num_frames}).")

# --- 1. Calculate Pairwise Distances (Physical Space) ---
print(f"Calculating pairwise distances based on coordinates in Frame {frame_for_distance}...")
frame_coords = coords[:, (2*frame_for_distance):(2*frame_for_distance+2)]
pairs, dists = pairwise_distances_unique(frame_coords)

df_dist = pd.DataFrame({
    "CellID": cell_ids[pairs[:, 0]],
    "TargetCellID": cell_ids[pairs[:, 1]],
    "Distance": dists,
    "Frame": frame_for_distance
})

# --- 2. Calculate Pairwise Trace Correlation (Functional Space) for segment before frame 5 ---
FRAME_STOP = 5
print(f"Calculating pairwise correlation for the segment from f0 up to f{FRAME_STOP-1} (5 frames)...")

# Check if we have at least FRAME_STOP frames in the luminosity data
lum_cols = [col for col in lum_df.columns if col.startswith('f')]
if not lum_cols:
     max_frame_index = -1
else:
    # Extract the numerical part of frame columns, e.g., 'f12' -> 12
    max_frame_index = max([int(col[1:]) for col in lum_cols if col[1:].isdigit()], default=-1)

if max_frame_index < FRAME_STOP - 1:
    raise ValueError(f"Requested segment up to frame {FRAME_STOP-1} is not available. Max frame index found is {max_frame_index}.")


df_corr_segment = calculate_pairwise_trace_correlation_segment(lum_df, cell_ids, frame_stop=FRAME_STOP)

# --- 3. Correlation Plot (Distance vs. Segment Correlation) ---
corr_plot(df_dist, df_corr_segment, data_path, analysis_frame=frame_for_distance, frame_stop=FRAME_STOP)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from itertools import combinations

# --- Data Preparation Functions ---
# Pure NumPy/Python implementation for pairwise distance
def pairwise_distances_unique(centers):
    n = centers.shape[0]
    pairs = []
    dists = []
    for i, j in combinations(range(n), 2):
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        d = np.sqrt(dx * dx + dy * dy)
        pairs.append((i, j))
        dists.append(d)
        
    return np.array(pairs), np.array(dists)

# GENERALIZED: Accepts 'frame_start' and 'frame_stop'
def calculate_pairwise_trace_correlation_segment(lum_df, cell_ids, frame_start, frame_stop):
    """
    Calculates the Pearson correlation (R) for the segment from f(frame_start) 
    up to f(frame_stop-1).
    """
    lum_cols = [f'f{i}' for i in range(frame_start, frame_stop)]
    if not lum_cols:
        return pd.DataFrame({"CellID": [], "TargetCellID": [], "TraceCorrelation_R": []})
        
    missing_cols = [col for col in lum_cols if col not in lum_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required frame columns: {missing_cols}")

    sorted_lum_df = lum_df.set_index("CellID").sort_index()
    traces_df = sorted_lum_df.loc[cell_ids, lum_cols].copy()
    
    # Filtering for valid traces (non-NaN and non-zero variance)
    cells_with_nan = traces_df.index[traces_df.isnull().any(axis=1)].tolist()
    trace_stdev = traces_df.std(axis=1)
    cells_with_zero_variance = trace_stdev.index[trace_stdev == 0].tolist()
    
    invalid_cell_ids = set(cells_with_nan + cells_with_zero_variance)
    valid_cell_ids = [cid for cid in cell_ids if cid not in invalid_cell_ids]
    
    print(f"Excluded {len(invalid_cell_ids)} cells due to invalid trace in the segment (f{frame_start} to f{frame_stop-1}).")
    
    valid_traces = traces_df.loc[valid_cell_ids].values
    n_valid_cells = len(valid_cell_ids)
    
    correlations = []
    cell_pairs = []

    for i, j in combinations(range(n_valid_cells), 2):
        r, _ = pearsonr(valid_traces[i], valid_traces[j])
        correlations.append(r)
        cell_pairs.append((valid_cell_ids[i], valid_cell_ids[j]))

    df_corr = pd.DataFrame({
        "CellID": [p[0] for p in cell_pairs],
        "TargetCellID": [p[1] for p in cell_pairs],
        "TraceCorrelation_R": correlations,
    })
    return df_corr

# GENERALIZED: Accepts 'frame_start' and 'frame_stop' for title/filename
def corr_plot(dist_df, corr_df, data_path, analysis_frame, frame_start, frame_stop):
    merged = pd.merge(
        dist_df[["CellID", "TargetCellID", "Distance"]],
        corr_df[["CellID", "TargetCellID", "TraceCorrelation_R"]],
        on=["CellID", "TargetCellID"],
        how="inner"
    )

    x = merged["Distance"].values
    y = merged["TraceCorrelation_R"].values

    if len(x) < 2:
        print("Warning: Insufficient cell pairs for correlation analysis and plotting.")
        return

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    print("--- Final Analysis: Distance vs. Trace Correlation ---")
    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.4f}")
    print(f"Spearman r={spearman_r:.4f}, p={spearman_p:.4f}")

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=15, alpha=0.6, label="Cell Pairs (Distance vs. R)")

    m_p, b_p = np.polyfit(x, y, 1)
    label_text = (
        f"Linear Fit (Pearson $r$={pearson_r:.3f}, $p$={pearson_p:.4f})\n"
        f"Spearman $\\rho$={spearman_r:.3f}, $p$={spearman_p:.4f}"
    )
    plt.plot(np.sort(x), m_p * np.sort(x) + b_p, color="red", linewidth=2, label=label_text)

    plt.xlabel("Pairwise Distance (pixels)")
    plt.ylabel("Luminosity Trace Correlation ($R_{trace}$)")
    
    correlation_segment_info = f"Frames {frame_start} to {frame_stop - 1}"
    title = f"Distance vs. Functional Connectivity ({correlation_segment_info} Correlation, Frame {analysis_frame} Distance)"
    plt.title(title)

    plt.legend()
    plt.ylim(-1.05, 1.05)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # out_png = f"dist_vs_traceCorrelation_segment_f{frame_start}_to_f{frame_stop-1}.png" 
    # plt.savefig(out_png, dpi=300)
    # print(f"Saved plot: {out_png}")

# -----------------------
# Main Script Execution for Segment AFTER Frame 5 (f6 to end)
# -----------------------
exp = "c2c12_carbachol_2"
data_path = f"{exp}/analysis"
traj_file = f"{data_path}/trajectories_complete.csv"
lum_file = f"{data_path}/luminosity_no_bground_complete.csv"

# --- Load Data (Requires files to be accessible) ---
try:
    traj_df = pd.read_csv(traj_file)
    lum_df = pd.read_csv(lum_file)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Error: The required file could not be found: '{e.filename}'. Please ensure the data files are accessible to run the code.")

# Data Cleaning
traj_df = traj_df[traj_df["CellID"] != 0]
lum_df = lum_df[lum_df["CellID"] != 0]
cell_ids = traj_df["CellID"].values
coords = traj_df.drop(columns=["CellID"]).values
num_frames = coords.shape[1] // 2
frame_for_distance = 0

# --- 1. Calculate Pairwise Distances (Physical Space) ---
# Distance calculation remains unchanged.
frame_coords = coords[:, (2*frame_for_distance):(2*frame_for_distance+2)]
pairs, dists = pairwise_distances_unique(frame_coords)

df_dist = pd.DataFrame({
    "CellID": cell_ids[pairs[:, 0]],
    "TargetCellID": cell_ids[pairs[:, 1]],
    "Distance": dists,
    "Frame": frame_for_distance
})

# --- 2. Calculate Pairwise Trace Correlation (Functional Space) ---
FRAME_START = 6 # Start after frame 5

# Determine FRAME_STOP (end exclusive)
lum_cols = [col for col in lum_df.columns if col.startswith('f')]
max_frame_index = max([int(col[1:]) for col in lum_cols if col[1:].isdigit()], default=-1)
FRAME_STOP = max_frame_index + 1 

if FRAME_START >= FRAME_STOP:
    raise ValueError(f"No valid frames available for correlation after frame {FRAME_START-1}. Max frame index is {max_frame_index}.")

print(f"Calculating pairwise correlation for the segment from f{FRAME_START} up to f{FRAME_STOP-1} (end)...")

df_corr_segment = calculate_pairwise_trace_correlation_segment(
    lum_df, cell_ids, 
    frame_start=FRAME_START, 
    frame_stop=FRAME_STOP
)

# --- 3. Correlation Plot (Distance vs. Segment Correlation) ---
corr_plot(
    df_dist, 
    df_corr_segment, 
    data_path, 
    analysis_frame=frame_for_distance, 
    frame_start=FRAME_START, 
    frame_stop=FRAME_STOP
)