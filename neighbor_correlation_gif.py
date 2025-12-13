import os
import numba
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import combinations
import matplotlib.animation as animation
from numba import int32, float64

# --- New Global Configuration for Region of Interest (ROI) ---
# Define the ROI rectangle based on pixel coordinates
ROI_X_CENTER = 700      # X-coordinate (Column) of the center
ROI_Y_CENTER = 1100      # Y-coordinate (Row) of the center
ROI_WIDTH = 1200        # Width (X-span) of the rectangle
ROI_HEIGHT = 2000        # Height (Y-span) of the rectangle

# --- Existing Configuration ---
CORR_WINDOW = 5 
MIN_CELL_DISTANCE_PIXELS = 10 

# --- Utility Function for Logging (Unchanged) ---
def log_message(log_file_path, message, print_to_console=False):
    """Appends a message to the log file and optionally prints it to the console."""
    with open(log_file_path, 'a') as f:
        f.write(message + '\n')
    if print_to_console:
        print(message)

# --- NEW Numba Filtering Function ---
@numba.jit(nopython=True)
def is_inside_rectangle(x, y, x_min, x_max, y_min, y_max):
    """Checks if a point (x, y) is within the defined rectangle bounds."""
    return (x >= x_min) and (x < x_max) and (y >= y_min) and (y < y_max)

# --- Existing Numba Core Functions (Unchanged) ---
@numba.jit(nopython=True)
def pairwise_distances_numba(centers):
    # ... (Unchanged)
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

@numba.jit(nopython=True)
def find_neighbors_numba(coords_2d, proximity, min_dist):
    # ... (Unchanged)
    n = coords_2d.shape[0]
    all_neighbors = [] 
    
    for i in range(n):
        cell_i_neighbors = []
        for j in range(n):
            if i == j:
                continue
            
            dx = coords_2d[i, 0] - coords_2d[j, 0]
            dy = coords_2d[i, 1] - coords_2d[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            
            if d <= proximity and d > min_dist:
                cell_i_neighbors.append(j) 
        all_neighbors.append(np.array(cell_i_neighbors, dtype=np.int32))
    
    return all_neighbors

@numba.jit(nopython=True)
def calculate_single_cell_rolling_correlation(
    cell_trace, neighbor_trace, frame_index, window_size, n_frames
):
    # ... (Unchanged)
    start = max(0, frame_index - window_size)
    stop = min(n_frames, frame_index + window_size + 1)
    
    trace_i = cell_trace[start:stop]
    trace_j = neighbor_trace[start:stop]
    
    if len(trace_i) < 2:
        return np.nan 
    
    mean_i = np.mean(trace_i)
    mean_j = np.mean(trace_j)
    
    std_i = np.std(trace_i)
    std_j = np.std(trace_j)
    
    if std_i == 0.0 or std_j == 0.0:
        return 0.0 
    
    covariance = np.mean((trace_i - mean_i) * (trace_j - mean_j))
    
    return covariance / (std_i * std_j)

# --- Main Analysis Function (Unchanged) ---
def calculate_dynamic_neighbor_correlation(
    traj_df, lum_df, cell_ids, proximity, corr_window, min_dist, log_file_path
):
    # ... (Unchanged - Now operates only on the provided cell_ids)
    log_message(log_file_path, f"Starting dynamic neighbor correlation for {len(cell_ids)} cells. Proximity={proximity}px, Window={2*corr_window+1} frames.")
    
    # 1. Prepare Data
    lum_cols = sorted([col for col in lum_df.columns if col.startswith('f')], key=lambda x: int(x[1:]))
    n_frames = len(lum_cols)
    if n_frames == 0:
        raise ValueError("No frame luminosity data found.")

    lum_matrix = lum_df.set_index("CellID").loc[cell_ids, lum_cols].values
    
    cell_lum_data = lum_df.set_index("CellID").loc[cell_ids].copy()
    invalid_cells = cell_lum_data.index[
        (cell_lum_data.isnull().any(axis=1)) | 
        (cell_lum_data.std(axis=1) == 0)
    ].tolist()
    
    log_message(log_file_path, f"Excluded {len(invalid_cells)} cells from correlation due to invalid trace (NaN or zero variance).")

    # 2. Extract Trajectory (Centers) and Prepare Result Array
    coords_cols = sorted([col for col in traj_df.columns if col.startswith('x') or col.startswith('y')])
    coords_matrix = traj_df.set_index("CellID").loc[cell_ids, coords_cols].values
    
    avg_corr_r_per_frame = np.full((len(cell_ids), n_frames), np.nan, dtype=np.float64)

    # 3. Main Loop: Iterate through all frames
    for t in range(n_frames):
        x_col_idx = coords_cols.index(f'x{t}')
        y_col_idx = coords_cols.index(f'y{t}')
        coords_2d = coords_matrix[:, [x_col_idx, y_col_idx]]

        all_neighbors = find_neighbors_numba(coords_2d, proximity, min_dist)
        
        for i, cell_id in enumerate(cell_ids):
            if cell_id in invalid_cells:
                continue

            neighbor_indices = all_neighbors[i]
            if neighbor_indices.size == 0:
                continue 

            correlations = []
            trace_i = lum_matrix[i, :] 
            
            for j in neighbor_indices:
                neighbor_id = cell_ids[j]
                if neighbor_id in invalid_cells:
                    continue 
                
                trace_j = lum_matrix[j, :] 
                
                r = calculate_single_cell_rolling_correlation(
                    trace_i, trace_j, t, corr_window, n_frames
                )
                if not np.isnan(r):
                    correlations.append(r)

            if correlations:
                avg_corr_r_per_frame[i, t] = np.mean(correlations)
    
    # 4. Convert Result to DataFrame
    df_list = []
    for i, cell_id in enumerate(cell_ids):
        row = {"CellID": cell_id}
        row.update({f"f{t}": avg_corr_r_per_frame[i, t] for t in range(n_frames)})
        df_list.append(row)
        
    return pd.DataFrame(df_list)

# --- Visualization Function (Unchanged - Now operates only on the provided cell_ids) ---
def create_correlation_movie(
    corr_per_frame_df, cell_ids, num_frames, exp_name, mask_dir, log_file_path
):
    # ... (Unchanged)
    log_message(log_file_path, f"Starting movie generation for {len(cell_ids)} cells over {num_frames} frames.")
    
    # 1. Prepare Data and Colormap
    frame_cols = [f'f{t}' for t in range(num_frames)]
    corr_data = corr_per_frame_df.set_index("CellID").loc[cell_ids, frame_cols].values
    
    # Set up the visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    norm = plt.Normalize(-1.0, 1.0)
    cmap = plt.colormaps.get_cmap('seismic') 
    
    def get_mask_path(t):
        return os.path.join(mask_dir, f"channel_2_image_0_a_timepoint_{t:05d}.npy")
        
    first_mask_path = get_mask_path(0)
    try:
        mask_image = np.load(first_mask_path, allow_pickle=True) 
    except FileNotFoundError:
        log_message(log_file_path, f"CRITICAL ERROR: Mask file not found: {first_mask_path}")
        return

    mask_shape = mask_image.shape

    im = ax.imshow(mask_image, cmap='gray', alpha=0.1) 
    
    colored_overlay = np.zeros((*mask_shape, 4)) 
    cell_image = ax.imshow(colored_overlay) 
    
    ax.set_title(f"Cell-Centric Neighbor Correlation (R) Movie\n{exp_name} (ROI-Filtered)") # Added (ROI-Filtered)
    ax.axis('off')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label='Average Neighbor Correlation (R)')
    
    # 2. Animation Update Function
    def update_frame(t):
        ax.set_title(f"Cell-Centric Neighbor Correlation (R) Movie\nFrame {t} (ROI-Filtered)")
        
        mask_path = get_mask_path(t)
        try:
            current_mask = np.load(mask_path, allow_pickle=True) 
        except FileNotFoundError:
            log_message(log_file_path, f"Warning: Mask file not found for frame {t}. Skipping.")
            return cell_image,

        new_overlay = np.zeros((*mask_shape, 4))
        
        for i, cell_id in enumerate(cell_ids): # Iterates only over filtered IDs
            r_value = corr_data[i, t]
            
            if not np.isnan(r_value):
                color = cmap(norm(r_value)) 
                
                # IMPORTANT: cell_mask logic remains the same, but it only runs for the filtered CellIDs
                cell_mask = (current_mask == cell_id)
                
                new_overlay[cell_mask, :3] = color[:3] 
                new_overlay[cell_mask, 3] = 0.7       
                
        cell_image.set_array(new_overlay)
        return cell_image,

    # 3. Create and Save Animation
    ani = animation.FuncAnimation(
        fig, update_frame, frames=num_frames, interval=100, blit=True
    )

    gif_filename = os.path.join(mask_dir, f"{exp_name}_ROI_filtered_correlation_movie.gif") # Updated filename
    writer = animation.PillowWriter(fps=10) 
    ani.save(gif_filename, writer=writer)
    log_message(log_file_path, f"Saved ROI-filtered correlation movie to: {gif_filename}")

    plt.close(fig)

# --- Unified Main Execution (MODIFIED) ---
def run_all_analyses(exp, proximity, corr_window, min_dist, log_file_path, roi_params):
    """Loads data, filters by ROI, calculates dynamic neighbor correlation, and generates a movie."""
    
    exp_name = exp
    data_path = f"{exp}/analysis"
    mask_dir = f"{exp}/masks"
    traj_file = f"{data_path}/trajectories_complete.csv"
    lum_file = f"{data_path}/luminosity_no_bground_complete.csv"
    
    # 1. Load and Clean Data
    try:
        traj_df = pd.read_csv(traj_file)
        lum_df = pd.read_csv(lum_file)
    except FileNotFoundError as e:
        log_message(log_file_path, f"CRITICAL ERROR: Data file not found: {e.filename}", print_to_console=True)
        raise e

    # Initial cleanup
    traj_df = traj_df[traj_df["CellID"] != 0].copy()
    lum_df = lum_df[lum_df["CellID"] != 0].copy()
    
    # Get all Cell IDs and determine number of frames
    all_cell_ids = sorted(traj_df["CellID"].unique())
    coords_cols = sorted([col for col in traj_df.columns if col.startswith('x') or col.startswith('y')])
    num_frames = len(coords_cols) // 2

    log_message(log_file_path, f"Loaded data for {len(all_cell_ids)} total cells over {num_frames} frames.")

    # 2. Filter Cells by ROI (at Frame 0)
    
    # Calculate ROI bounds
    x_min = roi_params['x_center'] - roi_params['width'] / 2
    x_max = roi_params['x_center'] + roi_params['width'] / 2
    y_min = roi_params['y_center'] - roi_params['height'] / 2
    y_max = roi_params['y_center'] + roi_params['height'] / 2

    log_message(log_file_path, f"Filtering cells based on initial position within ROI: X=[{x_min}, {x_max}), Y=[{y_min}, {y_max})")

    # Get x and y coordinates for frame 0
    df_frame_0 = traj_df.set_index("CellID").loc[all_cell_ids, [f'x0', f'y0']].copy()
    
    # Apply Numba-based filtering
    filtered_ids = []
    
    # Convert relevant data to NumPy arrays for Numba speed
    cell_ids_array = np.array(all_cell_ids, dtype=np.int32)
    x0_array = df_frame_0['x0'].values.astype(np.float64)
    y0_array = df_frame_0['y0'].values.astype(np.float64)

    # Note: Numba doesn't directly return list of Python objects (like CellIDs)
    # The filtering is performed in Pandas/Python logic for convenience here.
    
    for i, cell_id in enumerate(all_cell_ids):
        x = x0_array[i]
        y = y0_array[i]
        
        # Check if center is inside the rectangle
        if is_inside_rectangle(x, y, x_min, x_max, y_min, y_max):
            filtered_ids.append(cell_id)

    if not filtered_ids:
        log_message(log_file_path, "CRITICAL: No cells found within the specified ROI at Frame 0.", print_to_console=True)
        return

    log_message(log_file_path, f"Analysis will proceed with {len(filtered_ids)} cells (out of {len(all_cell_ids)} total).")
    
    # 3. Calculate Frame-wise Neighbor Correlation (using only filtered IDs)
    df_corr_per_frame = calculate_dynamic_neighbor_correlation(
        traj_df, lum_df, filtered_ids, proximity, corr_window, min_dist, log_file_path
    )
    
    # 4. Generate Correlation Movie (using only filtered IDs)
    create_correlation_movie(
        df_corr_per_frame, filtered_ids, num_frames, exp_name, mask_dir, log_file_path
    )


# --- Final Execution ---
exp = 'u87_pipettingControl_2'
LOG_FILE_PATH = os.path.join(".", "DELETE.txt") # Updated log file
proximity = 200 
CORR_WINDOW = 5 
MIN_CELL_DISTANCE_PIXELS = 10 

# Package ROI parameters
ROI_PARAMS = {
    'x_center': ROI_X_CENTER, 
    'y_center': ROI_Y_CENTER, 
    'width': ROI_WIDTH, 
    'height': ROI_HEIGHT
}

# Clear existing log for new run
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)
log_message(LOG_FILE_PATH, f"Starting ROI-Filtered Analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", print_to_console=True)

try:
    run_all_analyses(exp, proximity, CORR_WINDOW, MIN_CELL_DISTANCE_PIXELS, LOG_FILE_PATH, ROI_PARAMS)
    print("ROI-FILTERED ANALYSIS DONE! Check your masks directory for the GIF/Movie.")
except Exception as e:
    log_message(LOG_FILE_PATH, f"ANALYSIS FAILED: {e}", print_to_console=True)
    print("ANALYSIS FAILED! Check the log file for details.")