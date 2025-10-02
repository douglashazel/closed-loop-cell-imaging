import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
exp = "pc3_carbachol_2_NEW"

distance_threshold = 200  # pixels, adjust as needed
jump_length = 2  # number of frames to allow for jumps

# Shift configuration: apply this shift starting at the given frame
frame_shift = {1: (0, 0)}  # frame: (dx, dy)

data_path = f"{exp}/analysis"
input_csv = f"{data_path}/trajectories.csv"
output_csv = f"{data_path}/trajectories_merged.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(input_csv)

# Extract frame x/y columns
x_cols = [c for c in df.columns if 'x' in c]
y_cols = [c for c in df.columns if 'y' in c]

# Get actual frame numbers from column names
frame_numbers = [int(c.lstrip('x')) for c in x_cols]

# -----------------------------
# Identify incomplete tracks
# -----------------------------
present_frames = pd.DataFrame(False, index=df.index, columns=x_cols)
for x_col, y_col in zip(x_cols, y_cols):
    present_frames[x_col] = df[x_col].notna() & df[y_col].notna()
num_frames_per_cell = present_frames.sum(axis=1)
max_frames = len(x_cols)

# Keep only cells that don't appear in all frames
incomplete_df = df[num_frames_per_cell < max_frames].copy()
print(f"Found {len(incomplete_df)} incomplete tracks out of {len(df)} total cells.")

# -----------------------------
# Merge candidate cells based on distance + jump_length with shift
# -----------------------------
merged_ids = {}

for i, row_i in tqdm(incomplete_df.iterrows(), total=len(incomplete_df)):
    valid_i = [j for j, val in enumerate(row_i[x_cols]) if not pd.isna(val)]
    if len(valid_i) == 0:
        continue

    start_i_idx = valid_i[0]
    end_i_idx = valid_i[-1]
    start_i_frame = frame_numbers[start_i_idx]
    end_i_frame = frame_numbers[end_i_idx]

    pos_start_i = np.array([row_i[x_cols[start_i_idx]], row_i[y_cols[start_i_idx]]])
    pos_end_i   = np.array([row_i[x_cols[end_i_idx]], row_i[y_cols[end_i_idx]]])

    for j, row_j in incomplete_df.iterrows():
        if row_i['CellID'] == row_j['CellID']:
            continue

        valid_j = [jj for jj, val in enumerate(row_j[x_cols]) if not pd.isna(val)]
        if len(valid_j) == 0:
            continue

        start_j_idx = valid_j[0]
        end_j_idx = valid_j[-1]
        start_j_frame = frame_numbers[start_j_idx]
        end_j_frame   = frame_numbers[end_j_idx]

        pos_start_j = np.array([row_j[x_cols[start_j_idx]], row_j[y_cols[start_j_idx]]])
        pos_end_j   = np.array([row_j[x_cols[end_j_idx]], row_j[y_cols[end_j_idx]]])

        # --- forward merge: i ends before j starts ---
        if end_i_frame < start_j_frame and (start_j_frame - end_i_frame) <= jump_length:
            dx, dy = 0, 0
            for f, (sdx, sdy) in frame_shift.items():
                if start_j_frame >= f:
                    dx += sdx
                    dy += sdy
            pos_start_j_shifted = pos_start_j + np.array([dx, dy])
            dist = np.linalg.norm(pos_end_i - pos_start_j_shifted)
            if dist <= distance_threshold:
                merged_ids[row_j['CellID']] = row_i['CellID']

# -----------------------------
# Apply merged IDs
# -----------------------------
df_merged = df.copy()
for old_id, new_id in merged_ids.items():
    df_merged.loc[df_merged['CellID'] == old_id, 'CellID'] = new_id

# -----------------------------
# Save merged CSV
# -----------------------------
df_merged.to_csv(output_csv, index=False)
print(f"Merged trajectories saved to {output_csv}.")