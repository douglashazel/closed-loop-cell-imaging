import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
exp = "pc3_carbachol_1"

# -----------------------------
# Parameters
# -----------------------------
distance_threshold = 200  # pixels, adjust as needed
jump_length = 10  # number of frames to allow for jumps

data_path = f"{exp}/analysis"
input_csv = f"{data_path}/trajectories.csv"
output_csv = f"{data_path}/trajectories_merged.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(input_csv)
cell_ids = df['CellID'].values

# Extract frame x/y columns
x_cols = [c for c in df.columns if 'x' in c]
y_cols = [c for c in df.columns if 'y' in c]

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
# Merge candidate cells based on distance + jump_length
# -----------------------------
merged_ids = {}

for i, row_i in tqdm(incomplete_df.iterrows(), total=len(incomplete_df)):
    # Skip rows with no valid positions
    valid_i = [j for j, val in enumerate(row_i[x_cols]) if not pd.isna(val)]
    if len(valid_i) == 0:
        continue
    start_i = valid_i[0]
    end_i = valid_i[-1]
    pos_start_i = np.array([row_i[x_cols[start_i]], row_i[y_cols[start_i]]])
    pos_end_i = np.array([row_i[x_cols[end_i]], row_i[y_cols[end_i]]])

    # Compare against all other incomplete tracks (excluding the target itself)
    for j, row_j in incomplete_df.iterrows():
        if row_i['CellID'] == row_j['CellID']:
            continue

        valid_j = [jj for jj, val in enumerate(row_j[x_cols]) if not pd.isna(val)]
        if len(valid_j) == 0:
            continue
        start_j = valid_j[0]
        end_j = valid_j[-1]
        pos_start_j = np.array([row_j[x_cols[start_j]], row_j[y_cols[start_j]]])
        pos_end_j = np.array([row_j[x_cols[end_j]], row_j[y_cols[end_j]]])

        # --- forward merge: i ends before j starts ---
        if end_i < start_j and (start_j - end_i) <= jump_length:
            dist = np.linalg.norm(pos_end_i - pos_start_j)
            if dist <= distance_threshold:
                merged_ids[row_j['CellID']] = row_i['CellID']

        # --- backward merge: j ends before i starts ---
        elif end_j < start_i and (start_i - end_j) <= jump_length:
            dist = np.linalg.norm(pos_end_j - pos_start_i)
            if dist <= distance_threshold:
                merged_ids[row_i['CellID']] = row_j['CellID']

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