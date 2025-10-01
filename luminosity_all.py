import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# Plotting function
# -----------------------------
def plot_luminosities_from_csv(lum_csv, save_path, cmap_name="twilight"):
    df = pd.read_csv(lum_csv, index_col=0)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(df)))

    plt.figure(dpi=300)
    for (cell_id, row), color in tqdm(zip(df.iterrows(), colors), total=len(df), desc='Plotting cells...'):
        non_nan = row.dropna()
        frames = [int(str(x).lstrip("f")) for x in non_nan.index]
        plt.plot(frames, non_nan.values, alpha=0.7, color=color)

    plt.xlabel("Frame")
    plt.ylabel("Average luminosity")
    plt.title("Cell luminosity over time")
    plt.tight_layout()
    plot_name = f'average_luminosity_{lum_csv.split("_")[-1].replace(".csv","")}.png'
    plt.savefig(f"{save_path}/{plot_name}", dpi=300)
    plt.close()
    return plot_name

# -----------------------------
# Config / processing function
# -----------------------------
def process_movies(exp, traj_suffixes=["merged"]):
    data_path = f"{exp}/analysis"

    for suffix in traj_suffixes:
        trajectories_csv = f"{data_path}/trajectories_{suffix}.csv"
        luminosity_csv = f"{data_path}/luminosity.csv"
        output_lum_csv = f"{data_path}/luminosity_{suffix}.csv"

        traj_df = pd.read_csv(trajectories_csv)
        lum_df = pd.read_csv(luminosity_csv)

        # Ensure consistent types
        traj_df['CellID'] = traj_df['CellID'].astype(str)
        lum_df['CellID'] = lum_df['CellID'].astype(str)

        # Keep only CellIDs present in trajectories
        lum_df = lum_df[lum_df['CellID'].isin(traj_df['CellID'].unique())]

        # Collapse duplicates
        lum_merged = lum_df.groupby('CellID', as_index=False).mean()

        lum_merged.to_csv(output_lum_csv, index=False)
        print(f"Merged luminosity saved to {output_lum_csv}")

        plot_name = plot_luminosities_from_csv(output_lum_csv, f"{data_path}", cmap_name="twilight")
        print(f"Figure saved to {data_path}/{plot_name}")

# -----------------------------
# Main
# -----------------------------
exp = "pc3_carbachol_1"
process_movies(exp)