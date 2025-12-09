import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_luminosity_std(traj_csv, save_path, tag):
    df = pd.read_csv(traj_csv, index_col=0)
    df_transposed = df.T
    std_per_frame = df_transposed.std(axis=1)

    frames = [int(str(x).lstrip("f")) for x in std_per_frame.index]
    std_values = std_per_frame.values

    plt.figure(dpi=300)
    plt.plot(frames, std_values, 'o-', color='darkred', alpha=0.7, markersize=3)
    plt.xlabel("Frame")
    plt.ylabel("Standard Deviation of Luminosity (Across all cells)")
    plt.title("Luminosity Variability Over Time")
    plt.tight_layout()

    plot_dir = os.path.join(save_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_name = f'std_average_luminosity{tag}.png'
    plt.savefig(os.path.join(plot_dir, plot_name), dpi=300)
    plt.close()

experiments = [item for item in os.listdir() if os.path.isdir(item) and item not in ["gifs", "archive"]]

for exp in tqdm(experiments, desc='Calculating...'):
    analysis_dir_path = os.path.join(exp, 'analysis')
    if os.path.isdir(analysis_dir_path):# and not os.path.exists(f'{analysis_dir_path}/plots/luminosity_derivative_complete.png'):
        file_path = os.path.join(analysis_dir_path, 'luminosity_complete.csv')
        df = pd.read_csv(file_path)

        try:
            df_data = df.set_index('CellID')
            frame_columns = [col for col in df_data.columns if col.startswith('f')]
            df_frames = df_data[frame_columns]

            derivative_matrix = np.gradient(df_frames.values, axis=1)
            df_derivative = pd.DataFrame(derivative_matrix, index=df_frames.index, columns=df_frames.columns)

            plt.figure(dpi=300)
            df_derivative.T.plot(
                legend=False, 
                ax=plt.gca(), 
                linewidth=1,
                cmap='twilight_shifted', 
                alpha=0.7
            )

            plt.title('Derivative of cell luminosity over time')
            plt.xlabel('Frame')
            plt.ylabel('Derivative of Luminosity')

            plt.savefig(f'{analysis_dir_path}/plots/luminosity_derivative_complete.png')
            plt.close()

            plot_luminosity_std(file_path, analysis_dir_path, "")

        except TypeError:
            print(f"Error with {exp}")