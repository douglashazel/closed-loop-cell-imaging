import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io_utils import load_msgpack, lum_dict_to_df

def plot_luminosity_std(df, save_path, tag):
    df_indexed = df.set_index('CellID')
    df_transposed = df_indexed.T
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

    plt.savefig(os.path.join(plot_dir, f'std_average_luminosity{tag}.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--analysis_dir", required=True)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    file_path = os.path.join(analysis_dir, 'luminosity_complete.json')

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit(1)

    df = lum_dict_to_df(load_msgpack(file_path))

    try:
        df_data = df.set_index('CellID')
        frame_columns = [col for col in df_data.columns if col.startswith('f')]
        df_frames = df_data[frame_columns]

        derivative_matrix = np.gradient(df_frames.values, axis=1)
        df_derivative = pd.DataFrame(derivative_matrix, index=df_frames.index, columns=df_frames.columns)

        plot_dir = os.path.join(analysis_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

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
        plt.savefig(os.path.join(plot_dir, 'luminosity_derivative_complete.png'), dpi=300)
        plt.close()

        plot_luminosity_std(df, analysis_dir, "")

    except TypeError:
        print(f"Error with {args.exp}")
