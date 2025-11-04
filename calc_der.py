import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

experiments = [item for item in os.listdir() if os.path.isdir(item) and item not in ["gifs", "archive"]]

for exp in tqdm(experiments, desc='Calculating...'):
    analysis_dir_path = os.path.join(exp, 'analysis')
    if os.path.isdir(analysis_dir_path) and not os.path.exists(f'{analysis_dir_path}/plots/luminosity_derivative_complete.png'):
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
        except TypeError:
            print(f"Error with {exp}")