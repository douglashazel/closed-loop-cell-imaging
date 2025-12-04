import os
import shutil
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind as ttest

# ---------------------
# AGGREGATE DATA
# ---------------------
cell_types = {'hela': False, 'u87': False, 'nrk': False, 'pc3': False, 'c2c12': False, 'ht29': False}
tag = 'stimulus_delta_complete'
data_dirs = [d for d in os.listdir('.') if any(x for x in cell_types.keys() if d.startswith(x))]

for cell in cell_types.keys():
    save_path = f"{cell}_{tag}"
    os.makedirs(save_path, exist_ok=True)
    conditions = [d for d in data_dirs if d.startswith(cell)]
    for condition in conditions:
        file = f'{condition}/analysis/stimulus_delta_complete.csv'
        dst = f"{cell}_{tag}/{condition}_stimulus_delta_complete.csv"
        if os.path.isfile(file):
            if os.path.exists(dst):
                continue
            shutil.copy(file, dst)

# ---------------------
# CONFIG
# ---------------------
delta_dirs = [d for d in os.listdir('.') if tag in d and any(d.startswith(c) for c in cell_types.keys())]

for d in delta_dirs:
    cell = d.split('_')[0]
    if cell in cell_types:
        cell_types[cell] = True 

cells_to_keep = {}
for key, val in cell_types.items():
    if val:
        text = 'found.'
        cells_to_keep[key] = val
    else:
        text = 'not found.'
    print(key, text)
cell_types = cells_to_keep

results_list = [] 

# ---------------------
# ANALYZE DATA & POPULATE results_df
# ---------------------
for cell_type in cell_types.keys():
    dir_path = f'{cell_type}_{tag}'
    # Check if the directory exists before proceeding
    if not os.path.isdir(dir_path):
        print(f"Directory {dir_path} not found. Skipping.")
        continue

    all_files = os.listdir(dir_path)
    files = [f for f in all_files if f.endswith('.csv') and 'Control' not in f]
    controls = [f for f in all_files if f.endswith('.csv') and 'Control' in f]

    if len(controls) < 1:
        print(f"No control files found in {dir_path}. Skipping.")
        continue
    
    # Compute baseline data
    control_dfs = []
    for control_file in controls:
        df_control = pd.read_csv(f'{dir_path}/{control_file}').dropna()
        control_dfs.append(df_control)
    dfc = pd.concat(control_dfs, ignore_index=True)
    
    if dfc.empty or 'delta' not in dfc.columns:
        print(f"Control data is empty or missing 'delta' column in {dir_path}. Skipping.")
        continue
        
    control_mean_response = np.average(dfc['delta'])

    for file in files:
        condition = file.split(tag)[0].split(cell_type)[1][1:-1]
        file_path = f'{dir_path}/{file}'
        
        try:
            df = pd.read_csv(file_path).dropna()
            if 'delta' in df.columns and not df.empty:
                mean_response = np.average(df['delta'])
                
                # Perform the two-sample t-test
                t_statistic, p = ttest(df['delta'], dfc['delta'])
                
                new_row = {
                    'cell_type': cell_type,
                    'condition': condition,
                    'mean_response_pvalue': p, 
                    'mean_response_diff': mean_response - control_mean_response
                }
                
                results_list.append(new_row)
            else:
                print(f"Skipping {file_path}: 'delta' column not found or DataFrame is empty.")

        except Exception as e:
            print(f"Could not read or process file {file_path}. Error: {e}")

results_df = pd.DataFrame(results_list)
results_df.to_csv('aggregated_results.csv', index=False)