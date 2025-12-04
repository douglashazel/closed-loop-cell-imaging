import os
import shutil
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind as ttest

# ---------------------
# AGGREGATE DATA
# ---------------------
cell_types = {'hela': False, 'u87': False, 'nrk': False, 'pc3': False, 'c2c12': False, 'ht29': False}
controls = ['dmem', 'dmso', 'pipettingControl']
tag = 'stimulus_delta_complete'
data_dirs = [d for d in os.listdir('.') if any(d.startswith(x) for x in cell_types)]

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
    control_files = [f for f in all_files if f.endswith('.csv') and any(x for x in controls if x in f)]
    data_files = [f for f in all_files if f.endswith('.csv') and f not in control_files]

    for f in data_files:
        f_condition = f.split(tag)[0].split(cell_type)[1][1:-1]
        f_path = f'{dir_path}/{f}'
        df = pd.read_csv(f_path).dropna()
        f_mean_response = np.average(df['delta'])

        for c in control_files:
            c_condition = c.split(tag)[0].split(cell_type)[1][1:-1]
            c_path = f'{dir_path}/{c}'
            df_control = pd.read_csv(c_path).dropna()
            c_mean_response = np.average(df_control['delta'])
                        
            # Match sample sizes for less crazy small p-values (still crazy small)
            n = min(len(df_control), len(df))
            cond_sample = df.sample(n, random_state=0)
            ctrl_sample = df_control.sample(n, random_state=0)

            # Compute mean response difference
            resp_diff = f_mean_response - c_mean_response

            # Perform the two-sample t-test
            if len(df) < 2 or len(df_control) < 2:
                print(f"SKIPPING: cell={cell_type}, cond={f_condition}, control={c_condition}, "
                    f"n_cond={len(df)}, n_ctrl={len(df_control)}")
                continue
            _, p = ttest(cond_sample['delta'], ctrl_sample['delta'])
            
            measurements = {'mean_resp_diff': resp_diff, 'resp_pval': p}
            for measurement, result in measurements.items():
                new_row = {
                    'cell_type': cell_type,
                    'condition': f_condition,
                    'measurement': measurement,
                    'result': result,
                    'control': c_condition
                }
            
                results_list.append(new_row)

results_df = pd.DataFrame(results_list)
results_df.to_csv('aggregated_results.csv', index=False)