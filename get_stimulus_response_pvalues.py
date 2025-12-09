import os
import shutil
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind as ttest, levene 
from datetime import datetime

LOG_FILE = 'aggregated_results.log'

# Function to safely log messages to the file
def log_message(file_handle, message):
    """Writes a message with a timestamp to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_handle.write(f"[{timestamp}] {message}\n")

# Open the log file for writing (will overwrite existing content)
with open(LOG_FILE, 'w') as log:
    log_message(log, "--- STARTING T-TEST ANALYSIS LOG ---")
    
    # ---------------------
    # AGGREGATE DATA
    # ---------------------
    cell_types = {'hela': False, 'u87': False, 'nrk': False, 'pc3': False, 'c2c12': False, 'ht29': False}
    controls = ['dmem', 'dmso', 'pipettingControl']
    tag = 'stimulus_delta_complete'
    data_dirs = [d for d in os.listdir('.') if any(d.startswith(x) for x in cell_types)]

    log_message(log, "Aggregating data and copying files...")

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
                log_message(log, f"Copied: {file} to {dst}")

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
        log_message(log, f"Cell Type Check: {key} {text}") # Log status instead of print
    cell_types = cells_to_keep

    results_list = [] 

    # ---------------------
    # ANALYZE DATA & POPULATE results_df
    # ---------------------
    log_message(log, "--- BEGINNING STATISTICAL ANALYSIS ---")
    
    for cell_type in cell_types.keys():
        dir_path = f'{cell_type}_{tag}'
        # Check if the directory exists before proceeding
        if not os.path.isdir(dir_path):
            log_message(log, f"Directory {dir_path} not found. Skipping.") # Log skip
            continue

        all_files = os.listdir(dir_path)
        control_files = [f for f in all_files if f.endswith('.csv') and any(x for x in controls if x in f)]
        data_files = [f for f in all_files if f.endswith('.csv') and f not in control_files]

        for f in data_files:
            f_condition = f.split(tag)[0].split(cell_type)[1][1:-1]
            f_path = f'{dir_path}/{f}'
            df = pd.read_csv(f_path).dropna()
            if len(df) < 2:
                log_message(log, f"SKIPPING: cell={cell_type}, cond={f_condition}, n_cond={len(df)}") # Log skip
                continue
            f_mean_response = np.average(df['delta'])
            
            for c in control_files:
                c_condition = c.split(tag)[0].split(cell_type)[1][1:-1]
                c_path = f'{dir_path}/{c}'
                df_control = pd.read_csv(c_path).dropna()
                if len(df_control) < 2:
                    log_message(log, f"SKIPPING CONTROL: cell={cell_type}, cond={c_condition}, n_cond={len(df_control)}") # Log skip
                    continue
                c_mean_response = np.average(df_control['delta'])
                            
                # Match sample sizes for less crazy small p-values (still crazy small)
                n = min(len(df_control), len(df))
                cond_sample = df.sample(n, random_state=0)
                ctrl_sample = df_control.sample(n, random_state=0)
                
                # --- VARIANCE CHECK INTEGRATION ---
                
                cond_data = cond_sample['delta'].values
                ctrl_data = ctrl_sample['delta'].values
                
                # Perform Levene's Test
                _, p_levene = levene(cond_data, ctrl_data)
                
                # Set equal_var parameter based on the result
                # H0: Variances are equal. If p > 0.05, keep True.
                run_equal_var = p_levene > 0.05
                
                # Log the variance decision
                log_message(log, f"CHECK: Cell: {cell_type}, Cond: {f_condition} vs Ctrl: {c_condition} (N={n})")
                log_message(log, f"       Levene p-value: {p_levene:.4f}. Setting equal_var to: {run_equal_var}")
                
                # --- END VARIANCE CHECK INTEGRATION ---
                
                # Compute mean response difference
                resp_diff = f_mean_response - c_mean_response

                # Perform the two-sample t-test, passing the calculated 'run_equal_var'
                _, p = ttest(cond_data, ctrl_data, equal_var=run_equal_var)
                
                log_message(log, f"       T-Test p-value: {p:.4e}, Mean Diff: {resp_diff:.4f}")
                
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

    log_message(log, "--- STATISTICAL ANALYSIS COMPLETE ---")
    log_message(log, "Saving final aggregated results to aggregated_results.csv")
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('aggregated_results.csv', index=False)
    
    log_message(log, "--- END OF LOG ---")

print("Analysis complete. Results saved to 'aggregated_results.csv'.")
print(f"Full log of processing steps and t-test decisions saved to '{LOG_FILE}'.")