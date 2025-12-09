import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_number(filename):
    match = re.search(r'timepoint_(\d+)', filename)
    return int(match.group(1)) if match else -1

def process_tag(tag, analysis_dir, stim_frame, prev_frame):
    lum_csv = os.path.join(analysis_dir, f"luminosity{tag}.csv")

    if not os.path.exists(lum_csv):
        print(f"Skipping {tag}: {lum_csv} not found")
        return

    df = pd.read_csv(lum_csv)

    stim_col = f"f{stim_frame}"
    prev_col = f"f{prev_frame}"

    if stim_col not in df.columns or prev_col not in df.columns:
        raise ValueError(f"Columns {stim_col} and/or {prev_col} not in {lum_csv}")

    # compute differences
    df_delta = pd.DataFrame({
        "CellID": df["CellID"],
        "delta": df[stim_col] - df[prev_col]
    })

    vals = df_delta["delta"].to_numpy()
    rms_pos = np.sqrt(np.mean(vals[vals >= 0] ** 2))
    rms_neg = 0 - np.sqrt(np.mean(vals[vals < 0] ** 2))

    # calculate SNR (mean delta / std dev baseline)
    signal_mean = df_delta["delta"].mean()
    noise_std = df[prev_col].std()
    
    # POSITIVE SNR
    positive_deltas = df_delta[df_delta["delta"] >= 0]["delta"]
    signal_mean_pos = positive_deltas.mean()
    snr_pos = signal_mean_pos / noise_std if noise_std != 0 else np.nan

    # NEGATIVE SNR
    negative_deltas = df_delta[df_delta["delta"] < 0]["delta"]
    signal_mean_neg = abs(negative_deltas.mean()) 
    snr_neg = signal_mean_neg / noise_std if noise_std != 0 else np.nan

    # Z-Score (mean delta / std dev delta)
    delta_std = df_delta["delta"].std()
    z_score = signal_mean / delta_std if delta_std != 0 else np.nan

    # save CSV
    delta_csv = os.path.join(analysis_dir, f"stimulus_delta{tag}.csv")
    df_delta.to_csv(delta_csv, index=False)

    # ensure plots directory exists
    plot_dir = os.path.join(analysis_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # plot
    plt.figure(dpi=300)
    plt.bar(df_delta["CellID"], df_delta["delta"], color="black", alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.axhline(rms_pos, color="purple", linestyle="--", label='pos RMS')
    plt.axhline(rms_neg, color="orange", linestyle="--", label='neg RMS')
    plt.xlabel("Cell ID")
    plt.ylabel("Δ Luminosity (stim - pre-stim)")
    plt.title(f"Luminosity change at stimulus frame f{stim_frame}")
    plt.legend()
    fig_path = os.path.join(plot_dir, f"stimulus_delta{tag}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    # Define the path for the log file (in the same directory as the plot)
    log_file_path = os.path.join(plot_dir, "stimulus_delta_log.txt")
    
    # Create the content to be written to the log file
    log_content = (
        f"--- Metrics for tag: {tag} ---\n"
        f"Stimulus Frame: f{stim_frame}\n"
        f"Baseline Frame: f{prev_frame}\n"
        f"Positive SNR: {snr_pos:.2f}\n"
        f"Negative SNR: {snr_neg:.2f}\n"
        f"Z-Score: {z_score:.2f}\n"
        f"RMS Positive Delta: {rms_pos:.2f}\n"
        f"RMS Negative Delta: {rms_neg:.2f}\n"
        f"CSV Delta Saved To: {delta_csv}\n"
        f"Plot Saved To: {fig_path}\n\n"
    )

    with open(log_file_path, 'a') as f:
        f.write(log_content)

    print(f"Processed {tag}. Metrics saved to {log_file_path}. Plot saved to {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--stim_frame", type=int, default=5)
    args = parser.parse_args()

    exp = args.exp
    analysis_dir = args.analysis_dir
    stim_frame = args.stim_frame
    prev_frame = stim_frame - 1

    # Before processing tags, ensure the log file is clean (optional, but good practice)
    plot_dir = os.path.join(analysis_dir, "plots")
    log_file_path = os.path.join(plot_dir, "stimulus_delta_log.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path) # Delete old log file

    for tag in ["_no_bground_complete", "_complete"]:
        process_tag(tag, analysis_dir, stim_frame, prev_frame)