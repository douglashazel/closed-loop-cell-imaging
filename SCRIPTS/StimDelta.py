import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from io_utils import load_msgpack, save_msgpack, lum_dict_to_df


def process_tag(tag, analysis_dir, stim_frame, prev_frame):
    lum_json = os.path.join(analysis_dir, f"luminosity{tag}.json")

    if not os.path.exists(lum_json):
        print(f"Skipping {tag}: {lum_json} not found")
        return

    df = lum_dict_to_df(load_msgpack(lum_json))

    stim_col = f"f{stim_frame}"
    prev_col = f"f{prev_frame}"

    if stim_col not in df.columns or prev_col not in df.columns:
        raise ValueError(f"Columns {stim_col} and/or {prev_col} not in {lum_json}")

    # compute differences
    df_delta = df[["CellID"]].copy()
    df_delta["delta"] = df[stim_col] - df[prev_col]

    vals = df_delta["delta"].to_numpy()
    rms_pos = np.sqrt(np.mean(vals[vals >= 0] ** 2))
    rms_neg = 0 - np.sqrt(np.mean(vals[vals < 0] ** 2))

    signal_mean = df_delta["delta"].mean()
    noise_std = df[prev_col].std()

    positive_deltas = df_delta[df_delta["delta"] >= 0]["delta"]
    snr_pos = positive_deltas.mean() / noise_std if noise_std != 0 else np.nan

    negative_deltas = df_delta[df_delta["delta"] < 0]["delta"]
    snr_neg = abs(negative_deltas.mean()) / noise_std if noise_std != 0 else np.nan

    delta_std = df_delta["delta"].std()
    z_score = signal_mean / delta_std if delta_std != 0 else np.nan

    # save as msgpack JSON
    delta_dict = {str(int(row['CellID'])): {'delta': row['delta']} for _, row in df_delta.iterrows()}
    delta_json_path = os.path.join(analysis_dir, f"stimulus_delta{tag}.json")
    save_msgpack(delta_dict, delta_json_path)

    # plot
    plot_dir = os.path.join(analysis_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

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

    # log metrics
    log_file_path = os.path.join(plot_dir, "stimulus_delta_log.txt")
    with open(log_file_path, 'a') as f:
        f.write(
            f"--- Metrics for tag: {tag} ---\n"
            f"Stimulus Frame: f{stim_frame}\n"
            f"Baseline Frame: f{prev_frame}\n"
            f"Positive SNR: {snr_pos:.2f}\n"
            f"Negative SNR: {snr_neg:.2f}\n"
            f"Z-Score: {z_score:.2f}\n"
            f"RMS Positive Delta: {rms_pos:.2f}\n"
            f"RMS Negative Delta: {rms_neg:.2f}\n"
            f"JSON Delta Saved To: {delta_json_path}\n"
            f"Plot Saved To: {fig_path}\n\n"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--stim_frame", type=int, default=5)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    stim_frame = args.stim_frame
    prev_frame = stim_frame - 1

    plot_dir = os.path.join(analysis_dir, "plots")
    log_file_path = os.path.join(plot_dir, "stimulus_delta_log.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    for tag in ["_no_bground_complete", "_complete"]:
        process_tag(tag, analysis_dir, stim_frame, prev_frame)
