import os
import re
import argparse
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
    plt.xlabel("Cell ID")
    plt.ylabel("Δ Luminosity (stim - pre-stim)")
    plt.title(f"Luminosity change at stimulus frame f{stim_frame}")
    fig_path = os.path.join(plot_dir, f"stimulus_delta{tag}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Processed {tag} - saved: \n{delta_csv}\n{fig_path}")

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

    for tag in ["_no_bground_complete", "_complete"]:
        process_tag(tag, analysis_dir, stim_frame, prev_frame)