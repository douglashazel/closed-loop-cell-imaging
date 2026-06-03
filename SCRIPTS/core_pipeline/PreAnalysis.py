import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from io_utils import load_msgpack, lum_dict_to_df

# ── Style ─────────────────────────────────────────────────────────────────────
PLOT_PARAMS = {
    'figsize': (10, 6),
    'dpi': 300,
    'title_fontsize': 14,
    'title_fontweight': 'bold',
    'axis_label_fontsize': 13,
    'trace_cmap': 'twilight_shifted',
    'trace_alpha': 0.7,
}


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_luminosity_traces(lum_dict, analysis_dir, tag):
    cmap = plt.get_cmap(PLOT_PARAMS['trace_cmap'])
    colors = cmap(np.linspace(0, 1, len(lum_dict)))

    fig, ax = plt.subplots(figsize=PLOT_PARAMS['figsize'], dpi=PLOT_PARAMS['dpi'])
    ax.spines[['top', 'right']].set_visible(False)

    for (cell_id, frame_lums), color in zip(lum_dict.items(), colors):
        frames_vals = [(int(k.lstrip('f')), v) for k, v in frame_lums.items() if v is not None]
        if not frames_vals:
            continue
        frames_vals.sort()
        frames, vals = zip(*frames_vals)
        ax.plot(frames, vals, alpha=PLOT_PARAMS['trace_alpha'], color=color)

    ax.set_xlabel("Frame", fontsize=PLOT_PARAMS['axis_label_fontsize'])
    ax.set_ylabel("Average Luminosity", fontsize=PLOT_PARAMS['axis_label_fontsize'])
    ax.set_title("Cell Luminosity Over Time",
                 fontsize=PLOT_PARAMS['title_fontsize'], fontweight=PLOT_PARAMS['title_fontweight'])

    plot_dir = os.path.join(analysis_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    fig.savefig(os.path.join(plot_dir, f'average_luminosity{tag}.png'), dpi=PLOT_PARAMS['dpi'])
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--analysis_dir", required=True)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir

    for tag in ['', '_complete']:
        lum_path = os.path.join(analysis_dir, f'luminosity{tag}.json')
        if not os.path.exists(lum_path):
            print(f"  Skipping {lum_path} (not found)")
            continue

        lum_dict = load_msgpack(lum_path)
        n_cells = len(lum_dict)
        print(f">>> Plotting luminosity{tag} ({n_cells} cells)")
        plot_luminosity_traces(lum_dict, analysis_dir, tag)
        print(f"  Saved average_luminosity{tag}.png")

    print(f">>> Pre-analysis plots finished for {args.exp}")
