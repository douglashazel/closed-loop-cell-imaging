#!/usr/bin/env python3
"""Average-peak overlay figures (DMSO experiments only).

For each DMSO experiment every per-stimulus response segment — the dF/F0
trace from a stimulus onset to 10 min later (the inter-stimulus interval) —
is pooled across all cells and all channels, resampled onto a common
0-10 min grid and overlaid with the mean:
    * average_peak.png

C2C12 and PC3 each get their own figure (in their per-experiment folder), so
the two response shapes can be compared for the PC3-vs-C2C12 figure.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.io_paths import fig_path, save_fig
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.stim_helpers import compute_f0_baseline
from common.time_axis import frames_to_min

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


# Window grabbed after each stimulus onset — the DMSO inter-stimulus interval
# (stims within a train are 10 min apart, so this is "until the next peak").
SEGMENT_MINUTES = 10.0
# Common grid the per-stimulus segments are resampled onto. Channels / experi-
# ments have different frame rates, so segments hold different frame counts;
# np.interp puts them all on the same 0-SEGMENT_MINUTES axis.
GRID_POINTS = 100


def _channel_peak_segments(state, exp_name, ch, cfg, grid):
    """Resampled dF/F0 peak segments for one channel.

    Returns a list with one ``(n_cells, GRID_POINTS)`` array per stimulus —
    every cell's dF/F0 from the stim onset to ``SEGMENT_MINUTES`` later,
    resampled onto ``grid`` (relative minutes since onset). Grid points beyond
    a segment's actual extent (e.g. the final stim near the experiment end)
    are left NaN so they drop out of the mean.
    """
    stim_frames = cfg["stim_frames"][ch]
    df = lum_dict_to_df(
        state["corrected_lum"][exp_name][ch]
    ).set_index("CellID")
    frame_cols = sorted(
        [c for c in df.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    if not frame_cols:
        return []
    frame_nums = np.array([int(str(c).lstrip("f")) for c in frame_cols])
    frame_to_col = {int(f): i for i, f in enumerate(frame_nums)}
    mat = df[frame_cols].values

    F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
    F0_safe = np.where(F0 == 0, np.nan, F0)
    dff = (mat - F0) / F0_safe
    minutes = frames_to_min(state, exp_name, ch, frame_nums)

    segments = []
    for sf in stim_frames:
        start_col = frame_to_col.get(int(sf))
        if start_col is None:
            continue
        stim_min = minutes[start_col]
        seg_cols = np.where(
            (minutes >= stim_min) & (minutes <= stim_min + SEGMENT_MINUTES)
        )[0]
        if seg_cols.size < 2:
            continue
        rel = minutes[seg_cols] - stim_min
        seg = dff[:, seg_cols]
        resampled = np.full((seg.shape[0], grid.size), np.nan)
        for i in range(seg.shape[0]):
            resampled[i] = np.interp(
                grid, rel, seg[i], left=np.nan, right=np.nan,
            )
        segments.append(resampled)
    return segments


def plot_average_peak(experiments, state):
    """One average-peak overlay figure per DMSO experiment."""
    grid = np.linspace(0.0, SEGMENT_MINUTES, GRID_POINTS)
    for exp_name, cfg in experiments.items():
        if cfg.get("response_direction") != "increase":
            continue
        all_segments = []
        for ch in cfg["channels"]:
            all_segments.extend(
                _channel_peak_segments(state, exp_name, ch, cfg, grid)
            )
        if not all_segments:
            print(f"{exp_name}: no peak segments — skipping average-peak figure.")
            continue
        stacked = np.vstack(all_segments)
        mean_peak = np.nanmean(stacked, axis=0)
        n_seg = stacked.shape[0]

        fig, ax = plt.subplots(
            figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)
        for row in stacked:
            ax.plot(
                grid, row,
                color=PLOT_PARAMS["cell_color"], alpha=0.05,
                linewidth=PLOT_PARAMS["cell_lw"], zorder=1,
            )
        mean_lw = PLOT_PARAMS["mean_lw"] * 1.7
        mean_line, = ax.plot(
            grid, mean_peak,
            color=PLOT_PARAMS["pooled_mean_color"], linewidth=mean_lw,
            label=f"Average peak (n={n_seg} cell×stim segments)", zorder=3,
        )
        # White halo so the average reads clearly over the dense overlay.
        mean_line.set_path_effects([
            pe.Stroke(linewidth=mean_lw + 2.4, foreground="white"),
            pe.Normal(),
        ])
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
        ax.set_xlabel(
            "Time since stimulus onset (min)",
            fontsize=PLOT_PARAMS["axis_label_fontsize"],
        )
        ax.set_ylabel("dF/F₀", fontsize=PLOT_PARAMS["axis_label_fontsize"])
        ax.set_title(
            f"{exp_name} — average response peak\n"
            f"per-stimulus dF/F₀ segments pooled over "
            f"{len(cfg['channels'])} channel(s)",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"],
        )
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
        plt.tight_layout()
        save_fig(
            fig, fig_path(exp_name, "average_peak"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)
        print(
            f"  average peak: {exp_name} — {n_seg} segments pooled across "
            f"{len(cfg['channels'])} channel(s)."
        )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_average_peak(experiments, state)


if __name__ == "__main__":
    main()
