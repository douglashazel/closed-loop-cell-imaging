#!/usr/bin/env python3
"""Background-correction diagnostic figure (one per experiment).

Per-experiment 4-column diagnostic: probe + sample points, fitted background,
corrected probe, sampled-background-mean trace.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.time_axis import frames_to_min


def plot_bg_diagnostic(experiments, state):
    """Per-experiment diagnostic: rows=channels, cols=4 background-fit views."""
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]
        sample_points = state["bg_sample_points_exp"][exp_name]

        fig, axes = plt.subplots(
            len(channels), 4,
            figsize=(22, 5 * len(channels)),
            dpi=PLOT_PARAMS["dpi"],
        )
        if len(channels) == 1:
            axes = np.array([axes])

        col_titles = [
            "probe frame + cell-free samples",
            "fitted probe background",
            "corrected = probe - fitted bg",
            "per-frame sampled background mean",
        ]

        for row, ch in enumerate(channels):
            pd_ = state["probe_data"][exp_name][ch]
            bg = state["bg_map_by_ch"][exp_name][ch]
            trace = state["bg_trace"][exp_name][ch]

            ax = axes[row, 0]
            ax.imshow(pd_["probe"], cmap=PLOT_PARAMS["img_cmap"], vmin=0, vmax=255)
            ax.scatter(
                sample_points[:, 0], sample_points[:, 1],
                s=2, c=PLOT_PARAMS["roi_color"], alpha=0.6, linewidths=0,
            )
            ax.annotate(
                f"frame {pd_['probe_idx']}\nsample mean = {pd_['probe_sample_mean']:.1f}",
                xy=(0.02, 0.98), xycoords="axes fraction", va="top", ha="left",
                color=PLOT_PARAMS["roi_color"], fontweight="bold",
            )
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(
                ch,
                fontsize=PLOT_PARAMS["axis_label_fontsize"],
                fontweight="bold",
            )
            if row == 0:
                ax.set_title(
                    col_titles[0],
                    fontsize=PLOT_PARAMS["title_fontsize"],
                    fontweight=PLOT_PARAMS["title_fontweight"],
                )

            ax = axes[row, 1]
            im = ax.imshow(bg, cmap=PLOT_PARAMS["bg_cmap"])
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.annotate(
                f"bg range {pd_['probe_bg_min']:.1f}-{pd_['probe_bg_max']:.1f}",
                xy=(0.02, 0.98), xycoords="axes fraction", va="top", ha="left",
                color="white", fontweight="bold",
            )
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    col_titles[1],
                    fontsize=PLOT_PARAMS["title_fontsize"],
                    fontweight=PLOT_PARAMS["title_fontweight"],
                )

            ax = axes[row, 2]
            ax.imshow(pd_["corrected"], cmap=PLOT_PARAMS["img_cmap"], vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    col_titles[2],
                    fontsize=PLOT_PARAMS["title_fontsize"],
                    fontweight=PLOT_PARAMS["title_fontweight"],
                )

            ax = axes[row, 3]
            ax.spines[["top", "right"]].set_visible(False)
            mins = frames_to_min(state, exp_name, ch, np.arange(len(trace)))
            ax.plot(
                mins, trace,
                color=PLOT_PARAMS["colors"][1], linewidth=1.2,
                label="sample mean",
            )
            ax.axvline(
                frames_to_min(state, exp_name, ch, [pd_["probe_idx"]])[0],
                color=PLOT_PARAMS["colors"][3], linestyle="--",
                linewidth=PLOT_PARAMS["f0_lw"],
                label=f"probe frame {pd_['probe_idx']}",
            )
            ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax.set_ylabel(
                "Sampled bg mean (px intensity)",
                fontsize=PLOT_PARAMS["axis_label_fontsize"],
            )
            ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
            if row == 0:
                ax.set_title(
                    col_titles[3],
                    fontsize=PLOT_PARAMS["title_fontsize"],
                    fontweight=PLOT_PARAMS["title_fontweight"],
                )

        fig.suptitle(
            f"{exp_name} - per-frame sampled polynomial background correction diagnostic",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, "bg_diagnostic"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_bg_diagnostic(experiments, state)


if __name__ == "__main__":
    main()
