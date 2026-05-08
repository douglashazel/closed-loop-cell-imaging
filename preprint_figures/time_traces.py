#!/usr/bin/env python3
"""Per-cell corrected luminosity time traces.

Two figures per experiment:
    * time_traces.png      — with stim shading + NRK real-setpoint marker
    * corrected_traces.png — same content, no stim markers
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
from common.stim_helpers import draw_stim_spans, stim_spans_min
from common.time_axis import frames_to_min


def plot_time_traces(experiments, state):
    """One figure per experiment: per-cell corrected luminosity with stim shading."""
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]

        fig, axes = plt.subplots(
            1, len(channels),
            figsize=(6 * len(channels), 5),
            dpi=PLOT_PARAMS["dpi"], sharey=True,
        )
        if len(channels) == 1:
            axes = np.array([axes])

        for col, ch in enumerate(channels):
            ax = axes[col]
            ax.spines[["top", "right"]].set_visible(False)

            stim_frames = cfg["stim_frames"][ch]
            corr_ch = state["corrected_lum"][exp_name][ch]
            cmap = plt.get_cmap(PLOT_PARAMS["trace_cmap"])
            colors = cmap(np.linspace(0, 1, max(len(corr_ch), 1)))

            for (cid, frames), color in zip(corr_ch.items(), colors):
                items = sorted(
                    ((int(k[1:]), v) for k, v in frames.items() if v is not None),
                    key=lambda t: t[0],
                )
                if not items:
                    continue
                xs, ys = zip(*items)
                xs_min = frames_to_min(state, exp_name, ch, xs)
                ax.plot(
                    xs_min, ys,
                    alpha=0.7, color=color,
                    linewidth=PLOT_PARAMS["cell_lw"] * 1.2,
                )

            spans, stim_label = stim_spans_min(state, exp_name, ch, cfg)
            draw_stim_spans(
                ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18
            )

            rsp = state["real_setpoint_min"][exp_name].get(ch)
            if rsp is not None:
                ax.axvline(
                    rsp,
                    color="#000000",
                    linewidth=2.0, linestyle=":",
                    alpha=0.9, zorder=5,
                    label=f"Real setpoint ({rsp:.1f} min)",
                )

            ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            if col == 0:
                ax.set_ylabel(
                    "Corrected luminosity",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )
            ax.set_title(
                f"{ch}  ({len(corr_ch)} cells, {len(stim_frames)} stims)",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
            if stim_frames or rsp is not None:
                ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")

        fig.suptitle(
            f"{exp_name} — corrected cell luminosity over time",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, "time_traces"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


def plot_corrected_traces(experiments, state):
    """One figure per experiment, one subplot per channel — clean per-cell traces."""
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]

        fig, axes = plt.subplots(
            1, len(channels),
            figsize=(6 * len(channels), 5),
            dpi=PLOT_PARAMS["dpi"], sharey=True,
        )
        if len(channels) == 1:
            axes = np.array([axes])

        for col, ch in enumerate(channels):
            ax = axes[col]
            ax.spines[["top", "right"]].set_visible(False)

            corr_ch = state["corrected_lum"][exp_name][ch]
            cmap = plt.get_cmap(PLOT_PARAMS["trace_cmap"])
            colors = cmap(np.linspace(0, 1, max(len(corr_ch), 1)))

            for (cid, frames), color in zip(corr_ch.items(), colors):
                items = sorted(
                    ((int(k[1:]), v) for k, v in frames.items() if v is not None),
                    key=lambda t: t[0],
                )
                if not items:
                    continue
                xs, ys = zip(*items)
                xs_min = frames_to_min(state, exp_name, ch, xs)
                ax.plot(
                    xs_min, ys,
                    alpha=0.7, color=color,
                    linewidth=PLOT_PARAMS["cell_lw"] * 1.2,
                )

            rsp = state["real_setpoint_min"][exp_name].get(ch)
            if rsp is not None:
                ax.axvline(
                    rsp,
                    color="#000000",
                    linewidth=2.0, linestyle=":",
                    alpha=0.9, zorder=5,
                    label=f"Real setpoint ({rsp:.1f} min)",
                )
                ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")

            ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            if col == 0:
                ax.set_ylabel(
                    "Corrected luminosity",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )
            ax.set_title(
                f"{ch}  ({len(corr_ch)} cells)",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )

        fig.suptitle(
            f"{exp_name} — corrected cell luminosity over time",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, "corrected_traces"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_time_traces(experiments, state)
    plot_corrected_traces(experiments, state)


if __name__ == "__main__":
    main()
