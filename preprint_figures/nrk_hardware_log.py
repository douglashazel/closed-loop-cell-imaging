#!/usr/bin/env python3
"""NRK hardware-feedback luminosity log.

Per channel of the NRK acid experiment:
    * <ch>_hw_lum_log.png

For non-NRK selections this script silently no-ops (matches source behavior).
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.config import PULSE_DEDUP_FRAMES
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS_HW_LOG
from common.stim_resolve import _dedup_close_frames
from common.time_axis import frames_to_min, setpoint_regions_from_log


def plot_nrk_hardware_log(experiments, state, exp_name="nrk_acid_13APR26"):
    """Plot the hardware-feedback luminosity log for the NRK acid experiment."""
    pp = PLOT_PARAMS_HW_LOG
    if exp_name not in experiments:
        return
    cfg = experiments[exp_name]

    for ch in cfg["channels"]:
        log_path, ch_num = cfg["stim_logs"][ch]
        lum_log_path = os.path.join(
            os.path.dirname(log_path),
            f"luminosity_log_channel{ch_num}.json",
        )

        with open(lum_log_path) as f:
            entries = json.load(f)
        entries = [e for e in entries if e.get("channel") == ch_num]
        if not entries:
            print(f"NRK / {ch}: no entries in {lum_log_path} — skipping.")
            continue
        entries.sort(key=lambda e: e["frame"])

        regions = setpoint_regions_from_log(entries)
        if not regions:
            print(f"NRK / {ch}: no setpoint regions parsed — skipping.")
            continue

        frames = [e["frame"] for e in entries]
        luminosity = [e["mean_luminosity"] for e in entries]

        acid_frames_raw = [
            e["frame"] for e in entries if e.get("decision") == "add acidic media"
        ]
        acid_frames = _dedup_close_frames(acid_frames_raw, PULSE_DEDUP_FRAMES)

        frames_min = frames_to_min(state, exp_name, ch, frames)
        acid_min = frames_to_min(state, exp_name, ch, acid_frames) if acid_frames else []
        rsp = state["real_setpoint_min"][exp_name].get(ch)

        rsp_str = f"{rsp:.2f} min" if rsp is not None else "not detected"
        print(
            f"NRK / {ch}: {len(frames)} frames | "
            f"{len(acid_frames)} acidic pulses (raw {len(acid_frames_raw)}) | "
            f"{len(regions)} setpoint regions | real setpoint @ {rsp_str}"
        )

        fig, ax = plt.subplots(figsize=pp["figsize"], dpi=pp["dpi"])
        ax.spines[["top", "right"]].set_visible(False)

        seen_sp = {}
        for idx, (start_f, end_f, sp) in enumerate(regions):
            if idx == 0:
                continue  # skip the initial calibration region
            color = pp["setpoint_colors"][idx % len(pp["setpoint_colors"])]
            label = f"Setpoint {sp:.2f}" if sp not in seen_sp else None
            seen_sp[sp] = color
            start_m, end_m = frames_to_min(state, exp_name, ch, [start_f, end_f])
            ax.fill_between(
                [start_m, end_m], 0, sp,
                color=color, alpha=pp["setpoint_alpha"],
                label=label, zorder=1,
            )
            ax.hlines(
                sp, xmin=start_m, xmax=end_m,
                colors=color, linewidths=pp["setpoint_lw"], zorder=2,
            )

        ax.plot(
            frames_min, luminosity,
            color=pp["line_color"], linewidth=pp["line_lw"],
            label="Mean luminosity", zorder=3,
        )

        pulse_duration = float(cfg.get("stim_duration_minutes", 0.5) or 0.5)
        acid_label = f"Acidic pulse ({int(pulse_duration * 60)} s)"
        for i, m in enumerate(acid_min):
            ax.axvspan(
                m, m + pulse_duration,
                color=pp["acid_color"], alpha=0.30,
                linewidth=0, zorder=2,
                label=acid_label if i == 0 else None,
            )

        if rsp is not None:
            ax.axvline(
                rsp,
                color="#000000",
                linewidth=2.0, linestyle=":",
                alpha=0.9, zorder=5,
                label=f"Real setpoint ({rsp:.1f} min)",
            )

        x_lo = float(np.asarray(frames_min).min())
        x_hi = float(cfg.get("time_window_minutes", 30.0))
        in_window = [
            lum for lum, m in zip(luminosity, frames_min)
            if x_lo <= m <= x_hi
        ]
        if in_window:
            y_lo = min(in_window) - 2
            y_hi = max(in_window) * 1.05
        else:
            y_lo = min(luminosity) - 2
            y_hi = max(luminosity) * 1.05
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(x_lo, x_hi)

        ax.set_title(
            f"NRK / {ch} — hardware feedback luminosity log",
            fontsize=pp["title_fontsize"], fontweight=pp["title_fontweight"],
        )
        ax.set_xlabel("Time (min)", fontsize=pp["axis_label_fontsize"])
        ax.set_ylabel("Mean luminosity", fontsize=pp["axis_label_fontsize"])
        ax.legend(loc="upper left", fontsize=pp["legend_fontsize"])
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, f"{ch}_hw_lum_log"),
            dpi=pp["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_nrk_hardware_log(experiments, state)


if __name__ == "__main__":
    main()
