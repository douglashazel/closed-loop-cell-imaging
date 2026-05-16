#!/usr/bin/env python3
"""Diagnostic: is PC3's high responder rate biology or an artifact?

Compares PC3 vs C2C12 on the two quantities that decide responder calls:
    1. per-cell mean Δ dF/F0  (the responder statistic) — bimodal vs shifted?
    2. F0 (corrected baseline luminosity) — does a dimmer F0 inflate dF/F0?

Saves a figure and prints a numeric summary. Not a preprint figure — kept
out of run_figures.sh on purpose.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.config import EXPERIMENTS, OUT_ROOT
from common.pipeline import prepare_state
from common.responders import (
    _aggregate,
    _channel_dff,
    _delta_at,
    compute_responder_thresholds,
)
from common.stim_helpers import compute_f0_baseline
from common.time_axis import response_window_frames

PLOT_PARAMS = {
    "figsize": (16, 9),
    "dpi": 300,
    "title_fontsize": 14,
    "title_fontweight": "bold",
    "suptitle_fontsize": 16,
    "colors": ["#e74c3c", "#363fe9", "#e67e22", "#1a9d51"],
    "threshold_color": "#000000",
    "hist_alpha": 0.75,
    "bins": 60,
}

TARGETS = ["pc3_dmso_23MAR26", "c2c12_dmso_09APR26"]
ALPHA = 0.01
BASELINE_N_PRE = 5
STAT = "mean"


def per_cell_response_stat(state, exp_name, cfg, ch):
    """Replicate compute_responder_masks' per-cell aggregate Δ dF/F0."""
    window = response_window_frames(state, exp_name, ch, cfg)
    direction = cfg.get("response_direction", "increase")
    dff_mat, frame_to_col = _channel_dff(state, exp_name, ch, cfg)
    per_stim = [
        _delta_at(dff_mat, frame_to_col[p], direction, window, BASELINE_N_PRE)
        for p in cfg["stim_frames"][ch]
        if p in frame_to_col
    ]
    return _aggregate(np.vstack(per_stim), STAT)


def main():
    experiments = {k: EXPERIMENTS[k] for k in TARGETS}
    state = prepare_state(experiments, recompute_bg=False)
    thresholds = compute_responder_thresholds(
        experiments, state, alpha=ALPHA, baseline_n_pre=BASELINE_N_PRE, stat=STAT,
    )

    panels = []  # (exp, ch, per_cell_stat, F0, threshold)
    for exp_name, cfg in experiments.items():
        for ch in cfg["channels"]:
            stat = per_cell_response_stat(state, exp_name, cfg, ch)
            F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
            panels.append(
                (exp_name, ch, stat, np.asarray(F0).ravel(),
                 float(thresholds.get((exp_name, ch), 0.10)))
            )

    print("\n" + "=" * 78)
    print(f"{'exp / channel':<34}{'thr':>8}{'%resp':>8}"
          f"{'med Δ (non/resp)':>22}{'med F0':>10}")
    print("-" * 78)
    for exp_name, ch, stat, f0, thr in panels:
        good = stat[~np.isnan(stat)]
        resp = good >= thr
        med_non = np.median(good[~resp]) if (~resp).any() else float("nan")
        med_resp = np.median(good[resp]) if resp.any() else float("nan")
        print(f"{exp_name + ' / ' + ch:<34}{thr:>8.4f}"
              f"{100 * resp.mean():>7.1f}%"
              f"{med_non:>11.4f}/{med_resp:<10.4f}"
              f"{np.nanmedian(f0):>10.1f}")
    print("=" * 78 + "\n")

    n = len(panels)
    fig, axes = plt.subplots(
        2, n, figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    for col, (exp_name, ch, stat, f0, thr) in enumerate(panels):
        color = PLOT_PARAMS["colors"][col % len(PLOT_PARAMS["colors"])]

        ax = axes[0, col]
        good = stat[~np.isnan(stat)]
        clip = np.clip(good, np.nanpercentile(good, 0.5),
                       np.nanpercentile(good, 99.5))
        ax.hist(clip, bins=PLOT_PARAMS["bins"], color=color,
                alpha=PLOT_PARAMS["hist_alpha"])
        ax.axvline(thr, color=PLOT_PARAMS["threshold_color"], lw=2, ls="--",
                   label=f"threshold {thr:.3f}")
        ax.axvline(0.0, color="#888888", lw=1)
        pct_resp = 100.0 * np.mean(good >= thr)
        ax.set_title(f"{exp_name}\n{ch} — {pct_resp:.1f}% responders",
                     fontsize=PLOT_PARAMS["title_fontsize"],
                     fontweight=PLOT_PARAMS["title_fontweight"])
        ax.set_xlabel("per-cell mean Δ dF/F0")
        ax.set_ylabel("cell count")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend();

        ax = axes[1, col]
        f0_good = f0[np.isfinite(f0)]
        f0_clip = np.clip(f0_good, np.nanpercentile(f0_good, 0.5),
                          np.nanpercentile(f0_good, 99.5))
        ax.hist(f0_clip, bins=PLOT_PARAMS["bins"], color=color,
                alpha=PLOT_PARAMS["hist_alpha"])
        ax.axvline(np.nanmedian(f0_good), color=PLOT_PARAMS["threshold_color"],
                   lw=2, ls="--", label=f"median F0 {np.nanmedian(f0_good):.1f}")
        ax.set_title(f"{exp_name} / {ch} — F0 baseline",
                     fontsize=PLOT_PARAMS["title_fontsize"],
                     fontweight=PLOT_PARAMS["title_fontweight"])
        ax.set_xlabel("F0 (corrected baseline luminosity)")
        ax.set_ylabel("cell count")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend();

    fig.suptitle("Responder diagnostic — PC3 vs C2C12: response statistic & F0",
                 fontsize=PLOT_PARAMS["suptitle_fontsize"],
                 fontweight=PLOT_PARAMS["title_fontweight"])
    plt.tight_layout();
    out = os.path.join(OUT_ROOT, "responder_diagnostic.png")
    fig.savefig(out, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    print(f"saved figure → {out}")


if __name__ == "__main__":
    main()
