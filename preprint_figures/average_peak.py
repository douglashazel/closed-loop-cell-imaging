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
from common.responders import compute_responder_masks
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
# Frame timing means most segments stop just short of SEGMENT_MINUTES, so the
# extreme tail grid points are covered by only a handful of segments and their
# mean is unrepresentative (it dives toward baseline). Drop grid points whose
# segment coverage falls below this fraction of the best-covered point.
MIN_COVERAGE_FRAC = 0.5


def _channel_peak_segments(state, exp_name, ch, cfg, grid, *,
                           responder_mask=None):
    """Resampled dF/F0 peak segments for one channel.

    Returns a list with one ``(n_cells, GRID_POINTS)`` array per stimulus —
    every cell's dF/F0 from the stim onset to ``SEGMENT_MINUTES`` later,
    resampled onto ``grid`` (relative minutes since onset). Grid points beyond
    a segment's actual extent (e.g. the final stim near the experiment end)
    are left NaN so they drop out of the mean.

    ``responder_mask`` (aligned to the corrected-lum ``CellID`` order) filters
    the segments to responder cells only when given.
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
    if responder_mask is not None:
        dff = dff[np.asarray(responder_mask, dtype=bool)]
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


def _render_average_peak(exp_name, grid, all_segments, n_channels, *,
                         save_name, descriptor, show_cells=True):
    """Render and save one average-peak overlay figure from pooled segments.

    ``show_cells`` draws the faint per-cell×stim segment overlay; pass False
    to render just the average line. A shaded mean ± 3·SEM band is always
    drawn. Low-coverage tail grid points are clipped (see MIN_COVERAGE_FRAC).
    """
    stacked = np.vstack(all_segments)
    n_seg = stacked.shape[0]

    # Clip grid points covered by too few segments — their mean is unreliable.
    counts = np.sum(~np.isnan(stacked), axis=0)
    keep = counts >= MIN_COVERAGE_FRAC * counts.max()
    grid = grid[keep]
    stacked = stacked[:, keep]
    counts = counts[keep]

    mean_peak = np.nanmean(stacked, axis=0)
    sem = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(counts)

    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    if show_cells:
        for row in stacked:
            ax.plot(
                grid, row,
                color=PLOT_PARAMS["cell_color"], alpha=0.05,
                linewidth=PLOT_PARAMS["cell_lw"], zorder=1,
            )
    mean_lw = PLOT_PARAMS["mean_lw"] * 1.7
    ax.fill_between(
        grid, mean_peak - 3 * sem, mean_peak + 3 * sem,
        color=PLOT_PARAMS["pooled_mean_color"], alpha=0.25,
        linewidth=0, zorder=2, label="Mean ± 3 SEM",
    )
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
        f"{exp_name} — average response peak ({descriptor})\n"
        f"per-stimulus dF/F₀ segments pooled over {n_channels} channel(s)",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
    plt.tight_layout()
    save_fig(
        fig, fig_path(exp_name, save_name),
        dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
    )
    plt.close(fig)
    print(
        f"  average peak ({descriptor}): {exp_name} — {n_seg} segments "
        f"pooled across {n_channels} channel(s)."
    )


def plot_average_peak(experiments, state):
    """Average-peak overlay figures per DMSO experiment.

    Two figures per experiment: ``average_peak`` (all cells) and
    ``average_peak_responders`` (responder cells only) — the latter lets the
    forked C2C12/PC3 response shapes be compared without the non-responder
    cloud flattening the mean.
    """
    grid = np.linspace(0.0, SEGMENT_MINUTES, GRID_POINTS)
    responder_masks = compute_responder_masks(experiments, state)
    for exp_name, cfg in experiments.items():
        if cfg.get("response_direction") != "increase":
            continue
        n_channels = len(cfg["channels"])
        all_segments = []
        resp_segments = []
        for ch in cfg["channels"]:
            all_segments.extend(
                _channel_peak_segments(state, exp_name, ch, cfg, grid)
            )
            ch_mask = responder_masks.get((exp_name, ch))
            if ch_mask is not None and np.any(ch_mask):
                resp_segments.extend(
                    _channel_peak_segments(
                        state, exp_name, ch, cfg, grid,
                        responder_mask=ch_mask,
                    )
                )
        if not all_segments:
            print(f"{exp_name}: no peak segments — skipping average-peak figure.")
            continue
        _render_average_peak(
            exp_name, grid, all_segments, n_channels,
            save_name="average_peak", descriptor="all cells",
        )
        if resp_segments:
            _render_average_peak(
                exp_name, grid, resp_segments, n_channels,
                save_name="average_peak_responders",
                descriptor="responders only",
                show_cells=False,
            )
        else:
            print(
                f"{exp_name}: no responder peak segments — "
                "skipping responders-only average-peak figure."
            )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_average_peak(experiments, state)


if __name__ == "__main__":
    main()
