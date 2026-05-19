#!/usr/bin/env python3
"""Average-peak overlay figures (DMSO experiments only).

For each DMSO experiment every per-stimulus response segment — the dF/F0
trace from a stimulus onset to 10 min later (the inter-stimulus interval) —
is pooled across all cells and all channels, resampled onto a common
0-10 min grid and overlaid with the mean:
    * average_peak.png            — all cells
    * average_peak_responders.png — responder cells only

A stim-#8 derivative isolates just the 8th-pulse response window:
    * average_peak_responders_stim8.png  — one figure per DMSO experiment
    * dmso_stim8_comparison/average_peak_responders_stim8_combined.png
                                         — PC3 vs C2C12 overlaid

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
from common.config import EXPERIMENTS, OUT_ROOT
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

# Stimulus singled out for the stim-#8 derivative figures (1-indexed: the 8th
# DMSO pulse overall — the 3rd pulse of the 2nd train).
STIM8_INDEX = 8
# The cross-experiment PC3-vs-C2C12 stim-#8 overlay isn't tied to a single
# experiment, so it lands in its own results subfolder.
STIM8_COMBINED_DIR = os.path.join(OUT_ROOT, "dmso_stim8_comparison")


def _channel_peak_segments(state, exp_name, ch, cfg, grid, *,
                           responder_mask=None, stim_indices=None):
    """Resampled dF/F0 peak segments for one channel.

    Returns a list with one ``(n_cells, GRID_POINTS)`` array per stimulus —
    every cell's dF/F0 from the stim onset to ``SEGMENT_MINUTES`` later,
    resampled onto ``grid`` (relative minutes since onset). Grid points beyond
    a segment's actual extent (e.g. the final stim near the experiment end)
    are left NaN so they drop out of the mean.

    ``responder_mask`` (aligned to the corrected-lum ``CellID`` order) filters
    the segments to responder cells only when given.

    ``stim_indices`` (a set/sequence of 0-based stimulus indices) restricts the
    result to those stimuli — used by the stim-#8 derivative figures.
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
    for si, sf in enumerate(stim_frames):
        if stim_indices is not None and si not in stim_indices:
            continue
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


def _mean_sem_clipped(stacked, grid):
    """Mean ± SEM over pooled segments, dropping low-coverage tail grid points.

    Mirrors the clipping ``_render_average_peak`` applies (see
    ``MIN_COVERAGE_FRAC``): the extreme tail grid points are covered by only a
    handful of segments, so their mean dives unrepresentatively toward
    baseline. Returns ``(grid_kept, mean, sem)``.
    """
    counts = np.sum(~np.isnan(stacked), axis=0)
    keep = counts >= MIN_COVERAGE_FRAC * counts.max()
    grid = grid[keep]
    stacked = stacked[:, keep]
    counts = counts[keep]
    mean = np.nanmean(stacked, axis=0)
    sem = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(counts)
    return grid, mean, sem


def _pooled_stim8_segments(state, exp_name, cfg, grid, responder_masks):
    """Responder dF/F0 segments for stimulus ``STIM8_INDEX``, pooled per channel.

    Returns a stacked ``(n_segments, GRID_POINTS)`` array — one row per
    responder cell per channel for the single stim-#8 response window — or
    ``None`` when the stimulus is absent or has no responders.
    """
    rows = []
    for ch in cfg["channels"]:
        ch_mask = responder_masks.get((exp_name, ch))
        if ch_mask is None or not np.any(ch_mask):
            continue
        rows.extend(
            _channel_peak_segments(
                state, exp_name, ch, cfg, grid,
                responder_mask=ch_mask, stim_indices=(STIM8_INDEX - 1,),
            )
        )
    if not rows:
        return None
    return np.vstack(rows)


def _stim8_cache_path(exp_name):
    """Path to the per-experiment stim-#8 pooled-segment cache (.npz)."""
    return os.path.join(OUT_ROOT, exp_name, "average_peak_responders_stim8.npz")


def _render_stim8(exp_name, grid, stacked, n_channels):
    """Render the stim-#8 average-peak figure for one experiment (responders).

    A one-stimulus derivation of ``average_peak_responders.png``: same mean
    line + 3·SEM band styling, but pooling only the stim-#8 response window.
    """
    grid, mean_peak, sem = _mean_sem_clipped(stacked, grid)
    n_seg = stacked.shape[0]

    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    mean_lw = PLOT_PARAMS["mean_lw"] * 1.7
    ax.fill_between(
        grid, mean_peak - 3 * sem, mean_peak + 3 * sem,
        color=PLOT_PARAMS["pooled_mean_color"], alpha=0.25,
        linewidth=0, zorder=2, label="Mean ± 3 SEM",
    )
    mean_line, = ax.plot(
        grid, mean_peak,
        color=PLOT_PARAMS["pooled_mean_color"], linewidth=mean_lw,
        label=f"Average peak (n={n_seg} responder cell segments)", zorder=3,
    )
    # White halo so the average reads clearly.
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
        f"{exp_name} — average response peak, stimulus #{STIM8_INDEX} "
        f"(responders only)\nper-cell dF/F₀ segments pooled over "
        f"{n_channels} channel(s)",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
    plt.tight_layout()
    save_fig(
        fig, fig_path(exp_name, "average_peak_responders_stim8"),
        dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
    )
    plt.close(fig)
    print(
        f"  stim-#{STIM8_INDEX} average peak (responders): {exp_name} — "
        f"{n_seg} responder cell segments across {n_channels} channel(s)."
    )


def _render_stim8_combined(grid, per_exp):
    """Overlay every DMSO experiment's stim-#8 average peak on one figure.

    ``per_exp`` is a list of ``(exp_name, stacked)`` pairs. Each experiment
    gets its responder mean with a ±1 SEM band — lighter than the per-
    experiment figure's ±3 SEM so two bands stay legible together.
    """
    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
    mean_lw = PLOT_PARAMS["mean_lw"] * 1.7
    for i, (exp_name, stacked) in enumerate(per_exp):
        color = PLOT_PARAMS["colors"][i % len(PLOT_PARAMS["colors"])]
        g, mean_peak, sem = _mean_sem_clipped(stacked, grid)
        ax.fill_between(
            g, mean_peak - sem, mean_peak + sem,
            color=color, alpha=0.18, linewidth=0, zorder=2,
        )
        line, = ax.plot(
            g, mean_peak, color=color, linewidth=mean_lw, zorder=3,
            label=f"{exp_name}  (n={stacked.shape[0]} responder cell segments)",
        )
        # White halo so each mean reads clearly where the bands overlap.
        line.set_path_effects([
            pe.Stroke(linewidth=mean_lw + 2.4, foreground="white"),
            pe.Normal(),
        ])
    ax.set_xlabel(
        "Time since stimulus onset (min)",
        fontsize=PLOT_PARAMS["axis_label_fontsize"],
    )
    ax.set_ylabel("dF/F₀", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_title(
        f"DMSO stimulus #{STIM8_INDEX} average response peak — "
        f"PC3 vs C2C12 (responders only)\nmean ± 1 SEM per experiment",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
    plt.tight_layout()
    os.makedirs(STIM8_COMBINED_DIR, exist_ok=True)
    save_fig(
        fig,
        os.path.join(
            STIM8_COMBINED_DIR, "average_peak_responders_stim8_combined.png"
        ),
        dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
    )
    plt.close(fig)
    print(
        f"  stim-#{STIM8_INDEX} average peak (combined): "
        f"{len(per_exp)} experiment(s) overlaid → {STIM8_COMBINED_DIR}/"
    )


def plot_stim8_average_peak(experiments, state):
    """Stimulus-#8 derivative of ``average_peak_responders``.

    For each DMSO experiment passed in, pool the responder dF/F0 segments for
    the single stim-#8 response window and render
    ``average_peak_responders_stim8.png``. Each experiment's pooled segments
    are also cached (``average_peak_responders_stim8.npz``) so the cross-
    experiment PC3-vs-C2C12 overlay can be assembled even when the pipeline
    runs one experiment per process — the last DMSO experiment to finish sees
    every cache and renders the combined figure.
    """
    grid = np.linspace(0.0, SEGMENT_MINUTES, GRID_POINTS)
    responder_masks = compute_responder_masks(experiments, state)

    processed_dmso = 0
    for exp_name, cfg in experiments.items():
        if cfg.get("response_direction") != "increase":
            continue
        processed_dmso += 1
        stacked = _pooled_stim8_segments(
            state, exp_name, cfg, grid, responder_masks
        )
        if stacked is None:
            print(
                f"{exp_name}: no responder segments for stimulus "
                f"#{STIM8_INDEX} — skipping stim-#{STIM8_INDEX} figure."
            )
            continue
        _render_stim8(exp_name, grid, stacked, len(cfg["channels"]))
        cache = _stim8_cache_path(exp_name)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, stacked=stacked)

    if processed_dmso == 0:
        return

    # Assemble the cross-experiment overlay from whatever per-experiment
    # caches exist. With one experiment per process the second DMSO experiment
    # to finish is the first to see both caches and render this figure.
    per_exp = []
    for name, cfg in EXPERIMENTS.items():
        if cfg.get("response_direction") != "increase":
            continue
        cache = _stim8_cache_path(name)
        if os.path.exists(cache):
            with np.load(cache) as data:
                per_exp.append((name, data["stacked"]))
    if len(per_exp) >= 2:
        _render_stim8_combined(grid, per_exp)
    else:
        print(
            f"  stim-#{STIM8_INDEX} combined overlay: only {len(per_exp)} "
            "DMSO cache(s) present — deferring until the other DMSO "
            "experiment has run."
        )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_average_peak(experiments, state)
    plot_stim8_average_peak(experiments, state)


if __name__ == "__main__":
    main()
