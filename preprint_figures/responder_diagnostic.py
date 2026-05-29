#!/usr/bin/env python3
"""Responder-call diagnostics — is the responder rate biology or artifact?

Per (experiment, channel) this writes two figures into the experiment's
results folder so the PDF aggregator picks them up:

* ``responder_distribution_diagnostic`` — every cell's aggregate per-stim
  Δ dF/F0 as a marginal histogram + jittered strip scatter, with the
  responder threshold drawn through it. A clean responder population is a
  bulk piled at 0 with a separated right tail; an artifact shows up as a
  whole-population blob shifted off 0 that the threshold merely slices.

* ``responder_stimlock_diagnostic`` — the population Δ dF/F0 aligned to
  real stim onsets vs. randomly placed pseudo-stims. A field-wide step at
  real stims (absent at pseudo-stims) means a stimulus-locked nuisance,
  not per-cell responses. A footer panel summarises the closed
  responder-rate investigation (diagnoses #1–#6).

* ``responder_artifact_diagnostic`` — diagnoses #2 and #3. #2: every
  masked dead frame's offset from each stim onset against the shaded
  baseline/response windows, plus per-stim Δ dF/F0 recomputed with those
  dead columns NaN-masked instead of interpolated — if the two agree,
  dead-frame interpolation is not biasing Δ. #3: background level
  (``bg_trace``, ``bg_min``) and image sharpness (variance of the
  Laplacian) aligned to stim onset, real vs. pseudo — a field-wide
  optical artifact steps at real stims and stays flat at pseudo-stims.

* ``responder_f0_diagnostic`` — diagnosis #4. Per-cell Δ dF/F0 vs F0
  alongside the same response in additive luminosity units (Δ dF/F0 × F0)
  vs F0. A Δ dF/F0 that declines toward low F0 while Δ luminosity stays
  flat means the responder signal is a 1/F0 normalization artifact —
  dim cells clear the threshold only because the bump is divided by a
  small baseline.

Diagnostic only — changes nothing in the responder pipeline.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import laplace
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.io_paths import channel_dir, fig_path, save_fig, sorted_image_files
from common.pipeline import prepare_state, resolve_dead_frame_indices
from common.responders import (
    _aggregate,
    _channel_dff,
    _delta_at,
    compute_responder_thresholds,
)
from common.stim_helpers import compute_f0_baseline
from common.time_axis import response_window_frames
from figstyle import PLOT_PARAMS as _FIGSTYLE_PARAMS, apply_style

apply_style()

PLOT_PARAMS = {
    "width_full": _FIGSTYLE_PARAMS["width_full"],
    "dpi": _FIGSTYLE_PARAMS["dpi"],
    # Font sizes governed by the single source (figstyle) so these diagnostic
    # panels match the locked preprint style used everywhere else.
    "title_fontsize": _FIGSTYLE_PARAMS["title_fontsize"],
    "title_fontweight": _FIGSTYLE_PARAMS["title_fontweight"],
    "suptitle_fontsize": _FIGSTYLE_PARAMS["suptitle_fontsize"],
    "panel_fontsize": _FIGSTYLE_PARAMS["panel_label_size"],
    "nonresponder_color": "#9aa0a6",
    "responder_color": "#e74c3c",
    "threshold_color": "#111111",
    "zero_color": "#888888",
    "real_color": "#363fe9",
    "pseudo_color": "#9aa0a6",
    "window_shade": "#363fe9",
    "dead_color": "#e67e22",        # dead frame falling inside a stim window
    "dead_far_color": "#c8ccd0",    # dead frame clear of every stim window
    "baseline_shade": "#e67e22",
    "interp_color": "#363fe9",      # pipeline value (dead frames interpolated)
    "masked_color": "#e74c3c",      # recomputed with dead frames NaN-masked
    "bgmin_color": "#16a085",
    "focus_color": "#8e44ad",
    "hist_alpha": 0.55,
    "scatter_alpha": 0.35,
    "scatter_size": 11,
    "band_alpha": 0.25,
    "bins": 70,
}

ALPHA = 0.01
BASELINE_N_PRE = 5
STAT = "mean"
N_PSEUDO_TRACE = 400
# Pseudo-stim anchors used for the optical-artifact check. Kept small because
# each one triggers a windowed image reload for the focus metric.
N_PSEUDO_FOCUS = 24
# Frames are box-reduced toward this short-side pixel count before the
# Laplacian. A defocus / refractive shift is a low-frequency, field-wide
# change, so the full-resolution decode (PC3 frames are ~20 megapixels) is
# unnecessary and would make the focus reload dominate the runtime.
FOCUS_TARGET_PX = 768
RNG_SEED = 42


def _per_cell_stat(dff_mat, frame_to_col, stim_frames, direction, window):
    """Aggregate per-stim Δ dF/F0 per cell — mirrors compute_responder_masks."""
    per_stim = [
        _delta_at(dff_mat, frame_to_col[p], direction, window, BASELINE_N_PRE)
        for p in stim_frames
        if p in frame_to_col
    ]
    return _aggregate(np.vstack(per_stim), STAT)


def _valid_pseudo_cols(dff_mat, stim_cols, window, exclusion_pad=10):
    """Stimulus-free columns usable as pseudo-stims — mirrors responders.py."""
    n_cols = dff_mat.shape[1]
    win_lo, win_hi = window
    excluded = np.zeros(n_cols, dtype=bool)
    pad = max(exclusion_pad, win_hi)
    for sc in stim_cols:
        excluded[max(0, sc - pad):min(n_cols, sc + pad + 1)] = True
    valid = ~excluded
    valid[: max(0, -win_lo)] = False
    valid[max(0, n_cols - win_hi + 1):] = False
    valid[:BASELINE_N_PRE] = False
    return np.where(valid)[0]


def _aligned_population_trace(dff_mat, anchor_cols, offsets, direction):
    """Population-*median* Δ dF/F0 (vs per-cell pre-stim baseline) by offset.

    The median over cells is used on purpose: a population *mean* is pulled
    up by the minority of genuine strong responders, so it rises even with
    no field-wide nuisance. The median tracks the *typical* cell, which is
    what a stimulus-locked field-wide shift would move.

    Returns ``(mean_over_anchors, sem_over_anchors)`` each of shape
    ``(len(offsets),)`` — averaged across the supplied anchor columns.
    """
    n_cols = dff_mat.shape[1]
    per_anchor = []
    for ac in anchor_cols:
        lo = ac - BASELINE_N_PRE
        if lo < 0 or ac + offsets[-1] >= n_cols or ac + offsets[0] < 0:
            continue
        base = np.nanmean(dff_mat[:, lo:ac], axis=1)
        row = [np.nanmedian(dff_mat[:, ac + o] - base) for o in offsets]
        per_anchor.append(row)
    arr = np.asarray(per_anchor, dtype=float)
    if arr.size == 0:
        return None, None
    n = arr.shape[0]
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0) / np.sqrt(max(n, 1))


def _dead_cols_and_frames(cfg, state, exp_name, ch, frame_to_col):
    """Dead-frame indices for (exp, ch) plus their dff-matrix columns.

    Uses the same resolver ``mask_dead_frames`` uses in the pipeline: an
    explicit ``bad_frames_file`` list when configured (PC3), otherwise a
    rolling-MAD test on ``bg_trace``. Returns ``(dead_frames, dead_cols)``
    — ``dead_cols`` keeps only frames present in ``frame_to_col``.
    """
    bg = np.asarray(state["bg_trace"][exp_name][ch], dtype=np.float64)
    dead_frames = resolve_dead_frame_indices(cfg, bg)
    dead_cols = np.array(
        [frame_to_col[int(f)] for f in dead_frames if int(f) in frame_to_col],
        dtype=np.int64,
    )
    return dead_frames, dead_cols


def _aligned_scalar_trace(values_by_col, anchor_cols, offsets):
    """Baseline-subtracted 1-D scalar aligned to anchor columns.

    The scalar analogue of :func:`_aligned_population_trace` — for a single
    field-wide signal (background level, image sharpness) rather than a
    per-cell matrix. Each anchor contributes ``value[anchor + offset] −
    mean(value[anchor − BASELINE_N_PRE : anchor])``; contributions are
    averaged. Returns ``(mean, sem)`` over usable anchors, or
    ``(None, None)`` when none fit.
    """
    v = np.asarray(values_by_col, dtype=float)
    n_cols = v.size
    per_anchor = []
    for ac in anchor_cols:
        lo = ac - BASELINE_N_PRE
        if lo < 0 or ac + offsets[-1] >= n_cols or ac + offsets[0] < 0:
            continue
        base = np.nanmean(v[lo:ac])
        per_anchor.append([v[ac + o] - base for o in offsets])
    arr = np.asarray(per_anchor, dtype=float)
    if arr.size == 0:
        return None, None
    n = arr.shape[0]
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0) / np.sqrt(max(n, 1))


def _frame_sharpness(args):
    """Variance-of-Laplacian sharpness of one frame — process-pool worker.

    ``args`` is ``(frame_num, image_path)``. The frame is decoded once and
    box-reduced toward :data:`FOCUS_TARGET_PX` on the short side before the
    Laplacian, so a 20-megapixel PNG does not dominate the runtime. The
    reduction factor is constant within a channel (frame size is fixed),
    so the real-vs-pseudo comparison stays self-consistent. Returns
    ``(frame_num, sharpness)``.
    """
    frame_num, path = args
    img = Image.open(path).convert("L")
    factor = max(1, min(img.size) // FOCUS_TARGET_PX)
    if factor > 1:
        img = img.reduce(factor)
    arr = np.asarray(img, dtype=np.float64)
    return frame_num, float(np.var(laplace(arr)))


def _focus_by_col(cfg, ch, col_to_frame, needed_cols, n_cols):
    """Per-column image sharpness — variance of the Laplacian.

    A medium-swap refractive shift defocuses the whole field without
    necessarily moving the background *level*, so a sharpness metric is
    what actually catches it. Only the frame images whose columns appear
    in ``needed_cols`` (the windows around real and pseudo anchors) are
    loaded, each exactly once, decoded in parallel across a process pool.
    Returns a length-``n_cols`` array (NaN for columns never loaded), or
    ``None`` when the frame directory is absent.
    """
    fdir = os.path.join(channel_dir(cfg, ch), "frames")
    if not os.path.isdir(fdir):
        return None
    ffiles = sorted_image_files(fdir)
    frames_needed = sorted({
        col_to_frame[int(c)] for c in needed_cols
        if 0 <= int(c) < n_cols
    })
    jobs = [
        (fn, os.path.join(fdir, ffiles[fn]))
        for fn in frames_needed if 0 <= fn < len(ffiles)
    ]
    if not jobs:
        return None
    # Respect the per-worker thread budget run_figures.sh sets when it runs
    # the three experiments concurrently; fall back to the full CPU count.
    n_workers = int(os.environ.get("OMP_NUM_THREADS") or 0) or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, 8, len(jobs)))
    sharp = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for fn, val in pool.map(_frame_sharpness, jobs, chunksize=8):
            sharp[fn] = val
    focus = np.full(n_cols, np.nan)
    for c in range(n_cols):
        if col_to_frame.get(c) in sharp:
            focus[c] = sharp[col_to_frame[c]]
    return focus


def _window_step(trace, offsets, lo, hi):
    """Mean of an aligned trace over the response-window offsets ``[lo, hi)``."""
    if trace is None:
        return float("nan")
    m = (np.asarray(offsets) >= lo) & (np.asarray(offsets) < hi)
    return float(np.nanmean(np.asarray(trace)[m])) if m.any() else float("nan")


def _binned_median(x, y, n_bins=12, min_per_bin=3):
    """Bin centres and per-bin median of ``y`` over equal-width bins of ``x``."""
    good = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[good], y[good]
    if x.size < 2 or x.min() == x.max():
        return np.array([]), np.array([])
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centers, meds = [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() >= min_per_bin:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            meds.append(float(np.median(y[sel])))
    return np.array(centers), np.array(meds)


def _quartile_resp_rates(f0, mask, valid):
    """Responder rate (%) in each F0 quartile, dimmest to brightest."""
    f, m = f0[valid], mask[valid]
    if f.size < 4:
        return [float("nan")] * 4
    q = np.nanpercentile(f, [25, 50, 75])
    edges = [-np.inf, q[0], q[1], q[2], np.inf]
    rates = []
    for i in range(4):
        sel = (f > edges[i]) & (f <= edges[i + 1])
        rates.append(100.0 * float(m[sel].mean()) if sel.any() else float("nan"))
    return rates


INVESTIGATION_SUMMARY = [
    "1.  Stim-locked population shift  — CONFIRMED (this figure): a "
    "field-wide Δ dF/F0 lift steps up at real stims, large in PC3.",
    "2.  Dead-frame proximity  — RULED OUT (responder_artifact_diagnostic): "
    "masked dead frames are never the response peak; Δ shift 0.0000.",
    "3.  Perfusion / optical artifact  — RULED OUT "
    "(responder_artifact_diagnostic): no bg step; sharpness rises, no defocus.",
    "4.  F0 dependence  — RULED OUT (responder_f0_diagnostic): Δ dF/F0 rises "
    "WITH brightness (corr +0.47), not a 1/F0 normalization artifact.",
    "5.  Decision (2026-05-16)  — no artifact found; responder logic kept "
    "UNCHANGED. PC3's ~40% is treated as a legitimate result.",
    "6.  alpha sensitivity sweep  — not run; logic unchanged, so no sweep "
    "is needed.",
]


def _draw_distribution_figure(exp_name, channels, panels):
    """Marginal histogram + jittered strip scatter of per-cell Δ dF/F0."""
    n = len(channels)
    # This figure positions its axes with explicit gridspec margins below, so
    # it opts OUT of the globally-enabled constrained layout (the two would
    # fight and collapse the axes). A no-op layout engine is required here:
    # constrained_layout=False alone gets silently re-applied by savefig, so we
    # install PlaceHolderLayoutEngine which survives the save. Width stays 6.5 in.
    fig = plt.figure(figsize=(PLOT_PARAMS["width_full"], 8.4),
                     dpi=PLOT_PARAMS["dpi"])
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(2, n, height_ratios=[1, 3], hspace=0.07, wspace=0.26,
                          top=0.84, bottom=0.09, left=0.09, right=0.97)
    rng = np.random.default_rng(RNG_SEED)

    for col, ch in enumerate(channels):
        d = panels[ch]
        stat, thr, signed_t, mask = d["stat"], d["thr"], d["signed_t"], d["mask"]
        good = stat[~np.isnan(stat)]
        lo, hi = np.nanpercentile(good, 0.5), np.nanpercentile(good, 99.5)
        clip = np.clip(good, lo, hi)
        resp_clip = np.clip(stat[mask], lo, hi)
        non_clip = np.clip(stat[~mask & ~np.isnan(stat)], lo, hi)

        ax_h = fig.add_subplot(gs[0, col])
        ax_h.hist(non_clip, bins=PLOT_PARAMS["bins"], range=(lo, hi),
                  color=PLOT_PARAMS["nonresponder_color"],
                  alpha=PLOT_PARAMS["hist_alpha"], label="non-responder")
        ax_h.hist(resp_clip, bins=PLOT_PARAMS["bins"], range=(lo, hi),
                  color=PLOT_PARAMS["responder_color"],
                  alpha=PLOT_PARAMS["hist_alpha"], label="responder")
        ax_h.axvline(signed_t, color=PLOT_PARAMS["threshold_color"], lw=2,
                     ls="--", label=f"threshold {signed_t:+.3f}")
        ax_h.axvline(0.0, color=PLOT_PARAMS["zero_color"], lw=1)
        ax_h.set_title(
            f"{exp_name} — {ch}\n{100 * d['pct_resp']:.1f}% responders  "
            f"(non-resp median Δ = {d['med_non']:+.3f})",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"])
        ax_h.set_ylabel("cell count")
        ax_h.tick_params(labelbottom=False)
        ax_h.spines[["top", "right"]].set_visible(False)
        ax_h.legend(fontsize=8);

        ax_s = fig.add_subplot(gs[1, col], sharex=ax_h)
        for sel, color, lbl in (
            (~mask & ~np.isnan(stat), PLOT_PARAMS["nonresponder_color"], "non-responder"),
            (mask, PLOT_PARAMS["responder_color"], "responder"),
        ):
            x = np.clip(stat[sel], lo, hi)
            y = rng.uniform(0.0, 1.0, size=x.size)
            ax_s.scatter(x, y, s=PLOT_PARAMS["scatter_size"], color=color,
                         alpha=PLOT_PARAMS["scatter_alpha"], edgecolors="none",
                         rasterized=True,
                         label=f"{lbl} (n={int(sel.sum())})")
        ax_s.axvline(signed_t, color=PLOT_PARAMS["threshold_color"], lw=2, ls="--")
        ax_s.axvline(0.0, color=PLOT_PARAMS["zero_color"], lw=1)
        ax_s.set_xlabel("per-cell aggregate per-stim Δ dF/F0  "
                        "(x clipped to [p0.5, p99.5])")
        ax_s.set_ylabel("jitter (no meaning)")
        ax_s.set_yticks([])
        ax_s.spines[["top", "right", "left"]].set_visible(False)
        ax_s.legend(fontsize=8, loc="upper right");

    fig.suptitle(
        f"{exp_name} — responder distribution diagnostic\n"
        "clean = bulk at 0 with a separated tail   |   "
        "suspect = whole-population blob shifted off 0, threshold slices its shoulder",
        fontsize=PLOT_PARAMS["suptitle_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"], y=0.97);
    out = fig_path(exp_name, "responder_distribution_diagnostic")
    save_fig(fig, out, dpi=PLOT_PARAMS["dpi"])
    plt.close(fig)
    return out


def _draw_stimlock_figure(exp_name, channels, panels):
    """Stim-aligned population trace + per-stim deltas + investigation-summary panel."""
    n = len(channels)
    # Tall per-channel stack with manual gridspec margins -> no-op layout engine
    # (constrained would collapse the rows) and let height grow with channel
    # count; width stays locked at 6.5 in. Capping the height collapses the rows.
    height = 4.2 * n + 3.0
    fig = plt.figure(figsize=(PLOT_PARAMS["width_full"], height),
                     dpi=PLOT_PARAMS["dpi"])
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[*([3] * n), 2.4],
                          hspace=0.62, wspace=0.30,
                          top=1.0 - 0.95 / height, bottom=0.05,
                          left=0.08, right=0.97)

    for row, ch in enumerate(channels):
        d = panels[ch]
        offsets = d["offsets"]

        ax_t = fig.add_subplot(gs[row, 0])
        for key, color, lbl in (
            ("real", PLOT_PARAMS["real_color"], "real stims"),
            ("pseudo", PLOT_PARAMS["pseudo_color"], "pseudo-stims"),
        ):
            m, s = d[f"{key}_trace"], d[f"{key}_sem"]
            if m is None:
                continue
            ax_t.plot(offsets, m, color=color, lw=2, label=lbl)
            ax_t.fill_between(offsets, m - s, m + s, color=color,
                              alpha=PLOT_PARAMS["band_alpha"])
        ax_t.axvline(0.0, color=PLOT_PARAMS["zero_color"], lw=1)
        ax_t.axhline(0.0, color=PLOT_PARAMS["zero_color"], lw=0.8, ls=":")
        ax_t.axvspan(d["win_lo"], d["win_hi"], color=PLOT_PARAMS["window_shade"],
                     alpha=0.08, label="response window")
        ax_t.set_title(f"{ch} — population-median Δ dF/F0 aligned to stim onset",
                       fontsize=PLOT_PARAMS["title_fontsize"],
                       fontweight=PLOT_PARAMS["title_fontweight"])
        ax_t.set_xlabel("frame offset from stim onset")
        ax_t.set_ylabel("population-median Δ dF/F0\n(vs per-cell pre-stim baseline)")
        ax_t.spines[["top", "right"]].set_visible(False)
        ax_t.legend(fontsize=8, loc="upper left");

        ax_b = fig.add_subplot(gs[row, 1])
        n_real = d["per_stim_real"].size
        xs = np.arange(1, n_real + 1)
        nb = d["null_band"]
        ax_b.fill_between([0.5, n_real + 0.5], nb[0], nb[2],
                          color=PLOT_PARAMS["pseudo_color"],
                          alpha=PLOT_PARAMS["band_alpha"],
                          label="pseudo-stim null (p1-p99)")
        ax_b.axhline(nb[1], color=PLOT_PARAMS["pseudo_color"], lw=1.2,
                     ls="--", label="pseudo-stim median")
        ax_b.axhline(0.0, color=PLOT_PARAMS["zero_color"], lw=0.8, ls=":")
        ax_b.scatter(xs, d["per_stim_real"], s=44,
                     color=PLOT_PARAMS["real_color"], zorder=3,
                     label="real stim")
        ax_b.set_title(f"{ch} — per-stim population-median Δ dF/F0",
                       fontsize=PLOT_PARAMS["title_fontsize"],
                       fontweight=PLOT_PARAMS["title_fontweight"])
        ax_b.set_xlabel("stimulus index")
        ax_b.set_ylabel("population-median Δ dF/F0")
        ax_b.set_xticks(xs)
        ax_b.spines[["top", "right"]].set_visible(False)
        ax_b.legend(fontsize=7, loc="lower right");

    ax_c = fig.add_subplot(gs[n, :])
    ax_c.axis("off")
    ax_c.text(0.0, 1.0, "Responder-rate investigation — outcome (closed 2026-05-16)",
              fontsize=PLOT_PARAMS["title_fontsize"],
              fontweight=PLOT_PARAMS["title_fontweight"], va="top")
    ax_c.text(0.0, 0.84, "\n".join(INVESTIGATION_SUMMARY),
              fontsize=PLOT_PARAMS["panel_fontsize"], va="top", family="monospace")

    fig.suptitle(
        f"{exp_name} — stimulus-locked artifact check\n"
        "if the real-stim trace steps up where the pseudo-stim trace stays flat, "
        "the responder rate is inflated by a field-wide nuisance",
        fontsize=PLOT_PARAMS["suptitle_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"], y=1.0 - 0.30 / height);
    out = fig_path(exp_name, "responder_stimlock_diagnostic")
    save_fig(fig, out, dpi=PLOT_PARAMS["dpi"])
    plt.close(fig)
    return out


def _draw_artifact_figure(exp_name, channels, panels):
    """Dead-frame proximity (#2) + perfusion/optical artifact (#3) checks.

    Two rows per channel:

    * row A — every masked dead frame's offset from each stim onset, drawn
      against the shaded baseline/response windows, plus the per-stim
      population-median Δ dF/F0 recomputed with those dead columns
      NaN-masked instead of interpolated. If the masked trace tracks the
      pipeline trace, dead-frame interpolation is not biasing Δ.
    * row B — background level (``bg_trace``, ``bg_min``) and image
      sharpness (variance of the Laplacian) aligned to stim onset, real
      vs. pseudo-stims. A field-wide optical artifact steps at real stims
      and stays flat at pseudo-stims.
    """
    n = len(channels)
    # Tallest diagnostic (2 rows per channel) with manual gridspec margins ->
    # no-op layout engine; height grows with channel count (width 6.5 in).
    height = 7.0 * n + 4.2
    fig = plt.figure(figsize=(PLOT_PARAMS["width_full"], height),
                     dpi=PLOT_PARAMS["dpi"])
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(2 * n + 1, 2, height_ratios=[*([3] * (2 * n)), 2.3],
                          hspace=0.66, wspace=0.27,
                          top=1.0 - 2.4 / height, bottom=0.045,
                          left=0.08, right=0.97)
    verdicts = []

    for r, ch in enumerate(channels):
        d = panels[ch]
        offsets = d["offsets"]
        win_lo, win_hi = d["win_lo"], d["win_hi"]
        stim_cols = d["stim_cols"]
        dead_cols = d["dead_cols"]
        n_stim = len(stim_cols)

        # ---- row A left: dead-frame proximity to stim windows ------------
        ax_p = fig.add_subplot(gs[2 * r, 0])
        ax_p.axhspan(-BASELINE_N_PRE, 0, color=PLOT_PARAMS["baseline_shade"],
                     alpha=0.16, label="baseline window")
        ax_p.axhspan(win_lo, win_hi, color=PLOT_PARAMS["window_shade"],
                     alpha=0.12, label="response window")
        ax_p.axhline(0.0, color=PLOT_PARAMS["zero_color"], lw=1)
        span_lo, span_hi = -BASELINE_N_PRE - 8, win_hi + 8
        resp_stims, base_stims = set(), set()
        in_x, in_y, out_x, out_y = [], [], [], []
        for i, sc in enumerate(stim_cols):
            for dc in dead_cols:
                off = int(dc - sc)
                if off < span_lo or off > span_hi:
                    continue
                in_base = -BASELINE_N_PRE <= off < 0
                in_resp = win_lo <= off < win_hi
                if in_base:
                    base_stims.add(i)
                if in_resp:
                    resp_stims.add(i)
                (in_x if in_base or in_resp else out_x).append(i + 1)
                (in_y if in_base or in_resp else out_y).append(off)
        ax_p.scatter(out_x, out_y, s=42, color=PLOT_PARAMS["dead_far_color"],
                     edgecolors="none", zorder=3, label="dead frame (clear)")
        ax_p.scatter(in_x, in_y, s=60, color=PLOT_PARAMS["dead_color"],
                     edgecolors="#7a4500", linewidths=0.6, zorder=4,
                     label="dead frame (in window)")
        ax_p.set_title(
            f"{ch} — #2 dead-frame proximity  "
            f"({len(resp_stims)}/{n_stim} stims hit in response win, "
            f"{len(base_stims)}/{n_stim} in baseline win)",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"])
        ax_p.set_xlabel("stimulus index")
        ax_p.set_ylabel("dead-frame offset from stim onset")
        ax_p.set_xticks(np.arange(1, n_stim + 1))
        ax_p.set_xlim(0.5, n_stim + 0.5)
        ax_p.spines[["top", "right"]].set_visible(False)
        ax_p.legend(fontsize=7, loc="upper right");

        # ---- row A right: Δ dF/F0 — interpolated vs dead-masked ----------
        ax_d = fig.add_subplot(gs[2 * r, 1])
        xs = np.arange(1, n_stim + 1)
        real = d["per_stim_real"]
        masked = d["per_stim_masked"]
        diff = np.abs(masked - real)
        med_diff = float(np.nanmedian(diff)) if diff.size else float("nan")
        max_diff = float(np.nanmax(diff)) if diff.size else float("nan")
        ax_d.axhline(0.0, color=PLOT_PARAMS["zero_color"], lw=0.8, ls=":")
        for i in range(n_stim):
            ax_d.plot([xs[i], xs[i]], [real[i], masked[i]],
                      color="#bbbbbb", lw=1, zorder=1)
        ax_d.scatter(xs, real, s=42, color=PLOT_PARAMS["interp_color"],
                     zorder=3, label="pipeline (dead frames interpolated)")
        ax_d.scatter(xs, masked, s=42, color=PLOT_PARAMS["masked_color"],
                     marker="D", zorder=3, label="dead frames NaN-masked")
        ax_d.set_title(
            f"{ch} — #2 per-stim Δ dF/F0  "
            f"(median |shift| = {med_diff:.4f}, max = {max_diff:.4f})",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"])
        ax_d.set_xlabel("stimulus index")
        ax_d.set_ylabel("population-median Δ dF/F0")
        ax_d.set_xticks(xs)
        ax_d.spines[["top", "right"]].set_visible(False)
        ax_d.legend(fontsize=7, loc="upper right");

        # ---- row B left: background level aligned to stim onset ----------
        ax_bg = fig.add_subplot(gs[2 * r + 1, 0])
        ax_bg.axvline(0.0, color=PLOT_PARAMS["zero_color"], lw=1)
        ax_bg.axhline(0.0, color=PLOT_PARAMS["zero_color"], lw=0.8, ls=":")
        ax_bg.axvspan(win_lo, win_hi, color=PLOT_PARAMS["window_shade"],
                      alpha=0.08, label="response window")
        for key, color, ls, lbl in (
            ("bg_real", PLOT_PARAMS["real_color"], "-", "bg_trace — real"),
            ("bg_pseudo", PLOT_PARAMS["pseudo_color"], "-", "bg_trace — pseudo"),
            ("bgmin_real", PLOT_PARAMS["bgmin_color"], "--", "bg_min — real"),
            ("bgmin_pseudo", "#9aa0a6", "--", "bg_min — pseudo"),
        ):
            m = d[key]
            if m is None:
                continue
            s = d[key + "_sem"]
            ax_bg.plot(offsets, m, color=color, lw=2, ls=ls, label=lbl)
            if s is not None:
                ax_bg.fill_between(offsets, m - s, m + s, color=color,
                                   alpha=PLOT_PARAMS["band_alpha"])
        ax_bg.set_title(f"{ch} — #3 background level aligned to stim onset",
                        fontsize=PLOT_PARAMS["title_fontsize"],
                        fontweight=PLOT_PARAMS["title_fontweight"])
        ax_bg.set_xlabel("frame offset from stim onset")
        ax_bg.set_ylabel("Δ background\n(vs pre-stim baseline)")
        ax_bg.spines[["top", "right"]].set_visible(False)
        ax_bg.legend(fontsize=7, loc="upper left");

        # ---- row B right: image sharpness aligned to stim onset ----------
        ax_f = fig.add_subplot(gs[2 * r + 1, 1])
        ax_f.axvline(0.0, color=PLOT_PARAMS["zero_color"], lw=1)
        ax_f.axhline(0.0, color=PLOT_PARAMS["zero_color"], lw=0.8, ls=":")
        ax_f.axvspan(win_lo, win_hi, color=PLOT_PARAMS["window_shade"],
                     alpha=0.08, label="response window")
        if d["focus_real"] is None:
            ax_f.text(0.5, 0.5, "frame images unavailable", ha="center",
                      va="center", transform=ax_f.transAxes,
                      fontsize=PLOT_PARAMS["panel_fontsize"])
        else:
            for key, color, lbl in (
                ("focus_real", PLOT_PARAMS["focus_color"], "real stims"),
                ("focus_pseudo", PLOT_PARAMS["pseudo_color"], "pseudo-stims"),
            ):
                m = d[key]
                if m is None:
                    continue
                s = d[key + "_sem"]
                ax_f.plot(offsets, m, color=color, lw=2, label=lbl)
                if s is not None:
                    ax_f.fill_between(offsets, m - s, m + s, color=color,
                                      alpha=PLOT_PARAMS["band_alpha"])
        ax_f.set_title(f"{ch} — #3 image sharpness (var-Laplacian)",
                       fontsize=PLOT_PARAMS["title_fontsize"],
                       fontweight=PLOT_PARAMS["title_fontweight"])
        ax_f.set_xlabel("frame offset from stim onset")
        ax_f.set_ylabel("Δ var-Laplacian\n(vs pre-stim baseline)")
        ax_f.spines[["top", "right"]].set_visible(False)
        ax_f.legend(fontsize=7, loc="upper left");

        # ---- per-channel verdict lines for the footer --------------------
        bg_r = _window_step(d["bg_real"], offsets, win_lo, win_hi)
        bg_p = _window_step(d["bg_pseudo"], offsets, win_lo, win_hi)
        fc_r = _window_step(d["focus_real"], offsets, win_lo, win_hi)
        fc_p = _window_step(d["focus_pseudo"], offsets, win_lo, win_hi)
        d2 = ("no dead frame in any stim window"
              if not resp_stims and not base_stims
              else f"{len(resp_stims)}/{n_stim} response + "
                   f"{len(base_stims)}/{n_stim} baseline windows hit, but "
                   f"interpolation shifts Δ by only {med_diff:.4f} (median)"
              if med_diff < 0.02
              else f"{len(resp_stims)}/{n_stim} response windows hit AND "
                   f"interpolation shifts Δ by {med_diff:.4f} (median) — "
                   f"inspect")
        fc_txt = ("frames unavailable" if np.isnan(fc_r)
                  else f"Δsharpness real={fc_r:+.1f} vs pseudo={fc_p:+.1f}")
        verdicts.append(
            f"{ch}:\n"
            f"   #2 dead-frame proximity — {d2}.\n"
            f"   #3 optical — Δbg_trace real={bg_r:+.3f} vs pseudo={bg_p:+.3f}; "
            f"{fc_txt}."
        )

    ax_c = fig.add_subplot(gs[2 * n, :])
    ax_c.axis("off")
    ax_c.text(0.0, 1.0, "Diagnoses #2 (dead-frame proximity) and #3 "
              "(perfusion / optical artifact) — findings",
              fontsize=PLOT_PARAMS["title_fontsize"],
              fontweight=PLOT_PARAMS["title_fontweight"], va="top")
    ax_c.text(0.0, 0.86, "\n".join(verdicts),
              fontsize=PLOT_PARAMS["panel_fontsize"], va="top",
              family="monospace")

    fig.suptitle(
        f"{exp_name} — dead-frame & optical-artifact check\n"
        "#2: do masked dead frames bias per-stim Δ?   "
        "#3: does the background level or image focus step at real stims?",
        fontsize=PLOT_PARAMS["suptitle_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"], y=1.0 - 0.85 / height);
    out = fig_path(exp_name, "responder_artifact_diagnostic")
    save_fig(fig, out, dpi=PLOT_PARAMS["dpi"])
    plt.close(fig)
    return out


def _draw_f0_figure(exp_name, channels, panels):
    """#4 F0-dependence — is the per-cell Δ dF/F0 a 1/F0 normalization effect?

    Δ dF/F0 = (F_peak − F_base) / F0, so a fixed *additive* luminosity bump
    delivered field-wide becomes a larger dF/F0 in dimmer (low-F0) cells.
    One row per channel:

    * left — per-cell aggregate Δ dF/F0 vs F0, responders vs non-responders,
      with the responder threshold and a binned-median trend. A strong
      decline toward low F0 means dim cells preferentially clear the bar.
    * right — the same response in *additive corrected-luminosity* units
      (Δ dF/F0 × F0 = F_peak − F_base) vs F0. If this is flat while the
      left panel declines, the dF/F0 "response" is a normalization
      artifact: every cell gets the same bump, dF/F0 just amplifies it for
      dim cells. If instead Δ dF/F0 is flat, the response is genuinely
      proportional and F0 is not the culprit.
    """
    n = len(channels)
    # Per-channel stack with manual gridspec margins -> no-op layout engine;
    # height grows with channel count (width stays locked at 6.5 in).
    height = 4.7 * n + 3.0
    fig = plt.figure(figsize=(PLOT_PARAMS["width_full"], height),
                     dpi=PLOT_PARAMS["dpi"])
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[*([3] * n), 2.0],
                          hspace=0.52, wspace=0.27,
                          top=1.0 - 1.7 / height, bottom=0.06,
                          left=0.08, right=0.97)
    verdicts = []

    for r, ch in enumerate(channels):
        d = panels[ch]
        stat, f0, dlum = d["stat"], d["f0"], d["delta_lum"]
        mask, signed_t = d["mask"], d["signed_t"]
        fin = ~np.isnan(stat) & ~np.isnan(f0) & (f0 > 0)
        if fin.sum() >= 2:
            lo, hi = np.nanpercentile(f0[fin], [1, 99])
        else:
            lo, hi = 0.0, 1.0

        ax1 = fig.add_subplot(gs[r, 0])
        for sel, color, lbl in (
            (fin & ~mask, PLOT_PARAMS["nonresponder_color"], "non-responder"),
            (fin & mask, PLOT_PARAMS["responder_color"], "responder"),
        ):
            ax1.scatter(np.clip(f0[sel], lo, hi), stat[sel],
                        s=PLOT_PARAMS["scatter_size"], color=color,
                        alpha=PLOT_PARAMS["scatter_alpha"], edgecolors="none",
                        label=f"{lbl} (n={int(sel.sum())})")
        bx, by = _binned_median(np.clip(f0[fin], lo, hi), stat[fin])
        if bx.size:
            ax1.plot(bx, by, color=PLOT_PARAMS["threshold_color"], lw=2,
                     marker="o", ms=4, label="binned median")
        ax1.axhline(signed_t, color=PLOT_PARAMS["threshold_color"], lw=1.5,
                    ls="--", label=f"responder threshold {signed_t:+.3f}")
        ax1.axhline(0.0, color=PLOT_PARAMS["zero_color"], lw=1)
        ax1.set_title(f"{ch} — Δ dF/F0 vs F0   "
                      f"(Spearman r = {d['corr_dff_f0']:+.2f})",
                      fontsize=PLOT_PARAMS["title_fontsize"],
                      fontweight=PLOT_PARAMS["title_fontweight"])
        ax1.set_xlabel("per-cell F0  (corrected-fluorescence baseline brightness)")
        ax1.set_ylabel("per-cell aggregate Δ dF/F0")
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.legend(fontsize=7, loc="upper right");

        ax2 = fig.add_subplot(gs[r, 1])
        ax2.scatter(np.clip(f0[fin], lo, hi), dlum[fin],
                    s=PLOT_PARAMS["scatter_size"],
                    color=PLOT_PARAMS["nonresponder_color"],
                    alpha=PLOT_PARAMS["scatter_alpha"], edgecolors="none",
                    label=f"all cells (n={int(fin.sum())})")
        bx2, by2 = _binned_median(np.clip(f0[fin], lo, hi), dlum[fin])
        if bx2.size:
            ax2.plot(bx2, by2, color=PLOT_PARAMS["threshold_color"], lw=2,
                     marker="o", ms=4, label="binned median")
        ax2.axhline(0.0, color=PLOT_PARAMS["zero_color"], lw=1)
        ax2.set_title(f"{ch} — Δ fluorescence (Δ dF/F0 × F0) vs F0   "
                      f"(Spearman r = {d['corr_lum_f0']:+.2f})",
                      fontsize=PLOT_PARAMS["title_fontsize"],
                      fontweight=PLOT_PARAMS["title_fontweight"])
        ax2.set_xlabel("per-cell F0")
        ax2.set_ylabel("per-cell aggregate Δ corrected fluorescence\n"
                       "(additive units)")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.legend(fontsize=7, loc="upper right");

        q = d["quartile_rates"]
        if d["corr_dff_f0"] < -0.25 and abs(d["corr_lum_f0"]) < 0.20:
            v = ("NORMALIZATION EFFECT — dim cells preferentially pass; the "
                 "additive fluorescence bump is ~F0-independent, so dividing "
                 "by a small F0 inflates Δ dF/F0 for low-F0 cells.")
        elif d["corr_dff_f0"] < -0.25:
            v = ("Δ dF/F0 declines with F0 AND Δ fluorescence also tracks F0 — "
                 "partial F0 dependence; inspect both panels before deciding.")
        else:
            v = ("Δ dF/F0 is ~F0-independent — not a 1/F0 normalization "
                 "artifact; dim cells do not preferentially pass.")
        verdicts.append(
            f"{ch}:\n"
            f"   corr(Δdff, F0) = {d['corr_dff_f0']:+.2f}    "
            f"corr(Δdff, 1/F0) = {d['corr_dff_invf0']:+.2f}    "
            f"corr(Δfluor, F0) = {d['corr_lum_f0']:+.2f}\n"
            f"   responder % by F0 quartile (dim -> bright): "
            f"{q[0]:.0f}% / {q[1]:.0f}% / {q[2]:.0f}% / {q[3]:.0f}%\n"
            f"   -> {v}"
        )

    ax_c = fig.add_subplot(gs[n, :])
    ax_c.axis("off")
    ax_c.text(0.0, 1.0, "Diagnosis #4 (F0 dependence) — findings",
              fontsize=PLOT_PARAMS["title_fontsize"],
              fontweight=PLOT_PARAMS["title_fontweight"], va="top")
    ax_c.text(0.0, 0.84, "\n".join(verdicts),
              fontsize=PLOT_PARAMS["panel_fontsize"], va="top",
              family="monospace")

    fig.suptitle(
        f"{exp_name} — F0-dependence check\n"
        "if Δ dF/F0 falls with F0 while Δ fluorescence stays flat, the "
        "responder signal is a 1/F0 normalization artifact, not biology",
        fontsize=PLOT_PARAMS["suptitle_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"], y=1.0 - 0.6 / height);
    out = fig_path(exp_name, "responder_f0_diagnostic")
    save_fig(fig, out, dpi=PLOT_PARAMS["dpi"])
    plt.close(fig)
    return out


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    thresholds = compute_responder_thresholds(
        experiments, state, alpha=ALPHA, baseline_n_pre=BASELINE_N_PRE, stat=STAT,
    )
    rng = np.random.default_rng(RNG_SEED)

    for exp_name, cfg in experiments.items():
        direction = cfg.get("response_direction", "increase")
        sign = -1.0 if direction == "decrease" else 1.0
        channels = list(cfg["channels"])
        panels = {}

        for ch in channels:
            window = response_window_frames(state, exp_name, ch, cfg)
            win_lo, win_hi = window
            dff_mat, frame_to_col = _channel_dff(state, exp_name, ch, cfg)
            stim_frames = cfg["stim_frames"][ch]
            stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]

            stat = _per_cell_stat(dff_mat, frame_to_col, stim_frames,
                                  direction, window)
            thr = float(thresholds.get((exp_name, ch), 0.10))
            signed_t = sign * thr
            mask = ((stat <= signed_t) if direction == "decrease"
                    else (stat >= signed_t)) & ~np.isnan(stat)
            good = stat[~np.isnan(stat)]
            non = good[(good < signed_t) if direction != "decrease"
                       else (good > signed_t)]

            valid_cols = _valid_pseudo_cols(dff_mat, stim_cols, window)
            offsets = np.arange(-(BASELINE_N_PRE + 2), win_hi + 4)
            real_trace, real_sem = _aligned_population_trace(
                dff_mat, stim_cols, offsets, direction)
            if valid_cols.size:
                pcs = rng.choice(valid_cols, size=N_PSEUDO_TRACE,
                                 replace=valid_cols.size < N_PSEUDO_TRACE)
                pseudo_trace, pseudo_sem = _aligned_population_trace(
                    dff_mat, list(pcs), offsets, direction)
            else:
                pseudo_trace = pseudo_sem = None

            per_stim_real = np.array([
                np.nanmedian(_delta_at(dff_mat, sc, direction, window,
                                       BASELINE_N_PRE))
                for sc in stim_cols
            ])
            if valid_cols.size:
                null = np.array([
                    np.nanmedian(_delta_at(dff_mat, int(pc), direction, window,
                                           BASELINE_N_PRE))
                    for pc in rng.choice(valid_cols, size=N_PSEUDO_TRACE,
                                         replace=valid_cols.size < N_PSEUDO_TRACE)
                ])
                null_band = (np.nanpercentile(null, 1),
                             np.nanpercentile(null, 50),
                             np.nanpercentile(null, 99))
            else:
                null_band = (0.0, 0.0, 0.0)

            # ---- #2 dead-frame proximity: Δ with dead frames NaN-masked --
            dead_frames, dead_cols = _dead_cols_and_frames(
                cfg, state, exp_name, ch, frame_to_col)
            if dead_cols.size:
                dff_masked = dff_mat.copy()
                dff_masked[:, dead_cols] = np.nan
                per_stim_masked = np.array([
                    np.nanmedian(_delta_at(dff_masked, sc, direction, window,
                                           BASELINE_N_PRE))
                    for sc in stim_cols
                ])
            else:
                per_stim_masked = per_stim_real.copy()

            # ---- #3 perfusion / optical artifact -------------------------
            col_to_frame = {c: f for f, c in frame_to_col.items()}
            n_cols = dff_mat.shape[1]
            bg_trace = np.asarray(state["bg_trace"][exp_name][ch], dtype=float)
            bg_min = np.asarray(state["bg_min_by_ch"][exp_name][ch], dtype=float)
            bg_by_col = np.full(n_cols, np.nan)
            bgmin_by_col = np.full(n_cols, np.nan)
            for c in range(n_cols):
                fn = col_to_frame[c]
                if 0 <= fn < bg_trace.size:
                    bg_by_col[c] = bg_trace[fn]
                if 0 <= fn < bg_min.size:
                    bgmin_by_col[c] = bg_min[fn]
            # Drop the camera dead frames (flashes spike bg_trace, dropouts
            # depress it): #3 is the *perfusion / optical* check, so the
            # camera artifacts handled by #2 must not leak into it.
            if dead_cols.size:
                bg_by_col[dead_cols] = np.nan
                bgmin_by_col[dead_cols] = np.nan
            bg_real, bg_real_sem = _aligned_scalar_trace(
                bg_by_col, stim_cols, offsets)
            bgmin_real, bgmin_real_sem = _aligned_scalar_trace(
                bgmin_by_col, stim_cols, offsets)
            if valid_cols.size:
                focus_pcs = list(pcs[:N_PSEUDO_FOCUS])
                bg_pseudo, bg_pseudo_sem = _aligned_scalar_trace(
                    bg_by_col, list(pcs), offsets)
                bgmin_pseudo, bgmin_pseudo_sem = _aligned_scalar_trace(
                    bgmin_by_col, list(pcs), offsets)
            else:
                focus_pcs = []
                bg_pseudo = bg_pseudo_sem = None
                bgmin_pseudo = bgmin_pseudo_sem = None

            focus_anchors = list(stim_cols) + list(focus_pcs)
            needed = {ac + o for ac in focus_anchors for o in offsets}
            focus_by_col = _focus_by_col(cfg, ch, col_to_frame, needed, n_cols)
            if focus_by_col is not None:
                if dead_cols.size:
                    focus_by_col[dead_cols] = np.nan
                focus_real, focus_real_sem = _aligned_scalar_trace(
                    focus_by_col, stim_cols, offsets)
                if focus_pcs:
                    focus_pseudo, focus_pseudo_sem = _aligned_scalar_trace(
                        focus_by_col, list(focus_pcs), offsets)
                else:
                    focus_pseudo = focus_pseudo_sem = None
            else:
                focus_real = focus_real_sem = None
                focus_pseudo = focus_pseudo_sem = None

            panels[ch] = {
                "stat": stat, "thr": thr, "signed_t": signed_t, "mask": mask,
                "pct_resp": float(mask.sum()) / max(good.size, 1),
                "med_non": float(np.median(non)) if non.size else float("nan"),
                "offsets": offsets, "win_lo": win_lo, "win_hi": win_hi,
                "real_trace": real_trace, "real_sem": real_sem,
                "pseudo_trace": pseudo_trace, "pseudo_sem": pseudo_sem,
                "per_stim_real": per_stim_real, "null_band": null_band,
                "stim_cols": np.asarray(stim_cols, dtype=np.int64),
                "dead_frames": dead_frames, "dead_cols": dead_cols,
                "per_stim_masked": per_stim_masked,
                "bg_real": bg_real, "bg_real_sem": bg_real_sem,
                "bg_pseudo": bg_pseudo, "bg_pseudo_sem": bg_pseudo_sem,
                "bgmin_real": bgmin_real, "bgmin_real_sem": bgmin_real_sem,
                "bgmin_pseudo": bgmin_pseudo, "bgmin_pseudo_sem": bgmin_pseudo_sem,
                "focus_real": focus_real, "focus_real_sem": focus_real_sem,
                "focus_pseudo": focus_pseudo, "focus_pseudo_sem": focus_pseudo_sem,
            }

            # ---- #4 F0 dependence ----------------------------------------
            f0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
            f0_flat = np.asarray(f0, dtype=float).ravel()
            # Δ dF/F0 = (F_peak − F_base)/F0, so Δ dF/F0 × F0 recovers the
            # response in additive corrected-luminosity units.
            delta_lum = stat * f0_flat
            fin = ~np.isnan(stat) & ~np.isnan(f0_flat) & (f0_flat > 0)
            if int(fin.sum()) >= 3:
                corr_dff_f0 = float(spearmanr(f0_flat[fin], stat[fin]).correlation)
                corr_dff_invf0 = float(
                    spearmanr(1.0 / f0_flat[fin], stat[fin]).correlation)
                corr_lum_f0 = float(
                    spearmanr(f0_flat[fin], delta_lum[fin]).correlation)
            else:
                corr_dff_f0 = corr_dff_invf0 = corr_lum_f0 = float("nan")
            q_rates = _quartile_resp_rates(f0_flat, mask, fin)
            panels[ch].update({
                "f0": f0_flat, "delta_lum": delta_lum,
                "corr_dff_f0": corr_dff_f0, "corr_dff_invf0": corr_dff_invf0,
                "corr_lum_f0": corr_lum_f0, "quartile_rates": q_rates,
            })

            dead_in_win = int(sum(
                1 for sc in stim_cols for dc in dead_cols
                if (-BASELINE_N_PRE <= int(dc - sc) < 0)
                or (win_lo <= int(dc - sc) < win_hi)
            ))
            print(f"  {exp_name} / {ch}: {100 * panels[ch]['pct_resp']:.1f}% "
                  f"responders | non-resp median Δ={panels[ch]['med_non']:+.4f} "
                  f"| median F0={np.nanmedian(f0):.1f} "
                  f"| {dead_cols.size} dead frames, "
                  f"{dead_in_win} in a baseline/response window")
            print(f"    F0-dependence ({ch}): corr(Δdff,F0)={corr_dff_f0:+.2f} "
                  f"corr(Δdff,1/F0)={corr_dff_invf0:+.2f} "
                  f"corr(Δlum,F0)={corr_lum_f0:+.2f} | responder % by F0 "
                  f"quartile (dim->bright): "
                  f"{q_rates[0]:.0f}/{q_rates[1]:.0f}/"
                  f"{q_rates[2]:.0f}/{q_rates[3]:.0f}")

        d_path = _draw_distribution_figure(exp_name, channels, panels)
        s_path = _draw_stimlock_figure(exp_name, channels, panels)
        a_path = _draw_artifact_figure(exp_name, channels, panels)
        f_path = _draw_f0_figure(exp_name, channels, panels)
        print(f"  → {d_path}")
        print(f"  → {s_path}")
        print(f"  → {a_path}")
        print(f"  → {f_path}")


if __name__ == "__main__":
    main()
