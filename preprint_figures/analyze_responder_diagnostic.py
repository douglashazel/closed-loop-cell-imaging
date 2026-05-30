#!/usr/bin/env python3
"""Responder-call diagnostics analysis (no plotting).

Per (experiment, channel) this computes the four QC figures' intermediates —
the per-cell Δ dF/F0 distribution + responder threshold, the stim-locked
population trace, the dead-frame / optical-artifact checks, and the F0
dependence — and caches them to ``analysis_cache/<exp>/responder_diagnostic.pkl``
so the plotting layer never opens a raw image.

CRITICAL: the original ``responder_diagnostic.py`` computed image SHARPNESS by
reading frame PNGs *at plot time* (``_focus_by_col`` / variance of the
Laplacian, decoded across a ``ProcessPoolExecutor``). That frame-image compute
moves HERE: this script loads the windowed frames once, computes the focus
metric, and caches the resulting ``focus_real``/``focus_pseudo`` (+ ``_sem``)
arrays. The plotting layer (``plots/responder_diagnostic.py``) reads them back
and never touches an image.

The responder-threshold computation is the verbatim
``compute_responder_thresholds`` with its own ``alpha=0.01``,
``baseline_n_pre=5``, ``stat="mean"`` — identical to the original script.

Diagnostic only — changes nothing in the responder pipeline. NO matplotlib.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy.ndimage import laplace
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.io_paths import (
    channel_dir,
    save_analysis_cache,
    sorted_image_files,
)
from common.pipeline import prepare_state, resolve_dead_frame_indices
from common.responders import (
    _aggregate,
    _channel_dff,
    _delta_at,
    compute_responder_thresholds,
)
from common.stim_helpers import compute_f0_baseline
from common.time_axis import response_window_frames

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

# Static investigation outcome text rendered verbatim in the stimlock footer.
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
    # Decode frames in a THREAD pool, not a process pool: PIL decode and the
    # SciPy Laplacian/variance release the GIL, so threads parallelize the work
    # without fork()ing the multi-GB analysis process — which fails with
    # ENOMEM under strict memory overcommit when run_analysis.sh runs the
    # experiments concurrently. Respect the per-worker thread budget it sets;
    # fall back to the full CPU count.
    n_workers = int(os.environ.get("OMP_NUM_THREADS") or 0) or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, 8, len(jobs)))
    sharp = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for fn, val in pool.map(_frame_sharpness, jobs):
            sharp[fn] = val
    focus = np.full(n_cols, np.nan)
    for c in range(n_cols):
        if col_to_frame.get(c) in sharp:
            focus[c] = sharp[col_to_frame[c]]
    return focus


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


def _build_panels(exp_name, cfg, state, thresholds, rng):
    """Build the per-channel ``panels`` dict — verbatim from the source main().

    Returns ``(channels, panels)``. ``panels[ch]`` holds every array the four
    diagnostic figures display, including the cached ``focus_*`` arrays so the
    plotting layer never opens an image.
    """
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

        # Frame-image focus compute (moved here from plot time) — reads the
        # windowed frame PNGs once and caches the resulting focus arrays.
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

    return channels, panels


def analyze(experiments, state):
    """Compute + cache one ``responder_diagnostic.pkl`` per experiment."""
    thresholds = compute_responder_thresholds(
        experiments, state, alpha=ALPHA, baseline_n_pre=BASELINE_N_PRE, stat=STAT,
    )
    rng = np.random.default_rng(RNG_SEED)

    for exp_name, cfg in experiments.items():
        channels, panels = _build_panels(
            exp_name, cfg, state, thresholds, rng)
        data = {"channels": channels, "panels": panels}
        meta = {
            "exp_name": exp_name,
            "investigation_summary": list(INVESTIGATION_SUMMARY),
            "thresholds_params": {
                "alpha": ALPHA,
                "baseline_n_pre": BASELINE_N_PRE,
                "stat": STAT,
            },
        }
        save_analysis_cache(data, exp_name, "responder_diagnostic", meta=meta)
        print(f"  cached responder_diagnostic.pkl for {exp_name} "
              f"({len(channels)} channels)")


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    analyze(experiments, state)


if __name__ == "__main__":
    main()
