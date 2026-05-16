#!/usr/bin/env python3
"""Learning-score histograms (DMSO experiments only).

Per DMSO experiment: the summed-distribution figures for habituation and
sensitization (× height/width) plus a single anticipation figure, each
overlaid with a permutation null and a stats box reporting the FDR-corrected
per-cell hit count plus a population-level permutation test (pooled and per
biological replicate):
    * learning_habituation_height.png
    * learning_habituation_width.png
    * learning_sensitization_height.png
    * learning_sensitization_width.png
    * learning_anticipation.png

Habituation / sensitization count new running extrema across each train,
nulled by shuffling stim order. Anticipation is scored per train against a
post-train rest region: a cell's luminosity in a 5-frame window 10 min after
the train's last response peak is compared to mean ± 3·SD of that rest
region, nulled by permuting the rest region's luminosity values.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.config import LEARNING_STIMS_PER_TRAIN
from common.io_paths import fig_path
from common.permutation_null import permutation_null_distribution, pvalue_one_tailed
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.stats import bh_fdr, inferential_caveat, population_permutation_pvalue
from common.stim_helpers import compute_stim_caps, per_cell_response_delta
from common.time_axis import frames_to_min, response_window_frames

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


def _build_learning_inputs(state, exp_name, ch, cfg, *, metric):
    """Per-channel inputs needed by every learning-score function."""
    direction = cfg.get("response_direction", "increase")
    window = response_window_frames(state, exp_name, ch, cfg)
    stim_frames = cfg["stim_frames"][ch]
    df_indexed = lum_dict_to_df(
        state["corrected_lum"][exp_name][ch]
    ).set_index("CellID")
    frame_cols = sorted(
        [c for c in df_indexed.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    if not frame_cols:
        return None
    frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
    mat = df_indexed[frame_cols].values
    n_cells, n_cols = mat.shape
    frame_to_col = {f: i for i, f in enumerate(frame_nums)}
    valid_stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]
    if not valid_stim_cols:
        return None
    caps = compute_stim_caps(valid_stim_cols, n_cols)

    def f2m(frames):
        return frames_to_min(state, exp_name, ch, frames)

    per_stim_height = np.full((len(valid_stim_cols), n_cells), np.nan)
    per_stim_width = np.full((len(valid_stim_cols), n_cells), np.nan)
    for i, (sc, cap) in enumerate(zip(valid_stim_cols, caps)):
        d, w = per_cell_response_delta(
            mat, sc, direction, window,
            return_width=True, cap_col=cap, frame_to_min_fn=f2m,
        )
        per_stim_height[i] = d
        per_stim_width[i] = w

    per_stim = per_stim_width if metric == "width" else per_stim_height
    return {
        "mat": mat,
        "stim_cols": valid_stim_cols,
        "frame_to_min_fn": f2m,
        "per_stim": per_stim,
        "n_cells": n_cells,
        "direction": direction,
        "window": window,
        "cell_ids": list(df_indexed.index),
        "exp_name": exp_name,
        "ch": ch,
    }


def _score_running_extremum(per_stim, *, mode, n_per_train=LEARNING_STIMS_PER_TRAIN):
    """Habituation (``mode='min'``) or sensitization (``mode='max'``).

    Returns ``(summed, per_train)`` where ``summed.shape == (n_cells,)`` and
    ``per_train.shape == (n_trains, n_cells)``. ``summed == per_train.sum(0)``.
    """
    n_stims, n_cells = per_stim.shape
    n_trains = n_stims // n_per_train
    per_train = np.zeros((n_trains, n_cells), dtype=int)
    for t in range(n_trains):
        train = per_stim[t * n_per_train:(t + 1) * n_per_train]
        if mode == "min":
            current = np.full(n_cells, np.inf)
        else:
            current = np.full(n_cells, -np.inf)
        seen_any = np.zeros(n_cells, dtype=bool)
        for s in range(n_per_train):
            r = train[s]
            valid = ~np.isnan(r)
            if mode == "min":
                beats = valid & (r < current)
            else:
                beats = valid & (r > current)
            update_score = beats & seen_any
            update_value = beats | (valid & ~seen_any)
            per_train[t] += update_score.astype(int)
            current = np.where(update_value, r, current)
            seen_any = seen_any | valid
    return per_train.sum(axis=0), per_train


def _anticipation_train_spec(t, stim_cols, mat, all_minutes, direction,
                             window, n_trains, n_win=5):
    """Resolve train ``t``'s rest region and 5-frame anticipation window.

    Returns ``(ref_lo, ref_hi, win_cols)`` — the half-open column range of the
    post-train rest region and the integer columns of the anticipation window
    — or ``None`` when the experiment ends too early to score the train (or
    the window is not fully inside the rest region, which the permutation null
    requires).

    The rest region runs from the end of the train's last response (the last
    stim + the response-window upper bound) to 30 min later, or to the last
    frame for the final train. The anticipation window is centred 10 min after
    the train's last response peak (the median peak column across cells), with
    ``± n_win // 2`` frames either side.
    """
    n_cols = mat.shape[1]
    win_lo_off, win_hi_off = window
    train = stim_cols[
        t * LEARNING_STIMS_PER_TRAIN:(t + 1) * LEARNING_STIMS_PER_TRAIN
    ]
    last_sc = train[-1]

    # Last response peak of the train: median peak column across cells.
    pk_lo = max(0, last_sc + win_lo_off)
    pk_hi = min(n_cols, last_sc + win_hi_off)
    if pk_lo >= pk_hi:
        return None
    seg = mat[:, pk_lo:pk_hi]
    if direction == "decrease":
        peak_offsets = np.argmin(np.where(np.isnan(seg), np.inf, seg), axis=1)
    else:
        peak_offsets = np.argmax(np.where(np.isnan(seg), -np.inf, seg), axis=1)
    peak_col = int(np.round(np.nanmedian(pk_lo + peak_offsets)))

    # Rest region: end of last response → +30 min (→ experiment end if last).
    end_resp_col = min(last_sc + win_hi_off, n_cols - 1)
    ref_lo = end_resp_col
    if t == n_trains - 1:
        ref_hi = n_cols
    else:
        ref_end_min = float(all_minutes[end_resp_col]) + 30.0
        ref_hi = int(np.argmin(np.abs(all_minutes - ref_end_min))) + 1
    if ref_hi - ref_lo < n_win:
        return None

    # Anticipation window: 10 min after the peak, ± n_win // 2 frames.
    anticip_min = float(all_minutes[peak_col]) + 10.0
    anticip_col = int(np.argmin(np.abs(all_minutes - anticip_min)))
    half = n_win // 2
    win_cols = np.arange(anticip_col - half, anticip_col - half + n_win)

    # The null permutes within the rest region, so the window must sit inside.
    if win_cols[0] < ref_lo or win_cols[-1] >= ref_hi:
        return None
    return ref_lo, ref_hi, win_cols


def _score_anticipation(inputs, n_win=5):
    """Per-train positive / negative anticipation events. Returns ``(pos, neg)``.

    For each train the post-train rest region gives a per-cell mean and SD
    (see :func:`_anticipation_train_spec`). A cell scores one positive event
    for the train when any frame of the 5-frame anticipation window exceeds
    ``mean + 3·SD``, and one negative event when any frame falls below
    ``mean − 3·SD`` (the two are independent — a cell may score both). ``pos``
    / ``neg`` are the event counts summed across trains, each of shape
    ``(n_cells,)`` and ranging ``0..n_trains``.
    """
    mat = inputs["mat"]
    stim_cols = inputs["stim_cols"]
    direction = inputs["direction"]
    window = inputs["window"]
    n_cells, n_cols = mat.shape
    n_trains = len(stim_cols) // LEARNING_STIMS_PER_TRAIN
    all_minutes = inputs["frame_to_min_fn"](np.arange(n_cols))

    pos = np.zeros(n_cells, dtype=int)
    neg = np.zeros(n_cells, dtype=int)
    for t in range(n_trains):
        spec = _anticipation_train_spec(
            t, stim_cols, mat, all_minutes, direction, window, n_trains, n_win,
        )
        if spec is None:
            print(
                f"    anticipation: {inputs['exp_name']} / {inputs['ch']} — "
                f"train {t + 1}/{n_trains} not scorable "
                f"(rest region / window out of range)."
            )
            continue
        ref_lo, ref_hi, win_cols = spec
        ref = mat[:, ref_lo:ref_hi]
        ref_mean = np.nanmean(ref, axis=1)
        ref_std = np.nanstd(ref, axis=1)
        win_vals = mat[:, win_cols]
        pos += np.any(win_vals > (ref_mean + 3.0 * ref_std)[:, None], axis=1)
        neg += np.any(win_vals < (ref_mean - 3.0 * ref_std)[:, None], axis=1)
    return pos, neg


def _anticipation_null(inputs, *, n_perm=200, rng_seed=44, n_win=5):
    """Per-cell null for the anticipation scores.

    The 5-frame anticipation window is a subset of each train's rest region,
    so the null permutes that region's luminosity values: every permutation a
    cell draws ``n_win`` values without replacement from its rest region and
    the same ``mean ± 3·SD`` test is applied (mean and SD are unchanged by a
    permutation). Returns ``(pos_null, neg_null)``, each of shape
    ``(n_perm, n_cells)``.
    """
    mat = inputs["mat"]
    stim_cols = inputs["stim_cols"]
    direction = inputs["direction"]
    window = inputs["window"]
    n_cells, n_cols = mat.shape
    n_trains = len(stim_cols) // LEARNING_STIMS_PER_TRAIN
    all_minutes = inputs["frame_to_min_fn"](np.arange(n_cols))
    rng = np.random.default_rng(rng_seed)

    pos_null = np.zeros((n_perm, n_cells), dtype=float)
    neg_null = np.zeros((n_perm, n_cells), dtype=float)
    for t in range(n_trains):
        spec = _anticipation_train_spec(
            t, stim_cols, mat, all_minutes, direction, window, n_trains, n_win,
        )
        if spec is None:
            continue
        ref_lo, ref_hi, _ = spec
        ref = mat[:, ref_lo:ref_hi]
        n_ref = ref.shape[1]
        ref_mean = np.nanmean(ref, axis=1)
        ref_std = np.nanstd(ref, axis=1)
        hi_thr = (ref_mean + 3.0 * ref_std)[:, None]
        lo_thr = (ref_mean - 3.0 * ref_std)[:, None]
        for i in range(n_perm):
            # Independent per-cell permutation; the first n_win draws stand
            # in for the values that would land in the anticipation window.
            order = np.argsort(rng.random((n_cells, n_ref)), axis=1)
            draws = np.take_along_axis(ref, order[:, :n_win], axis=1)
            pos_null[i] += np.any(draws > hi_thr, axis=1)
            neg_null[i] += np.any(draws < lo_thr, axis=1)
    return pos_null, neg_null


def _compute_anticipation_blob(state, exp_name, cfg, *, n_perm=200):
    """Pool the per-train anticipation scores across channels for one expt.

    Anticipation is metric-independent (it is scored on raw luminosity, height
    only), so it is computed once per experiment rather than per metric.
    Returns a blob with the observed ``anticipation_pos`` / ``anticipation_neg``
    scores plus, when ``n_perm > 0``, the permutation null, per-cell p-values
    and population-level stats — or ``None`` when no channel yields data.
    """
    pos_chunks, neg_chunks = [], []
    pos_null_chunks, neg_null_chunks = [], []
    used_channels = []
    for ch in cfg["channels"]:
        inputs = _build_learning_inputs(
            state, exp_name, ch, cfg, metric="height",
        )
        if inputs is None:
            continue
        used_channels.append(ch)
        pos, neg = _score_anticipation(inputs)
        pos_chunks.append(pos)
        neg_chunks.append(neg)
        if n_perm and n_perm > 0:
            pos_null, neg_null = _anticipation_null(
                inputs, n_perm=n_perm, rng_seed=44,
            )
            pos_null_chunks.append(pos_null)
            neg_null_chunks.append(neg_null)

    if not pos_chunks:
        return None
    blob = {
        "anticipation_pos": np.concatenate(pos_chunks),
        "anticipation_neg": np.concatenate(neg_chunks),
    }
    # Channels are biological replicates; the null matrices are concatenated
    # in the same channel order, so this index aligns scores and null columns.
    blob["channel_index"] = np.concatenate(
        [np.full(len(c), ci, dtype=int) for ci, c in enumerate(pos_chunks)]
    )
    blob["channel_names"] = list(used_channels)
    if pos_null_chunks:
        blob["anticipation_pos_null"] = np.concatenate(pos_null_chunks, axis=1)
        blob["anticipation_neg_null"] = np.concatenate(neg_null_chunks, axis=1)
        blob["anticipation_pos_pvalue"] = pvalue_one_tailed(
            blob["anticipation_pos"], blob["anticipation_pos_null"],
        )
        blob["anticipation_neg_pvalue"] = pvalue_one_tailed(
            blob["anticipation_neg"], blob["anticipation_neg_null"],
        )
        _add_population_stats(blob, blob["channel_index"], len(used_channels))
    print(
        f"  anticipation scores: {exp_name} — "
        f"{len(blob['anticipation_pos'])} cells pooled across "
        f"{len(used_channels)} channels."
    )
    return blob


def _add_population_stats(blob, channel_index, n_channels):
    """Augment a learning blob with FDR q-values + population-level tests.

    Per-cell permutation p-values answer "which cells learn" but, pooled over
    hundreds of cells, need an FDR correction (``<measure>_qvalue``) before the
    significant count means anything. The population-level question — "does the
    cell *population* learn more than under shuffled stim order" — is answered
    by comparing the observed mean score to the per-permutation mean of the
    existing null (``<measure>_pop``), and again within each channel /
    biological replicate (``<measure>_pop_by_channel``) so a single dish
    driving the pooled result stays visible.
    """
    for m in ("habituation", "sensitization",
              "anticipation_pos", "anticipation_neg"):
        pvals = blob.get(f"{m}_pvalue")
        null = blob.get(f"{m}_null")
        if pvals is None or null is None:
            continue
        blob[f"{m}_qvalue"] = bh_fdr(pvals)
        blob[f"{m}_pop"] = population_permutation_pvalue(blob[m], null)
        per_channel = []
        for ci in range(n_channels):
            sel = channel_index == ci
            per_channel.append(
                population_permutation_pvalue(blob[m][sel], null[:, sel])
                if np.any(sel) else None
            )
        blob[f"{m}_pop_by_channel"] = per_channel


def compute_learning_scores(experiments, state, *, n_perm=200):
    """Compute habituation + sensitization + anticipation scores per DMSO expt.

    Returns a nested dict ``out[exp]`` keyed by ``"height"`` / ``"width"``
    (a habituation + sensitization blob per metric) and ``"anticipation"`` (a
    single metric-independent anticipation blob). Each blob holds the observed
    scores plus, when ``n_perm > 0``, the permutation null distributions and
    one-tailed per-cell p-values. Experiments are filtered to
    ``response_direction == "increase"`` (DMSO).
    """
    out = {}
    for exp_name, cfg in experiments.items():
        if cfg.get("response_direction") != "increase":
            continue
        out[exp_name] = {}
        for metric in ("height", "width"):
            hab_chunks, sen_chunks = [], []
            hab_null_chunks, sen_null_chunks = [], []
            used_channels = []
            for ch in cfg["channels"]:
                inputs = _build_learning_inputs(
                    state, exp_name, ch, cfg, metric=metric,
                )
                if inputs is None:
                    continue
                used_channels.append(ch)
                hab, _ = _score_running_extremum(inputs["per_stim"], mode="min")
                sen, _ = _score_running_extremum(inputs["per_stim"], mode="max")
                hab_chunks.append(hab)
                sen_chunks.append(sen)

                if n_perm and n_perm > 0:
                    hab_null = permutation_null_distribution(
                        lambda ps: _score_running_extremum(ps, mode="min")[0],
                        inputs["per_stim"], n_perm=n_perm,
                        rng_seed=42, mode="per_cell",
                    )
                    sen_null = permutation_null_distribution(
                        lambda ps: _score_running_extremum(ps, mode="max")[0],
                        inputs["per_stim"], n_perm=n_perm,
                        rng_seed=43, mode="per_cell",
                    )
                    hab_null_chunks.append(hab_null)
                    sen_null_chunks.append(sen_null)

            if not hab_chunks:
                out[exp_name][metric] = None
                continue
            blob = {
                "habituation": np.concatenate(hab_chunks),
                "sensitization": np.concatenate(sen_chunks),
            }
            # Keep replicate identity: which channel each pooled cell came
            # from (channels are biological replicates). Null matrices below
            # are concatenated in the same channel order, so this one index
            # aligns the score vectors and the null columns alike.
            blob["channel_index"] = np.concatenate(
                [np.full(len(c), ci, dtype=int)
                 for ci, c in enumerate(hab_chunks)]
            )
            blob["channel_names"] = list(used_channels)
            if hab_null_chunks:
                blob["habituation_null"] = np.concatenate(hab_null_chunks, axis=1)
                blob["sensitization_null"] = np.concatenate(sen_null_chunks, axis=1)
                blob["habituation_pvalue"] = pvalue_one_tailed(
                    blob["habituation"], blob["habituation_null"],
                )
                blob["sensitization_pvalue"] = pvalue_one_tailed(
                    blob["sensitization"], blob["sensitization_null"],
                )
                _add_population_stats(
                    blob, blob["channel_index"], len(used_channels),
                )
            out[exp_name][metric] = blob
            print(
                f"  learning scores: {exp_name} ({metric}) — "
                f"{len(blob['habituation'])} cells pooled across "
                f"{len(used_channels)} channels."
            )
        out[exp_name]["anticipation"] = _compute_anticipation_blob(
            state, exp_name, cfg, n_perm=n_perm,
        )
    return out


def _format_score_stats(qvalues, pop_result, pop_by_channel, q_alpha=0.05):
    """Compose the multi-line stats annotation for a learning histogram.

    Reports the FDR-significant cell count (per-cell q-values), the
    population-level permutation test (p + z effect size) and, when more than
    one biological replicate is present, the per-replicate z so a single dish
    driving the pooled result stays visible.
    """
    lines = []
    if qvalues is not None and np.size(qvalues):
        finite = np.isfinite(qvalues)
        n_sig = int(np.sum(qvalues[finite] < q_alpha))
        lines.append(f"FDR q<{q_alpha}: {n_sig}/{int(finite.sum())} cells")
    if pop_result is not None:
        lines.append(
            f"population: p={pop_result['p_value']:.3g}, "
            f"z={pop_result['z']:.2f}"
        )
    if pop_by_channel:
        parts = [
            f"c{ci + 1} z={res['z']:.1f}"
            for ci, res in enumerate(pop_by_channel) if res is not None
        ]
        if len(parts) > 1:
            lines.append("per replicate: " + "  ".join(parts))
    return "\n".join(lines)


def _plot_score_histogram(scores, *, title, xlabel, save_path,
                          bins=None, color=None, null_dist=None, x_max=None,
                          qvalues=None, pop_result=None, pop_by_channel=None,
                          caveat=None):
    """Single-distribution histogram with optional permutation-null overlay.

    When ``null_dist`` (shape ``(n_perm, n_cells)``) is provided, plot the
    pooled null as a back-layer histogram normalized to the same total cell
    count. The stats box reports FDR-corrected per-cell hits plus the
    population-level test (see :func:`_format_score_stats`); ``caveat`` is
    drawn as a figure footnote naming the unit of inference.

    When ``x_max`` is given the score axis is fixed to the discrete whole
    numbers ``0..x_max``.
    """
    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    if bins is None:
        if x_max is not None:
            max_v = int(x_max)
        else:
            max_v = int(np.nanmax(scores)) if scores.size else 0
            if null_dist is not None and null_dist.size:
                max_v = max(max_v, int(np.nanmax(null_dist)))
        bins = np.arange(-0.5, max_v + 1.5, 1)
    if null_dist is not None and null_dist.size:
        n_perm = null_dist.shape[0]
        ax.hist(
            null_dist.ravel(), bins=bins,
            color="#7a7a7a", alpha=0.55,
            edgecolor="#444444", linewidth=0.4,
            weights=np.full(null_dist.size, 1.0 / max(n_perm, 1)),
            label=f"Shuffled null (mean over {n_perm} perms)",
            zorder=1,
        )
    ax.hist(
        scores, bins=bins,
        color=color or PLOT_PARAMS["fit_color"],
        alpha=0.55, edgecolor="#222222", linewidth=0.6,
        label=f"Observed (n={scores.size})",
        zorder=2,
    )
    stats_text = _format_score_stats(qvalues, pop_result, pop_by_channel)
    if stats_text:
        ax.text(
            0.98, 0.95, stats_text,
            ha="right", va="top",
            transform=ax.transAxes,
            fontsize=PLOT_PARAMS["legend_fontsize"],
            bbox=dict(facecolor="white", edgecolor="#999999", alpha=0.9),
        )
    if x_max is not None:
        ax.set_xticks(np.arange(0, int(x_max) + 1))
    ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    if null_dist is not None and null_dist.size:
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
    plt.tight_layout(rect=(0, 0.035, 1, 1));
    if caveat:
        fig.text(
            0.5, 0.008, caveat, ha="center", va="bottom",
            fontsize=PLOT_PARAMS["legend_fontsize"] - 2,
            style="italic", color="#555555",
        )
    fig.savefig(save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)


def _plot_anticipation_histogram(pos, neg, *, title, save_path,
                                 pos_null=None, neg_null=None,
                                 pos_stats=None, neg_stats=None,
                                 caveat=None, n_trains=3):
    """Side-by-side negative / positive anticipation histograms.

    Each cell contributes one negative-event count and one positive-event
    count, so the two panels hold the same set of cells. The left panel plots
    the negative counts on a discrete ``-n_trains..0`` axis, the right panel
    the positive counts on ``0..n_trains``; both panels share the y-axis.

    ``pos_stats`` / ``neg_stats`` are ``(qvalues, pop_result, pop_by_channel)``
    tuples rendered as the per-panel stats box; ``caveat`` is the figure
    footnote naming the unit of inference.
    """
    fig, (ax_neg, ax_pos) = plt.subplots(
        1, 2, sharey=True,
        figsize=(PLOT_PARAMS["figsize"][0] * 1.7, PLOT_PARAMS["figsize"][1]),
        dpi=PLOT_PARAMS["dpi"],
    )
    for ax in (ax_neg, ax_pos):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)

    neg_bins = np.arange(-n_trains - 0.5, 1.5, 1)
    pos_bins = np.arange(-0.5, n_trains + 1.5, 1)

    if neg_null is not None and neg_null.size:
        n_perm = neg_null.shape[0]
        ax_neg.hist(
            -neg_null.ravel(), bins=neg_bins,
            color="#eda396", alpha=0.45,
            edgecolor="#6e2418", linewidth=0.4,
            weights=np.full(neg_null.size, 1.0 / max(n_perm, 1)),
            label=f"null ({n_perm} perms)",
            zorder=1,
        )
    ax_neg.hist(
        -neg, bins=neg_bins, color="#e74c3c", alpha=0.55,
        edgecolor="#7a1a12", linewidth=0.8,
        label=f"negative (n={neg.size})",
        zorder=2,
    )

    if pos_null is not None and pos_null.size:
        n_perm = pos_null.shape[0]
        ax_pos.hist(
            pos_null.ravel(), bins=pos_bins,
            color="#7fc8a0", alpha=0.45,
            edgecolor="#2f6e49", linewidth=0.4,
            weights=np.full(pos_null.size, 1.0 / max(n_perm, 1)),
            label=f"null ({n_perm} perms)",
            zorder=1,
        )
    ax_pos.hist(
        pos, bins=pos_bins, color="#1a9d51", alpha=0.55,
        edgecolor="#0d4f29", linewidth=0.8,
        label=f"positive (n={pos.size})",
        zorder=2,
    )

    ax_neg.set_xticks(np.arange(-n_trains, 1))
    ax_pos.set_xticks(np.arange(0, n_trains + 1))

    neg_text = _format_score_stats(*neg_stats) if neg_stats else ""
    if neg_text:
        ax_neg.text(
            0.02, 0.95, neg_text,
            ha="left", va="top", transform=ax_neg.transAxes,
            fontsize=PLOT_PARAMS["legend_fontsize"],
            bbox=dict(facecolor="white", edgecolor="#999999", alpha=0.9),
        )
    pos_text = _format_score_stats(*pos_stats) if pos_stats else ""
    if pos_text:
        ax_pos.text(
            0.98, 0.95, pos_text,
            ha="right", va="top", transform=ax_pos.transAxes,
            fontsize=PLOT_PARAMS["legend_fontsize"],
            bbox=dict(facecolor="white", edgecolor="#999999", alpha=0.9),
        )

    ax_neg.set_xlabel("Negative anticipation score (events / 3 trains)",
                      fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax_pos.set_xlabel("Positive anticipation score (events / 3 trains)",
                      fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax_neg.set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax_neg.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best");
    ax_pos.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best");
    fig.suptitle(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1));
    if caveat:
        fig.text(
            0.5, 0.008, caveat, ha="center", va="bottom",
            fontsize=PLOT_PARAMS["legend_fontsize"] - 2,
            style="italic", color="#555555",
        )
    fig.savefig(save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_learning_score_histograms(experiments, state, scores=None):
    """Emit summed habituation / sensitization / anticipation histograms (DMSO).

    For habituation / sensitization: ``learning_<measure>_<metric>.png`` per
    metric. For anticipation: a single ``learning_anticipation.png`` per
    experiment. The permutation null is overlaid when it was computed.
    """
    if scores is None:
        scores = compute_learning_scores(experiments, state)
    for exp_name, by_key in scores.items():
        for metric in ("height", "width"):
            blob = by_key.get(metric)
            if blob is None:
                continue
            n_ch = len(blob.get("channel_names", []))
            caveat = inferential_caveat(exp_name, n_ch, unit="cell")
            for measure_key, label_word in (
                ("habituation", "Habituation"),
                ("sensitization", "Sensitization"),
            ):
                summed = blob[measure_key]
                null = blob.get(f"{measure_key}_null")
                _plot_score_histogram(
                    summed,
                    title=(
                        f"{exp_name} — {label_word.lower()} score ({metric})\n"
                        f"new {'minimums' if measure_key == 'habituation' else 'maximums'} "
                        f"per cell across trains × increments"
                    ),
                    xlabel=f"{label_word} score ({metric})",
                    save_path=fig_path(
                        exp_name, f"learning_{measure_key}_{metric}",
                    ),
                    null_dist=null,
                    qvalues=blob.get(f"{measure_key}_qvalue"),
                    pop_result=blob.get(f"{measure_key}_pop"),
                    pop_by_channel=blob.get(f"{measure_key}_pop_by_channel"),
                    caveat=caveat,
                    x_max=12,
                )

        ablob = by_key.get("anticipation")
        if ablob is not None:
            n_ch = len(ablob.get("channel_names", []))
            caveat = inferential_caveat(exp_name, n_ch, unit="cell")
            _plot_anticipation_histogram(
                ablob["anticipation_pos"],
                ablob["anticipation_neg"],
                title=(
                    f"{exp_name} — anticipation score\n"
                    f"luminosity beyond mean ± 3·SD of post-train rest, "
                    f"5-frame window 10 min after each train's last peak"
                ),
                save_path=fig_path(exp_name, "learning_anticipation"),
                pos_null=ablob.get("anticipation_pos_null"),
                neg_null=ablob.get("anticipation_neg_null"),
                pos_stats=(
                    ablob.get("anticipation_pos_qvalue"),
                    ablob.get("anticipation_pos_pop"),
                    ablob.get("anticipation_pos_pop_by_channel"),
                ),
                neg_stats=(
                    ablob.get("anticipation_neg_qvalue"),
                    ablob.get("anticipation_neg_pop"),
                    ablob.get("anticipation_neg_pop_by_channel"),
                ),
                caveat=caveat,
            )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    scores = compute_learning_scores(experiments, state)
    plot_learning_score_histograms(experiments, state, scores=scores)


if __name__ == "__main__":
    main()
