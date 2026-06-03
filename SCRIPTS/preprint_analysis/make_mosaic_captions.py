"""Write a caption-information ``.txt`` beside every saved mosaic PNG.

The mosaics were deliberately simplified — most of the descriptive numbers that
used to live in each subplot's legend (fit slope/r, pair counts, permutation /
Mantel p-values, cell counts, setpoints, …) were removed so the panels read
cleanly. This script recovers exactly that information and lays it out
per-panel, keyed by the same a/b/c… letters the mosaic uses, so it can be
dropped straight into a figure caption.

It is a READ-ONLY companion to ``plots/mosaics.py``: it reuses that module's
``MOSAICS`` definitions and ``_find_instance`` resolver, and recomputes the
descriptive fits / permutation p-values with the *same* code paths the render
functions use (``scipy.linregress`` for the correlation fits;
``learning_scores._permutation_mean_pvalue`` for the permutation tests), so
every number here matches what the figure either draws or used to draw.

For each mosaic whose ``<OUT_ROOT>/mosaics/<name>.png`` exists, a sibling
``<name>.txt`` is written. Run from the repo root (where ``OUT_ROOT`` resolves):

    python SCRIPTS/preprint_analysis/make_mosaic_captions.py            # all saved mosaics
    python SCRIPTS/preprint_analysis/make_mosaic_captions.py nrk_chambers_dff_corr  # one
"""
import os
import sys

import numpy as np
from scipy.stats import linregress

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.config import OUT_ROOT  # noqa: E402
from common.io_paths import load_analysis_cache  # noqa: E402
from plots._base import title_of, xlabel_of, ylabel_of  # noqa: E402
from plots.learning_scores import _permutation_mean_pvalue  # noqa: E402
from plots.mosaics import MOSAICS, _find_instance  # noqa: E402

# Real experiments to probe when a mosaic doesn't pin one via "experiments"
# (mirrors make_figures: the driver builds for every experiment, and only the
# one whose caches resolve all cells succeeds). Discovered from the cache dir.
def _known_experiments():
    cache_root = os.path.join(OUT_ROOT, "analysis_cache")
    if not os.path.isdir(cache_root):
        return []
    return sorted(
        d for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d))
    )


# =============================================================================
# Small formatters — mirror the figure's own display logic exactly.
# =============================================================================
def _fmt_p_mantel(p):
    """Mantel p as the figure prints it (``correlation_distance._fmt_p``)."""
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.2g}"


def _perm_p_disp(p_value, n_perm):
    """Permutation p as the figure prints it
    (``learning_scores.render_permutation_mean_test``)."""
    return f"< {1.0 / n_perm:.0e}" if p_value == 0.0 else f"= {p_value:.4g}"


def _fit_stats(dists, corrs):
    """Least-squares fit on the non-NaN pairs (matches ``_fit_and_plot_subset``).

    Returns ``(n_pairs, slope, r)`` or ``(n_pairs, None, None)`` when fewer than
    3 valid pairs (the figure draws no line in that case).
    """
    d = np.asarray(dists, dtype=float)
    c = np.asarray(corrs, dtype=float)
    valid = ~np.isnan(d) & ~np.isnan(c)
    n = int(valid.sum())
    if n < 3:
        return n, None, None
    res = linregress(d[valid], c[valid])
    return n, float(res.slope), float(res.rvalue)


def _n_valid(mask, dists, corrs):
    d = np.asarray(dists, dtype=float)
    c = np.asarray(corrs, dtype=float)
    m = np.asarray(mask, dtype=bool) & ~np.isnan(d) & ~np.isnan(c)
    return int(m.sum())


# =============================================================================
# Per-spec fact extractors.  Each returns (kind, [bullet strings]).
# `kind` is a short description used when the panel has no drawn title.
# =============================================================================
def _facts_dff_trace(payload, fill, spec):
    n_cells = payload["mat"].shape[0]
    is_norm = spec.id.startswith("{ch}_dff_norm")
    kind = ("dF/F₀ trace stack" if is_norm
            else "Corrected-fluorescence trace stack")
    facts = [
        f"{n_cells} cells, {fill['ch']}; faint blue lines = individual cell "
        "traces, black line = population mean.",
        f"{fill['n_stims']} stimulus pulses shaded "
        f"(stimulus = {payload['stim_label']}).",
    ]
    if is_norm and fill.get("f0_note"):
        facts.append(f"dF/F₀ baseline: {fill['f0_note']}.")
    rsp = payload.get("real_setpoint_min")
    if rsp is not None:
        facts.append(f"Dotted vertical line = real setpoint ({rsp:.1f} min).")
    return kind, facts


def _facts_dff_pooled(payload, fill, spec):
    n_all = int(payload["pooled_dff_all"].shape[0])
    n_ch = len(payload["per_channel_counts"])
    facts = [
        f"Pooled mean dF/F₀ of {fill['n_total']} responder cells "
        f"({fill['cell_line']}); shaded band = ±1 SEM.",
        f"Responders pooled across {n_ch} channel(s) "
        f"({fill['n_total']} of {n_all} cells imaged were responders).",
    ]
    if payload.get("stim_aligned"):
        facts.append(
            f"{len(payload['spans'])} stimulus pulses shaded "
            f"(stimulus = {payload['stim_label']}); dashed line at 0."
        )
    return "Pooled responder mean dF/F₀", facts


def _facts_pca(payload, fill, spec):
    facts = [
        f"PCA of {fill['n_cells']} cells, {fill['ch_label']}; one point per "
        "cell, single colour (no clustering applied).",
        f"PC 1 explains {fill['evr0_pct']:.1f}% of variance; "
        f"PC 2 explains {fill['evr1_pct']:.1f}%.",
    ]
    return "PCA scatter (PC1 vs PC2)", facts


def _corr_fit_line(label, n_pairs, slope, r):
    if slope is None:
        return f"{label}: <3 valid pairs, no fit drawn ({n_pairs} pairs)."
    return (f"{label} fit: slope = {slope:.2e} Δr/μm, fit r = {r:.3f} "
            f"({n_pairs} cell pairs).")


def _facts_corr_channel(payload, fill, spec):
    """NRK per-chamber Pearson r vs distance (clean / caption legend)."""
    d, c = payload["pw_dist"], payload["pw_corr"]
    pc = payload.get("pair_classes") or {}
    facts = [
        f"Pairwise Pearson r vs. distance, chamber {fill['chamber']} "
        f"({fill['ch']}); {fill['n_cells']} cells, "
        f"{_n_valid(np.ones(len(d), bool), d, c)} cell pairs.",
        "Each point = one cell pair; correlation over "
        f"{fill['window_label']}.",
    ]
    # Responders-only (RR) fit + per-cell Mantel p.
    if pc.get("RR") is not None and np.asarray(pc["RR"]).any():
        n, s, r = _fit_stats(d[pc["RR"]], c[pc["RR"]])
        facts.append(_corr_fit_line("Responders-only (blue)", n, s, r))
        m_rr = payload.get("mantel_rr")
        if m_rr and not m_rr.get("insufficient"):
            facts.append(
                f"    → Mantel p = {_fmt_p_mantel(m_rr['p_value'])} "
                f"({m_rr['n_perm']} perms, {m_rr['n_cells']} responder cells)."
            )
    # All-cells fit + per-cell Mantel p.
    n, s, r = _fit_stats(d, c)
    facts.append(_corr_fit_line("All cells (grey line)", n, s, r))
    m_all = payload.get("mantel_all")
    if m_all and not m_all.get("insufficient"):
        facts.append(
            f"    → Mantel p = {_fmt_p_mantel(m_all['p_value'])} "
            f"({m_all['n_perm']} perms, {m_all['n_cells']} cells)."
        )
    # Non-responder cloud.
    if pc.get("NN") is not None:
        n_nn = _n_valid(pc["NN"], d, c)
        if n_nn:
            facts.append(
                f"Grey points = non-responder pairs ({n_nn}), no fit line.")
    facts.append("Shaded band around each fit line = ±3 SEM.")
    return "Pearson r vs distance (per chamber)", facts


def _facts_corr_combined(payload, fill, spec):
    """c2c12 pooled Pearson r vs distance, all channels combined."""
    d, c = payload["pw_dist"], payload["pw_corr"]
    pc = payload.get("pair_classes") or {}
    facts = [
        "Pairwise Pearson r vs. distance, pooled across all channels; "
        f"each point = one cell pair, correlation over {fill['window_label']}.",
    ]
    if pc.get("RR") is not None and np.asarray(pc["RR"]).any():
        n, s, r = _fit_stats(d[pc["RR"]], c[pc["RR"]])
        facts.append(_corr_fit_line("Responders-only (blue)", n, s, r))
        p_rr = payload.get("p_rr")
        if p_rr is not None:
            facts.append(
                f"    → replicate-level Mantel p = "
                f"{_fmt_p_mantel(p_rr['p_value'])} (n = {p_rr['n']} channels, "
                f"mean per-channel r = {p_rr['mean_r']:.3f})."
            )
    n, s, r = _fit_stats(d, c)
    facts.append(_corr_fit_line("All cells (grey line)", n, s, r))
    p_all = payload.get("p_all")
    if p_all is not None:
        facts.append(
            f"    → replicate-level Mantel p = "
            f"{_fmt_p_mantel(p_all['p_value'])} (n = {p_all['n']} channels, "
            f"mean per-channel r = {p_all['mean_r']:.3f})."
        )
    if pc.get("NN") is not None:
        n_nn = _n_valid(pc["NN"], d, c)
        if n_nn:
            facts.append(
                f"Grey points = non-responder pairs ({n_nn}), no fit line.")
    facts.append("Shaded band around each fit line = ±3 SEM.")
    return "Pearson r vs distance (pooled)", facts


def _facts_violin(payload, fill, spec):
    vd = payload["violin_data"]
    highlight = bool(payload.get("highlight_responders"))
    n_stim = len(vd)
    onsets = ", ".join(str(x) for x in payload["x_labels"])
    per_stim = [len(v) for v in vd]
    n_cells = per_stim[0] if per_stim else 0
    same = len(set(per_stim)) == 1
    facts = [
        f"Per-stimulus distribution of {fill['metric']} "
        f"({fill['y_label'].strip()}); {fill['n_total_cells']} cells pooled "
        f"across {fill['n_channels']} channels.",
        f"Metric: {fill['title_core']}.",
        f"{n_stim} stimuli, onsets (min): {onsets}.",
        (f"{n_cells} cells per stimulus."
         if same else f"cells per stimulus: {per_stim}."),
    ]
    if highlight:
        facts.append(
            f"{fill['n_total_responders']} of {fill['n_total_cells']} cells "
            "are responders (highlighted points); grey points = all other "
            "cells.")
    else:
        facts.append(
            "Grey points = individual cells (responders are NOT highlighted in "
            f"this panel; {fill['n_total_responders']} of "
            f"{fill['n_total_cells']} cells are responders).")
    facts.append(
        "Half-violin = kernel density; notched box = median + IQR; "
        "horizontal marker = mean.")
    return "Per-stimulus response violins", facts


def _facts_train_means(payload, fill, spec):
    from plots.response_violins import _sig_stars
    ctm = np.asarray(payload["chan_train_means"], dtype=float)
    n_ch, n_tr = ctm.shape
    per_train = np.nanmean(ctm, axis=0)
    train_p = payload["train_p"]
    means_str = ", ".join(
        f"train {i + 1} = {v:.3f}" for i, v in enumerate(per_train))
    facts = [
        f"Mean {fill['metric']} response per stimulus train; one line per "
        f"biological replicate (channel), {n_ch} replicates × {n_tr} trains.",
        f"Train means (averaged over replicates): {means_str}.",
        f"Bracket = first→last train change, replicate-level one-sample "
        f"t-test: p = {train_p:.3g} ({_sig_stars(train_p)}).",
    ]
    return "Per-replicate train means", facts


def _facts_learning_hist(payload, fill, spec):
    scores = np.asarray(payload["scores"])
    facts = [
        f"Distribution of per-cell {fill['label_word_lower']} scores; "
        f"n = {scores.size} cells, mean = {np.nanmean(scores):.3f}.",
        "Blue = observed; grey = one representative shuffled null draw "
        "(significance is in the paired permutation-test panel).",
    ]
    return f"{fill['label_word']} score histogram", facts


def _facts_learning_permtest(payload, fill, spec):
    obs = np.asarray(payload["observed"])
    null = np.asarray(payload["null"])
    M_real, M_shuf, p = _permutation_mean_pvalue(obs, null)
    n_perm = int(M_shuf.size)
    measure = fill.get("label_word") or f"anticipation train {fill.get('train_idx')}"
    facts = [
        f"Permutation test on the population mean {measure.lower()} score.",
        f"Observed mean = {M_real:.3f}; two-tailed permutation "
        f"p {_perm_p_disp(p, n_perm)} ({n_perm} permutations).",
        "Grey histogram = shuffled-null means; vertical line = observed mean.",
    ]
    return f"{measure} permutation test", facts


def _facts_anticipation_hist(payload, fill, spec):
    real = np.asarray(payload["real"])
    shuf = np.asarray(payload["shuffled"])
    facts = [
        f"Rest-region z-scores, train {fill['train_idx']}; n = {real.size} "
        "cells.",
        f"Mean observed z = {np.nanmean(real):.3f} (blue dashed); "
        f"mean shuffled z = {np.nanmean(shuf):.3f} (grey dashed).",
        "Blue = observed rest-region z-scores; grey = shuffled null.",
    ]
    return f"Anticipation z-score histogram: train {fill['train_idx']}", facts


def _facts_avg_peak_stim8(payload, fill, spec):
    grid = payload["grid"]
    facts = [
        f"Mean response to stimulus #8 ({fill['cell_line']}); mean ± 3 SEM "
        f"over {fill['n_seg']} responder cell segments "
        f"(pooled over {fill['n_channels']} channel(s)).",
        f"x-axis = time since stimulus onset, {grid.min():.0f}–"
        f"{grid.max():.0f} min.",
    ]
    return "Stimulus-#8 responder average peak", facts


def _facts_nrk_hw_log(payload, fill, spec):
    reg = payload["setpoint_regions_min"]
    setpoints = ", ".join(f"{r['setpoint']:.2f}" for r in reg)
    secs = int(payload["pulse_duration"] * 60)
    n_acid = len(payload["acid_min"])
    y0, y1 = payload["y_lim"]
    x0, x1 = payload["x_lim"]
    facts = [
        f"Hardware-feedback controlled mean fluorescence, chamber "
        f"{fill['chamber']} ({fill['ch']}).",
        f"Setpoint band(s) at {setpoints}; line = measured mean fluorescence.",
        f"{n_acid} acidic pulse(s) shaded ({secs} s each).",
        f"x-axis {x0:.0f}–{x1:.0f} min; y-axis {y0:.2f}–{y1:.2f} "
        "(per-chamber scale, not shared).",
    ]
    rsp = payload.get("real_setpoint_min")
    if rsp is not None:
        facts.append(f"Dotted line = real setpoint ({rsp:.1f} min).")
    return "NRK hardware-feedback log", facts


FACTS_BY_SPEC = {
    "dff_raw": _facts_dff_trace,
    "dff_norm": _facts_dff_trace,
    "dff_mean_pooled_responders": _facts_dff_pooled,
    "pooled_pca_only": _facts_pca,
    "corr_vs_dist_channel_pearson": _facts_corr_channel,
    "corr_vs_dist_combined_pearson": _facts_corr_combined,
    "response_violin": _facts_violin,
    "response_violin_train_means": _facts_train_means,
    "learning_score_hist": _facts_learning_hist,
    "learning_score_permtest": _facts_learning_permtest,
    "learning_anticipation_hist": _facts_anticipation_hist,
    "learning_anticipation_permtest": _facts_learning_permtest,
    "average_peak_responders_stim8": _facts_avg_peak_stim8,
    "nrk_hardware_log": _facts_nrk_hw_log,
}


# =============================================================================
# Mosaic-level prose (the one-paragraph "what is this figure" overview).
# =============================================================================
OVERVIEWS = {
    "c2c12_chambers_dff_stack": (
        "C2C12 chamber stack: the three per-channel dF/F₀ trace panels "
        "(relabelled Chamber A/B/C) over the pooled responder mean. Panels "
        "a–c share one x- and y-scale; the pooled panel d has its own "
        "zoomed y-range."),
    "c2c12_corr_pca_responses": (
        "C2C12 overview: pooled correlation-vs-distance and PCA on the top "
        "row; the single-cell response violins and per-replicate train means "
        "on the bottom."),
    "c2c12_learning_scores": (
        "C2C12 learning scores: one row per measure, each pairing the "
        "observed-vs-shuffled score histogram (left) with its permutation "
        "test (right) — habituation, sensitization, then anticipation "
        "trains 1 and 2 (height metric)."),
    "dmso_responder_overview": (
        "Cross-experiment DMSO responder overview: one cell line per row — "
        "the pooled responder-mean trace (left) beside its mean response to "
        "stimulus #8 (right); C2C12 on top, PC-3 below. All y-axes relabelled "
        "“fluorescence”."),
    "c2c12_ch3_dff_pair": (
        "C2C12 channel-3 dF/F₀ pair, schematic style: the corrected-"
        "fluorescence trace stack (left) beside its normalized dF/F₀ stack "
        "(right). Stripped to bare traces — no title, legend, panel "
        "letters, or ticks; x clipped to the first 90 min."),
    "nrk_chambers_hw_log": (
        "NRK hardware-feedback fluorescence logs as a 2×2 chamber grid: "
        "channel 1 (chambers A, C) on top, channel 2 (chambers B, D) below. "
        "Each panel keeps its own per-chamber y-scale."),
    "nrk_chambers_dff_corr": (
        "NRK per-chamber dF/F₀ + correlation-vs-distance, one chamber per "
        "row: the chamber's dF/F₀ trace stack (left) beside its pairwise "
        "Pearson-r vs distance scatter (right). Rows: channel 1 chambers A, C "
        "then channel 2 chambers B, D. No shared y-scale."),
}


# =============================================================================
# Assembly.
# =============================================================================
def _resolve_exp(name, m, experiments):
    """The single experiment this mosaic builds for (mirrors build_mosaic)."""
    allowed = m.get("experiments")
    candidates = allowed if allowed else experiments
    for exp in candidates:
        try:
            for cell in m["cells"].values():
                cell_exp = cell[2] if len(cell) > 2 else exp
                _find_instance(cell_exp, cell[0], cell[1])
            return exp
        except (KeyError, FileNotFoundError):
            continue
    return None


def _layout_sketch(layout):
    return "\n".join(
        "  " + "".join(f"[{k}]" if k != "." else "[ ]" for k in row)
        for row in layout
    )


def _effective_label(spec, fill, m, mkey):
    """Drawn title for the panel: mosaic override > spec title (formatted)."""
    if mkey in m.get("titles", {}):
        return m["titles"][mkey]
    try:
        return title_of(spec, fill)
    except (KeyError, IndexError):
        return ""


def build_caption(name, experiments):
    m = MOSAICS[name]
    exp = _resolve_exp(name, m, experiments)
    if exp is None:
        return None

    panel_labels = m.get("panel_labels", True)
    hide_leg = m.get("hide_legend")
    hide_tk = m.get("hide_ticks")
    xlims = m.get("xlims", {})
    titles_ov = m.get("titles", {})
    xlabels_ov = m.get("xlabels", {})
    ylabels_ov = m.get("ylabels", {})

    lines = []
    lines.append("=" * 78)
    lines.append(f"FIGURE: {name}   (file: {name}.png)")
    src_exps = sorted({(c[2] if len(c) > 2 else exp) for c in m["cells"].values()})
    lines.append("Experiment(s): " + ", ".join(src_exps))
    lines.append("=" * 78)
    lines.append("")
    if name in OVERVIEWS:
        lines.append("Overview:")
        for ln in _wrap(OVERVIEWS[name], 74):
            lines.append("  " + ln)
        lines.append("")
    lines.append("Layout (reading order):")
    lines.append(_layout_sketch(m["layout"]))
    lines.append("")
    if not panel_labels:
        lines.append("(No a/b/c… letters are drawn on this figure; panels "
                     "are keyed below by their layout position / title.)")
        lines.append("")

    for mkey, cell in m["cells"].items():
        if mkey == ".":
            continue
        cell_exp = cell[2] if len(cell) > 2 else exp
        spec, payload, fill = _find_instance(cell_exp, cell[0], cell[1])
        fn = FACTS_BY_SPEC.get(cell[0])
        kind, facts = fn(payload, fill, spec) if fn else (cell[0], [])

        label = _effective_label(spec, fill, m, mkey).replace("\n", " ").strip()
        header = f"Panel {mkey}"
        if label:
            header += f" — {label}"
        else:
            header += f" — {kind}"
        if len(src_exps) > 1:
            header += f"   [{cell_exp}]"
        lines.append("-" * 78)
        lines.append(header)
        lines.append("-" * 78)
        for f in facts:
            for j, ln in enumerate(_wrap(f, 72)):
                lines.append(("  • " if j == 0 else "    ") + ln)
        # Mosaic-level overrides worth flagging for the caption.
        notes = []
        if mkey in titles_ov and titles_ov[mkey] == "":
            notes.append("panel title removed in this mosaic.")
        elif mkey in titles_ov:
            notes.append(f"panel re-titled “{titles_ov[mkey]}”.")
        if mkey in ylabels_ov:
            notes.append(f"y-axis relabelled “{ylabels_ov[mkey]}”.")
        if mkey in xlabels_ov:
            notes.append(f"x-axis relabelled “{xlabels_ov[mkey]}”.")
        if mkey in xlims:
            lo, hi = xlims[mkey]
            notes.append(f"x-axis clipped to {lo:g}–{hi:g}.")
        if hide_leg is True or (isinstance(hide_leg, list) and mkey in hide_leg):
            notes.append("legend hidden.")
        if hide_tk is True or (isinstance(hide_tk, list) and mkey in hide_tk):
            notes.append("axis ticks/tick-labels hidden.")
        for nt in notes:
            lines.append(f"  · (mosaic) {nt}")
        lines.append("")

    lines.append("-" * 78)
    lines.append("Auto-generated by make_mosaic_captions.py from the analysis "
                 "caches; numbers match the figure's own computations.")
    return "\n".join(lines) + "\n"


def _wrap(text, width):
    """Tiny word-wrap (avoids importing textwrap config; keeps unicode width)."""
    words, line, out = text.split(), "", []
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        out.append(line)
    return out or [""]


def main(argv):
    experiments = _known_experiments()
    mosaics_dir = os.path.join(OUT_ROOT, "mosaics")
    requested = argv[1:] if len(argv) > 1 else list(MOSAICS)
    written = 0
    for name in requested:
        if name not in MOSAICS:
            print(f"  skip {name}: unknown mosaic")
            continue
        png = os.path.join(mosaics_dir, f"{name}.png")
        if not os.path.exists(png):
            if len(argv) > 1:
                print(f"  skip {name}: no saved PNG at {png}")
            continue
        text = build_caption(name, experiments)
        if text is None:
            print(f"  skip {name}: could not resolve an experiment")
            continue
        out = os.path.join(mosaics_dir, f"{name}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        written += 1
        print(f"  wrote {out}")
    print(f"done: {written} caption file(s).")


if __name__ == "__main__":
    main(sys.argv)
