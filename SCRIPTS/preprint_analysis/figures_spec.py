"""Central registry of every figure's label / title / legend text.

This is the ONE scannable place to read or edit axis labels, titles, suptitles,
legend strings, and footnotes. Each entry is a :class:`FigureSpec` keyed by the
short spec id that a ``plots/<group>.py`` ``iter_figures()`` generator yields;
the render callable itself lives in that module. Templates use ``str.format``
placeholders filled at plot time from cached metadata (n cells, % variance,
p-values, ...).

The four ``responder_diagnostic`` figures are the documented multi-panel
exception (``multi_panel=True``): their render functions receive a whole
``Figure`` and own their gridspec + in-figure titles, so the strings here are
documentation for the registry rather than the drawn text.
"""
from plots._base import FigureSpec
from plots import dff as _dff
from plots import clustering as _clustering
from plots import average_peak as _avg
from plots import correlation_distance as _corr
from plots import response_violins as _violins
from plots import learning_scores as _learning
from plots import nrk_hardware_log as _nrk_hw
from plots import responder_diagnostic as _diag

FIGURES = {}


def _add(key, spec):
    FIGURES[key] = spec


# =============================================================================
# dF/F0  (plots/dff.py)
# =============================================================================
# Shared standalone suptitle for the decomposed corrected / dF/F0 trace panels.
_SUP_DFF = "{exp_name} / {ch} — {cell_str}, {n_stims} stims"
_LEGEND_DFF_TRACE = {"mean": "Mean", "setpoint": "Real setpoint ({rsp:.1f} min)"}

_add("dff_raw", FigureSpec(
    id="{ch}_dff_raw{subset_suffix}", analysis="dff", scope="channel_subset",
    render=_dff.render_dff_trace,
    title="Corrected fluorescence",
    xlabel="time (min)", ylabel="corrected fluorescence",
    suptitle=_SUP_DFF, legend=_LEGEND_DFF_TRACE,
))
_add("dff_norm", FigureSpec(
    id="{ch}_dff_norm{subset_suffix}", analysis="dff", scope="channel_subset",
    render=_dff.render_dff_trace,
    title="dF/F₀  ({f0_note})",
    xlabel="time (min)", ylabel="fluorescence",
    suptitle=_SUP_DFF, legend=_LEGEND_DFF_TRACE,
))
_add("dff_mean_pooled_responders", FigureSpec(
    id="dff_mean_pooled_responders", analysis="dff", scope="experiment",
    render=_dff.render_dff_mean_pooled,
    title="Mean fluorescence of responder cells ({cell_line})",
    xlabel="time (min)", ylabel="fluorescence",
    legend={"sem": "±1 SEM", "mean": "Pooled mean ({n_total} cells)"},
))
_add("dff_pooled_traces", FigureSpec(
    id="dff_pooled_traces", analysis="dff", scope="experiment",
    render=_dff.render_dff_pooled_traces,
    title="{exp_name} — pooled dF/F₀ traces, all cells\n[{per_ch_str}]",
    xlabel="time (min)", ylabel="dF/F₀  (pooled across channels)",
    legend={"mean": "Pooled mean ({n_total} cells)"},
))


# =============================================================================
# Clustering — PCA + UMAP  (plots/clustering.py)
# =============================================================================
_SUP_CLUSTERING = "{exp_name} / {ch_label} — PCA + UMAP (no clustering, n={n_cells})"

_add("pooled_pca_only", FigureSpec(
    id="pooled_pca_only", analysis="clustering", scope="experiment",
    render=_clustering.render_pca_only,
    title="PCA: PC1 vs PC2",
    xlabel="PC 1 ({evr0_pct:.1f}% var)", ylabel="PC 2 ({evr1_pct:.1f}% var)",
    suptitle=_SUP_CLUSTERING,
))
_add("pooled_umap_only", FigureSpec(
    id="pooled_umap_only", analysis="clustering", scope="experiment",
    render=_clustering.render_umap_only,
    title="UMAP embedding",
    xlabel="UMAP 1", ylabel="UMAP 2",
    suptitle=_SUP_CLUSTERING,
))


# =============================================================================
# Average peak  (plots/average_peak.py) — DMSO experiments only
# The two cross-experiment combined figures build their FigureSpec inline in
# plots/average_peak.build_combined (registered as a CROSS_EXPERIMENT_BUILDER).
# =============================================================================
_LEGEND_AVG = {"band": "Mean ± 3 SEM",
               "mean": "Average peak (n={n_seg} cell×stim segments)"}

_add("average_peak", FigureSpec(
    id="average_peak", analysis="average_peak", scope="experiment",
    render=_avg.render_average_peak,
    title="{exp_name} — average response peak (all cells)\n"
          "per-stimulus dF/F₀ segments pooled over {n_channels} channel(s)",
    xlabel="Time since stimulus onset (min)", ylabel="dF/F₀",
    legend=_LEGEND_AVG,
))
_add("average_peak_responders", FigureSpec(
    id="average_peak_responders", analysis="average_peak", scope="experiment",
    render=_avg.render_average_peak,
    title="{exp_name} — average response peak (responders only)\n"
          "per-stimulus dF/F₀ segments pooled over {n_channels} channel(s)",
    xlabel="Time since stimulus onset (min)", ylabel="dF/F₀",
    legend=_LEGEND_AVG,
))
_add("average_peak_responders_stim8", FigureSpec(
    id="average_peak_responders_stim8", analysis="average_peak",
    scope="experiment", render=_avg.render_average_peak_stim8,
    title="Mean response to stimulus #8 ({cell_line})",
    xlabel="Time since stimulus onset (min)", ylabel="dF/F₀",
    legend={"band": "Mean ± 3 SEM",
            "mean": "Average peak (n={n_seg} responder cell segments)"},
))


# =============================================================================
# Correlation vs distance  (plots/correlation_distance.py)
# Each spec has a render-time _log1p twin: iter_figures yields each with
# apply_log1p True/False and fills {log1p_suffix} (filename) + {log1p_note}
# (title). The clean (NRK-chamber Pearson) vs verbose title/ylabel is resolved
# in iter_figures into {title_main}/{ylabel_main}, so these stay static.
# =============================================================================
_CORR_CAVEAT = "{caveat}"   # cached inferential_caveat string, via fill

_add("corr_vs_dist_channel_pearson", FigureSpec(
    id="{ch}_corr_vs_dist_pearson{log1p_suffix}",
    analysis="correlation_distance", scope="channel",
    render=_corr.render_corr_vs_dist_channel,
    title="{title_main}{log1p_note}",
    xlabel="Pairwise distance (μm)", ylabel="{ylabel_main}",
    caveat=_CORR_CAVEAT,
))
_add("corr_vs_dist_channel_spearman", FigureSpec(
    id="{ch}_corr_vs_dist_spearman{log1p_suffix}",
    analysis="correlation_distance", scope="channel",
    render=_corr.render_corr_vs_dist_channel,
    title="{title_main}{log1p_note}",
    xlabel="Pairwise distance (μm)", ylabel="{ylabel_main}",
    caveat=_CORR_CAVEAT,
))
_add("corr_vs_dist_combined_pearson", FigureSpec(
    id="corr_vs_dist_combined_pearson{log1p_suffix}",
    analysis="correlation_distance", scope="experiment",
    render=_corr.render_corr_vs_dist_combined,
    title="Pairwise Pearson r vs. distance{log1p_note}",
    xlabel="Pairwise distance (μm)", ylabel="{method_label} ({window_label})",
    caveat=_CORR_CAVEAT,
))
_add("corr_vs_dist_combined_spearman", FigureSpec(
    id="corr_vs_dist_combined_spearman{log1p_suffix}",
    analysis="correlation_distance", scope="experiment",
    render=_corr.render_corr_vs_dist_combined,
    title="{exp_name_disp} — pairwise Spearman ρ vs distance "
          "({window_label}, all channels combined){log1p_note}",
    xlabel="Pairwise distance (μm)", ylabel="{method_label} ({window_label})",
    caveat=_CORR_CAVEAT,
))


# =============================================================================
# Response violins  (plots/response_violins.py) — DMSO experiments
# title_core / y_label / width_cap_note / n_* are filled from the cache so the
# verbatim source title strings are preserved exactly. The source keeps the
# stats box and caveat footnote commented out, so neither is drawn here.
# =============================================================================
_VIOLIN_TITLE = "{exp_name} — pooled per-stimulus {metric} ({title_core}){width_cap_note}"
_VIOLIN_TITLE_RESP = (
    "{exp_name} — pooled per-stimulus {metric} ({title_core})\n"
    "responders highlighted — {n_total_responders}/{n_total_cells} cells "
    "(Bonferroni |Δ dF/F₀| threshold){width_cap_note}"
)
_VIOLIN_LEGEND = {"mean": "Mean", "median": "Median", "responder": "Responder"}

_add("response_violin", FigureSpec(
    id="pooled_response_violin_{metric}_dff", analysis="response_violins",
    scope="experiment", render=_violins.render_violin,
    title=_VIOLIN_TITLE,
    xlabel="Stimulus onset (min)", ylabel="{y_label}",
    legend=_VIOLIN_LEGEND, figsize=(6.5, 4.5),
))
_add("response_violin_responders", FigureSpec(
    id="pooled_response_violin_{metric}_dff_responders", analysis="response_violins",
    scope="experiment", render=_violins.render_violin,
    title=_VIOLIN_TITLE_RESP,
    xlabel="Stimulus onset (min)", ylabel="{y_label}",
    legend=_VIOLIN_LEGEND, figsize=(6.5, 4.5),
))
_add("response_violin_train_means", FigureSpec(
    id="pooled_response_violin_{metric}_dff_train_means", analysis="response_violins",
    scope="experiment", render=_violins.render_train_means,
    title="{exp_name} — per-replicate train mean ({metric})\n"
          "each biological replicate's mean response per stimulus train",
    xlabel="Stimulus train", ylabel="Mean response ({metric})",
    legend={"rep": "Rep {rep_num}"},
))


# =============================================================================
# Learning scores  (plots/learning_scores.py) — DMSO experiments
# =============================================================================
_LEGEND_LEARNING_HIST = {
    "observed": "Observed (n={n})",
    "shuffled": "Shuffled null distribution",
}
_LEGEND_LEARNING_PERMTEST = {
    "observed": "Observed mean = {m_real:.3f}\ntwo-tailed permutation p {p_disp}",
    "shuffled": "Shuffled mean ({n_perm} perms)",
}

_add("learning_score_hist", FigureSpec(
    id="learning_{measure_key}_{metric}", analysis="learning_scores",
    scope="experiment", render=_learning.render_score_histogram,
    title="{label_word} score distribution{metric_suffix}",
    xlabel="{label_word_lower} score{metric_suffix}", ylabel="number of cells",
    legend=_LEGEND_LEARNING_HIST,
))
_add("learning_score_permtest", FigureSpec(
    id="learning_{measure_key}_{metric}_permtest", analysis="learning_scores",
    scope="experiment", render=_learning.render_permutation_mean_test,
    title="{label_word} score permutation test{metric_suffix}",
    xlabel="Mean {label_word_lower} score{metric_suffix}", ylabel="Permutations",
    legend=_LEGEND_LEARNING_PERMTEST,
))
_add("learning_anticipation_hist", FigureSpec(
    id="learning_anticipation_train{train_idx}", analysis="learning_scores",
    scope="experiment", render=_learning.render_anticipation_zscore_histogram,
    title="Anticipation score distribution: train {train_idx}",
    xlabel="Rest-region z-score", ylabel="number of cells",
    legend=_LEGEND_LEARNING_HIST,
))
_add("learning_anticipation_permtest", FigureSpec(
    id="learning_anticipation_train{train_idx}_permtest", analysis="learning_scores",
    scope="experiment", render=_learning.render_permutation_mean_test,
    title="Anticipation permutation test",
    xlabel="Mean rest-region z-score", ylabel="Permutations",
    legend=_LEGEND_LEARNING_PERMTEST,
))


# =============================================================================
# NRK hardware-feedback luminosity log  (plots/nrk_hardware_log.py) — NRK only
# =============================================================================
_add("nrk_hardware_log", FigureSpec(
    id="{ch}_hw_lum_log", analysis="nrk_hardware_log", scope="channel",
    render=_nrk_hw.render_nrk_hardware_log, figsize=(6.5, 2.8),
    title="Controlled mean fluorescence chamber {chamber}",
    xlabel="time (min)", ylabel="fluorescence",
    legend={
        "setpoint_band": "Setpoint {sp:.2f}",
        "mean": "Mean fluorescence",
        "acid": "Acidic pulse ({secs} s)",
        "real_setpoint": "Real setpoint ({rsp:.1f} min)",
    },
))


# =============================================================================
# Responder QC diagnostics  (plots/responder_diagnostic.py)
# THE MULTI-PANEL EXCEPTION — each render fn receives a Figure and builds its
# own gridspec (set_layout_engine("none") + manual margins). The title/suptitle
# strings below document the figure; the render fns own the drawn text. Height
# grows with channel count inside the render fn; figsize is the 1-channel
# default the driver hands to plt.figure().
# =============================================================================
_add("responder_distribution_diagnostic", FigureSpec(
    id="responder_distribution_diagnostic", analysis="responder_diagnostic",
    render=_diag.render_distribution, scope="experiment", multi_panel=True,
    title="Responder distribution diagnostic",
    suptitle="{exp_name} — responder distribution diagnostic",
    figsize=(6.5, 8.4),
))
_add("responder_stimlock_diagnostic", FigureSpec(
    id="responder_stimlock_diagnostic", analysis="responder_diagnostic",
    render=_diag.render_stimlock, scope="experiment", multi_panel=True,
    title="Stimulus-locked artifact check",
    suptitle="{exp_name} — stimulus-locked artifact check",
    figsize=(6.5, 7.2),
))
_add("responder_artifact_diagnostic", FigureSpec(
    id="responder_artifact_diagnostic", analysis="responder_diagnostic",
    render=_diag.render_artifact, scope="experiment", multi_panel=True,
    title="Dead-frame & optical-artifact check",
    suptitle="{exp_name} — dead-frame & optical-artifact check",
    figsize=(6.5, 11.2),
))
_add("responder_f0_diagnostic", FigureSpec(
    id="responder_f0_diagnostic", analysis="responder_diagnostic",
    render=_diag.render_f0, scope="experiment", multi_panel=True,
    title="F0-dependence check",
    suptitle="{exp_name} — F0-dependence check",
    figsize=(6.5, 7.7),
))
