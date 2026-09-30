"""Render functions for the learning-score figures.

Reads ``analysis_cache/<exp>/learning_scores.pkl`` (written by
``analyze_learning_scores.py``). Every figure here is single-axis and draws onto
a passed-in ``ax``. The source ``learning_scores.py`` is split into four render
functions:

  * ``render_score_histogram``     — observed-vs-shuffled score histogram
        (``_plot_score_histogram``)
  * ``render_permutation_mean_test`` — shuffled-mean permutation test with the
        observed mean line + two-tailed p (``_plot_permutation_mean_test``);
        serves both the habituation/sensitization permtests and the
        anticipation permtests.
  * ``render_anticipation_zscore_histogram`` — real-vs-shuffled rest-region
        z-score histogram (``_plot_anticipation_zscore_histogram``)

Title/axis-label text lives centrally in ``figures_spec.py`` and is pulled from
the passed ``spec``. The two number-bearing legend labels of the permutation
test (observed mean = …, two-tailed p …; shuffled mean (N perms)) are recomputed
from the cached null/observed inside the render — deterministic, since the null
matrices are cached verbatim.
"""
import numpy as np

from plots._base import (
    PLOT_PARAMS,
    clean_axes,
    legend_text,
    title_of,
    xlabel_of,
    ylabel_of,
)

NAME = "learning_scores"

# Fixed discrete score axis for the habituation/sensitization histograms — the
# source hard-codes x_max=12 (0..12 whole-number bins).
_SCORE_X_MAX = 12

# Null-overlay gray (matches the source's "#7a7a7a" example-shuffle histogram).
_NULL_COLOR = "#7a7a7a"
_NULL_EDGE = "#444444"
_OBS_EDGE = "#222222"


def render_score_histogram(ax, payload, spec, *, fill):
    """Observed score distribution with a single representative shuffled null.

    Port of ``_plot_score_histogram`` with ``x_max=12``. ``payload`` supplies
    ``scores`` (n_cells,) and ``null`` (n_perm × n_cells); the back-layer null
    is the FIRST row of ``null`` (one shuffled score per cell — an example
    draw, not the mean over permutations).
    """
    P = PLOT_PARAMS
    clean_axes(ax)
    scores = np.asarray(payload["scores"])
    null_dist = payload.get("null")

    max_v = int(_SCORE_X_MAX)
    bins = np.arange(-0.5, max_v + 1.5, 1)

    if null_dist is not None and np.size(null_dist):
        # A single representative shuffle (one shuffled score per cell), not
        # the mean over all permutations — an example null draw.
        example_null = np.asarray(null_dist)[0]
        ax.hist(
            example_null, bins=bins,
            color=_NULL_COLOR, alpha=0.55,
            edgecolor=_NULL_EDGE, linewidth=0.4,
            label=legend_text(spec, "shuffled", fill),
            zorder=1,
        )
    ax.hist(
        scores, bins=bins,
        color=P["fit_color"],
        alpha=0.55, edgecolor=_OBS_EDGE, linewidth=0.6,
        label=legend_text(spec, "observed", {**fill, "n": int(scores.size)}),
        zorder=2,
    )
    ax.set_xticks(np.arange(0, max_v + 1))
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])
    if null_dist is not None and np.size(null_dist):
        ax.legend(fontsize=P["legend_fontsize"], loc="best");


def render_anticipation_zscore_histogram(ax, payload, spec, *, fill):
    """Real (blue) vs shuffled (gray) rest-region z-scores for one train.

    Port of ``_plot_anticipation_zscore_histogram``: shared bins from the
    pooled finite values; dashed vertical lines mark the two means.
    """
    P = PLOT_PARAMS
    clean_axes(ax)
    real = np.asarray(payload["real"])
    shuffled = np.asarray(payload["shuffled"])

    pooled = np.concatenate([real, shuffled])
    finite = pooled[np.isfinite(pooled)]
    if finite.size:
        bins = np.histogram_bin_edges(finite, bins=40)
    else:
        bins = 40
    ax.hist(
        shuffled, bins=bins, color=_NULL_COLOR, alpha=0.55,
        edgecolor=_NULL_EDGE, linewidth=0.4,
        label=legend_text(spec, "shuffled", fill),
        zorder=1,
    )
    ax.hist(
        real, bins=bins, color=P["fit_color"], alpha=0.6,
        edgecolor=_OBS_EDGE, linewidth=0.6,
        label=legend_text(spec, "observed", {**fill, "n": int(real.size)}),
        zorder=2,
    )
    mean_real = float(np.nanmean(real))
    mean_shuf = float(np.nanmean(shuffled))
    ax.axvline(
        mean_real, color=P["fit_color"],
        linewidth=1.8, linestyle="--", zorder=3,
    )
    ax.axvline(
        mean_shuf, color=_NULL_EDGE,
        linewidth=1.4, linestyle="--", zorder=3,
    )
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])
    ax.legend(fontsize=P["legend_fontsize"], loc="best");


def render_permutation_mean_test(ax, payload, spec, *, fill):
    """Histogram of shuffled mean scores with the observed mean overlaid.

    Port of ``_plot_permutation_mean_test``. The observed mean and the
    two-tailed permutation p are read from the cached population test
    (``payload["pop"]``, from ``common.stats.population_permutation_pvalue``);
    the histogram shows the per-permutation means of the cached ``null``. The
    two number-bearing legend labels are filled from spec templates.
    """
    P = PLOT_PARAMS
    clean_axes(ax)
    null_mat = np.asarray(payload["null"])
    pop = payload["pop"]

    M_real, p_value, n_perm = pop["obs_stat"], pop["p_value"], pop["n_perm"]
    M_shuffled = np.nanmean(null_mat, axis=1)
    p_disp = (f"< {1.0 / n_perm:.0e}" if p_value == 0.0
              else f"= {p_value:.4g}")

    ax.hist(
        M_shuffled, bins=40,
        color=_NULL_COLOR, alpha=0.65,
        edgecolor=_NULL_EDGE, linewidth=0.4,
        label=legend_text(spec, "shuffled", {**fill, "n_perm": n_perm}),
        zorder=1,
    )
    ax.axvline(
        M_real, color=P["fit_color"], linewidth=2.4,
        label=legend_text(
            spec, "observed",
            {**fill, "m_real": M_real, "p_disp": p_disp},
        ),
        zorder=3,
    )
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])
    ax.legend(fontsize=P["legend_fontsize"], loc="best");


# (measure_key, label_word) for the two running-extremum learning scores.
_MEASURES = (
    ("habituation", "Habituation"),
    ("sensitization", "Sensitization"),
)


def iter_figures(blob, exp_name):
    """Yield ``(spec_key, payload, fill)`` for every learning-score figure.

    Skips metrics/trains that are ``None`` (a channel with no scoreable data).
    """
    data = blob["data"]

    for metric in ("height", "width"):
        mblob = data.get(metric)
        if mblob is None:
            continue
        # Height carries the bare titles/labels requested for the preprint;
        # width keeps a suffix so the two metrics stay distinct.
        metric_suffix = "" if metric == "height" else f" ({metric})"
        for measure_key, label_word in _MEASURES:
            summed = mblob[measure_key]
            null = mblob.get(f"{measure_key}_null")
            fill = {
                "exp_name": exp_name,
                "measure_key": measure_key,
                "metric": metric,
                "label_word": label_word,
                "label_word_lower": label_word.lower(),
                "metric_suffix": metric_suffix,
            }
            yield (
                "learning_score_hist",
                {"scores": summed, "null": null},
                dict(fill),
            )
            if null is not None and np.size(null):
                yield (
                    "learning_score_permtest",
                    {"observed": summed, "null": null,
                     "pop": mblob[f"{measure_key}_pop"]},
                    dict(fill),
                )

    ablob = data.get("anticipation")
    if ablob is None or not ablob.get("trains"):
        return
    for train_idx in (1, 2):
        train_blob = ablob["trains"].get(train_idx)
        if train_blob is None:
            continue
        fill = {"exp_name": exp_name, "train_idx": train_idx}
        yield (
            "learning_anticipation_hist",
            {"real": train_blob["real"], "shuffled": train_blob["shuffled"]},
            dict(fill),
        )
        null = train_blob.get("null")
        if null is not None and np.size(null):
            yield (
                "learning_anticipation_permtest",
                {"observed": train_blob["real"], "null": null,
                 "pop": train_blob["pop"]},
                dict(fill),
            )
