"""Render functions for the dF/F0 figures (reads analysis_cache/<exp>/dff.pkl).

The old 2-row ``<ch>_dff`` (corrected + dF/F0, shared x) is decomposed into two
standalone single-axis figures — ``{ch}_dff_raw`` and ``{ch}_dff_norm`` — each
in three cell subsets (all / responders / non-responders). Pooled figures are
unchanged single-axis plots. Label/title/legend text lives centrally in
``figures_spec.py``; these functions pull it from the passed ``spec``.
"""
import numpy as np

from plots._base import (
    PLOT_PARAMS,
    clean_axes,
    draw_stim_spans,
    legend_text,
    title_of,
    xlabel_of,
    ylabel_of,
)

NAME = "dff"

# (filename suffix, subset key) for the three cell subsets of each trace figure.
SUBSETS = [("", None), ("_responders", "responders"),
           ("_non_responders", "non_responders")]


def render_dff_trace(ax, payload, spec, *, fill):
    """One trace-stack panel: every cell faint + the mean, with stim spans.

    Serves both the corrected-fluorescence and the dF/F0 figures — ``payload``
    supplies the matrix (``mat``); ``spec`` supplies the title/ylabel/legend.
    """
    P = PLOT_PARAMS
    clean_axes(ax)
    fmin = payload["frame_min"]
    data = payload["mat"]
    mean_trace = np.nanmean(data, axis=0)
    for row in data:
        ax.plot(fmin, row, color=P["cell_color"], alpha=P["cell_alpha"],
                linewidth=P["cell_lw"], zorder=1)
    ax.plot(fmin, mean_trace, color=P["mean_color"], linewidth=P["mean_lw"],
            zorder=3, label=legend_text(spec, "mean", fill))
    draw_stim_spans(ax, payload["spans"], payload["stim_label"],
                    P["stim_color"], alpha=0.18)
    rsp = payload.get("real_setpoint_min")
    if rsp is not None:
        ax.axvline(rsp, color="#000000", linewidth=2.0, linestyle=":",
                   alpha=0.9, zorder=5, label=legend_text(spec, "setpoint", fill))
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])
    ax.legend(fontsize=P["legend_fontsize_large"], loc="upper right");


def render_dff_mean_pooled(ax, payload, spec, *, fill):
    """Pooled responder dF/F0 mean ± 1 SEM."""
    P = PLOT_PARAMS
    clean_axes(ax)
    fmin = payload["frame_min"]
    mean_trace = payload["mean_responders"]
    sem = payload["sem_responders"]
    ax.fill_between(fmin, mean_trace - sem, mean_trace + sem,
                    color=P["pooled_sem_color"], alpha=0.18, linewidth=0,
                    label=legend_text(spec, "sem", fill))
    ax.plot(fmin, mean_trace, color=P["pooled_mean_color"], linewidth=P["mean_lw"],
            label=legend_text(spec, "mean", fill), zorder=3)
    if payload["stim_aligned"]:
        draw_stim_spans(ax, payload["spans"], payload["stim_label"],
                        P["stim_color"], alpha=0.18)
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])
    ax.legend(fontsize=P["legend_fontsize"], loc="best");


def render_dff_pooled_traces(ax, payload, spec, *, fill):
    """Every dF/F0 trace pooled across channels + the pooled mean (all cells)."""
    P = PLOT_PARAMS
    clean_axes(ax)
    fmin = payload["frame_min"]
    pooled = payload["pooled_dff_all"]
    mean_trace = payload["mean_all"]
    for row in pooled:
        ax.plot(fmin, row, color=P["cell_color"], alpha=P["cell_alpha"],
                linewidth=P["cell_lw"], zorder=1)
    ax.plot(fmin, mean_trace, color=P["pooled_mean_color"], linewidth=P["mean_lw"],
            label=legend_text(spec, "mean", fill), zorder=3)
    if payload["stim_aligned"]:
        draw_stim_spans(ax, payload["spans"], payload["stim_label"],
                        P["stim_color"], alpha=0.18)
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])
    ax.legend(fontsize=P["legend_fontsize"], loc="best");


def iter_figures(blob, exp_name):
    """Yield ``(spec_key, payload, fill)`` for every dff figure of one exp."""
    data = blob["data"]
    cell_line = blob["meta"]["cell_line"]

    for ch, d in data["per_channel"].items():
        n_all = d["n_all"]
        rsp = d["real_setpoint_min"]
        common = {"frame_min": d["frame_min"], "spans": d["spans"],
                  "stim_label": d["stim_label"], "real_setpoint_min": rsp}
        for suffix, subset in SUBSETS:
            if subset is None:
                sel = slice(None)
                cell_str = f"{n_all} cells"
            else:
                mask = d["mask"] if subset == "responders" else ~d["mask"]
                if not mask.any():
                    continue
                sel = mask
                word = "responder" if subset == "responders" else "non-responder"
                cell_str = f"{int(mask.sum())}/{n_all} {word} cells"
            fill = {"exp_name": exp_name, "ch": ch, "subset_suffix": suffix,
                    "cell_str": cell_str, "n_stims": d["n_stims"],
                    "rsp": (0.0 if rsp is None else rsp)}
            yield ("dff_raw", {**common, "mat": d["raw_mat"][sel]}, dict(fill))
            yield ("dff_norm", {**common, "mat": d["dff_mat"][sel]},
                   dict(fill, f0_note=d["f0_note"]))

    pooled = data["pooled"]
    if pooled is not None:
        if pooled["mean_responders"] is not None:
            yield ("dff_mean_pooled_responders", pooled,
                   {"exp_name": exp_name, "cell_line": cell_line,
                    "n_total": pooled["n_total_responders"]})
        per_ch_str = ", ".join(f"{ch}: {n}" for ch, n in pooled["per_channel_counts"])
        yield ("dff_pooled_traces", pooled,
               {"exp_name": exp_name, "per_ch_str": per_ch_str,
                "n_total": int(pooled["pooled_dff_all"].shape[0])})
