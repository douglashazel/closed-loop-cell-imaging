"""Render the NRK hardware-feedback luminosity log (reads
``analysis_cache/<exp>/nrk_hardware_log.pkl``).

One standalone single-axis figure per channel of the NRK acid experiment
(``<ch>_hw_lum_log``): the setpoint bands (one legend entry per unique setpoint),
the mean-fluorescence line, the acidic-pulse axvspans (a single legend entry),
and the optional dotted real-setpoint vline. Styling comes from
``PLOT_PARAMS_HW_LOG``; label/title/legend text comes from the passed ``spec``.
"""
from plots._base import (
    clean_axes,
    legend_text,
    title_of,
    xlabel_of,
    ylabel_of,
)
from style import PLOT_PARAMS_HW_LOG

NAME = "nrk_hardware_log"


def render_nrk_hardware_log(ax, payload, spec, *, fill):
    """Draw one channel's hardware-feedback luminosity log onto ``ax``."""
    pp = PLOT_PARAMS_HW_LOG
    clean_axes(ax)

    # --- setpoint bands: one legend entry per unique setpoint value ---
    seen_sp = set()
    for region in payload["setpoint_regions_min"]:
        idx = region["idx"]
        sp = region["setpoint"]
        start_m, end_m = region["start_min"], region["end_min"]
        color = pp["setpoint_colors"][idx % len(pp["setpoint_colors"])]
        label = (
            legend_text(spec, "setpoint_band", dict(fill, sp=sp))
            if sp not in seen_sp
            else None
        )
        seen_sp.add(sp)
        ax.fill_between(
            [start_m, end_m], 0, sp,
            color=color, alpha=pp["setpoint_alpha"],
            label=label, zorder=1,
        )
        ax.hlines(
            sp, xmin=start_m, xmax=end_m,
            colors=color, linewidths=pp["setpoint_lw"], zorder=2,
        )

    # --- mean-fluorescence trace ---
    ax.plot(
        payload["frames_min"], payload["luminosity"],
        color=pp["line_color"], linewidth=pp["line_lw"],
        label=legend_text(spec, "mean", fill), zorder=3,
    )

    # --- acidic-pulse spans: single legend entry on the first span ---
    pulse_duration = payload["pulse_duration"]
    secs = int(pulse_duration * 60)
    acid_label = legend_text(spec, "acid", dict(fill, secs=secs))
    for i, m in enumerate(payload["acid_min"]):
        ax.axvspan(
            m, m + pulse_duration,
            color=pp["acid_color"], alpha=0.30,
            linewidth=0, zorder=2,
            label=acid_label if i == 0 else None,
        )

    # --- optional real-setpoint marker ---
    rsp = payload["real_setpoint_min"]
    if rsp is not None:
        ax.axvline(
            rsp,
            color="#000000",
            linewidth=2.0, linestyle=":",
            alpha=0.9, zorder=5,
            label=legend_text(spec, "real_setpoint", dict(fill, rsp=rsp)),
        )

    ax.set_ylim(*payload["y_lim"])
    ax.set_xlim(*payload["x_lim"])

    ax.set_title(title_of(spec, fill), fontsize=pp["title_fontsize"],
                 fontweight=pp["title_fontweight"])
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=pp["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=pp["axis_label_fontsize"])
    ax.legend(loc="upper left", fontsize=pp["legend_fontsize"]);


def iter_figures(blob, exp_name):
    """Yield ``(spec_key, payload, fill)`` for every hw-log figure of one exp."""
    data = blob["data"]
    for ch, d in data["per_channel"].items():
        fill = {"exp_name": exp_name, "ch": ch, "chamber": d["chamber"]}
        yield ("nrk_hardware_log", d, fill)
