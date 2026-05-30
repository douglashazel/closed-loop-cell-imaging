"""Render the four responder-call QC diagnostics from the analysis cache.

These four figures are the DOCUMENTED multi-panel EXCEPTION to the
one-figure-one-axes rule: each render function receives the whole ``Figure``
and builds its own gridspec exactly as the original
``responder_diagnostic.py``'s ``_draw_*_figure`` did (preserving
``fig.set_layout_engine("none")`` + manual gridspec margins). They are ported
verbatim, reading every array from the cached ``panels`` dict instead of
recomputing — in particular the image-sharpness ``focus_*`` arrays are read
from cache, so NO frame image is opened here (that compute moved to
``analyze_responder_diagnostic.py``).

``render_distribution``  -> ``responder_distribution_diagnostic``
``render_stimlock``      -> ``responder_stimlock_diagnostic``
``render_artifact``      -> ``responder_artifact_diagnostic``
``render_f0``            -> ``responder_f0_diagnostic``
"""
import numpy as np

from style import PLOT_PARAMS as _STYLE_PARAMS

NAME = "responder_diagnostic"

# Base font sizes come from the single locked style source; the diagnostic
# colour palette below is plotting-only (it lived as a LOCAL PLOT_PARAMS in the
# original combined script and has no place in the analysis layer).
PLOT_PARAMS = {
    "width_full": _STYLE_PARAMS["width_full"],
    "dpi": _STYLE_PARAMS["dpi"],
    # Font sizes governed by the single source (style) so these diagnostic
    # panels match the locked preprint style used everywhere else.
    "title_fontsize": _STYLE_PARAMS["title_fontsize"],
    "title_fontweight": _STYLE_PARAMS["title_fontweight"],
    "suptitle_fontsize": _STYLE_PARAMS["suptitle_fontsize"],
    "panel_fontsize": _STYLE_PARAMS["panel_label_size"],
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

BASELINE_N_PRE = 5
RNG_SEED = 42

# Per-figure figsize HEIGHTS scale with channel count; only the distribution
# figure has a fixed height. Widths stay locked at the full text width (6.5 in).
DIST_HEIGHT = 8.4


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


# =============================================================================
# Render functions — each receives the whole FIGURE (multi-panel exception) and
# owns its gridspec/suptitle, ported verbatim from the source _draw_* functions.
# `payload` is {"channels": [...], "panels": {ch: {...}}}; `fill` carries
# exp_name (+ investigation_summary for the stimlock footer).
# =============================================================================
def render_distribution(fig, payload, spec, *, fill):
    """Marginal histogram + jittered strip scatter of per-cell Δ dF/F0."""
    exp_name = fill["exp_name"]
    channels = payload["channels"]
    panels = payload["panels"]
    n = len(channels)
    # This figure positions its axes with explicit gridspec margins below, so
    # it opts OUT of the globally-enabled constrained layout (the two would
    # fight and collapse the axes). A no-op layout engine is required here:
    # constrained_layout=False alone gets silently re-applied by savefig, so we
    # install PlaceHolderLayoutEngine which survives the save. Width stays 6.5 in.
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


def render_stimlock(fig, payload, spec, *, fill):
    """Stim-aligned population trace + per-stim deltas + investigation-summary panel."""
    exp_name = fill["exp_name"]
    channels = payload["channels"]
    panels = payload["panels"]
    investigation_summary = fill["investigation_summary"]
    n = len(channels)
    # Tall per-channel stack with manual gridspec margins -> no-op layout engine
    # (constrained would collapse the rows) and let height grow with channel
    # count; width stays locked at 6.5 in. Capping the height collapses the rows.
    height = 4.2 * n + 3.0
    fig.set_size_inches(PLOT_PARAMS["width_full"], height)
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
    ax_c.text(0.0, 0.84, "\n".join(investigation_summary),
              fontsize=PLOT_PARAMS["panel_fontsize"], va="top", family="monospace")

    fig.suptitle(
        f"{exp_name} — stimulus-locked artifact check\n"
        "if the real-stim trace steps up where the pseudo-stim trace stays flat, "
        "the responder rate is inflated by a field-wide nuisance",
        fontsize=PLOT_PARAMS["suptitle_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"], y=1.0 - 0.30 / height);


def render_artifact(fig, payload, spec, *, fill):
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
    exp_name = fill["exp_name"]
    channels = payload["channels"]
    panels = payload["panels"]
    n = len(channels)
    # Tallest diagnostic (2 rows per channel) with manual gridspec margins ->
    # no-op layout engine; height grows with channel count (width 6.5 in).
    height = 7.0 * n + 4.2
    fig.set_size_inches(PLOT_PARAMS["width_full"], height)
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


def render_f0(fig, payload, spec, *, fill):
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
    exp_name = fill["exp_name"]
    channels = payload["channels"]
    panels = payload["panels"]
    n = len(channels)
    # Per-channel stack with manual gridspec margins -> no-op layout engine;
    # height grows with channel count (width stays locked at 6.5 in).
    height = 4.7 * n + 3.0
    fig.set_size_inches(PLOT_PARAMS["width_full"], height)
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


def iter_figures(blob, exp_name):
    """Yield ``(spec_key, payload, fill)`` for the 4 responder diagnostics.

    ``payload`` is the whole cached ``{channels, panels}`` slice (the render
    fns own their gridspec and read every panel directly). ``fill`` carries the
    exp_name plus the static investigation-summary text the stimlock footer
    prints; ``exp_name`` also selects the output directory.
    """
    data = blob["data"]
    meta = blob["meta"]
    payload = {"channels": data["channels"], "panels": data["panels"]}
    fill = {
        "exp_name": exp_name,
        "investigation_summary": meta["investigation_summary"],
    }
    for spec_key in ("responder_distribution_diagnostic",
                     "responder_stimlock_diagnostic",
                     "responder_artifact_diagnostic",
                     "responder_f0_diagnostic"):
        yield (spec_key, payload, dict(fill))
