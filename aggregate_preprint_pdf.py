#!/usr/bin/env python3
"""
Aggregate every figure under ``April28_preprint_results/<experiment>/`` into a
single, organized multi-page PDF.

Layout
------
- One cover page per experiment.
- Per-experiment "overview" section: corrected_traces, corr_vs_dist,
  learning scores.
- Per-channel section: dff, pca_umap_uncolored, hw_lum_log.

Channels are sorted naturally — c2c12 uses ``channel_<n>_*.png``; nrk uses
``channel_<n>_<letter>_*.png`` (letter encodes the stimulus). Any unrecognised
PNG in a subfolder is appended at the end of its experiment so nothing is lost.

Run from the project root:
    python aggregate_preprint_pdf.py
"""

import matplotlib
matplotlib.use("Agg")

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


RESULTS_DIR = Path("April28_preprint_results")
OUTPUT_PDF = RESULTS_DIR / "April28_preprint_figures.pdf"
SKIP_DIRS = {"bg_cache", "archive", "state_cache"}

OVERVIEW_ORDER = [
    "corrected_traces",
    "dff_mean_combined",
    "corr_vs_dist",
    "corr_vs_dist_combined",
    "learning_habituation_height",
    "learning_habituation_height_per_train",
    "learning_habituation_height_split",
    "learning_habituation_width",
    "learning_habituation_width_per_train",
    "learning_habituation_width_split",
    "learning_sensitization_height",
    "learning_sensitization_height_per_train",
    "learning_sensitization_height_split",
    "learning_sensitization_width",
    "learning_sensitization_width_per_train",
    "learning_sensitization_width_split",
    "learning_anticipation_height",
    "learning_anticipation_height_per_train",
    "learning_anticipation_height_split",
    "learning_anticipation_width",
    "learning_anticipation_width_per_train",
    "learning_anticipation_width_split",
]
PER_CHANNEL_ORDER = [
    "dff",
    "dff_responders",
    "dff_non_responders",
    "dff_response_breakdown",
    "response_violin",
    "response_violin_responders",
    "response_violin_width",
    "response_violin_width_baselines",
    "amplitude_width_scatter_lum",
    "amplitude_width_scatter_dff",
    "pca_umap_uncolored",
    "hw_lum_log",
]

CATEGORY_LABELS = {
    "corrected_traces": "Corrected per-cell traces",
    "dff_mean_combined": "Mean dF/F0 — all channels combined",
    "corr_vs_dist": "Pairwise correlation vs. distance",
    "corr_vs_dist_combined": "Pairwise correlation vs. distance (combined)",
    "dff": "dF/F0 normalized traces",
    "dff_responders": "dF/F0 normalized traces — responders only",
    "dff_non_responders": "dF/F0 normalized traces — non-responders only",
    "dff_response_breakdown": "dF/F0 response breakdown (mean/median/percentile + per-cell peak histogram)",
    "response_violin": "Per-stimulus Δ luminosity (peak − baseline)",
    "response_violin_responders": "Per-stimulus Δ luminosity — responders only (|peak Δ dF/F0| ≥ 0.10)",
    "response_violin_width": "Per-stimulus response width (frame-of-stim baseline)",
    "response_violin_width_baselines": "Per-stimulus response width — baseline-mode comparison",
    "amplitude_width_scatter_lum": "Amplitude × width per stim (raw luminosity)",
    "amplitude_width_scatter_dff": "Amplitude × width per stim (dF/F0)",
    "pca_umap_uncolored": "PCA + UMAP scatter (no clustering)",
    "hw_lum_log": "Hardware feedback luminosity (log)",
}

PLOT_PARAMS = {
    "page_size": (8.5, 11.0),
    "cover_title_size": 26,
    "cover_subtitle_size": 14,
    "section_title_size": 22,
    "section_subtitle_size": 13,
    "page_title_size": 13,
    "page_subtitle_size": 10,
}


CHANNEL_RE = re.compile(r"^channel_(\d+)(?:_([A-Za-z]))?_(.+)$")


def classify_png(path: Path):
    """Return (kind, channel_key, category) for a PNG.

    kind is "overview" or "channel" or "other".
    channel_key is e.g. "1" or "1_A" (only for channel kind).
    category is the trailing suffix, e.g. "dff", "bg_diagnostic".
    """
    stem = path.stem
    if stem in OVERVIEW_ORDER:
        return ("overview", None, stem)
    m = CHANNEL_RE.match(stem)
    if m:
        ch_num, ch_letter, cat = m.group(1), m.group(2), m.group(3)
        ch_key = f"{ch_num}_{ch_letter}" if ch_letter else ch_num
        return ("channel", ch_key, cat)
    return ("other", None, stem)


def channel_sort_key(ch_key: str):
    parts = ch_key.split("_")
    num = int(parts[0])
    letter = parts[1] if len(parts) > 1 else ""
    return (num, letter)


def category_sort_key(cat: str, order_list):
    return order_list.index(cat) if cat in order_list else len(order_list)


def add_text_page(pdf, title, subtitle=None, title_size=26, subtitle_size=14):
    fig, ax = plt.subplots(figsize=PLOT_PARAMS["page_size"])
    ax.axis("off")
    ax.text(
        0.5, 0.6, title,
        ha="center", va="center",
        fontsize=title_size, fontweight="bold",
        transform=ax.transAxes,
    )
    if subtitle:
        ax.text(
            0.5, 0.5, subtitle,
            ha="center", va="center",
            fontsize=subtitle_size,
            transform=ax.transAxes,
        )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf, image_path: Path, title: str, subtitle: str = None):
    img = Image.open(image_path)
    w, h = img.size
    aspect = w / h
    page_w, page_h = PLOT_PARAMS["page_size"]
    # Reserve top strip for title; layout figure proportional to image.
    fig_w = page_w
    fig_h = min(page_h - 1.2, fig_w / aspect + 0.8)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
    ax.imshow(img)
    ax.axis("off")
    fig.suptitle(
        title,
        fontsize=PLOT_PARAMS["page_title_size"],
        fontweight="bold",
        y=0.985,
    )
    if subtitle:
        fig.text(
            0.5, 0.945, subtitle,
            ha="center", va="top",
            fontsize=PLOT_PARAMS["page_subtitle_size"],
        )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def collect_experiment(exp_dir: Path):
    """Group PNGs in an experiment dir into overview + per-channel buckets."""
    overview = {}
    by_channel = {}
    other = []
    for png in sorted(exp_dir.glob("*.png")):
        kind, ch_key, cat = classify_png(png)
        if kind == "overview":
            overview[cat] = png
        elif kind == "channel":
            by_channel.setdefault(ch_key, {})[cat] = png
        else:
            other.append(png)
    return overview, by_channel, other


def build_pdf():
    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"Missing {RESULTS_DIR.resolve()}")

    experiments = sorted(
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    )
    if not experiments:
        raise SystemExit(f"No experiment subfolders under {RESULTS_DIR}")

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PDF) as pdf:
        # Document cover.
        add_text_page(
            pdf,
            "April 28 Preprint Figures",
            subtitle=(
                f"Aggregated from {len(experiments)} experiment(s): "
                + ", ".join(e.name for e in experiments)
            ),
            title_size=PLOT_PARAMS["cover_title_size"],
            subtitle_size=PLOT_PARAMS["cover_subtitle_size"],
        )

        for exp in experiments:
            overview, by_channel, other = collect_experiment(exp)
            n_pages = (
                len(overview) + sum(len(v) for v in by_channel.values()) + len(other)
            )
            print(f"[{exp.name}] {n_pages} figure pages")

            add_text_page(
                pdf,
                exp.name,
                subtitle=f"{n_pages} figures",
                title_size=PLOT_PARAMS["section_title_size"],
                subtitle_size=PLOT_PARAMS["section_subtitle_size"],
            )

            # Overview section.
            ordered_overview = sorted(
                overview.items(),
                key=lambda kv: category_sort_key(kv[0], OVERVIEW_ORDER),
            )
            for cat, png in ordered_overview:
                add_image_page(
                    pdf, png,
                    title=f"{exp.name} — {CATEGORY_LABELS.get(cat, cat)}",
                    subtitle=png.name,
                )

            # Per-channel sections.
            for ch_key in sorted(by_channel.keys(), key=channel_sort_key):
                cats = by_channel[ch_key]
                ordered_cats = sorted(
                    cats.items(),
                    key=lambda kv: category_sort_key(kv[0], PER_CHANNEL_ORDER),
                )
                for cat, png in ordered_cats:
                    add_image_page(
                        pdf, png,
                        title=(
                            f"{exp.name} — channel {ch_key.replace('_', ' / ')}"
                            f" — {CATEGORY_LABELS.get(cat, cat)}"
                        ),
                        subtitle=png.name,
                    )

            # Anything that didn't match either pattern.
            for png in sorted(other):
                add_image_page(
                    pdf, png,
                    title=f"{exp.name} — {png.stem}",
                    subtitle=png.name,
                )

    print(f"\nWrote {OUTPUT_PDF.resolve()}")


if __name__ == "__main__":
    build_pdf()
