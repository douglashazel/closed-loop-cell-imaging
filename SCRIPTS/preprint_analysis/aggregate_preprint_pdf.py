#!/usr/bin/env python3
"""
Aggregate every figure under ``May29_preprint_figures/<experiment>/`` into a
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


RESULTS_DIR = Path("May29_preprint_figures")
OUTPUT_PDF = RESULTS_DIR / "May29_preprint_figures.pdf"
# Skip non-experiment subfolders (caches, logs, sibling SVG archive). The
# cross-experiment "dmso_stim8_comparison" bucket is intentionally NOT skipped.
SKIP_DIRS = {"bg_cache", "analysis_cache", "run_logs", "svg",
             "archive", "state_cache"}

# Stem ordering after the analysis/plotting split decomposed the multi-panel
# figures into standalone single-axis PNGs. Unknown stems still append
# gracefully (category_sort_key sends them to the end).
OVERVIEW_ORDER = [
    "dff_pooled_traces",
    "dff_mean_pooled_responders",
    "pooled_pca_only",
    "pooled_umap_only",
    "average_peak",
    "average_peak_responders",
    "average_peak_responders_stim8",
    "average_peak_responders_combined",
    "average_peak_responders_stim8_combined",
    "corr_vs_dist_combined_pearson",
    "corr_vs_dist_combined_pearson_log1p",
    "corr_vs_dist_combined_spearman",
    "corr_vs_dist_combined_spearman_log1p",
    "pooled_response_violin_height_dff",
    "pooled_response_violin_height_dff_responders",
    "pooled_response_violin_height_dff_train_means",
    "pooled_response_violin_width_dff",
    "pooled_response_violin_width_dff_responders",
    "pooled_response_violin_width_dff_train_means",
    "learning_habituation_height",
    "learning_habituation_height_permtest",
    "learning_habituation_width",
    "learning_habituation_width_permtest",
    "learning_sensitization_height",
    "learning_sensitization_height_permtest",
    "learning_sensitization_width",
    "learning_sensitization_width_permtest",
    "learning_anticipation_train1",
    "learning_anticipation_train1_permtest",
    "learning_anticipation_train2",
    "learning_anticipation_train2_permtest",
    "responder_distribution_diagnostic",
    "responder_stimlock_diagnostic",
    "responder_artifact_diagnostic",
    "responder_f0_diagnostic",
]
PER_CHANNEL_ORDER = [
    "dff_raw",
    "dff_norm",
    "dff_raw_responders",
    "dff_norm_responders",
    "dff_raw_non_responders",
    "dff_norm_non_responders",
    "corr_vs_dist_pearson",
    "corr_vs_dist_pearson_log1p",
    "corr_vs_dist_spearman",
    "corr_vs_dist_spearman_log1p",
    "hw_lum_log",
]

CATEGORY_LABELS = {
    "dff_pooled_traces": "dF/F0 traces pooled across channels (all cells) + mean",
    "dff_mean_pooled_responders": "Mean dF/F0 — responders, pooled across channels",
    "pooled_pca_only": "PCA scatter (PC1 vs PC2, no clustering)",
    "pooled_umap_only": "UMAP embedding (no clustering)",
    "average_peak": "Average response peak — per-stimulus dF/F0 segments + mean",
    "average_peak_responders": (
        "Average response peak — responders only (per-stimulus dF/F0 segments + mean)"
    ),
    "average_peak_responders_stim8": (
        "Average response peak — stimulus #8 only, responders"
    ),
    "average_peak_responders_combined": (
        "Average response peak — responders, PC3 vs C2C12 (combined)"
    ),
    "average_peak_responders_stim8_combined": (
        "Average response peak — stimulus #8, PC3 vs C2C12 (combined)"
    ),
    "corr_vs_dist_combined_pearson": "Pairwise Pearson r vs. distance (channels combined)",
    "corr_vs_dist_combined_spearman": "Pairwise Spearman ρ vs. distance (channels combined)",
    "corr_vs_dist_combined_pearson_log1p": "Pairwise Pearson r vs. distance (combined, log1p axis)",
    "corr_vs_dist_combined_spearman_log1p": "Pairwise Spearman ρ vs. distance (combined, log1p axis)",
    "pooled_response_violin_height_dff": "Per-stimulus response height (dF/F0)",
    "pooled_response_violin_height_dff_responders": "Per-stimulus response height — responders highlighted",
    "pooled_response_violin_height_dff_train_means": "Per-replicate train mean (height)",
    "pooled_response_violin_width_dff": "Per-stimulus response width (dF/F0)",
    "pooled_response_violin_width_dff_responders": "Per-stimulus response width — responders highlighted",
    "pooled_response_violin_width_dff_train_means": "Per-replicate train mean (width)",
    "dff_raw": "Corrected per-cell traces",
    "dff_norm": "dF/F0 normalized traces",
    "dff_raw_responders": "Corrected per-cell traces — responders only",
    "dff_norm_responders": "dF/F0 normalized traces — responders only",
    "dff_raw_non_responders": "Corrected per-cell traces — non-responders only",
    "dff_norm_non_responders": "dF/F0 normalized traces — non-responders only",
    "corr_vs_dist_pearson": "Pairwise Pearson r vs. distance",
    "corr_vs_dist_spearman": "Pairwise Spearman ρ vs. distance",
    "corr_vs_dist_pearson_log1p": "Pairwise Pearson r vs. distance (log1p axis)",
    "corr_vs_dist_spearman_log1p": "Pairwise Spearman ρ vs. distance (log1p axis)",
    "hw_lum_log": "Hardware feedback fluorescence (log)",
    "learning_habituation_height_permtest": (
        "Habituation permutation test (height) — observed vs. shuffled mean score"
    ),
    "learning_habituation_width_permtest": (
        "Habituation permutation test (width) — observed vs. shuffled mean score"
    ),
    "learning_sensitization_height_permtest": (
        "Sensitization permutation test (height) — observed vs. shuffled mean score"
    ),
    "learning_sensitization_width_permtest": (
        "Sensitization permutation test (width) — observed vs. shuffled mean score"
    ),
    "learning_anticipation_train1_permtest": (
        "Anticipation permutation test — train 1 (observed vs. shuffled mean z)"
    ),
    "learning_anticipation_train2_permtest": (
        "Anticipation permutation test — train 2 (observed vs. shuffled mean z)"
    ),
    "responder_distribution_diagnostic": (
        "Responder diagnostic — per-cell Δ dF/F0 distribution vs. threshold"
    ),
    "responder_stimlock_diagnostic": (
        "Responder diagnostic — stimulus-locked artifact check + open questions"
    ),
    "responder_artifact_diagnostic": (
        "Responder diagnostic — dead-frame proximity + perfusion/optical artifact"
    ),
    "responder_f0_diagnostic": (
        "Responder diagnostic — F0 dependence (1/F0 normalization check)"
    ),
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
            "Preprint Figures",
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
