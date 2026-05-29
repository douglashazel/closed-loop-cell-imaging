"""
Template for a multi-panel preprint figure (panels a, b, c, d).

Two routes, one shared style:
  - All-plot figures: build the whole figure here with subplot_mosaic, label
    the panels, and save once. Fully reproducible; resizing = edit figsize
    and rerun.
  - Mixed figures (plots + micrographs/schematics): either drop the image into
    an axes with imshow so it stays one reproducible file (panel c below), or
    export each plot panel with make_single_panel_for_inkscape() at its placed
    size and assemble it with the images in Inkscape.

Run locally:  conda activate DOUG  &&  python preprint_figures/figure_template.py
(or from inside preprint_figures/:  python figure_template.py)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the shared style the same way the analysis scripts do: the directory
# holding figstyle.py (this file's directory) is put on sys.path so
# ``from figstyle import ...`` resolves regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import PLOT_PARAMS, apply_style, save_figure, add_panel_label  # noqa: E402

apply_style()
rng = np.random.default_rng(0)


def make_figure_1():
    # Build at the final on-page width. A 2x2 layout at full width -> 6.5 x 5.0 in.
    fig, axd = plt.subplot_mosaic(
        [['a', 'b'],
         ['c', 'd']],
        figsize=(PLOT_PARAMS['width_full'], 5.0),
        layout='constrained',
    )
    c = PLOT_PARAMS['colors']

    # a: line plot (e.g. train / val curves)
    x = np.arange(1, 51)
    axd['a'].plot(x, 1.0 / np.sqrt(x), color=c[0], label='train')
    axd['a'].plot(x, 1.0 / np.sqrt(x) + 0.05, color=c[1], label='val')
    axd['a'].set_xlabel('epoch')
    axd['a'].set_ylabel('loss')
    axd['a'].legend(frameon=False);

    # b: hexbin (dense-scatter alternative)
    xb = rng.normal(0, 1, 5000)
    yb = xb * 0.6 + rng.normal(0, 0.8, 5000)
    hb = axd['b'].hexbin(xb, yb, gridsize=30, cmap='viridis')
    axd['b'].set_xlabel('feature 1')
    axd['b'].set_ylabel('feature 2')
    fig.colorbar(hb, ax=axd['b'], label='count')

    # c: a micrograph / image panel embedded with imshow (this covers MIXED figures)
    img = rng.random((100, 100))
    im = axd['c'].imshow(img, cmap='magma')
    axd['c'].set_xticks([]); axd['c'].set_yticks([])
    axd['c'].set_xlabel('100 \u00b5m')
    fig.colorbar(im, ax=axd['c'], label='intensity')

    # d: violinplot with jittered points
    groups = [rng.normal(m, s, 150) for m, s in [(0, 1), (1.5, 1.2), (3, 0.8)]]
    parts = axd['d'].violinplot(groups, positions=[1, 2, 3])
    for i, (pc, g) in enumerate(zip(parts['bodies'], groups)):
        pc.set_facecolor(c[i]); pc.set_alpha(0.8)
        jit = rng.uniform(-PLOT_PARAMS['jitter_strength'],
                          PLOT_PARAMS['jitter_strength'], len(g))
        axd['d'].scatter(np.full(len(g), i + 1) + jit, g,
                         color=PLOT_PARAMS['scatter_color'],
                         alpha=PLOT_PARAMS['scatter_alpha'],
                         s=PLOT_PARAMS['scatter_size'], zorder=2)
    axd['d'].set_xticks([1, 2, 3]); axd['d'].set_xticklabels(['A', 'B', 'C'])
    axd['d'].set_xlabel('group')
    axd['d'].set_ylabel('value')

    for label, ax in axd.items():
        add_panel_label(ax, label)

    save_figure(fig, 'figure1')
    plt.close(fig)


def make_single_panel_for_inkscape():
    # For a MIXED figure assembled in Inkscape: build ONE panel at the exact
    # physical size it will occupy in the final layout (here ~half width).
    # Do NOT add the panel letter here; add it in Inkscape with the same spec
    # (Arial bold 11 pt) so it matches your all-plot figures. Import the SVG
    # into Inkscape at 100% and never rescale it.
    fig, ax = plt.subplots(figsize=(PLOT_PARAMS['width_half'], 2.4),
                           layout='constrained')
    x = np.linspace(0, 10, 200)
    ax.plot(x, np.sin(x), color=PLOT_PARAMS['colors'][1])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('signal')
    save_figure(fig, 'fig2_panel_b', outdir='panels')
    plt.close(fig)


if __name__ == '__main__':
    make_figure_1()
    make_single_panel_for_inkscape()
