"""Path / I/O utilities. Copied verbatim from april28_final_figures.py."""

import os
import re

import matplotlib as mpl
import numpy as np

from common.config import OUT_ROOT


def _slug(s):
    """Convert ``s`` into a filesystem-safe slug by collapsing whitespace to ``_``."""
    return re.sub(r"\s+", "_", s.strip())


def fig_path(exp_name, name, ext="png"):
    """Return the save path ``April28_preprint_results/<exp_name>/<name>.<ext>``."""
    out_dir = os.path.join(OUT_ROOT, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{_slug(name)}.{ext}")


def save_fig(fig, png_path, **savefig_kwargs):
    """Save *fig* as a PNG at *png_path* and as an editable SVG alongside it.

    The SVG is written to a sibling ``svg/`` directory with the same basename
    (``<dir>/svg/<name>.svg``). SVG text is kept as live ``<text>`` elements
    (``svg.fonttype='none'``) so fonts/labels can be edited in Illustrator or
    Inkscape. ``savefig_kwargs`` (e.g. ``dpi``, ``bbox_inches``) are passed to
    both saves — ``dpi`` still controls the resolution of any rasterized layers
    embedded in the SVG.
    """
    fig.savefig(png_path, **savefig_kwargs)
    svg_dir = os.path.join(os.path.dirname(png_path), "svg")
    os.makedirs(svg_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(png_path))[0]
    svg_path = os.path.join(svg_dir, base + ".svg")
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig.savefig(svg_path, **savefig_kwargs)


def load_segmentation(path):
    """Load a Cellpose / mask ``.npy`` file as a 2-D mask array."""
    seg = np.load(path, allow_pickle=True)
    if isinstance(seg, np.ndarray) and seg.dtype == object:
        try:
            seg = seg.item()["masks"]
        except Exception:
            pass
    return np.asarray(seg)


def sorted_image_files(frame_dir):
    """Return the sorted list of ``.png``/``.jpg`` filenames in ``frame_dir``."""
    return sorted([f for f in os.listdir(frame_dir) if f.endswith((".png", ".jpg"))])


def channel_dir(cfg, ch):
    """Resolve the on-disk root for a channel.

    Most experiments store data under ``<cfg['dir']>/<channel>/{frames,masks,analysis}``
    (one subdirectory per channel). For datasets like PC3 23MAR26 the data is
    flat — a single channel lives at ``<cfg['dir']>/{frames,masks,analysis}``
    directly. Setting ``cfg['single_channel_root'] = True`` selects that flat
    layout.
    """
    if cfg.get("single_channel_root"):
        return cfg["dir"]
    return os.path.join(cfg["dir"], ch)
