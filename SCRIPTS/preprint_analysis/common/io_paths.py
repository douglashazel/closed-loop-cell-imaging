"""Path / I/O utilities. Originally copied from april28_final_figures.py;
extended with the analysis-cache layer that decouples analysis from plotting."""

import os
import pickle
import re

import matplotlib as mpl
import numpy as np

from common.config import OUT_ROOT


# =============================================================================
# Analysis cache — figure-ready intermediates written by analyze_*.py and read
# by the plotting layer (make_figures.py + plots/). This is distinct from the
# big per-experiment background pickles in bg_cache/ (see common.pipeline): the
# analysis cache sits *between* prepare_state() and matplotlib, holding only the
# small arrays/stats each figure needs.
# =============================================================================
ANALYSIS_CACHE_DIR = os.path.join(OUT_ROOT, "analysis_cache")
# Bump whenever a cached schema changes (mirrors pipeline.PIPELINE_VERSION). The
# plotting layer passes require_version so a stale cache fails loud instead of
# rendering wrong numbers.
ANALYSIS_VERSION = 1
ANALYSIS_CACHE_PROTOCOL = pickle.HIGHEST_PROTOCOL  # matches bg_cache pickling


def _slug(s):
    """Convert ``s`` into a filesystem-safe slug by collapsing whitespace to ``_``."""
    return re.sub(r"\s+", "_", s.strip())


def analysis_cache_path(exp_name, analysis):
    """Return ``results/analysis_cache/<exp>/<analysis>.pkl``.

    Creates the per-experiment directory if needed.
    """
    out_dir = os.path.join(ANALYSIS_CACHE_DIR, _slug(exp_name))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{_slug(analysis)}.pkl")


def save_analysis_cache(obj, exp_name, analysis, *, meta=None):
    """Pickle a figure-ready bundle for ``(exp_name, analysis)``.

    The on-disk blob wraps the payload so the plotting layer can version-check
    and introspect::

        {"ANALYSIS_VERSION": int, "analysis": str, "exp_name": str,
         "meta": {...scalars for label templates...}, "data": <obj>}

    ``meta`` holds the small scalars used to fill title/label templates at plot
    time (n cells, % variance, p-values, ...); ``obj`` (``data``) holds the
    arrays / ragged lists / nested dicts. Returns the written path.
    """
    blob = {
        "ANALYSIS_VERSION": ANALYSIS_VERSION,
        "analysis": analysis,
        "exp_name": exp_name,
        "meta": meta or {},
        "data": obj,
    }
    path = analysis_cache_path(exp_name, analysis)
    with open(path, "wb") as f:
        pickle.dump(blob, f, protocol=ANALYSIS_CACHE_PROTOCOL)
    return path


def load_analysis_cache(exp_name, analysis, *, require_version=ANALYSIS_VERSION):
    """Load and return the full blob for ``(exp_name, analysis)``.

    Returns the wrapper dict (use ``blob["data"]`` / ``blob["meta"]``). Raises
    ``FileNotFoundError`` with a run-this-first hint when the cache is missing,
    and ``RuntimeError`` on a version mismatch when ``require_version`` is set
    (pass ``require_version=None`` to skip the check).
    """
    path = analysis_cache_path(exp_name, analysis)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No analysis cache for ({exp_name}, {analysis}) at {path}. "
            f"Run analyze_{analysis}.py --experiments {exp_name} first."
        )
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if require_version is not None and blob.get("ANALYSIS_VERSION") != require_version:
        raise RuntimeError(
            f"analysis cache {path} is version {blob.get('ANALYSIS_VERSION')}, "
            f"expected {require_version}. Re-run analyze_{analysis}.py."
        )
    return blob


def fig_path(exp_name, name, ext="png"):
    """Return the save path ``results/<exp_name>/<name>.<ext>``."""
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
    # svg_dir = os.path.join(os.path.dirname(png_path), "svg")
    # os.makedirs(svg_dir, exist_ok=True)
    # base = os.path.splitext(os.path.basename(png_path))[0]
    # svg_path = os.path.join(svg_dir, base + ".svg")
    # with mpl.rc_context({"svg.fonttype": "none"}):
    #     fig.savefig(svg_path, **savefig_kwargs)


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
