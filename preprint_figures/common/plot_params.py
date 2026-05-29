"""Backward-compatible re-export of the shared plot styling dicts.

The single source of truth is now ``preprint_figures/figstyle.py``. This module
exists only so the analysis scripts can keep doing

    from common.plot_params import PLOT_PARAMS

unchanged while every value is governed from ``figstyle``. Edit ``figstyle.py``
to change fonts, sizes, line widths, or figure dimensions — never here.
"""
import os
import sys

# Make the top-level ``figstyle`` module importable even if the caller did not
# already put the preprint_figures directory on sys.path.
_PREPRINT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PREPRINT_DIR not in sys.path:
    sys.path.insert(0, _PREPRINT_DIR)

from figstyle import (  # noqa: E402,F401  (re-export)
    PLOT_PARAMS,
    PLOT_PARAMS_SLIDING,
    PLOT_PARAMS_HW_LOG,
)
