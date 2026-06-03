#!/usr/bin/env python3
"""Plotting orchestrator — render every figure from the analysis cache.

Reads ``analysis_cache/<exp>/<analysis>.pkl`` (written by the ``analyze_*.py``
scripts) and renders each figure as a standalone single-axis PNG (the four
``responder_diagnostic`` figures are the multi-panel exception). No heavy
compute and no raw-data access happen here — only matplotlib.

    python make_figures.py --experiments all
    python make_figures.py --experiments c2c12_dmso_09APR26 --figures dff
    python make_figures.py --mosaics fig2_dff_overview

Per-experiment figures render first; cross-experiment figures and mosaics run
last (they need every experiment's cache loaded).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.config import EXPERIMENTS
from common.io_paths import load_analysis_cache
from figures_spec import FIGURES
from plots._base import render_one

# Per-analysis plotting modules. Each exposes NAME + iter_figures(blob, exp).
from plots import dff as plots_dff  # noqa: E402
from plots import clustering as plots_clustering  # noqa: E402
from plots import average_peak as plots_avg  # noqa: E402
from plots import correlation_distance as plots_corr  # noqa: E402
from plots import response_violins as plots_violins  # noqa: E402
from plots import learning_scores as plots_learning  # noqa: E402
from plots import nrk_hardware_log as plots_nrk_hw  # noqa: E402
from plots import responder_diagnostic as plots_diag  # noqa: E402

MODULES = [plots_dff, plots_clustering, plots_avg, plots_corr, plots_violins,
           plots_learning, plots_nrk_hw, plots_diag]
MODULES_BY_NAME = {m.NAME: m for m in MODULES}

# Cross-experiment builders (run after every per-experiment cache is loaded).
CROSS_EXPERIMENT_BUILDERS = [plots_avg.build_combined]
# Named Nature-style mosaics live in plots/mosaics.py (MOSAICS dict) and are
# requested with --mosaics <name>.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+", default=["all"],
                   help="Experiment names from EXPERIMENTS, or 'all'.")
    p.add_argument("--figures", nargs="+", default=["all"],
                   help="Analysis groups to render (e.g. dff correlation_distance), or 'all'.")
    p.add_argument("--mosaics", nargs="+", default=[],
                   help="Named mosaics to assemble (default: none).")
    args = p.parse_args()
    if args.experiments == ["all"]:
        experiments = list(EXPERIMENTS)
    else:
        unknown = [e for e in args.experiments if e not in EXPERIMENTS]
        if unknown:
            raise SystemExit(f"Unknown experiments: {unknown}. Known: {list(EXPERIMENTS)}")
        experiments = args.experiments
    return experiments, args.figures, args.mosaics


def render_experiment(exp_name, which):
    """Render the requested analysis groups for one experiment."""
    names = list(MODULES_BY_NAME) if which == ["all"] else which
    for name in names:
        mod = MODULES_BY_NAME.get(name)
        if mod is None:
            print(f"  [{exp_name}] unknown figure group '{name}' — skipping")
            continue
        try:
            blob = load_analysis_cache(exp_name, mod.NAME)
        except FileNotFoundError as e:
            print(f"  [{exp_name}] no {mod.NAME} cache — skipping ({e})")
            continue
        n = 0
        for spec_key, payload, fill in mod.iter_figures(blob, exp_name):
            render_one(FIGURES[spec_key], payload, fill)
            n += 1
        print(f"  [{exp_name}] {mod.NAME}: {n} figures")


def main():
    experiments, which_figures, mosaics = parse_args()
    for exp_name in experiments:
        render_experiment(exp_name, which_figures)

    if which_figures == ["all"]:
        for build in CROSS_EXPERIMENT_BUILDERS:
            build(experiments)

    if mosaics:
        from plots import mosaics as mosaics_mod
        for name in mosaics:
            if name not in mosaics_mod.MOSAICS:
                print(f"  unknown mosaic '{name}' — skipping "
                      f"(known: {list(mosaics_mod.MOSAICS)})")
                continue
            for exp_name in experiments:
                try:
                    mosaics_mod.build_mosaic(name, exp_name)
                    print(f"  [{exp_name}] mosaic: {name}")
                except (KeyError, FileNotFoundError) as e:
                    print(f"  [{exp_name}] mosaic '{name}' skipped ({e})")


if __name__ == "__main__":
    main()
