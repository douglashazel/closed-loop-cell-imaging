"""Shared argparse helper.

Every analysis script accepts:
    --experiments <name1> <name2> ...   (or 'all', the default)
    --recompute-bg                      (force background-cache rebuild)

Returns ``(experiments_subset_dict, recompute_bg_flag)``. The subset preserves
``EXPERIMENTS`` insertion order.
"""

import argparse

from common.config import EXPERIMENTS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiments", nargs="+", default=["all"],
        help="Experiment names from the EXPERIMENTS dict, or 'all'.",
    )
    p.add_argument(
        "--recompute-bg", action="store_true",
        help="Force background-cache rebuild for the selected experiments.",
    )
    args = p.parse_args()

    if args.experiments == ["all"]:
        return dict(EXPERIMENTS), args.recompute_bg

    unknown = [e for e in args.experiments if e not in EXPERIMENTS]
    if unknown:
        raise SystemExit(
            f"Unknown experiments: {unknown}. "
            f"Known: {list(EXPERIMENTS)}"
        )
    subset = {e: EXPERIMENTS[e] for e in args.experiments}
    return subset, args.recompute_bg
