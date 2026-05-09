"""Empirical sanity check for ``cfg["response_direction"]`` per (exp, ch).

Pools per-stim post-window peak (max-base) and trough (min-base) deltas across
every cell and every stim, then compares the magnitudes. Whichever direction
has the larger pooled magnitude is the empirical direction. Logs a warning if
that disagrees with the configured direction. Diagnostic only — config still
wins, since the rest of the pipeline reads ``cfg["response_direction"]``.

Called from ``pipeline.prepare_state`` after BG fit + dead-frame fill.
"""

import numpy as np

from common.config import PEAK_OFFSET
from common.stim_helpers import per_cell_response_delta

import sys
sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


def confirm_response_direction(experiments, state):
    """Return ``{(exp, ch): {"configured", "empirical", "agreement",
    "abs_increase", "abs_decrease"}}`` and log a warning per disagreement."""
    results = {}
    print()
    print("=== response-direction sanity check ===")
    for exp_name, cfg in experiments.items():
        configured = cfg.get("response_direction", "increase")
        window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))

        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"].get(ch, [])
            if not stim_frames:
                continue
            df_indexed = lum_dict_to_df(
                state["corrected_lum"][exp_name][ch]
            ).set_index("CellID")
            frame_cols = sorted(
                [c for c in df_indexed.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
            mat = df_indexed[frame_cols].values
            frame_to_col = {f: i for i, f in enumerate(frame_nums)}
            stim_cols = [
                frame_to_col[p] for p in stim_frames if p in frame_to_col
            ]
            if not stim_cols:
                continue

            up_deltas = []
            down_deltas = []
            for sc in stim_cols:
                up_deltas.append(
                    per_cell_response_delta(mat, int(sc), "increase", window)
                )
                down_deltas.append(
                    per_cell_response_delta(mat, int(sc), "decrease", window)
                )
            up = np.concatenate(up_deltas)
            down = np.concatenate(down_deltas)
            abs_up = float(np.nanmean(np.abs(up)))
            abs_down = float(np.nanmean(np.abs(down)))
            empirical = "increase" if abs_up >= abs_down else "decrease"
            agreement = empirical == configured
            results[(exp_name, ch)] = {
                "configured": configured,
                "empirical": empirical,
                "agreement": agreement,
                "abs_increase": abs_up,
                "abs_decrease": abs_down,
            }
            tag = "OK" if agreement else "WARN"
            print(
                f"  [{tag}] {exp_name} / {ch}: configured={configured}, "
                f"empirical={empirical}  "
                f"|mean(max-base)|={abs_up:.3f}  "
                f"|mean(min-base)|={abs_down:.3f}"
            )
            if not agreement:
                print(
                    f"    >>> empirical direction disagrees with "
                    f"cfg['response_direction']={configured!r}; "
                    f"config still applied — verify intent."
                )
    print()
    return results
