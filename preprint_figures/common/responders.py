"""Per-(experiment, channel) Bonferroni responder threshold computation.

Copied verbatim from april28_final_figures.py (compute_responder_thresholds).
"""

import sys

import numpy as np

from common.config import PEAK_OFFSET
from common.stim_helpers import compute_f0_baseline, per_cell_response_delta

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


def compute_responder_thresholds(
    experiments, state,
    alpha=0.01,
    exclusion_pad=10,
    n_pseudo=500,
    rng_seed=42,
):
    """Per-(experiment, channel) Bonferroni-corrected responder threshold.

    Returns ``{(exp_name, ch_name): threshold_magnitude}``.
    """
    rng = np.random.default_rng(rng_seed)
    thresholds = {}

    for exp_name, cfg in experiments.items():
        direction = cfg.get("response_direction", "increase")
        window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
        win_lo, win_hi = window

        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            n_real = len(stim_frames)
            if n_real == 0:
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
            n_cells, n_cols = mat.shape
            frame_to_col = {f: i for i, f in enumerate(frame_nums)}

            F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe

            stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]
            excluded = np.zeros(n_cols, dtype=bool)
            pad = max(exclusion_pad, win_hi)
            for sc in stim_cols:
                lo = max(0, sc - pad)
                hi = min(n_cols, sc + pad + 1)
                excluded[lo:hi] = True
            valid_mask = ~excluded
            valid_mask[: max(0, -win_lo)] = False
            valid_mask[max(0, n_cols - win_hi + 1):] = False
            valid_cols = np.where(valid_mask)[0]

            ch_null = []
            if len(valid_cols) >= 1:
                replace = len(valid_cols) < n_pseudo
                pseudo_cols = rng.choice(valid_cols, size=n_pseudo, replace=replace)
                for pc in pseudo_cols:
                    deltas = per_cell_response_delta(dff_mat, int(pc), direction, window)
                    deltas = deltas[~np.isnan(deltas)]
                    ch_null.append(deltas)

            per_stim_alpha = alpha / n_real
            pct = 100.0 * (1.0 - per_stim_alpha)
            if not ch_null:
                thresholds[(exp_name, ch)] = 0.10
                print(
                    f"  responder threshold ({exp_name} / {ch}): "
                    f"|Δ dF/F0| ≥ 0.1000  (fallback: empty null; "
                    f"N_real={n_real}, per-cell α={alpha:g})"
                )
                continue
            ch_abs = np.abs(np.concatenate(ch_null))
            threshold = float(np.nanpercentile(ch_abs, pct))
            thresholds[(exp_name, ch)] = threshold
            print(
                f"  responder threshold ({exp_name} / {ch}): "
                f"|Δ dF/F0| ≥ {threshold:.4f}  "
                f"(Bonferroni: {pct:.4f}th pct of per-stim null, "
                f"N_real={n_real}, per-cell α={alpha:g})"
            )

    return thresholds
