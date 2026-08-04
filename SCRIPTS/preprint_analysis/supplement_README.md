# Supplementary data — per-chamber single-cell fluorescence

Processed single-cell time series, Cellpose segmentation masks, and the
metadata needed to reproduce the secondary analyses in the accompanying
manuscript, for each of the 8 CellASIC chambers featured in the paper.

The raw microscope frames (~100 GB) are not redistributed. **Everything in the
manuscript downstream of segmentation can still be reproduced from this
bundle** — see [Reproducing the analyses](#reproducing-the-analyses).

---

## Chambers

| Directory | Cell line | Stimulus | Cells tracked | Cells analysed | Frames analysed |
|---|---|---|---|---|---|
| `C2C12_A` | C2C12 | DMSO pulse (2 min) x15 | 417 | 185 | 553 |
| `C2C12_B` | C2C12 | DMSO pulse (2 min) x15 | 364 | 276 | 553 |
| `C2C12_C` | C2C12 | DMSO pulse (2 min) x15 | 245 | 232 | 553 |
| `PC3` | PC-3 | DMSO pulse (2 min) x15 | 338 | 338 | 1952 |
| `NRK_A` | NRK | Acid pulse (30 s), closed loop | 64 | 64 | 94 |
| `NRK_B` | NRK | Acid pulse (30 s), closed loop | 76 | 76 | 94 |
| `NRK_C` | NRK | Acid pulse (30 s), closed loop | 59 | 59 | 93 |
| `NRK_D` | NRK | Acid pulse (30 s), closed loop | 42 | 42 | 93 |

C2C12 and PC-3 report a calcium indicator; NRK reports a genetically encoded
pH/voltage reporter. NRK chambers are analysed over the first 30 min only, so
`n_frames_analyzed` is shorter than the full recording (`n_frames_total`).

**Tracked vs analysed.** *Cells tracked* is every cell segmentation + tracking
produced, and is the row count of `*_fluorescence_raw.csv`; *cells analysed* is
the row count of `*_fluorescence_bgcorrected.csv` and `*_dff.csv`. The two
differ only for C2C12, where a manually drawn selection mask
(`*_mask_analyzed_cells.npz`) restricts the analysis to the usable field of
view, excluding the chamber periphery, out-of-focus regions, and debris: a cell
is kept iff its frame-0 centroid lies on a non-zero pixel of that mask. No
other chamber applies a cell filter, so their two counts are equal. See
production step 4 below.

## Files per chamber

Every file is prefixed with the chamber name (e.g. `C2C12_A_dff.csv`).

| File | Contents |
|---|---|
| `*_fluorescence_raw.csv` | **Raw** per-cell mean ROI intensity. All segmented+tracked cells x all frames. No background correction, no normalisation. |
| `*_fluorescence_bgcorrected.csv` | Background-subtracted fluorescence. Analysed cells x analysed frames. **This is the table the published analyses consume.** |
| `*_dff.csv` | Normalised (F − F₀)/F₀ of the table above. |
| `*_cell_positions.csv` | Tidy `cell_id, frame, x, y` tracks (pixels) for the analysed cells. Needed for the correlation-vs-distance / Mantel analysis. |
| `*_background.csv` | Per-frame fitted-background scalars (`bg_sampled_mean`, `bg_fit_min`). |
| `*_time_axis.csv` | Per-frame `minutes` since perfusion start, stimulus-onset flag, and whether the frame is inside the analysis window. |
| `*_mask.npz`, `*_mask.png` | Frame-0 Cellpose label mask (`uint16`; label *L* = cell *L*, 0 = background). NPZ key `masks`; the PNG is a lossless 16-bit copy. **These look black in an image viewer — see [Viewing the masks](#viewing-the-masks).** |
| `*_mask_preview.png` | Colour rendering of the mask, one colour per cell. For eyeballing the segmentation only — never use it as analysis input. |
| `*_mask_analyzed_cells.npz/.png`, `*_mask_analyzed_cells_preview.png` | *C2C12 only* — the circle+area selection mask defining which segmented cells entered the analysis. |
| `*_hardware_feedback_log.json` | *NRK only* — closed-loop controller record: per-frame mean luminosity, active setpoint, and the delivered decision. |
| `*_metadata.json` | Stimulus frames/minutes, F₀ window, response window and direction, dead frames, analysed cell IDs, mask provenance, and the frame→minutes interpolation knots. |

### Table layout

`*_fluorescence_*.csv` and `*_dff.csv` are wide matrices: one row per cell, a
leading `cell_id` column, then one column per frame named `f0, f1, …`. Empty
cells are missing values (`NaN`). Column `fN` corresponds to row `frame == N`
in `*_time_axis.csv`.

```python
import pandas as pd
dff = pd.read_csv("C2C12_A/C2C12_A_dff.csv", index_col="cell_id")
t   = pd.read_csv("C2C12_A/C2C12_A_time_axis.csv")
minutes = t.loc[t.in_analysis_window, "minutes"].values   # x-axis for dff
```

> **Exact reload:** the tables are written with Python's shortest round-trip
> float repr. pandas' default CSV reader is fast but not correctly rounded and
> loses ~1 unit in the last place. Pass `float_precision="round_trip"` to
> `pd.read_csv` to recover the exact `float64` values. This matters only for
> bit-exact reproduction, not for any scientific conclusion.

### Viewing the masks

`*_mask.png` / `*_mask.npz` are **label images, not pictures**: each pixel holds
the integer ID of the cell covering it, and 0 is background. The IDs run 1…N
with N ≈ 40–550 per chamber, stored in a 16-bit container, so the brightest
cell sits below 1 % of full scale and the file appears uniformly black in any
viewer that does not auto-scale. The pixel data is intact — only the display is
misleading.

To see the cells, either open `*_mask_preview.png` (already colourised), or
auto-scale the label image:

- **ImageJ / Fiji** — open `*_mask.png`, then Image ▸ Adjust ▸
  Brightness/Contrast ▸ Auto (Image ▸ Lookup Tables ▸ Glasbey gives one colour
  per cell).
- **Python**

  ```python
  import numpy as np, matplotlib.pyplot as plt
  masks = np.load("C2C12_A/C2C12_A_mask.npz")["masks"]     # uint16 labels
  plt.imshow(np.ma.masked_equal(masks, 0), cmap="nipy_spectral", interpolation="nearest")
  ```

Use the NPZ (or the 16-bit PNG) for anything quantitative — the preview is
lossy by construction, since it maps many cell IDs onto a repeating palette.

## How the columns were produced

1. **Segmentation** — Cellpose label masks per frame.
2. **Tracking** — cells linked across frames; per-cell mean ROI intensity →
   `*_fluorescence_raw.csv`.
3. **Background correction** — for every frame, a 2-D polynomial is fitted to
   cell-free background sample points and evaluated at each cell's (x, y), then
   subtracted → `*_fluorescence_bgcorrected.csv`.
4. **Cell selection** *(C2C12 only)* — cells whose frame-0 centroid falls
   outside `*_mask_analyzed_cells` are dropped (417→185, 364→276, 245→232 for
   chambers A/B/C). This is the only reason the raw and corrected tables differ
   in cell count; PC-3 and NRK apply no cell filter.
5. **Dead-frame repair** *(PC-3)* — dropped/flash camera frames (listed in
   `metadata.json → dead_frames`) are masked and linearly interpolated.
6. **Time clipping** *(NRK)* — analysis restricted to the first 30 min.
7. **Normalisation** — F₀ is each cell's mean background-corrected
   fluorescence over the frames before the first stimulus
   (`metadata.json → f0_baseline_frames`); `dF/F0 = (F − F₀)/F₀` → `*_dff.csv`.

Step 3 is the reason this bundle ships corrected values rather than only raw
ones: reproducing it requires the raw frames, which are too large to
redistribute.

## Reproducing the analyses

`load_supplement.py` (included) rebuilds the analysis pipeline's internal state
from these tables, so the repository's analysis scripts run unchanged without
the raw frames. From the analysis repository root, with this bundle at
`results/supplement_export/`:

```bash
python SCRIPTS/preprint_analysis/load_supplement.py \
    --root results/supplement_export \
    --analyses responders dff average_peak correlation_distance \
               clustering response_violins learning_scores

./run_aggregate_plots.sh        # render the figures
```

This reproduces the responder classification, dF/F0 traces, per-stimulus peak
analysis, PCA/UMAP embeddings, correlation-vs-distance and Mantel tests,
response violins, and the habituation / sensitization / anticipation learning
scores.

**Not reproducible from this bundle** (requires the raw frames): the
frame-sharpness panel of the responder diagnostic, and the frame mosaics.
Segmentation and tracking themselves likewise cannot be re-run.

## Notes

- Cell IDs are per chamber and are **not** comparable across chambers. A cell's
  ID matches its label in that chamber's frame-0 mask only where the two were
  produced together; use `*_cell_positions.csv` (frame 0) to locate a cell on
  the mask.
- `*_fluorescence_raw.csv` contains *all* segmented+tracked cells, at full
  recording length. The corrected and dF/F0 tables contain only the analysed
  subset over the analysed window — see `n_cells_raw` vs `n_cells_analyzed` and
  `analyzed_cell_ids` in `metadata.json`.
- Positions are in pixels; the imaging calibration is 0.555 μm/pixel.
- `CHECKSUMS.sha256` covers every file in the bundle:
  `sha256sum -c CHECKSUMS.sha256`.
