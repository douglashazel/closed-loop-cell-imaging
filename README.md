# Cell Trainer

Segmentation, single-cell tracking, fluorescence extraction, and downstream
statistical/figure analysis for time-lapse fluorescence microscopy of cultured
cells under repeated stimulation.

## About the study

Cells have a remarkable ability to adapt to the perturbations we impose on them —
often faster than mutation and selection can explain — producing challenges such
as drug resistance and transgene silencing. Accumulating evidence suggests that
this adaptation resembles classical forms of learning defined in behavioral
science, motivating the idea that **training techniques can be brought to bear as
a complementary approach for controlling cell physiology**. This project supports
a device that runs automated training experiments on non-neural mammalian cells,
using timed drug pulses as the stimulus and a moving fluorescence microscope to
image responses across replicate cultures. The device operates in either a
feedforward or feedback-controlled (closed-loop) manner, and the analysis code in
this repository reports the behavior of individual cells throughout each
experiment and characterizes population heterogeneity.

Two experiments are showcased. First, repeated pulses of DMSO produce
**sensitization-like behavior in the calcium response** of myoblast cells. Second,
the device maintains the **fluorescence of kidney cells carrying a pH/voltage
reporter** within a specified range by administering pulses of acidic medium in a
closed loop. The imaging modalities are therefore a calcium indicator (myoblast
experiment) and a genetically encoded pH/voltage reporter (kidney experiment).
Stage 1 of the pipeline segments, tracks, and extracts per-cell fluorescence
(dF/F0) from the raw frames; Stage 2 pools many experiments into the published
figures and statistics (responders, clustering, correlation, learning scores).
The device schematics and software are shared openly to accelerate research on
cell training, learning, and memory.

> 📄 **Preprint:** Erickson, Hazel, Martinez, *et al.* "A platform for automated
> training of mammalian cell physiology," bioRxiv (2026).
> <https://doi.org/10.64898/2026.08.13.744473>
>
> See also the [Citation](#citation) section.

## Authors

Patrick Erickson¹, Douglas Hazel¹, Ramses Martinez², Kostyantyn Shcherbina²,
Susan L. Marquez², Thomas Ferrante², Katarina Johnson¹, Angelina Pimkina¹,
Hananel Hazan¹, Juanita Mathews¹, Adama Marie Sesay², Michael Levin¹˒²

1. Allen Discovery Center at Tufts University, Medford, Massachusetts, USA
2. Wyss Institute for Biologically Inspired Engineering, Harvard University, Boston, Massachusetts, USA

### Software authorship & contact

The analysis and closed-loop code in this repository was written by
**Douglas Hazel**, with contributions from **Hananel Hazan**. For questions
about the code, please open a GitHub issue or contact Douglas Hazel
(douglas.hazel@tufts.edu).

The code is organized as a **two-stage workflow**:

```
                 ┌───────────────────────────────────────────────┐
  raw frames ──▶ │ STAGE 1 — core per-experiment pipeline        │ ──▶ per-cell
  (microscope)   │ SCRIPTS/core_pipeline/ + run_*.sh             │     fluorescence,
                 │ segment → track → pre-analysis → post-analysis│     dF/F0, plots
                 └───────────────────────────────────────────────┘
                                      │
                                      ▼  (many experiments)
                 ┌──────────────────────────────────────────────┐
                 │ STAGE 2 — detailed preprint analysis         │ ──▶ figures,
                 │ SCRIPTS/preprint_analysis/ +                 │     stats,
                 │ run_aggregate_results.sh,                    │     mosaics
                 │ run_aggregate_plots.sh  (responders, dF/F0,  │
                 │ clustering, correlation, learning scores, …) │
                 └──────────────────────────────────────────────┘
```

Run **Stage 1** on each experiment to turn raw frames into per-cell fluorescence
traces; then run **Stage 2** to pool many experiments into the published figures
and statistics.

The raw frames of the feedback experiment were produced by a separate **live
closed-loop acquisition + perturbation system** (`CLOSED_LOOP_GUI/`), published
here alongside the analysis code. It is optional and hardware-dependent (a Flask
web GUI + Cellpose + an ONIX microscopy controller) and is **not required to
reproduce the figures**.

This README has two parts: [Part I — Analysis](#part-i--analysis) (Stage 1,
Stage 2, the supplementary data and the tests) and
[Part II — Instrument and lab tooling](#part-ii--instrument-and-lab-tooling)
(the closed-loop controller and the parameter-tuning GUIs). Each part lists
its own install requirements.

---

## Repository layout

```
.
├── run_processes.sh           # Stage 1 driver: segmentation + tracking + pre-analysis
├── run_post_processes.sh      # Stage 1 driver: post-analysis (QC background correction, dF/F0)
├── run_segmentation.sh        # Stage 1: segmentation only
├── run_trajectories.sh        # Stage 1: tracking only
├── configs/                   # documented example pipeline_config.yaml (C2C12 chamber A values)
├── run_aggregate_results.sh   # Stage 2 driver: compute + cache figure intermediates
├── run_aggregate_plots.sh     # Stage 2 driver: render figures + mosaics from caches
│
├── SCRIPTS/
│   ├── core_pipeline/         # STAGE 1 code
│   │   ├── segmentation.py        # Cellpose segmentation → masks/*.npy
│   │   ├── trajectories.py        # link cells, extract fluorescence → analysis/*.json (msgpack)
│   │   ├── PreAnalysis.py         # QC luminosity plots
│   │   ├── PostAnalysis.py        # QC background correction, derivative/STD, dF/F0
│   │   ├── io_utils.py            # shared msgpack/DataFrame helpers
│   │   ├── pipeline_config.py     # reads/validates pipeline_config.yaml for the run_*.sh drivers
│   │   └── CreateGifsJson.py      # optional per-cell GIF renderer
│   │
│   └── preprint_analysis/     # STAGE 2 code; driven by run_aggregate_results.sh & run_aggregate_plots.sh
│       ├── analyze_*.py           # main analyses (responders runs first)
│       ├── make_figures.py        # plotting orchestrator
│       ├── make_mosaic_captions.py
│       ├── aggregate_preprint_pdf.py
│       ├── figures_spec.py, style.py
│       ├── common/                # shared config + analysis library
│       ├── plots/                 # figure render modules
│       └── data/                  # small Stage 2 inputs (timestamps, cell-selection masks, NRK logs)
│
├── supplement/                # PUBLISHED supplementary data (8 chambers, tracked in git)
├── tests/                     # pytest: supplement consistency, tracking, pipeline config
│
├── CLOSED_LOOP_GUI/           # Part II: live closed-loop microscopy + ONIX perturbation
├── TUNE_GUI/                  # Part II: browser GUI for parameter tuning
├── preprocess_gui.py          # Part II: napari tuning GUI (deprecated; use TUNE_GUI)
│
├── requirements.txt           # pip dependencies
├── environment.yml            # conda environment
└── CITATION.cff               # citation metadata
```

Input/output **data directories** (`EXPERIMENTS/`, `results/`,
`gifs/`, …) are git-ignored and not redistributed — see
[Expected data layout](#expected-data-layout). The one exception is
`supplement/`, the published supplementary data bundle, which **is** tracked
here — see [Supplementary data](#supplementary-data).

---

# Part I — Analysis

## Installation

Python 3.11 or 3.12. Either conda or pip:

```bash
# conda
conda env create -f environment.yml
conda activate cell_trainer

# or pip (into a fresh venv)
pip install -r requirements.txt
```

The analysis needs only the "Core analysis" and Cellpose blocks of
`requirements.txt`; Stage 2 and the supplement route do not need Cellpose
either. The napari/Qt, Flask, `requests` and `tomlkit` packages are for Part II.

**GPU:** Stage 1 segmentation runs Cellpose on the GPU when CUDA is available
(default `CUDA_VISIBLE_DEVICES=0`; pass `--cpu` to `segmentation.py` to force
the CPU, which is much slower). Everything else runs on the CPU.

> **Run all commands from the project root.** Several Stage 2 modules add
> `SCRIPTS/core_pipeline` and `SCRIPTS/preprint_analysis` to `sys.path` using
> paths relative to the current directory. The two Stage 2 drivers
> (`run_aggregate_results.sh`, `run_aggregate_plots.sh`) and the four Stage 1
> drivers `cd` to the project root themselves.

---

## Expected data layout

The pipeline reads/writes a per-experiment tree under `EXPERIMENTS/` (git-ignored):

```
EXPERIMENTS/<group>/<experiment>/[<channel>/]
├── pipeline_config.yaml  # Stage 1 parameters for this experiment (TUNE_GUI writes it)
├── frames/      # input images, named "...timepoint_NNNNN.png" (or .jpg), ordered by N
├── masks/       # Cellpose label masks, one .npy per frame (written by Stage 1)
└── analysis/    # Stage-1 outputs: trajectories_complete.json, luminosity_complete.json,
                 #   *_complete.csv, run_history/, run_params.json, bg_values_cache.npy, plots/
                 #   (runs before pipeline_config.yaml wrote config.txt instead of run_history/)
```

Multi-channel experiments use a `<channel>/` level (e.g. `channel 1 A/`); single
field-of-view experiments put `frames/masks/analysis` directly under the
experiment. Stage 2's experiment registry lives in
[`SCRIPTS/preprint_analysis/common/config.py`](SCRIPTS/preprint_analysis/common/config.py)
(`EXPERIMENTS` dict: data dir, channels, stim schedule, timestamps, masks).

> **The Stage 1 `.json` files are msgpack, not JSON.** Read them with
> `SCRIPTS/core_pipeline/io_utils.load_msgpack`. Stage 2 reads only
> `trajectories_complete.json` and `luminosity_complete.json`.

---

## Stage 1 — core per-experiment pipeline

1. **Set the parameters** for the experiment in
   `<experiment>/pipeline_config.yaml`. The browser GUI
   (`python TUNE_GUI/app.py`, see [Part II](#tune_gui)) writes this file when
   you save or run from its Review & Run tab. It holds the Cellpose params
   (`flow_threshold`, `cellprob_threshold`, `niter`, `diameter`), the tracking
   params (`max_distance`, frame shift, ROI radius, `save_interval`) and the
   post-analysis frames (`f0_frame`, `stim_frames`). To write it by hand, copy
   [configs/example_c2c12_chamber_A.yaml](configs/example_c2c12_chamber_A.yaml),
   which documents every key, into the experiment folder and edit it. Check
   it with `python3 SCRIPTS/core_pipeline/pipeline_config.py check <experiment>`.

2. **Run segmentation + tracking + pre-analysis.** The drivers take the
   experiment folder (or a config file) and need no edits:

   ```bash
   bash run_processes.sh "EXPERIMENTS/<group>/<experiment>"
   ```

   `run_segmentation.sh` and `run_trajectories.sh` run those stages
   individually, and `run_processes.sh --skip-segmentation` tracks existing
   masks. In the config, `radius: 0` disables the ROI filter and `shift_frame`
   is a `timepoint_NNNNN` frame number. The example config holds the values
   recorded for C2C12 chamber A of the paper (its `analysis/config.txt`).

   **Which parameters made these results?** Each driver saves the values it
   used, with the date, user and git commit, to
   `<experiment>/analysis/run_history/<timestamp>_<driver>.yaml` when it
   starts, so the newest record for a stage describes its current outputs
   if that run finished. The config is read once at start, so editing it
   during a run does not affect that run.

3. **Post-analysis** — set `post_analysis` (`f0_frame`, `stim_frames`) in the
   config, then:

   ```bash
   bash run_post_processes.sh "EXPERIMENTS/<group>/<experiment>"
   ```

   This produces background-corrected traces, derivative/STD, and dF/F0 plots
   under `analysis/plots/`. It is a quick-look QC step: it uses a 6×6 spline
   background and F0 from a single frame, while the published figures use
   Stage 2's per-frame polynomial background fit and F0 as the mean of the
   pre-stimulus frames. Its numbers therefore differ from the figures.

4. **(Optional) Per-cell GIFs** —
   `python SCRIPTS/core_pipeline/CreateGifsJson.py --experiment_dir <dir>`
   (see `--help`).

`trajectories.py` records its tracking parameters in `analysis/run_params.json`
and refuses to resume a run made with different ones (`--force_resume`
overrides). To start over with new parameters, delete the experiment's
`analysis/` directory, but never for the eight published chambers: their
`analysis/` outputs are the inputs to the published results. See
[SCRIPTS/core_pipeline/README.md](SCRIPTS/core_pipeline/README.md) for the
flags and the output contract.

---

## Stage 2 — preprint analysis

Pools the Stage-1 outputs of many experiments into the published figures.

1. **Register experiments** in
   [`SCRIPTS/preprint_analysis/common/config.py`](SCRIPTS/preprint_analysis/common/config.py)
   (the `EXPERIMENTS` dict). The small non-image inputs of the published
   experiments (frame timestamps, the C2C12 cell-selection masks, the PC-3
   bad-frame list, and the NRK closed-loop controller logs) are in
   [`SCRIPTS/preprint_analysis/data/`](SCRIPTS/preprint_analysis/data/README.md).

2. **Compute + cache** the figure intermediates (the shared `responders` step
   runs first):

   ```bash
   ./run_aggregate_results.sh
   ```

3. **Render** figures and mosaics from the caches:

   ```bash
   ./run_aggregate_plots.sh
   ```

   Outputs land in `results/<experiment>/` and
   `results/mosaics/`. Set `AGGREGATE_PDF=true` in `run_aggregate_plots.sh`
   to also build a combined PDF. Both scripts have a `CONFIG` block at the top to
   select a subset of analyses/experiments/figures.

See [SCRIPTS/preprint_analysis/README.md](SCRIPTS/preprint_analysis/README.md)
for the analysis → cache → plot contract.

---

## Supplementary data

The processed single-cell data behind the manuscript is published **in this
repository** under [`supplement/`](supplement/) (~89 MB), covering all 8
CellASIC chambers in the paper. Per chamber it contains raw and
background-corrected fluorescence tables, dF/F0, cell-position tracks, the
per-frame background and time axis, the frame-0 Cellpose masks, and a
`metadata.json` recording the stimulus schedule, F₀ window, and response window.

The raw microscope frames (~100 GB) are **not** redistributed. They are only
needed for Stage 1 (segmentation and tracking); everything the manuscript
reports downstream of segmentation is reproducible from the bundle alone:

```bash
python SCRIPTS/preprint_analysis/load_supplement.py \
    --analyses responders dff average_peak correlation_distance \
               clustering response_violins learning_scores nrk_hardware_log

./run_aggregate_plots.sh        # render the figures
```

`load_supplement.py` rebuilds the Stage-2 pipeline state from the exported
tables, so the `analyze_*.py` scripts run unchanged without the raw frames, and
all seven figure mosaics build from the bundle. One thing cannot be reproduced
from the bundle, because it reads the images directly: the frame-sharpness
panel of the responder diagnostic.

See [`supplement/README.md`](supplement/README.md) for the full column-by-column
description, the chamber table, and notes on reading the label masks.
Integrity: `cd supplement && sha256sum -c CHECKSUMS.sha256`.

Maintainers regenerate the bundle with
`python SCRIPTS/preprint_analysis/export_supplement.py` (writes `supplement/`);
this requires the raw experiment tree and the warm `results/bg_cache/` pickles.

---

## Tests

```bash
pip install pytest
pytest -q tests/
```

`tests/test_supplement.py` checks the bundle's internal consistency (checksums,
cell and frame counts, masks, dF/F0 recomputed exactly, no duplicated cells)
and reruns the responder classification from the bundle against the published
counts. `tests/test_tracking.py` covers the Stage 1 tracker. They need only the
core analysis packages, and run in CI on every push
(`.github/workflows/tests.yml`).

---

# Part II — Instrument and lab tooling

The code below runs the hardware and helps pick Stage 1 parameters. It is lab
tooling: none of it is needed to reproduce the figures. On top of the core
packages it needs Flask (both browser GUIs) and `requests` + `tomlkit` (the
closed-loop controller); napari/Qt only for the deprecated `preprocess_gui.py`.

## Live closed-loop microscopy / perturbation pipeline

`CLOSED_LOOP_GUI/` is the **live acquisition + feedback system** that generated the
acid-feedback experiments analysed above. A Flask web GUI watches incoming microscope
frames, segments cells with Cellpose, measures per-cell fluorescence, and drives an
**ONIX hardware controller** (over HTTP) to dose acidic / neutral media in a closed
loop. It needs a CUDA GPU and a networked ONIX2 server.

```bash
python CLOSED_LOOP_GUI/LaunchWebGUI.py   # Flask "Closed-Loop Bio-Control Hub" on http://127.0.0.1:5000
```

The GUI bootstraps `CLOSED_LOOP_GUI/config.json`, lets you set the run parameters, and
starts/stops the pipeline (`run_system.sh`, which launches the segmentation,
decision, ONIX-actuation, and monitoring daemons). All paths derive from a single
`global_path` in `CLOSED_LOOP_GUI/config.py`; edit it (and the ONIX endpoint /
experiment templates) before first use. Without hardware, set `"dry_run": true`
in `CLOSED_LOOP_GUI/config.json` to run the loop against a simulated ONIX and
synthetic frames.

See [CLOSED_LOOP_GUI/README.md](CLOSED_LOOP_GUI/README.md) for file roles, the data-flow
contract, configuration knobs, how the loop behaves, and hardware/security notes.

> ⚠️ **Security:** `CLOSED_LOOP_GUI/` issues HTTP requests that create and run
> experiments on networked lab hardware, and the GUI spawns subprocesses. It
> has no authentication: it listens on 127.0.0.1 by default (`HOST=0.0.0.0`
> exposes it, trusted networks only) and rejects cross-site POSTs. Point it
> only at hardware you control.

## TUNE_GUI

`TUNE_GUI/` is a Flask app for interactively tuning Stage-1 parameters and
launching the pipeline from a browser (`python TUNE_GUI/app.py`, then
http://127.0.0.1:5001). See [TUNE_GUI/README.md](TUNE_GUI/README.md).

> ⚠️ **Security:** the tuning GUI writes each experiment's
> `pipeline_config.yaml` and runs the Stage 1 drivers on it. It binds
> `127.0.0.1:5001` by default; `HOST=0.0.0.0` exposes it to the network, so do
> that only on a trusted network. POSTs must be same-origin JSON, and
> experiment paths must be inside `EXPERIMENTS/`.

## preprocess_gui.py (deprecated)

`preprocess_gui.py` is the older napari version of the tuning GUI. It has
drifted from `TUNE_GUI/` and is no longer maintained; use `TUNE_GUI/` instead.

---

## Citation

If you use this code or the accompanying data, please cite the preprint:

> Erickson, P., Hazel, D., Martinez, R., Shcherbina, K., Marquez, S. L.,
> Ferrante, T., Johnson, K., Pimkina, A., Hazan, H., Mathews, J., Sesay, A. M.,
> & Levin, M. (2026). *A platform for automated training of mammalian cell
> physiology.* bioRxiv. https://doi.org/10.64898/2026.08.13.744473

```bibtex
@article{erickson2026celltrainer,
  title   = {A platform for automated training of mammalian cell physiology},
  author  = {Erickson, Patrick and Hazel, Douglas and Martinez, Ramses and
             Shcherbina, Kostyantyn and Marquez, Susan L. and Ferrante, Thomas and
             Johnson, Katarina and Pimkina, Angelina and Hazan, Hananel and
             Mathews, Juanita and Sesay, Adama Marie and Levin, Michael},
  year    = {2026},
  journal = {bioRxiv},
  doi     = {10.64898/2026.08.13.744473},
  url     = {https://doi.org/10.64898/2026.08.13.744473}
}
```

For the software specifically, please cite this repository (GitHub's "Cite
this repository" button reads [`CITATION.cff`](CITATION.cff)) and contact
Douglas Hazel (douglas.hazel@tufts.edu).
