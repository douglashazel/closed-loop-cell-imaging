# Cell Analysis Pipeline - Cell Trainer

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

> **TODO (authors):** add a link to the preprint once a DOI is available (see also
> the [Citation](#citation) section).

## Authors

Patrick Erickson¹, Douglas Hazel¹, Ramses Martinez², Kostyantyn Shcherbina²,
Susan Marquez², Thomas Ferrante², Hananel Hazan¹, Katarina Johnson¹,
Angelina Pimkina¹, Juanita Mathews¹, Adama Sesay², Michael Levin¹˒²

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

The raw frames are produced upstream by a separate **live closed-loop acquisition
+ perturbation system** (`CLOSED_LOOP_GUI/`), published here alongside the analysis
code. It is optional and hardware-dependent (a Flask web GUI + Cellpose GPU + an
ONIX microscopy controller) and is **not required to reproduce the figures** — see
[CLOSED_LOOP_GUI/README.md](CLOSED_LOOP_GUI/README.md).

---

## Repository layout

```
.
├── run_processes.sh           # Stage 1 driver: segmentation + tracking + pre-analysis
├── run_post_processes.sh      # Stage 1 driver: post-analysis (bg correction, dF/F0)
├── run_segmentation.sh        # Stage 1: segmentation only
├── run_trajectories.sh        # Stage 1: tracking only
├── run_aggregate_results.sh   # Stage 2 driver: compute + cache figure intermediates
├── run_aggregate_plots.sh     # Stage 2 driver: render figures + mosaics from caches
├── preprocess_gui.py          # Optional napari GUI to tune parameters
│
├── SCRIPTS/
│   ├── core_pipeline/         # STAGE 1 code
│   │   ├── segmentation.py        # Cellpose segmentation  → masks/*.npy
│   │   ├── trajectories.py        # link cells, extract fluorescence → analysis/*.json
│   │   ├── PreAnalysis.py         # QC luminosity plots
│   │   ├── PostAnalysis.py        # background correction, derivative/STD, dF/F0
│   │   ├── io_utils.py            # shared msgpack/DataFrame helpers
│   │   └── CreateGifs*.py         # optional per-cell GIF renderers (edit-then-run)
│   │
│   └── preprint_analysis/     # STAGE 2 code (was "preprint_figures/"); driven by
│       │                      #   the root run_aggregate_results.sh / run_aggregate_plots.sh
│       ├── analyze_*.py           # one analysis each (responders runs first)
│       ├── make_figures.py        # plotting orchestrator
│       ├── make_mosaic_captions.py
│       ├── aggregate_preprint_pdf.py
│       ├── figures_spec.py, style.py
│       ├── common/                # shared config + analysis library
│       └── plots/                 # figure render modules
│
├── CLOSED_LOOP_GUI/           # Live closed-loop microscopy + ONIX perturbation (Flask web GUI; optional, hardware)
├── TUNE_GUI/                  # Optional browser GUI for parameter tuning (Flask)
├── requirements.txt           # pip dependencies
└── environment.yml            # conda environment
```

Input/output **data directories** (`EXPERIMENTS/`, `results/`,
`gifs/`, …) are git-ignored and not redistributed — see
[Expected data layout](#expected-data-layout).

---

## Installation

Python 3.12. Either conda (recommended, for napari/Cellpose) or pip:

```bash
# conda
conda env create -f environment.yml
conda activate cell_trainer

# or pip (into a fresh venv)
pip install -r requirements.txt
```

**GPU:** Stage-1 segmentation uses Cellpose with `gpu=True` and defaults to
`CUDA_VISIBLE_DEVICES=0`; a CUDA-capable GPU is strongly recommended. Everything
else runs on CPU.

> **Run all commands from the project root.** Several Stage-2 modules add
> `SCRIPTS/core_pipeline` and `SCRIPTS/preprint_analysis` to `sys.path` using
> paths relative to the current directory. The provided `run_*.sh` scripts
> `cd` to the project root automatically.

---

## Expected data layout

The pipeline reads/writes a per-experiment tree under `EXPERIMENTS/` (git-ignored):

```
EXPERIMENTS/<group>/<experiment>/[<channel>/]
├── frames/      # input images, named "...timepoint_NNNNN.png" (or .jpg), sorted by N
├── masks/       # Cellpose label masks, one .npy per frame (written by Stage 1)
└── analysis/    # Stage-1 outputs: trajectories_complete.json, luminosity_complete.json,
                 #   *_complete.csv, config.txt, bg_values_cache.npy, plots/
```

Multi-channel experiments use a `<channel>/` level (e.g. `channel 1 A/`); single
field-of-view experiments put `frames/masks/analysis` directly under the
experiment. Stage 2's experiment registry lives in
[`SCRIPTS/preprint_analysis/common/config.py`](SCRIPTS/preprint_analysis/common/config.py)
(`EXPERIMENTS` dict: data dir, channels, stim schedule, timestamps, masks).

---

## Stage 1 — core per-experiment pipeline

1. **(Optional) Tune parameters** for a new experiment with either GUI:
   - napari: `python preprocess_gui.py`
   - browser: `python TUNE_GUI/app.py` → http://localhost:5001 (see [TUNE_GUI/README.md](TUNE_GUI/README.md))

   Determine the Cellpose params (`flow_threshold`, `cellprob_threshold`,
   `niter`, `diameter`) and tracking params (`max_distance`, frame shift, ROI
   radius, `save_interval`).

2. **Edit the driver** — set `GLOBAL_DIR` and the parameters at the top of
   `run_processes.sh` (these scripts are templates pinned to example
   experiments). Then run segmentation + tracking + pre-analysis:

   ```bash
   bash run_processes.sh
   ```

   (`run_segmentation.sh` and `run_trajectories.sh` run those stages
   individually.)

3. **Post-analysis** — edit `GLOBAL_DIR`/`STIM_FRAMES` at the top of
   `run_post_processes.sh`, then:

   ```bash
   bash run_post_processes.sh
   ```

   This produces background-corrected traces, derivative/STD, and dF/F0 plots
   under `analysis/plots/`.

4. **(Optional) Per-cell GIFs** — edit the `CHANGE HERE` block at the top of
   `SCRIPTS/core_pipeline/CreateGifsJson.py` (or `CreateGifs.py`) and run it from
   the project root.

---

## Stage 2 — preprint analysis

Pools the Stage-1 outputs of many experiments into the published figures.

1. **Register experiments** in
   [`SCRIPTS/preprint_analysis/common/config.py`](SCRIPTS/preprint_analysis/common/config.py)
   (the `EXPERIMENTS` dict). For the NRK acid-feedback experiment, point the
   external feedback-log location via an env var:
   `export PE_PIPELINE=/path/to/PE_Pipeline/V5`.

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

## Live closed-loop microscopy / perturbation pipeline

`CLOSED_LOOP_GUI/` is the **live acquisition + feedback system** that generated the
acid-feedback experiments analysed above. A Flask web GUI watches incoming microscope
frames, segments cells with Cellpose, measures per-cell fluorescence, and drives an
**ONIX hardware controller** (over HTTP) to dose acidic / neutral media in a closed
loop. It is **optional and hardware-dependent** — it needs a CUDA GPU and a
networked ONIX2 server — and is **not required to reproduce the figures**.

```bash
python CLOSED_LOOP_GUI/LaunchWebGUI.py   # Flask "Closed-Loop Bio-Control Hub" on http://localhost:5000
```

The GUI bootstraps `CLOSED_LOOP_GUI/config.json`, lets you set the run parameters, and
starts/stops the pipeline (`run_system.sh`, which launches the segmentation,
decision, ONIX-actuation, and monitoring daemons). All paths derive from a single
`global_path` in `CLOSED_LOOP_GUI/config.py`; edit it (and the ONIX endpoint /
experiment templates) before first use.

See [CLOSED_LOOP_GUI/README.md](CLOSED_LOOP_GUI/README.md) for file roles, the data-flow
contract, configuration knobs, and hardware/security notes.

> ⚠️ **Security:** `CLOSED_LOOP_GUI/` issues HTTP requests that create and run
> experiments on networked lab hardware, and the GUI spawns subprocesses. Point it
> only at hardware you control, on a trusted network.

---

## Optional: TUNE_GUI

`TUNE_GUI/` is a Flask app for interactively tuning Stage-1 parameters and
launching the pipeline from a browser. It is a **development tool, not part of
the published analysis** — the figures are fully reproducible from the shell
scripts above.

> ⚠️ **Security:** the tuning GUI binds `0.0.0.0:5001` and launches subprocesses with
> user-supplied paths. Run it only on `localhost` or a trusted machine; do **not**
> expose it to an untrusted network.

---

## Citation

**TODO:** Add the preprint citation (and a `CITATION.cff`) once a DOI is
available.

For the software specifically, please cite this repository and contact
Douglas Hazel (douglas.hazel@tufts.edu).
