# TUNE_GUI — browser-based parameter tuning (optional)

A Flask single-page app for interactively tuning Stage-1 parameters (Cellpose
preview, ROI, frame-shift, Delaunay max-distance) and launching the core
pipeline from a browser. It ports `../preprocess_gui.py` (napari) to the browser
and adds a Run & Monitor tab.

**This is a development/convenience tool, not part of the published analysis** —
the figures are fully reproducible from the project-root `run_*.sh` scripts
without it.

## Launch

```bash
python TUNE_GUI/app.py        # then open http://localhost:5001
```

Override the port with `PORT=<n>` and the bind address with `HOST=<addr>`
(default `127.0.0.1`). The app scans `EXPERIMENTS/` and launches
`SCRIPTS/core_pipeline/` scripts as subprocesses; run it from the project root.

## Tabs

Choose Experiment → Tune Segmentation → Set Tracking → Review & Run → Results
(with a live SSE log/progress monitor).

Review & Run saves the parameters to the experiment's `pipeline_config.yaml`
(the file the `run_*.sh` drivers read; see `configs/example_c2c12_chamber_A.yaml`)
and then runs `run_processes.sh` or `run_post_processes.sh` on it. "Save config
only" writes the file without running anything, and "Preview config" shows it
without writing. Saving keeps keys the GUI doesn't manage, such as
`tracking.workers`, but not comments. Choosing an experiment loads its
`pipeline_config.yaml`, or else the `analysis/config.txt` older runs wrote.

Frame numbers entered in the GUI (representative frame, shift frame, f0_frame)
are positions in the sorted frame list. The config stores the shift frame and
f0_frame as that frame's `timepoint_NNNNN` number, which is what Stage 1
compares against; `stim_frames` is stored as typed. Turning the ROI off writes
`radius: 0`, the `trajectories.py` value for "no ROI".

## Dependencies

Adds `Flask` to the core requirements (plus `Pillow`, `msgpack`, and
`cellpose`+GPU for the segmentation preview). See the repo `requirements.txt` /
`environment.yml`.

## ⚠️ Security

`app.py` writes config files and runs the Stage 1 drivers, so treat it as a local tool:

- It binds `127.0.0.1:5001` by default, reachable only from this machine.
  `HOST=0.0.0.0` exposes it to the network, where anyone who can reach the
  port can run the pipeline. Do that only on a trusted network.
- Every POST must be JSON and carry an `Origin` (or `Referer`) header matching
  the host it was sent to, so a page on another site cannot drive the API.
- Experiments must live under `EXPERIMENTS/`. Paths come only from choosing
  an experiment; `/api/session` rejects path fields, and `session.json` path
  fields outside `EXPERIMENTS/` are dropped at startup.
- Parameters are coerced to numbers before they reach the config, and
  `stim_frames` must be comma-separated integers. The driver is started with
  an argument list, not a shell string, and `pipeline_config.py` shell-quotes
  every value it hands the driver.

## Runtime artifacts

`TUNE_GUI/tmp/` (the pipeline log and image caches) and
`TUNE_GUI/session.json` are created at runtime and are git-ignored — do not commit
them.
