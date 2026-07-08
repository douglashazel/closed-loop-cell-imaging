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
python WEBGUI/app.py        # then open http://localhost:5001
```

Override the port with `PORT=<n>`. The app scans `EXPERIMENTS/` and launches
`SCRIPTS/core_pipeline/` scripts as subprocesses; run it from the project root.

## Tabs

Choose Experiment → Tune Segmentation → Set Tracking → Review & Run → Results
(with a live SSE log/progress monitor).

## Dependencies

Adds `Flask` to the core requirements (plus `Pillow`, `msgpack`, and
`cellpose`+GPU for the segmentation preview). See the repo `requirements.txt` /
`environment.yml`.

## ⚠️ Security

`app.py` binds `0.0.0.0:5001` (all interfaces) and writes/executes generated
shell scripts via `subprocess` using user-supplied experiment paths. This is a
remote-code-execution surface. **Run only on `localhost` or a trusted machine;
never expose it to an untrusted network.**

## Runtime artifacts

`WEBGUI/tmp/` (logs, caches, the generated `current_run.sh`) and
`WEBGUI/session.json` are created at runtime and are git-ignored — do not commit
them.
