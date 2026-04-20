"""Flask web GUI for the Patrick cell-analysis preprocessing pipeline.

Launch:  python preprocess_webgui/app.py
Browse:  http://localhost:5001

Ports the functionality of ../preprocess_gui.py (napari) to a browser and adds
a "Run & Monitor" tab that launches run_processes.sh / run_post_processes.sh
directly with the parameters the user tuned.
"""

import atexit
import io
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image

from cellpose_worker import CellposeJob
from pipeline_logic import (
    all_mean_neighbor_distances,
    array_to_png_bytes,
    centroids_from_seg,
    colorize_labels,
    downsample_to_width,
    get_delaunay_neighbors,
    list_frames_in_dir,
    list_masks_in_dir,
    load_segmentation,
    normalize_gray,
    phase_correlation_shift,
    roi_filter,
    scan_experiments,
    split_frames_png,
)
from state import PipelineState, SessionStore, parse_config_txt

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_EXPERIMENTS_ROOT = os.path.join(_PROJECT_ROOT, "EXPERIMENTS")
_SESSION_JSON = os.path.join(_HERE, "session.json")
_TMP_DIR = os.path.join(_HERE, "tmp")
_PIPELINE_LOG = os.path.join(_TMP_DIR, "pipeline.log")
_PIPELINE_SCRIPT = os.path.join(_TMP_DIR, "current_run.sh")

os.makedirs(_TMP_DIR, exist_ok=True)

# ── Global runtime state ───────────────────────────────────────────────────
session = SessionStore(_SESSION_JSON)
cellpose_job = CellposeJob()
_state_lock = threading.Lock()

pipeline_proc: subprocess.Popen | None = None
pipeline_started_at: float | None = None
pipeline_kind: str | None = None  # "run_processes" | "run_post_processes"
pipeline_log_fh = None

dup_status = {"state": "idle", "message": "", "progress": 0, "total": 0}
dup_thread: threading.Thread | None = None


# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(_HERE, "templates"),
    static_folder=os.path.join(_HERE, "static"),
)


@app.route("/")
def index():
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════════════════
# Session / experiment
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/session", methods=["GET"])
def api_session_get():
    return jsonify(session.snapshot())


@app.route("/api/session", methods=["POST"])
def api_session_post():
    body = request.get_json(force=True) or {}
    return jsonify(session.apply_patch(body))


@app.route("/api/experiments", methods=["GET"])
def api_experiments():
    rows = scan_experiments(_EXPERIMENTS_ROOT)
    return jsonify({"root": _EXPERIMENTS_ROOT, "experiments": rows})


@app.route("/api/experiment", methods=["POST"])
def api_experiment_select():
    body = request.get_json(force=True) or {}
    path = body.get("path") or ""
    if not path:
        return jsonify({"ok": False, "error": "path required"}), 400

    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)
    path = os.path.normpath(path)

    frames_dir = os.path.join(path, "frames")
    masks_dir = os.path.join(path, "masks")
    analysis_dir = os.path.join(path, "analysis")
    if not os.path.isdir(frames_dir):
        return jsonify({"ok": False,
                        "error": f"No 'frames/' subdirectory: {path}"}), 400

    all_frames = list_frames_in_dir(frames_dir)
    all_masks = list_masks_in_dir(masks_dir)

    patch = {
        "global_dir": path,
        "frames_dir": frames_dir,
        "masks_dir": masks_dir,
        "save_path": analysis_dir,
        "all_frames": all_frames,
        "all_masks": all_masks,
        "frame_idx": 0,
        "shift_frame_idx": 1 if len(all_frames) > 1 else 0,
    }

    # Resume-from-config prefill
    cfg_path = os.path.join(analysis_dir, "config.txt")
    resumed = False
    if os.path.isfile(cfg_path):
        prefill = parse_config_txt(cfg_path)
        if prefill:
            patch.update(prefill)
            resumed = True

    snapshot = session.apply_patch(patch)
    snapshot["resumed_from_config"] = resumed
    snapshot["config_txt_path"] = cfg_path if resumed else None
    return jsonify({"ok": True, **snapshot})


# ═══════════════════════════════════════════════════════════════════════════
# Frame + mask imagery
# ═══════════════════════════════════════════════════════════════════════════
def _read_frame(frame_idx: int) -> np.ndarray | None:
    s = session.snapshot()
    frames = s.get("all_frames") or []
    if frame_idx < 0 or frame_idx >= len(frames):
        return None
    path = os.path.join(s["frames_dir"], frames[frame_idx])
    try:
        return np.array(Image.open(path))
    except Exception:
        return None


@app.route("/api/frame/<int:idx>.png", methods=["GET"])
def api_frame_png(idx):
    img = _read_frame(idx)
    if img is None:
        abort(404)
    gray = normalize_gray(img)
    return send_file(array_to_png_bytes(gray, mode="L"), mimetype="image/png")


@app.route("/api/thumbnail/<int:idx>.png", methods=["GET"])
def api_thumbnail_png(idx):
    w = int(request.args.get("w", 120))
    img = _read_frame(idx)
    if img is None:
        abort(404)
    gray = normalize_gray(img)
    small = downsample_to_width(gray, w)
    return send_file(array_to_png_bytes(small, mode="L"), mimetype="image/png")


@app.route("/api/mask/preview.png", methods=["GET"])
def api_mask_preview_png():
    if session.temp_segmentation is None:
        abort(404)
    rgba = colorize_labels(session.temp_segmentation)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/frames/split", methods=["GET"])
def api_frames_split():
    idx = int(request.args.get("idx", 1))
    prev = _read_frame(idx - 1)
    curr = _read_frame(idx)
    if prev is None or curr is None:
        abort(404)
    combined, left_w = split_frames_png(prev, curr)
    buf = array_to_png_bytes(combined, mode="L")
    resp = send_file(buf, mimetype="image/png")
    resp.headers["X-Left-Width"] = str(left_w)
    resp.headers["X-Frame-Height"] = str(combined.shape[0])
    return resp


# ═══════════════════════════════════════════════════════════════════════════
# Cellpose
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/cellpose/run", methods=["POST"])
def api_cellpose_run():
    body = request.get_json(force=True) or {}
    patch = {
        "frame_idx": int(body.get("frame_idx", session.snapshot()["frame_idx"])),
        "flow_threshold": float(body["flow_threshold"]),
        "cellprob_threshold": float(body["cellprob_threshold"]),
        "niter": int(body["niter"]),
        "diameter": int(body["diameter"]),
    }
    s = session.apply_patch(patch)

    img = _read_frame(s["frame_idx"])
    if img is None:
        return jsonify({"ok": False, "error": "no frame loaded"}), 400

    def _on_done(masks):
        session.temp_segmentation = masks

    started = cellpose_job.start(
        img, s["frame_idx"],
        s["flow_threshold"], s["cellprob_threshold"], s["niter"], s["diameter"],
        on_done=_on_done,
    )
    if not started:
        return jsonify({"ok": False, "error": "already running"})
    return jsonify({"ok": True})


@app.route("/api/cellpose/status", methods=["GET"])
def api_cellpose_status():
    with cellpose_job.lock:
        out = dict(cellpose_job.status)
    out["has_mask"] = session.temp_segmentation is not None
    if session.temp_segmentation is not None:
        out["n_cells"] = int(session.temp_segmentation.max())
    return jsonify(out)


@app.route("/api/cellpose/stats.png", methods=["GET"])
def api_cellpose_stats_png():
    """Histogram of cell areas for the current temp_segmentation."""
    seg = session.temp_segmentation
    if seg is None:
        abort(404)
    ids = np.unique(seg)
    ids = ids[ids != 0]
    if len(ids) == 0:
        abort(404)
    areas = np.array([(seg == c).sum() for c in ids], dtype=np.int64)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 2.2), dpi=140)
    ax.hist(areas, bins=30, color="#4f9aa8", edgecolor="#1d3b44")
    ax.set_xlabel("Cell area (px)")
    ax.set_ylabel("Count")
    ax.set_title(f"{len(ids)} cells — mean area {areas.mean():.0f} px",
                 fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ═══════════════════════════════════════════════════════════════════════════
# Shift
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/shift/auto", methods=["POST"])
def api_shift_auto():
    body = request.get_json(force=True) or {}
    idx = int(body.get("idx", session.snapshot()["shift_frame_idx"]))
    prev = _read_frame(idx - 1)
    curr = _read_frame(idx)
    if prev is None or curr is None:
        return jsonify({"ok": False, "error": "frames unavailable"}), 400
    dx, dy = phase_correlation_shift(prev, curr)
    session.apply_patch({"shift_xy": (dx, dy), "shift_frame_idx": idx})
    return jsonify({"ok": True, "dx": dx, "dy": dy})


# ═══════════════════════════════════════════════════════════════════════════
# Max distance (Delaunay)
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/maxdistance/compute", methods=["POST"])
def api_maxdistance_compute():
    body = request.get_json(force=True) or {}
    seg = session.temp_segmentation
    if seg is None:
        return jsonify({"ok": False,
                        "error": "Run Cellpose Preview first (Tab 1)"}), 400

    centroids = centroids_from_seg(seg)
    if len(centroids) < 4:
        return jsonify({"ok": False, "error": "Need ≥4 cells for Delaunay"}), 400

    cell_id = body.get("cell_id")
    if cell_id is None:
        cell_id = random.choice(list(centroids.keys()))
    cell_id = int(cell_id)
    if cell_id not in centroids:
        return jsonify({"ok": False, "error": f"Cell {cell_id} not found"}), 400

    neighbour_ids = get_delaunay_neighbors(cell_id, centroids)
    cy, cx = centroids[cell_id]
    chosen = np.array([cy, cx])
    nbr_coords = np.array([centroids[n] for n in neighbour_ids])
    dists = np.linalg.norm(nbr_coords - chosen, axis=1)
    mean_dist = float(dists.mean()) if len(dists) else 0.0

    # Whole-population distribution for the improved histogram UI
    all_means = all_mean_neighbor_distances(centroids)
    all_vals = sorted(all_means.values()) if all_means else []

    session.apply_patch({"max_distance": mean_dist})
    return jsonify({
        "ok": True,
        "chosen_id": cell_id,
        "chosen_xy": [float(cx), float(cy)],
        "neighbors": [{
            "id": int(n),
            "xy": [float(centroids[n][1]), float(centroids[n][0])],
            "distance": float(d),
        } for n, d in zip(neighbour_ids, dists)],
        "mean_distance": mean_dist,
        "percentile_95": float(np.percentile(all_vals, 95)) if all_vals else 0.0,
        "percentile_75": float(np.percentile(all_vals, 75)) if all_vals else 0.0,
        "median": float(np.percentile(all_vals, 50)) if all_vals else 0.0,
        "population": all_vals[:500],  # cap payload size
    })


# ═══════════════════════════════════════════════════════════════════════════
# ROI
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/roi/count", methods=["POST"])
def api_roi_count():
    body = request.get_json(force=True) or {}
    seg = session.temp_segmentation
    if seg is None:
        return jsonify({"ok": False,
                        "error": "Run Cellpose Preview first (Tab 1)"}), 400

    radius = int(body.get("radius", session.snapshot()["radius"]))
    y_shift = int(body.get("y_shift", session.snapshot()["y_shift"]))
    x_shift = int(body.get("x_shift", session.snapshot()["x_shift"]))
    session.apply_patch({"radius": radius, "y_shift": y_shift, "x_shift": x_shift})

    _, n_inside, (cx, cy) = roi_filter(seg, radius, y_shift, x_shift)
    h, w = seg.shape[:2]
    return jsonify({
        "ok": True,
        "n_inside": n_inside,
        "center_xy": [float(cx), float(cy)],
        "radius": radius,
        "image_size": [int(w), int(h)],
    })


@app.route("/api/roi/mask.png", methods=["GET"])
def api_roi_mask_png():
    seg = session.temp_segmentation
    if seg is None:
        abort(404)
    s = session.snapshot()
    filtered, _, _ = roi_filter(seg, s["radius"], s["y_shift"], s["x_shift"])
    rgba = colorize_labels(filtered)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ═══════════════════════════════════════════════════════════════════════════
# Duplicate (Tab 6)
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/duplicate/run", methods=["POST"])
def api_duplicate_run():
    global dup_thread, dup_status
    body = request.get_json(force=True) or {}
    mode = body.get("mode", "both")  # centers | masks | both
    source = body.get("source", "temp")  # temp | masks_dir

    s = session.snapshot()
    frames = s.get("all_frames") or []
    if not frames:
        return jsonify({"ok": False, "error": "No experiment loaded"}), 400

    masks_dir = s["masks_dir"]
    analysis_dir = s["save_path"]

    mask = None
    if mode in ("masks", "both"):
        if source == "temp":
            mask = session.temp_segmentation
            if mask is None:
                return jsonify({"ok": False,
                                "error": "No temp mask (run Cellpose first)"}), 400
        else:
            if not os.path.isdir(masks_dir):
                return jsonify({"ok": False,
                                "error": f"No masks dir: {masks_dir}"}), 400
            files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".npy")])
            if not files:
                return jsonify({"ok": False, "error": "No mask files found"}), 400
            mask = load_segmentation(os.path.join(masks_dir, files[0]))

    centers = None
    centers_dir = os.path.join(analysis_dir, "cellpose_centers")
    if mode in ("centers", "both"):
        if not os.path.isdir(centers_dir):
            return jsonify({"ok": False,
                            "error": f"No cellpose_centers dir: {centers_dir}"}), 400
        cfs = sorted([f for f in os.listdir(centers_dir) if f.endswith(".npy")])
        if not cfs:
            return jsonify({"ok": False, "error": "No center files found"}), 400
        centers = np.load(os.path.join(centers_dir, cfs[0]), allow_pickle=True)

    with _state_lock:
        if dup_thread is not None and dup_thread.is_alive():
            return jsonify({"ok": False, "error": "already running"})
        dup_status = {"state": "running", "message": f"Duplicating ({mode})...",
                      "progress": 0, "total": len(frames)}
        dup_thread = threading.Thread(
            target=_duplicate_worker,
            args=(frames, masks_dir, centers_dir, mask, centers, mode),
            daemon=True,
        )
        dup_thread.start()
    return jsonify({"ok": True})


def _duplicate_worker(frames, masks_dir, centers_dir, mask, centers, mode):
    global dup_status
    try:
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(centers_dir, exist_ok=True)
        for i, fname in enumerate(frames):
            base = os.path.splitext(fname)[0]
            if mode in ("centers", "both") and centers is not None:
                np.save(os.path.join(centers_dir, f"{base}_centers.npy"), centers)
            if mode in ("masks", "both") and mask is not None:
                np.save(os.path.join(masks_dir, f"{base}.npy"), mask)
            dup_status["progress"] = i + 1
        dup_status = {"state": "done",
                      "message": f"Duplicated to {len(frames)} frames",
                      "progress": len(frames), "total": len(frames)}
    except Exception as e:
        dup_status = {"state": "error", "message": f"Error: {e}",
                      "progress": dup_status.get("progress", 0),
                      "total": len(frames)}


@app.route("/api/duplicate/status", methods=["GET"])
def api_duplicate_status():
    return jsonify(dup_status)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline runner (run_processes.sh / run_post_processes.sh)
# ═══════════════════════════════════════════════════════════════════════════
def _build_run_processes_script(s: dict) -> str:
    """Render a shell script equivalent to run_processes.sh but with the
    session parameters baked in. Mirrors preprocess_gui.py::_generate_script."""
    global_dir = s["global_dir"]
    rel_dir = os.path.relpath(global_dir, _PROJECT_ROOT)
    shift_x, shift_y = s["shift_xy"]
    return f"""#!/bin/bash
set -euo pipefail

cd "{_PROJECT_ROOT}"

GLOBAL_DIR="{rel_dir}"
IMAGE_DIR="${{GLOBAL_DIR}}/frames"
MASK_DIR="${{GLOBAL_DIR}}/masks"
SAVE_PATH="${{GLOBAL_DIR}}/analysis"

SCRIPT1="SCRIPTS/segmentation.py"
SCRIPT2="SCRIPTS/trajectories.py"

FLOW_THRESHOLD={s['flow_threshold']}
CELLPROB_THRESHOLD={s['cellprob_threshold']}
NITER={s['niter']}
DIAMETER={s['diameter']}

MAX_DISTANCE={s['max_distance']:.1f}
GRACE_PERIOD={s['grace_period']}
RADIUS={s['radius']}
RADIUS_Y={s['y_shift']}
RADIUS_X={s['x_shift']}
SHIFT_FRAME={s['shift_frame_idx']}
SHIFT_XY="{shift_x} {shift_y}"
SAVE_INTERVAL={s['save_interval']}

mkdir -p "$SAVE_PATH"
CONFIG_FILE="${{SAVE_PATH}}/config.txt"
cat > "$CONFIG_FILE" <<CFGEOF
Run date: $(date)

[PATHS]
GLOBAL_DIR=$GLOBAL_DIR
IMAGE_DIR=$IMAGE_DIR
MASK_DIR=$MASK_DIR
SAVE_PATH=$SAVE_PATH

[CELLPOSE]
FLOW_THRESHOLD=$FLOW_THRESHOLD
CELLPROB_THRESHOLD=$CELLPROB_THRESHOLD
NITER=$NITER
DIAMETER=$DIAMETER

[TRAJECTORIES]
MAX_DISTANCE=$MAX_DISTANCE
GRACE_PERIOD=$GRACE_PERIOD
RADIUS=$RADIUS
RADIUS_Y=$RADIUS_Y
RADIUS_X=$RADIUS_X
SHIFT_FRAME=$SHIFT_FRAME
SHIFT_XY=$SHIFT_XY
SAVE_INTERVAL=$SAVE_INTERVAL
CFGEOF
echo "Config saved to $CONFIG_FILE"

echo "--- Accessing ${{GLOBAL_DIR}} ---"

echo ">>> STAGE: SEGMENTATION <<<"
python3 -u "$SCRIPT1" \\
    --image_dir "$IMAGE_DIR" \\
    --mask_dir "$MASK_DIR" \\
    --flow_threshold "$FLOW_THRESHOLD" \\
    --cellprob_threshold "$CELLPROB_THRESHOLD" \\
    --niter "$NITER" \\
    --diameter "$DIAMETER" &
PID1=$!

sleep 5

echo ">>> STAGE: TRAJECTORIES <<<"
python3 -u "$SCRIPT2" \\
    --mask_dir "$MASK_DIR" \\
    --image_dir "$IMAGE_DIR" \\
    --save_path "$SAVE_PATH" \\
    --max_distance "$MAX_DISTANCE" \\
    --grace_period "$GRACE_PERIOD" \\
    --radius "$RADIUS" \\
    --radius_y "$RADIUS_Y" \\
    --radius_x "$RADIUS_X" \\
    --shift_frame "$SHIFT_FRAME" \\
    --shift_xy $SHIFT_XY \\
    --save_interval "$SAVE_INTERVAL" &
PID2=$!

wait $PID1 $PID2

echo ">>> STAGE: PRE-ANALYSIS <<<"
python3 -u SCRIPTS/PreAnalysis.py \\
    --exp "$GLOBAL_DIR" \\
    --analysis_dir "$SAVE_PATH"

echo ">>> DONE <<<"
"""


def _build_run_post_processes_script(s: dict, f0_frame: int, stim_frames: str) -> str:
    global_dir = s["global_dir"]
    rel_dir = os.path.relpath(global_dir, _PROJECT_ROOT)
    return f"""#!/bin/bash
set -euo pipefail

cd "{_PROJECT_ROOT}"

GLOBAL_DIR="{rel_dir}"
IMAGE_DIR="${{GLOBAL_DIR}}/frames"
ANALYSIS_DIR="${{GLOBAL_DIR}}/analysis"

F0_FRAME={f0_frame}
STIM_FRAMES="{stim_frames}"

echo "--- Accessing ${{GLOBAL_DIR}} ---"
echo ">>> STAGE: POST-ANALYSIS <<<"
python3 -u SCRIPTS/PostAnalysis.py \\
    --exp "$GLOBAL_DIR" \\
    --image_dir "$IMAGE_DIR" \\
    --analysis_dir "$ANALYSIS_DIR" \\
    --f0_frame "$F0_FRAME" \\
    --stim_frames "$STIM_FRAMES"

echo ">>> DONE <<<"
"""


@app.route("/api/pipeline/run", methods=["POST"])
def api_pipeline_run():
    global pipeline_proc, pipeline_started_at, pipeline_kind, pipeline_log_fh
    body = request.get_json(force=True) or {}
    kind = body.get("kind", "run_processes")
    if kind not in ("run_processes", "run_post_processes"):
        return jsonify({"ok": False, "error": f"unknown kind: {kind}"}), 400

    with _state_lock:
        if pipeline_proc is not None and pipeline_proc.poll() is None:
            return jsonify({"ok": False, "error": "already running",
                            "pid": pipeline_proc.pid})

        s = session.snapshot()
        if not s.get("global_dir"):
            return jsonify({"ok": False, "error": "no experiment loaded"}), 400

        if kind == "run_processes":
            script_text = _build_run_processes_script(s)
        else:
            f0 = int(body.get("f0_frame", s.get("f0_frame", 1)))
            stim = str(body.get("stim_frames", s.get("stim_frames", "")))
            session.apply_patch({"f0_frame": f0, "stim_frames": stim})
            script_text = _build_run_post_processes_script(s, f0, stim)

        with open(_PIPELINE_SCRIPT, "w") as f:
            f.write(script_text)
        os.chmod(_PIPELINE_SCRIPT, 0o755)

        # Truncate log
        if pipeline_log_fh is not None:
            try:
                pipeline_log_fh.close()
            except Exception:
                pass
        pipeline_log_fh = open(_PIPELINE_LOG, "w")
        pipeline_log_fh.write(f"=== {kind} started at {time.ctime()} ===\n")
        pipeline_log_fh.flush()

        pipeline_proc = subprocess.Popen(
            ["bash", _PIPELINE_SCRIPT],
            cwd=_PROJECT_ROOT,
            stdout=pipeline_log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        pipeline_started_at = time.time()
        pipeline_kind = kind
        return jsonify({"ok": True, "pid": pipeline_proc.pid,
                        "kind": kind, "script": _PIPELINE_SCRIPT})


@app.route("/api/pipeline/stop", methods=["POST"])
def api_pipeline_stop():
    global pipeline_proc, pipeline_log_fh
    with _state_lock:
        if pipeline_proc is None or pipeline_proc.poll() is not None:
            return jsonify({"ok": False, "error": "not running"})
        try:
            os.killpg(os.getpgid(pipeline_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(30):
            if pipeline_proc.poll() is not None:
                break
            time.sleep(0.1)
        else:
            try:
                os.killpg(os.getpgid(pipeline_proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if pipeline_log_fh is not None:
            pipeline_log_fh.write(f"\n=== STOPPED at {time.ctime()} ===\n")
            pipeline_log_fh.flush()
        pipeline_proc = None
        return jsonify({"ok": True})


@app.route("/api/pipeline/status", methods=["GET"])
def api_pipeline_status():
    global pipeline_proc
    out = {
        "running": False, "pid": None, "exit_code": None,
        "kind": pipeline_kind,
        "started_at": pipeline_started_at,
        "uptime": (time.time() - pipeline_started_at) if pipeline_started_at else 0,
    }
    if pipeline_proc is None:
        return jsonify(out)
    rc = pipeline_proc.poll()
    if rc is None:
        out.update({"running": True, "pid": pipeline_proc.pid})
    else:
        out.update({"pid": pipeline_proc.pid, "exit_code": rc})
    out["stage"] = _infer_stage()
    return jsonify(out)


_STAGE_MARKERS = [
    ("post_analysis", "POST-ANALYSIS"),
    ("pre_analysis",  "PRE-ANALYSIS"),
    ("trajectories",  "TRAJECTORIES"),
    ("segmentation",  "SEGMENTATION"),
]


def _infer_stage() -> str | None:
    if not os.path.isfile(_PIPELINE_LOG):
        return None
    try:
        with open(_PIPELINE_LOG) as f:
            text = f.read()
    except Exception:
        return None
    for key, marker in _STAGE_MARKERS:
        if f">>> STAGE: {marker} <<<" in text:
            return key
    return None


@app.route("/api/pipeline/log", methods=["GET"])
def api_pipeline_log():
    pos = int(request.args.get("pos", 0))
    if not os.path.isfile(_PIPELINE_LOG):
        return jsonify({"text": "", "pos": 0, "exists": False})
    try:
        size = os.path.getsize(_PIPELINE_LOG)
        if pos > size:
            pos = 0
        with open(_PIPELINE_LOG, "r") as f:
            f.seek(pos)
            text = f.read()
            new_pos = f.tell()
        return jsonify({"text": text, "pos": new_pos, "exists": True})
    except Exception as e:
        return jsonify({"text": "", "pos": pos, "exists": True, "error": str(e)})


@app.route("/api/pipeline/progress", methods=["GET"])
def api_pipeline_progress():
    s = session.snapshot()
    total = len(s.get("all_frames") or [])
    masks_dir = s.get("masks_dir") or ""
    save_path = s.get("save_path") or ""

    n_masks = 0
    if masks_dir and os.path.isdir(masks_dir):
        n_masks = sum(1 for f in os.listdir(masks_dir) if f.endswith(".npy"))

    traj_pct = 0.0
    lumi_path = os.path.join(save_path, "luminosity.json")
    if os.path.isfile(lumi_path):
        traj_pct = 100.0
    else:
        lumi_partial = os.path.join(save_path, "luminosity_partial.json")
        if os.path.isfile(lumi_partial):
            traj_pct = 50.0

    pre_done = os.path.isdir(os.path.join(save_path, "plots"))
    post_done = os.path.isfile(os.path.join(save_path, "post_analysis_complete.txt"))

    return jsonify({
        "total_frames": total,
        "segmentation": {
            "done": n_masks, "total": total,
            "pct": (100.0 * n_masks / total) if total else 0.0,
        },
        "trajectories": {"pct": traj_pct},
        "pre_analysis": {"pct": 100.0 if pre_done else 0.0},
        "post_analysis": {"pct": 100.0 if post_done else 0.0},
    })


@app.route("/api/pipeline/luminosity.png", methods=["GET"])
def api_pipeline_luminosity_png():
    """Render matplotlib plot of the in-progress luminosity.json if available."""
    s = session.snapshot()
    save_path = s.get("save_path") or ""
    candidates = [
        os.path.join(save_path, "luminosity_complete.json"),
        os.path.join(save_path, "luminosity.json"),
        os.path.join(save_path, "luminosity_partial.json"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        abort(404)

    try:
        import msgpack
        with open(path, "rb") as f:
            data = msgpack.unpackb(f.read(), raw=False)
    except Exception:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            abort(500)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3), dpi=140)
    if isinstance(data, dict) and data:
        # data is typically {cell_id: [luminosity_per_frame]}
        n_cells = 0
        for cid, trace in list(data.items())[:200]:
            if not isinstance(trace, (list, tuple)):
                continue
            ax.plot(trace, linewidth=0.6, alpha=0.4)
            n_cells += 1
        if n_cells:
            arr = []
            for trace in data.values():
                if isinstance(trace, (list, tuple)):
                    arr.append(list(trace))
            if arr:
                min_len = min(len(x) for x in arr)
                mat = np.array([x[:min_len] for x in arr], dtype=np.float32)
                ax.plot(mat.mean(axis=0), linewidth=1.8, color="black",
                        label="mean")
                ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Luminosity — {n_cells} cells (live)",
                     fontsize=10, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean pixel intensity")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ═══════════════════════════════════════════════════════════════════════════
# Script preview (Tab 7 summary / export)
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/script/preview", methods=["GET"])
def api_script_preview():
    kind = request.args.get("kind", "run_processes")
    s = session.snapshot()
    if not s.get("global_dir"):
        return jsonify({"ok": False, "error": "no experiment"}), 400
    if kind == "run_post_processes":
        text = _build_run_post_processes_script(
            s, s.get("f0_frame", 1), s.get("stim_frames", ""),
        )
    else:
        text = _build_run_processes_script(s)
    return jsonify({"ok": True, "script": text})


# ═══════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════
def _cleanup():
    global pipeline_proc
    if pipeline_proc is not None and pipeline_proc.poll() is None:
        print("[web gui] cleaning up pipeline subprocess", flush=True)
        try:
            os.killpg(os.getpgid(pipeline_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass


atexit.register(_cleanup)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"[web gui] starting on http://0.0.0.0:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False, use_reloader=False)
