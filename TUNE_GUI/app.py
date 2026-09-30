"""Flask web GUI for the Cell Trainer preprocessing pipeline.

Launch:  python TUNE_GUI/app.py
Browse:  http://localhost:5001

"""

import atexit
import hashlib
import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

import numpy as np
from flask import (
    Flask, Response, abort, jsonify, make_response, render_template, request,
    send_file, stream_with_context,
)
from PIL import Image

from cellpose_worker import CellposeJob
from pipeline_logic import (
    centroids_from_seg,
    colorize_labels,
    downsample_to_width,
    get_delaunay_neighbors,
    list_frames_in_dir,
    list_masks_in_dir,
    load_segmentation,
    normalize_gray,
    scan_experiments,
    split_frames_png,
    timepoint_token,
)
from state import (
    PATH_FIELDS, PipelineState, SessionStore, parse_config_txt, patch_from_config, within_root,
)

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_EXPERIMENTS_ROOT = os.path.realpath(os.path.join(_PROJECT_ROOT, "EXPERIMENTS"))
_SESSION_JSON = os.path.join(_HERE, "session.json")
_TMP_DIR = os.path.join(_HERE, "tmp")
_CACHE_DIR = os.path.join(_TMP_DIR, "cache")
_PIPELINE_LOG = os.path.join(_TMP_DIR, "pipeline.log")

# The pipeline_config.yaml schema the run_*.sh drivers read lives with the core scripts
sys.path.append(os.path.join(_PROJECT_ROOT, "SCRIPTS", "core_pipeline"))
import pipeline_config

os.makedirs(_TMP_DIR, exist_ok=True)
os.makedirs(_CACHE_DIR, exist_ok=True)

# ── Global runtime state ───────────────────────────────────────────────────
session = SessionStore(_SESSION_JSON, _EXPERIMENTS_ROOT)
cellpose_job = CellposeJob()
_state_lock = threading.Lock()

pipeline_proc: subprocess.Popen | None = None
pipeline_pgid: int | None = None  # outlives pipeline_proc: see api_pipeline_stop
pipeline_started_at: float | None = None
pipeline_kind: str | None = None  # "run_processes" | "run_post_processes"
pipeline_log_fh = None

dup_status = {"state": "idle", "message": "", "progress": 0, "total": 0}
dup_thread: threading.Thread | None = None

stage_cache = {"sig": None, "stage": None}


# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(_HERE, "templates"),
    static_folder=os.path.join(_HERE, "static"),
)


@app.before_request
def _refuse_cross_site_writes():
    """CSRF guard: a state-changing request must be JSON and come from a page
    this server served (Origin, else Referer, host:port equals Host). A form
    post or fetch from another site cannot satisfy both."""
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return None
    source = request.headers.get("Origin") or request.headers.get("Referer") or ""
    if not request.is_json or urlsplit(source).netloc.lower() != request.host.lower():
        return jsonify({"ok": False,
                        "error": "refused: POST must be JSON from this app's own page"}), 403
    return None


def _json_body() -> dict:
    """The request's JSON object; aborts with 400 if missing or malformed."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        abort(make_response(
            jsonify({"ok": False, "error": "expected a JSON object body"}), 400))
    return body


def _cache_namespace() -> str:
    s = session.snapshot()
    root = s.get("global_dir") or "no-experiment"
    return hashlib.sha1(root.encode("utf-8")).hexdigest()[:16]


def _cache_path(kind: str, *parts) -> str:
    ns_dir = os.path.join(_CACHE_DIR, _cache_namespace())
    os.makedirs(ns_dir, exist_ok=True)
    key = json.dumps([kind, *parts], sort_keys=True, default=str)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(ns_dir, f"{kind}-{digest}.png")


def _file_sig(path: str) -> tuple:
    try:
        st = os.stat(path)
        return (path, int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return (path, 0, 0)


def _png_response(path: str, max_age: int = 86400):
    resp = send_file(path, mimetype="image/png", max_age=max_age, conditional=True)
    resp.headers["Cache-Control"] = f"public, max-age={max_age}"
    return resp


def _write_png_if_missing(path: str, arr: np.ndarray, mode: str = "L") -> None:
    if os.path.isfile(path):
        return
    tmp = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    Image.fromarray(arr, mode=mode).save(tmp, format="PNG")
    os.replace(tmp, path)


def _frame_path(frame_idx: int) -> str | None:
    s = session.snapshot()
    frames = s.get("all_frames") or []
    if frame_idx < 0 or frame_idx >= len(frames):
        return None
    return os.path.join(s["frames_dir"], frames[frame_idx])


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
    body = _json_body()
    blocked = sorted(set(body) & set(PATH_FIELDS))
    if blocked:
        return jsonify({"ok": False,
                        "error": f"{', '.join(blocked)} can only be set by "
                                 f"choosing an experiment"}), 400
    try:
        return jsonify(session.apply_patch(body))
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/experiments", methods=["GET"])
def api_experiments():
    rows = scan_experiments(_EXPERIMENTS_ROOT)
    return jsonify({"root": _EXPERIMENTS_ROOT, "experiments": rows})


@app.route("/api/experiment", methods=["POST"])
def api_experiment_select():
    body = _json_body()
    path = body.get("path") or ""
    if not path or not isinstance(path, str):
        return jsonify({"ok": False, "error": "path required"}), 400

    if cellpose_job.is_running():
        return jsonify({
            "ok": False,
            "error": "Cellpose is still running on the current experiment. "
                     "Wait for it to finish before switching.",
        }), 409

    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)
    path = os.path.realpath(path)
    if not within_root(path, _EXPERIMENTS_ROOT):
        return jsonify({"ok": False,
                        "error": f"Experiment must be inside {_EXPERIMENTS_ROOT}"}), 400

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
        "last_preview_frame": -1,
        "preview_mask_source": "",
        "segmentation_reviewed": False,
        "last_roi_count": 0,
        "validation_warnings": [],
    }

    # Stimulus frames belong to one experiment
    patch.update({"f0_frame": 1, "stim_frames": ""})

    # Resume from the experiment's pipeline_config.yaml
    cfg_path = os.path.join(path, pipeline_config.CONFIG_NAME)
    prefill, config_error = {}, None
    if os.path.isfile(cfg_path):
        try:
            values = pipeline_config.validate(
                pipeline_config.read_config(cfg_path), [], cfg_path)
            prefill = patch_from_config(values, all_frames)
        except pipeline_config.ConfigError as e:
            config_error = str(e)
    elif os.path.isfile(os.path.join(analysis_dir, "config.txt")):
        cfg_path = os.path.join(analysis_dir, "config.txt")
        prefill = parse_config_txt(cfg_path, all_frames)

    session.temp_segmentation = None
    session.temp_segmentation_version += 1
    snapshot = session.apply_patch(patch)
    if prefill:
        try:
            snapshot = session.apply_patch(prefill)
        except ValueError as e:  # e.g. a fractional shift_xy
            config_error, prefill = f"{cfg_path}: {e}", {}
    snapshot["resumed_from_config"] = bool(prefill)
    snapshot["config_path"] = cfg_path if prefill else None
    snapshot["config_error"] = config_error
    return jsonify({"ok": True, **snapshot})


# ═══════════════════════════════════════════════════════════════════════════
# Frame + mask imagery
# ═══════════════════════════════════════════════════════════════════════════
def _read_frame(frame_idx: int) -> np.ndarray | None:
    path = _frame_path(frame_idx)
    if path is None:
        return None
    try:
        return np.array(Image.open(path))
    except Exception:
        return None


@app.route("/api/frame/<int:idx>.png", methods=["GET"])
def api_frame_png(idx):
    path = _frame_path(idx)
    if path is None:
        abort(404)
    width = int(request.args.get("w", 0))
    cache = _cache_path("frame", idx, width, _file_sig(path))
    if not os.path.isfile(cache):
        img = _read_frame(idx)
        if img is None:
            abort(404)
        gray = normalize_gray(img)
        if width > 0:
            gray = downsample_to_width(gray, width)
        _write_png_if_missing(cache, gray, mode="L")
    return _png_response(cache)


@app.route("/api/thumbnail/<int:idx>.png", methods=["GET"])
def api_thumbnail_png(idx):
    w = int(request.args.get("w", 120))
    path = _frame_path(idx)
    if path is None:
        abort(404)
    cache = _cache_path("thumb", idx, w, _file_sig(path))
    if not os.path.isfile(cache):
        img = _read_frame(idx)
        if img is None:
            abort(404)
        gray = normalize_gray(img)
        small = downsample_to_width(gray, w)
        _write_png_if_missing(cache, small, mode="L")
    return _png_response(cache)


@app.route("/api/mask/preview.png", methods=["GET"])
def api_mask_preview_png():
    if session.temp_segmentation is None:
        abort(404)
    # Default alpha is 255 (fully opaque color fill)
    alpha = int(request.args.get("alpha", 255))
    cache = _cache_path("mask", session.temp_segmentation_version, alpha)
    if not os.path.isfile(cache):
        rgba = colorize_labels(session.temp_segmentation, alpha=alpha)
        _write_png_if_missing(cache, rgba, mode="RGBA")
    return _png_response(cache, max_age=3600)


@app.route("/api/frames/split", methods=["GET"])
def api_frames_split():
    idx = int(request.args.get("idx", 1))
    prev_path = _frame_path(idx - 1)
    curr_path = _frame_path(idx)
    if prev_path is None or curr_path is None:
        abort(404)
    cache = _cache_path("split", idx, _file_sig(prev_path), _file_sig(curr_path))
    left_w_path = cache + ".json"
    if not os.path.isfile(cache):
        prev = _read_frame(idx - 1)
        curr = _read_frame(idx)
        if prev is None or curr is None:
            abort(404)
        combined, left_w = split_frames_png(prev, curr)
        _write_png_if_missing(cache, combined, mode="L")
        with open(left_w_path, "w") as f:
            json.dump({"left_w": left_w, "height": int(combined.shape[0])}, f)
    meta = {"left_w": 0, "height": 0}
    if os.path.isfile(left_w_path):
        with open(left_w_path) as f:
            meta = json.load(f)
    resp = _png_response(cache)
    resp.headers["X-Left-Width"] = str(meta.get("left_w", 0))
    resp.headers["X-Frame-Height"] = str(meta.get("height", 0))
    return resp


# ═══════════════════════════════════════════════════════════════════════════
# Cellpose
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/cellpose/run", methods=["POST"])
def api_cellpose_run():
    body = _json_body()
    try:
        patch = {
            "frame_idx": int(body.get("frame_idx", session.snapshot()["frame_idx"])),
            "flow_threshold": float(body["flow_threshold"]),
            "cellprob_threshold": float(body["cellprob_threshold"]),
            "niter": int(body["niter"]),
            "diameter": int(body["diameter"]),
        }
        s = session.apply_patch(patch)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": f"bad Cellpose parameters: {e}"}), 400

    img_path = _frame_path(s["frame_idx"])
    if img_path is None or not os.path.isfile(img_path):
        return jsonify({"ok": False, "error": "no frame loaded"}), 400

    def _on_done(masks):
        session.temp_segmentation = masks
        session.temp_segmentation_version += 1
        session.apply_patch({
            "last_preview_frame": s["frame_idx"],
            "preview_mask_source": "cellpose_preview",
            "segmentation_reviewed": False,
            "last_roi_count": 0,
        })
        # Pre-render the default-alpha preview PNG so the first GET hits the disk cache instead of paying the colorize + encode cost inline
        try:
            cache = _cache_path("mask", session.temp_segmentation_version, 255)
            if not os.path.isfile(cache):
                rgba = colorize_labels(masks, alpha=255)
                _write_png_if_missing(cache, rgba, mode="RGBA")
        except Exception as e:
            print(f"[mask preview pre-render] {e}", flush=True)

    started = cellpose_job.start(
        img_path, s["frame_idx"],
        s["flow_threshold"], s["cellprob_threshold"], s["niter"], s["diameter"],
        on_done=_on_done,
    )
    if not started:
        return jsonify({"ok": False, "error": "already running"})
    return jsonify({"ok": True})


def _cellpose_status_snapshot() -> dict:
    with cellpose_job.lock:
        out = dict(cellpose_job.status)
        out["status_version"] = cellpose_job.status_version
    out["has_mask"] = session.temp_segmentation is not None
    out["mask_version"] = session.temp_segmentation_version
    if session.temp_segmentation is not None:
        out["n_cells"] = int(session.temp_segmentation.max())
    return out


@app.route("/api/cellpose/stream", methods=["GET"])
def api_cellpose_stream():
    """Server-Sent Events stream of cellpose worker status.

    Pushes a new event whenever the worker's status_version or the mask
    version advance. Used by the frontend to react instantly to running →
    done transitions without polling.
    """
    @stream_with_context
    def gen():
        last_status_version = -1
        last_mask_version = -1
        last_keepalive = time.time()
        # Send an initial snapshot so the client doesn't have to GET /status
        snap = _cellpose_status_snapshot()
        last_status_version = snap.get("status_version", 0)
        last_mask_version = snap.get("mask_version", 0)
        yield f"event: status\ndata: {json.dumps(snap)}\n\n"
        while True:
            with cellpose_job.lock:
                sv = cellpose_job.status_version
            mv = session.temp_segmentation_version
            if sv != last_status_version or mv != last_mask_version:
                last_status_version = sv
                last_mask_version = mv
                snap = _cellpose_status_snapshot()
                yield f"event: status\ndata: {json.dumps(snap)}\n\n"
                last_keepalive = time.time()
            else:
                now = time.time()
                if now - last_keepalive > 15:
                    yield ": keepalive\n\n"
                    last_keepalive = now
            time.sleep(0.15)

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/cellpose/centroids", methods=["GET"])
def api_cellpose_centroids():
    """Return centroid coords for every cell in the current preview.

    Used by the ROI tab to count cells inside the circle entirely
    client-side — no per-drag server round trip.
    """
    seg = session.temp_segmentation
    if seg is None:
        return jsonify({"ok": False,
                        "error": "Run Cellpose Preview first"}), 400
    centroids = centroids_from_seg(seg)
    h, w = seg.shape[:2]
    items = [{"id": int(cid), "cx": float(rc[1]), "cy": float(rc[0])}
             for cid, rc in centroids.items()]
    return jsonify({
        "ok": True,
        "mask_version": session.temp_segmentation_version,
        "image_size": [int(w), int(h)],
        "centroids": items,
    })


@app.route("/api/cellpose/review", methods=["POST"])
def api_cellpose_review():
    body = _json_body()
    reviewed = bool(body.get("reviewed", True))
    return jsonify(session.apply_patch({"segmentation_reviewed": reviewed}))


# ═══════════════════════════════════════════════════════════════════════════
# Max distance (Delaunay)
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/maxdistance/compute", methods=["POST"])
def api_maxdistance_compute():
    body = _json_body()
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
        "visualization_url": (
            f"/api/maxdistance/visualization.png"
            f"?cell_id={cell_id}&v={session.temp_segmentation_version}"
        ),
    })


@app.route("/api/maxdistance/visualization.png", methods=["GET"])
def api_maxdistance_visualization_png():
    seg = session.temp_segmentation
    if seg is None:
        abort(404)
    cell_id = int(request.args.get("cell_id", 0))
    centroids = centroids_from_seg(seg)
    if cell_id not in centroids:
        abort(404)

    frame_idx = session.snapshot().get("last_preview_frame", -1)
    if frame_idx < 0:
        frame_idx = session.snapshot().get("frame_idx", 0)
    frame_path = _frame_path(frame_idx)
    if frame_path is None:
        abort(404)

    neighbor_ids = get_delaunay_neighbors(cell_id, centroids)
    cy, cx = centroids[cell_id]
    chosen_rc = np.array([cy, cx])
    nbr_coords = np.array([centroids[n] for n in neighbor_ids])
    dists = np.linalg.norm(nbr_coords - chosen_rc, axis=1) if len(neighbor_ids) else np.array([])
    mean_dist = float(dists.mean()) if len(dists) else 0.0

    cache = _cache_path(
        "maxdistance-viz",
        session.temp_segmentation_version,
        cell_id,
        _file_sig(frame_path),
    )
    if os.path.isfile(cache):
        return _png_response(cache, max_age=3600)

    image = _read_frame(frame_idx)
    if image is None:
        abort(404)
    image = image.astype(np.float32)
    if image.ndim == 3:
        image = image.mean(axis=-1)
    lo, hi = float(image.min()), float(image.max())
    img_norm = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    ax.imshow(img_norm, cmap="gray", interpolation="none")

    overlay = np.zeros((*seg.shape, 4), dtype=float)
    overlay[seg == cell_id] = [0.0, 1.0, 1.0, 0.45]
    for nid in neighbor_ids:
        overlay[seg == nid] = [1.0, 0.55, 0.0, 0.45]
    ax.imshow(overlay, interpolation="none")

    ax.scatter(cx, cy, color="cyan", s=80, zorder=5,
               label=f"Chosen cell (ID {cell_id})")
    for i, nid in enumerate(neighbor_ids):
        nr, nc = centroids[nid]
        ax.scatter(nc, nr, color="orange", s=50, zorder=5,
                   label="Neighbours" if i == 0 else None)
        ax.plot([cx, nc], [cy, nr], color="white", linewidth=0.8, alpha=0.65)

    all_rc = np.vstack([chosen_rc, nbr_coords]) if len(neighbor_ids) else np.array([chosen_rc])
    pad = 150
    rmin, cmin = (all_rc.min(axis=0) - pad).clip(min=0)
    rmax, cmax = np.minimum(all_rc.max(axis=0) + pad, [seg.shape[0], seg.shape[1]])
    ax.set_xlim(cmin, cmax)
    ax.set_ylim(rmax, rmin)
    ax.set_title(
        f"Frame {frame_idx} | Cell ID {cell_id} | "
        f"{len(neighbor_ids)} Delaunay neighbours | Mean dist = {mean_dist:.1f} px",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(cache, format="png", bbox_inches="tight")
    plt.close(fig)
    return _png_response(cache, max_age=3600)


# ═══════════════════════════════════════════════════════════════════════════
# Duplicate (Tab 6)
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/duplicate/run", methods=["POST"])
def api_duplicate_run():
    global dup_thread, dup_status
    body = _json_body()
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
# Pipeline validation + runner (run_processes.sh / run_post_processes.sh)
# ═══════════════════════════════════════════════════════════════════════════
def _validation_state(run_mode: str = "full") -> dict:
    s = session.snapshot()
    checks = []
    warnings = []

    def add(key, ok, label, detail):
        row = {"key": key, "ok": bool(ok), "label": label, "detail": detail}
        checks.append(row)
        if not ok:
            warnings.append(row)

    frames = s.get("all_frames") or []
    masks = s.get("all_masks") or []
    save_path = s.get("save_path") or ""
    save_parent = os.path.dirname(save_path) if save_path else ""
    needs_existing_masks = run_mode in ("existing_masks", "post")

    add("experiment", bool(s.get("global_dir")), "Experiment selected",
        s.get("global_dir") or "Choose an experiment folder first.")
    add("frames", bool(frames), "Frames found",
        f"{len(frames)} frame files found." if frames else "No images were found in frames/.")

    if needs_existing_masks:
        add("masks", bool(masks), "Existing masks available",
            f"{len(masks)} mask files found." if masks else
            "Run segmentation first or choose Full analysis.")
    elif run_mode != "preview_only":
        add("preview", session.temp_segmentation is not None, "Cellpose preview made",
            "Preview segmentation is available." if session.temp_segmentation is not None else
            "Run Cellpose on a representative frame before launching.")
        add("reviewed", bool(s.get("segmentation_reviewed")), "Segmentation reviewed",
            "Marked as looks good." if s.get("segmentation_reviewed") else
            "Use the overlay review button to confirm the preview.")

    if run_mode != "preview_only":
        roi_on = bool(s.get("roi_enabled", True))
        if roi_on:
            add("roi", int(s.get("last_roi_count") or 0) > 0, "ROI has cells",
                f"{s.get('last_roi_count')} cells inside ROI." if s.get("last_roi_count") else
                "Open Set Tracking and refresh the ROI.")
        else:
            add("roi", True, "ROI disabled", "Pipeline will include every cell.")
        add("max_distance", float(s.get("max_distance") or 0) > 0, "Track distance set",
            f"{float(s.get('max_distance') or 0):.1f} px.")
        shift_idx = int(s.get("shift_frame_idx") or 0)
        shift_ok = (tuple(s.get("shift_xy") or (0, 0)) == (0, 0)
                    or 0 <= shift_idx < len(frames))
        add("shift", shift_ok, "Frame shift reviewed",
            f"shift_frame {shift_idx}, shift_xy {s.get('shift_xy')}." if shift_ok else
            f"shift_frame {shift_idx} is outside frames 0..{len(frames) - 1}.")

    writable = bool(save_parent and os.path.isdir(save_parent) and os.access(save_parent, os.W_OK))
    add("writable", writable, "Output folder writable",
        save_path if writable else "The experiment folder is not writable.")

    session.apply_patch({"validation_warnings": warnings})
    return {"ok": not warnings, "checks": checks, "warnings": warnings}


@app.route("/api/validation", methods=["GET"])
def api_validation():
    mode = request.args.get("mode", session.snapshot().get("last_run_mode", "full"))
    return jsonify(_validation_state(mode))


_RUN_MODES = ("full", "existing_masks", "preview_only", "post")


def _frame_token(frames: list, idx: int) -> int:
    """Map a 0-based position in the sorted frame list to that frame's
    timepoint_NNNNN token, which is what trajectories.py --shift_frame and
    PostAnalysis.py --f0_frame compare against."""
    if not 0 <= idx < len(frames):
        raise ValueError(f"frame index {idx} is outside 0..{len(frames) - 1}")
    token = timepoint_token(frames[idx])
    if token < 0:
        raise ValueError(f"{frames[idx]} has no timepoint_NNNNN token")
    return token


def _check_stim_frames(text: str) -> str:
    """Comma-separated frame numbers (whitespace ignored); "" means none."""
    stim = re.sub(r"\s+", "", str(text))
    if stim and not re.fullmatch(r"\d+(,\d+)*", stim):
        raise ValueError(f"stim_frames must be comma-separated frame numbers, "
                         f"got {text!r:.80}")
    return stim


def _config_updates(s: dict) -> dict:
    """The session's parameters as pipeline_config.yaml sections. Frame
    positions become timepoint_NNNNN tokens, which is what trajectories.py
    --shift_frame and PostAnalysis.py --f0_frame compare against.

    Raises ValueError/TypeError/KeyError if the session holds something
    unusable (the calling route returns 400)."""
    frames = s["all_frames"]
    shift_x, shift_y = (int(v) for v in s["shift_xy"])
    shift_idx = int(s["shift_frame_idx"])
    if (shift_x, shift_y) == (0, 0) and not 0 <= shift_idx < len(frames):
        shift_frame = shift_idx
    else:
        shift_frame = _frame_token(frames, shift_idx)
    stim = _check_stim_frames(s.get("stim_frames", ""))
    return {
        "segmentation": {
            "flow_threshold": float(s["flow_threshold"]),
            "cellprob_threshold": float(s["cellprob_threshold"]),
            "niter": int(s["niter"]),
            "diameter": int(s["diameter"]),
        },
        "tracking": {
            "max_distance": round(float(s["max_distance"]), 1),
            "grace_period": int(s["grace_period"]),
            # trajectories.py treats radius 0 as "no ROI filter"
            "radius": int(s["radius"]) if s.get("roi_enabled", True) else 0,
            "radius_y": int(s["y_shift"]),
            "radius_x": int(s["x_shift"]),
            "shift_frame": shift_frame,
            "shift_xy": [shift_x, shift_y],
            "save_interval": int(s["save_interval"]),
        },
        "post_analysis": {
            "f0_frame": _frame_token(frames, int(s["f0_frame"])),
            "stim_frames": [int(x) for x in stim.split(",")] if stim else [],
        },
    }


def _config_path(s: dict) -> str:
    return os.path.join(s["global_dir"], pipeline_config.CONFIG_NAME)


def _driver_command(s: dict, kind: str, run_mode: str) -> list:
    """The run_*.sh call for this run, from the project root."""
    exp = os.path.relpath(s["global_dir"], _PROJECT_ROOT)
    if kind == "run_post_processes":
        return ["bash", "run_post_processes.sh", exp]
    if run_mode == "full":
        return ["bash", "run_processes.sh", exp]
    return ["bash", "run_processes.sh", "--skip-segmentation", exp]


def _pipeline_group_alive() -> bool:
    """True while any process in the last run's process group is alive.
    bash can exit before its children (trajectories.py polling for masks),
    so pipeline_proc.poll() alone is not enough."""
    global pipeline_pgid
    pgid = pipeline_pgid
    if pgid is None:
        return False
    if pipeline_proc is not None:
        pipeline_proc.poll()
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        if pipeline_pgid == pgid:
            pipeline_pgid = None
        return False


@app.route("/api/pipeline/run", methods=["POST"])
def api_pipeline_run():
    global pipeline_proc, pipeline_pgid, pipeline_started_at, pipeline_kind, pipeline_log_fh
    body = _json_body()
    kind = body.get("kind", "run_processes")
    run_mode = body.get("run_mode", "full")
    if kind not in ("run_processes", "run_post_processes"):
        return jsonify({"ok": False, "error": f"unknown kind: {kind}"}), 400
    if run_mode not in _RUN_MODES:
        return jsonify({"ok": False, "error": f"unknown run_mode: {run_mode}"}), 400

    with _state_lock:
        if _pipeline_group_alive():
            return jsonify({"ok": False, "error": "already running",
                            "pid": pipeline_pgid})

        s = session.snapshot()
        if not s.get("global_dir"):
            return jsonify({"ok": False, "error": "no experiment loaded"}), 400
        session.apply_patch({"last_run_mode": run_mode})

        try:
            if kind == "run_post_processes":
                f0 = int(body.get("f0_frame", s.get("f0_frame", 1)))
                stim = str(body.get("stim_frames", s.get("stim_frames", "")))
                s = session.apply_patch({"f0_frame": f0, "stim_frames": stim})
            validation = _validation_state("post" if kind == "run_post_processes" else run_mode)
            if not validation["ok"]:
                return jsonify({"ok": False, "error": "validation failed",
                                "validation": validation}), 400
            # Every run, and "Save config only", saves the parameters first
            cfg_path = _config_path(s)
            pipeline_config.write_config(cfg_path, _config_updates(s), "TUNE_GUI")
        except (KeyError, TypeError, ValueError, pipeline_config.ConfigError) as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except OSError as e:
            return jsonify({"ok": False, "error": f"could not save the config: {e}"}), 400

        if kind == "run_processes" and run_mode == "preview_only":
            return jsonify({"ok": True, "saved_only": True, "config_path": cfg_path})
        cmd = _driver_command(s, kind, run_mode)

        # Truncate log
        if pipeline_log_fh is not None:
            try:
                pipeline_log_fh.close()
            except Exception:
                pass
        pipeline_log_fh = open(_PIPELINE_LOG, "w")
        pipeline_log_fh.write(f"=== {kind} started at {time.ctime()} ===\n"
                              f"$ {shlex.join(cmd)}\n")
        pipeline_log_fh.flush()

        pipeline_proc = subprocess.Popen(
            cmd,
            cwd=_PROJECT_ROOT,
            stdout=pipeline_log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        pipeline_pgid = pipeline_proc.pid
        pipeline_started_at = time.time()
        pipeline_kind = run_mode if kind == "run_processes" else "post"
        return jsonify({"ok": True, "pid": pipeline_proc.pid,
                        "kind": kind, "run_mode": run_mode,
                        "config_path": cfg_path, "command": shlex.join(cmd)})


@app.route("/api/pipeline/stop", methods=["POST"])
def api_pipeline_stop():
    global pipeline_proc, pipeline_pgid, pipeline_log_fh
    with _state_lock:
        # Signal the stored group even if bash itself has already exited
        pgid = pipeline_pgid
        if not _pipeline_group_alive():
            return jsonify({"ok": False, "error": "not running"})
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(30):
            if not _pipeline_group_alive():
                break
            time.sleep(0.1)
        else:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if pipeline_log_fh is not None:
            pipeline_log_fh.write(f"\n=== STOPPED at {time.ctime()} ===\n")
            pipeline_log_fh.flush()
        pipeline_proc = None
        pipeline_pgid = None
        return jsonify({"ok": True})


def _pipeline_status_snapshot() -> dict:
    global pipeline_proc
    out = {
        "running": False, "pid": None, "exit_code": None,
        "kind": pipeline_kind,
        "started_at": pipeline_started_at,
        "uptime": (time.time() - pipeline_started_at) if pipeline_started_at else 0,
    }
    if pipeline_proc is None:
        out["stage"] = _infer_stage()
        return out
    rc = pipeline_proc.poll()
    if rc is None:
        out.update({"running": True, "pid": pipeline_proc.pid})
    else:
        out.update({"pid": pipeline_proc.pid, "exit_code": rc})
        if _pipeline_group_alive():
            out["running"] = True
    out["stage"] = _infer_stage()
    return out


def _pipeline_progress_snapshot() -> dict:
    s = session.snapshot()
    total = len(s.get("all_frames") or [])
    masks_dir = s.get("masks_dir") or ""
    save_path = s.get("save_path") or ""

    n_masks = 0
    if masks_dir and os.path.isdir(masks_dir):
        n_masks = sum(1 for f in os.listdir(masks_dir) if f.endswith(".npy"))

    traj_pct = 0.0
    if os.path.isfile(os.path.join(save_path, "trajectories_complete.json")):
        traj_pct = 100.0
    elif os.path.isfile(os.path.join(save_path, "trajectories.json")):
        traj_pct = 50.0

    pre_done = os.path.isdir(os.path.join(save_path, "plots"))
    # Written by PostAnalysis.py as its last step
    post_done = os.path.isfile(os.path.join(save_path, "post_analysis_complete.txt"))
    return {
        "total_frames": total,
        "segmentation": {
            "done": n_masks, "total": total,
            "pct": (100.0 * n_masks / total) if total else 0.0,
        },
        "trajectories": {"pct": traj_pct},
        "pre_analysis": {"pct": 100.0 if pre_done else 0.0},
        "post_analysis": {"pct": 100.0 if post_done else 0.0},
    }


@app.route("/api/pipeline/stream", methods=["GET"])
def api_pipeline_stream():
    """Combined SSE stream for pipeline status, log delta, and progress.

    The browser opens this once at boot and consumes three event types:
      - status   : process up/down/exit + inferred stage
      - log      : new lines appended to pipeline.log since last event
      - progress : segmentation/trajectories/pre/post percentages

    We push only when something changes (or every 15s as keepalive), so the
    UI no longer suffers the 1-second poll latency.
    """
    @stream_with_context
    def gen():
        last_status_key = None
        last_log_pos = 0
        last_progress = None
        last_keepalive = time.time()

        # Emit baselines so the client has state without a GET
        status = _pipeline_status_snapshot()
        last_status_key = (status["running"], status["stage"],
                            status["pid"], status["exit_code"])
        yield f"event: status\ndata: {json.dumps(status)}\n\n"

        if os.path.isfile(_PIPELINE_LOG):
            try:
                with open(_PIPELINE_LOG) as f:
                    text = f.read()
                    last_log_pos = f.tell()
                if text:
                    yield (
                        f"event: log\n"
                        f"data: {json.dumps({'text': text, 'pos': last_log_pos})}\n\n"
                    )
            except Exception:
                pass

        progress = _pipeline_progress_snapshot()
        last_progress = json.dumps(progress, sort_keys=True)
        yield f"event: progress\ndata: {last_progress}\n\n"

        while True:
            sent = False

            status = _pipeline_status_snapshot()
            key = (status["running"], status["stage"],
                   status["pid"], status["exit_code"])
            if key != last_status_key:
                last_status_key = key
                yield f"event: status\ndata: {json.dumps(status)}\n\n"
                sent = True

            if os.path.isfile(_PIPELINE_LOG):
                try:
                    size = os.path.getsize(_PIPELINE_LOG)
                    if size < last_log_pos:
                        # file was truncated (new run); restart
                        last_log_pos = 0
                    if size > last_log_pos:
                        with open(_PIPELINE_LOG) as f:
                            f.seek(last_log_pos)
                            chunk = f.read()
                            last_log_pos = f.tell()
                        if chunk:
                            yield (
                                f"event: log\n"
                                f"data: {json.dumps({'text': chunk, 'pos': last_log_pos})}\n\n"
                            )
                            sent = True
                except Exception:
                    pass

            progress = _pipeline_progress_snapshot()
            pkey = json.dumps(progress, sort_keys=True)
            if pkey != last_progress:
                last_progress = pkey
                yield f"event: progress\ndata: {pkey}\n\n"
                sent = True

            now = time.time()
            if sent:
                last_keepalive = now
            elif now - last_keepalive > 15:
                yield ": keepalive\n\n"
                last_keepalive = now

            # Cadence: faster while running for snappy logs, slower when idle
            time.sleep(0.25 if status.get("running") else 1.0)

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


_STAGE_MARKERS = [
    ("post_analysis", "POST-ANALYSIS"),
    ("pre_analysis",  "PRE-ANALYSIS"),
    ("trajectories",  "TRAJECTORIES"),
    ("segmentation",  "SEGMENTATION"),
]


def _infer_stage() -> str | None:
    if not os.path.isfile(_PIPELINE_LOG):
        return None
    sig = _file_sig(_PIPELINE_LOG)
    if stage_cache["sig"] == sig:
        return stage_cache["stage"]
    try:
        with open(_PIPELINE_LOG) as f:
            text = f.read()
    except Exception:
        return None
    for key, marker in _STAGE_MARKERS:
        if f">>> STAGE: {marker} <<<" in text:
            stage_cache.update({"sig": sig, "stage": key})
            return key
    stage_cache.update({"sig": sig, "stage": None})
    return None


@app.route("/api/pipeline/luminosity.png", methods=["GET"])
def api_pipeline_luminosity_png():
    """Render matplotlib plot of the in-progress luminosity.json if available."""
    s = session.snapshot()
    save_path = s.get("save_path") or ""
    candidates = [
        os.path.join(save_path, "luminosity_complete.json"),
        os.path.join(save_path, "luminosity.json"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        abort(404)
    cache = _cache_path("luminosity", _file_sig(path))
    if os.path.isfile(cache):
        return _png_response(cache, max_age=10)

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

    # trajectories.py writes {cell_id: {"f<frame>": value or None}}. Cells
    # first seen mid-run have no keys for earlier frames, so place each
    # value by its frame number, not by list position
    traces = []
    if isinstance(data, dict):
        for trace in data.values():
            if not isinstance(trace, dict):
                continue
            pts = sorted((int(k[1:]), float(v)) for k, v in trace.items()
                         if v is not None and k[1:].isdigit())
            if pts:
                traces.append(pts)

    fig, ax = plt.subplots(figsize=(7, 3), dpi=140)
    if traces:
        n_cells = len(traces)
        for pts in traces[:200]:
            ax.plot(*zip(*pts), linewidth=0.6, alpha=0.4)
        frames = sorted({f for pts in traces for f, _ in pts})
        col = {f: i for i, f in enumerate(frames)}
        mat = np.full((n_cells, len(frames)), np.nan, dtype=np.float32)
        for row, pts in enumerate(traces):
            for f, v in pts:
                mat[row, col[f]] = v
        ax.plot(frames, np.nanmean(mat, axis=0), linewidth=1.8, color="black",
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

    fig.savefig(cache, format="png", bbox_inches="tight")
    plt.close(fig)
    return _png_response(cache, max_age=10)


# ═══════════════════════════════════════════════════════════════════════════
# Config preview
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/config/preview", methods=["GET"])
def api_config_preview():
    """The pipeline_config.yaml a run would save, and the command it would
    launch, without writing anything."""
    kind = request.args.get("kind", "run_processes")
    run_mode = request.args.get("run_mode", session.snapshot().get("last_run_mode", "full"))
    if run_mode not in _RUN_MODES:
        return jsonify({"ok": False, "error": f"unknown run_mode: {run_mode}"}), 400
    s = session.snapshot()
    if not s.get("global_dir"):
        return jsonify({"ok": False, "error": "no experiment"}), 400
    cfg_path = _config_path(s)
    try:
        text = pipeline_config.render_config(cfg_path, _config_updates(s), "TUNE_GUI")
    except (KeyError, TypeError, ValueError, pipeline_config.ConfigError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    command = None
    if not (kind == "run_processes" and run_mode == "preview_only"):
        command = shlex.join(_driver_command(s, kind, run_mode))
    return jsonify({"ok": True, "config_path": cfg_path, "config": text, "command": command})


# ═══════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════
def _cleanup():
    pgid = pipeline_pgid
    if _pipeline_group_alive():
        print("[web gui] cleaning up pipeline subprocess", flush=True)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass


atexit.register(_cleanup)


if __name__ == "__main__":
    # Loopback only by default; HOST=0.0.0.0 exposes it to the network
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5001))
    print(f"[web gui] starting on http://{host}:{port}", flush=True)
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
