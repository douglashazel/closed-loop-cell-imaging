"""Flask web GUI for the Patrick cell-analysis preprocessing pipeline.

Launch:  python WEBGUI/app.py
Browse:  http://localhost:5001

Ports the functionality of ../preprocess_gui.py (napari) to a browser and adds
a "Run & Monitor" tab that launches run_processes.sh / run_post_processes.sh
directly with the parameters the user tuned.
"""

import atexit
import hashlib
import json
import os
import random
import signal
import subprocess
import threading
import time

import numpy as np
from flask import (
    Flask, Response, abort, jsonify, render_template, request, send_file,
    stream_with_context,
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
_CACHE_DIR = os.path.join(_TMP_DIR, "cache")
_PIPELINE_LOG = os.path.join(_TMP_DIR, "pipeline.log")
_PIPELINE_SCRIPT = os.path.join(_TMP_DIR, "current_run.sh")

os.makedirs(_TMP_DIR, exist_ok=True)
os.makedirs(_CACHE_DIR, exist_ok=True)

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

progress_cache = {"ts": 0.0, "data": None}
stage_cache = {"sig": None, "stage": None}


# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(_HERE, "templates"),
    static_folder=os.path.join(_HERE, "static"),
)


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

    if cellpose_job.is_running():
        return jsonify({
            "ok": False,
            "error": "Cellpose is still running on the current experiment. "
                     "Wait for it to finish before switching.",
        }), 409

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
        "last_preview_frame": -1,
        "preview_mask_source": "",
        "segmentation_reviewed": False,
        "last_roi_count": 0,
        "validation_warnings": [],
    }

    # Resume-from-config prefill
    cfg_path = os.path.join(analysis_dir, "config.txt")
    resumed = False
    if os.path.isfile(cfg_path):
        prefill = parse_config_txt(cfg_path)
        if prefill:
            patch.update(prefill)
            resumed = True

    session.temp_segmentation = None
    session.temp_segmentation_version += 1
    snapshot = session.apply_patch(patch)
    snapshot["resumed_from_config"] = resumed
    snapshot["config_txt_path"] = cfg_path if resumed else None
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
    # Default alpha is 255 (fully opaque colour fill); the browser applies the
    # opacity slider via CSS, so we don't want to bake a fractional alpha into
    # the PNG and end up multiplying it.
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
        session.temp_segmentation_version += 1
        session.apply_patch({
            "last_preview_frame": s["frame_idx"],
            "preview_mask_source": "cellpose_preview",
            "segmentation_reviewed": False,
            "last_roi_count": 0,
        })
        # Pre-render the default-alpha preview PNG so the first GET hits the
        # disk cache instead of paying the colorize + encode cost inline.
        try:
            cache = _cache_path("mask", session.temp_segmentation_version, 255)
            if not os.path.isfile(cache):
                rgba = colorize_labels(masks, alpha=255)
                _write_png_if_missing(cache, rgba, mode="RGBA")
        except Exception as e:
            print(f"[mask preview pre-render] {e}", flush=True)

    started = cellpose_job.start(
        img, s["frame_idx"],
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


@app.route("/api/cellpose/status", methods=["GET"])
def api_cellpose_status():
    return jsonify(_cellpose_status_snapshot())


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
        # Send an initial snapshot so the client doesn't have to GET /status.
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


@app.route("/api/cellpose/stats.png", methods=["GET"])
def api_cellpose_stats_png():
    """Histogram of cell areas for the current temp_segmentation."""
    seg = session.temp_segmentation
    if seg is None:
        abort(404)
    cache = _cache_path("cellpose-stats", session.temp_segmentation_version)
    if os.path.isfile(cache):
        return _png_response(cache, max_age=3600)
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
    fig.savefig(cache, format="png", bbox_inches="tight")
    plt.close(fig)
    return _png_response(cache, max_age=3600)


@app.route("/api/cellpose/review", methods=["POST"])
def api_cellpose_review():
    body = request.get_json(force=True) or {}
    reviewed = bool(body.get("reviewed", True))
    return jsonify(session.apply_patch({"segmentation_reviewed": reviewed}))


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
    _, n_inside, (cx, cy) = roi_filter(seg, radius, y_shift, x_shift)
    session.apply_patch({
        "radius": radius,
        "y_shift": y_shift,
        "x_shift": x_shift,
        "last_roi_count": n_inside,
    })
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
    alpha = int(request.args.get("alpha", 140))
    cache = _cache_path(
        "roi-mask",
        session.temp_segmentation_version,
        s["radius"],
        s["y_shift"],
        s["x_shift"],
        alpha,
    )
    if not os.path.isfile(cache):
        filtered, _, _ = roi_filter(seg, s["radius"], s["y_shift"], s["x_shift"])
        rgba = colorize_labels(filtered, alpha=alpha)
        _write_png_if_missing(cache, rgba, mode="RGBA")
    return _png_response(cache, max_age=3600)


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
        add("shift", int(s.get("shift_frame_idx") or 0) >= 0, "Frame shift reviewed",
            f"shift_frame {s.get('shift_frame_idx')}, shift_xy {s.get('shift_xy')}.")

    writable = bool(save_parent and os.path.isdir(save_parent) and os.access(save_parent, os.W_OK))
    add("writable", writable, "Output folder writable",
        save_path if writable else "The experiment folder is not writable.")

    session.apply_patch({"validation_warnings": warnings})
    return {"ok": not warnings, "checks": checks, "warnings": warnings}


@app.route("/api/validation", methods=["GET"])
def api_validation():
    mode = request.args.get("mode", session.snapshot().get("last_run_mode", "full"))
    return jsonify(_validation_state(mode))


def _build_run_processes_script(s: dict, run_mode: str = "full") -> str:
    """Render a shell script equivalent to run_processes.sh but with the
    session parameters baked in. Mirrors preprocess_gui.py::_generate_script."""
    global_dir = s["global_dir"]
    rel_dir = os.path.relpath(global_dir, _PROJECT_ROOT)
    shift_x, shift_y = s["shift_xy"]
    # When the user disables the ROI, send a sentinel radius huge enough to
    # include every cell in any plausibly-sized image so trajectories.py
    # behaves as if no ROI filter were active.
    effective_radius = s["radius"] if s.get("roi_enabled", True) else 999999999
    return f"""#!/bin/bash
set -euo pipefail

cd "{_PROJECT_ROOT}"

GLOBAL_DIR="{rel_dir}"
IMAGE_DIR="${{GLOBAL_DIR}}/frames"
MASK_DIR="${{GLOBAL_DIR}}/masks"
SAVE_PATH="${{GLOBAL_DIR}}/analysis"

SCRIPT1="SCRIPTS/core_pipeline/segmentation.py"
SCRIPT2="SCRIPTS/core_pipeline/trajectories.py"

FLOW_THRESHOLD={s['flow_threshold']}
CELLPROB_THRESHOLD={s['cellprob_threshold']}
NITER={s['niter']}
DIAMETER={s['diameter']}

MAX_DISTANCE={s['max_distance']:.1f}
GRACE_PERIOD={s['grace_period']}
RADIUS={effective_radius}
RADIUS_Y={s['y_shift']}
RADIUS_X={s['x_shift']}
SHIFT_FRAME={s['shift_frame_idx']}
SHIFT_XY="{shift_x} {shift_y}"
SAVE_INTERVAL={s['save_interval']}
RUN_MODE="{run_mode}"

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
echo "Run mode: $RUN_MODE"

if [[ "$RUN_MODE" == "preview_only" ]]; then
    echo "Preview only: config was generated, but no analysis jobs were launched."
else
    PID1=""
    if [[ "$RUN_MODE" == "full" ]]; then
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
    else
        echo ">>> STAGE: SEGMENTATION <<<"
        echo "Skipping segmentation; using existing mask files in $MASK_DIR"
    fi

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

    if [[ -n "$PID1" ]]; then
        wait "$PID1"
    fi
    wait "$PID2"

    echo ">>> STAGE: PRE-ANALYSIS <<<"
    python3 -u SCRIPTS/core_pipeline/PreAnalysis.py \\
        --exp "$GLOBAL_DIR" \\
        --analysis_dir "$SAVE_PATH"
fi

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
python3 -u SCRIPTS/core_pipeline/PostAnalysis.py \\
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
    run_mode = body.get("run_mode", "full")
    if kind not in ("run_processes", "run_post_processes"):
        return jsonify({"ok": False, "error": f"unknown kind: {kind}"}), 400
    if run_mode not in ("full", "existing_masks", "preview_only", "post"):
        return jsonify({"ok": False, "error": f"unknown run_mode: {run_mode}"}), 400

    with _state_lock:
        if pipeline_proc is not None and pipeline_proc.poll() is None:
            return jsonify({"ok": False, "error": "already running",
                            "pid": pipeline_proc.pid})

        s = session.snapshot()
        if not s.get("global_dir"):
            return jsonify({"ok": False, "error": "no experiment loaded"}), 400
        session.apply_patch({"last_run_mode": run_mode})

        if kind == "run_processes":
            validation = _validation_state(run_mode)
            if not validation["ok"]:
                return jsonify({"ok": False, "error": "validation failed",
                                "validation": validation}), 400
            script_text = _build_run_processes_script(s, run_mode)
        else:
            f0 = int(body.get("f0_frame", s.get("f0_frame", 1)))
            stim = str(body.get("stim_frames", s.get("stim_frames", "")))
            session.apply_patch({"f0_frame": f0, "stim_frames": stim})
            validation = _validation_state("post")
            if not validation["ok"]:
                return jsonify({"ok": False, "error": "validation failed",
                                "validation": validation}), 400
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
        pipeline_kind = run_mode if kind == "run_processes" else "post"
        return jsonify({"ok": True, "pid": pipeline_proc.pid,
                        "kind": kind, "run_mode": run_mode,
                        "script": _PIPELINE_SCRIPT})


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
    out["stage"] = _infer_stage()
    return out


@app.route("/api/pipeline/status", methods=["GET"])
def api_pipeline_status():
    return jsonify(_pipeline_status_snapshot())


def _pipeline_progress_snapshot() -> dict:
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

        # Emit baselines so the client has state without a GET.
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

            # Cadence: faster while running for snappy logs, slower when idle.
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
    now = time.time()
    if progress_cache["data"] is not None and now - progress_cache["ts"] < 1.5:
        return jsonify(progress_cache["data"])
    data = _pipeline_progress_snapshot()
    progress_cache.update({"ts": now, "data": data})
    return jsonify(data)


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

    fig.savefig(cache, format="png", bbox_inches="tight")
    plt.close(fig)
    return _png_response(cache, max_age=10)


# ═══════════════════════════════════════════════════════════════════════════
# Script preview (Tab 7 summary / export)
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/script/preview", methods=["GET"])
def api_script_preview():
    kind = request.args.get("kind", "run_processes")
    run_mode = request.args.get("run_mode", session.snapshot().get("last_run_mode", "full"))
    s = session.snapshot()
    if not s.get("global_dir"):
        return jsonify({"ok": False, "error": "no experiment"}), 400
    if kind == "run_post_processes":
        text = _build_run_post_processes_script(
            s, s.get("f0_frame", 1), s.get("stim_frames", ""),
        )
    else:
        text = _build_run_processes_script(s, run_mode)
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
