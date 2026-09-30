"""PipelineState for the preprocess web GUI.

Mirrors the fields of preprocess_gui.py's PipelineState but is JSON-serializable
so it can persist to session.json between page reloads and be round-tripped
through the Flask endpoints.
"""

import json
import math
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from pipeline_logic import timepoint_token


# Set only by /api/experiment (or session.json at startup, confined to the
# experiments root). /api/session refuses them.
PATH_DIR_FIELDS = ("global_dir", "frames_dir", "masks_dir", "save_path")
PATH_LIST_FIELDS = ("all_frames", "all_masks")
PATH_FIELDS = PATH_DIR_FIELDS + PATH_LIST_FIELDS


def within_root(path: str, root: str) -> bool:
    """True if `path` resolves (symlinks followed) to `root` or below it.
    `root` must already be a realpath."""
    return os.path.commonpath([os.path.realpath(path), root]) == root


def _coerce(name: str, typ: type, v: Any) -> Any:
    """Return `v` converted to the field type `typ`, or raise ValueError."""
    out = None
    if typ is bool or isinstance(v, bool):
        # bool is an int subclass; accept it only for bool fields.
        if typ is bool and isinstance(v, bool):
            out = v
    elif typ is int:
        if isinstance(v, (int, str)) or (isinstance(v, float) and v.is_integer()):
            try:
                out = int(v)
            except ValueError:
                pass
    elif typ is float:
        if isinstance(v, (int, float, str)):
            try:
                f = float(v)
                out = f if math.isfinite(f) else None
            except ValueError:
                pass
    elif typ is str:
        out = v if isinstance(v, str) else None
    elif typ is list:
        out = v if isinstance(v, list) else None
    elif typ is tuple:
        # shift_xy: two whole-pixel offsets
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                out = (_coerce(name, int, v[0]), _coerce(name, int, v[1]))
            except ValueError:
                pass
    if out is None:
        raise ValueError(f"{name}: expected {typ.__name__}, got {v!r:.80}")
    return out


def _check_path_field(name: str, v: Any, root: str) -> None:
    """Raise ValueError unless a path field stays inside `root`."""
    if name in PATH_DIR_FIELDS:
        if not isinstance(v, str):
            raise ValueError(f"{name}: expected str")
        if v and not within_root(v, root):
            raise ValueError(f"{name}: {v} is outside {root}")
    elif name in PATH_LIST_FIELDS:
        # Joined onto frames_dir / masks_dir, so plain file names only.
        if not isinstance(v, list) or not all(
                isinstance(x, str) and x not in ("", ".", "..")
                and os.path.basename(x) == x for x in v):
            raise ValueError(f"{name}: expected a list of plain file names")


@dataclass
class PipelineState:
    # Paths
    global_dir: str = ""
    frames_dir: str = ""
    masks_dir: str = ""
    save_path: str = ""
    all_frames: list = field(default_factory=list)
    all_masks: list = field(default_factory=list)

    # Cellpose
    flow_threshold: float = 0.975
    cellprob_threshold: float = -4.0
    niter: int = 3000
    diameter: int = 26
    frame_idx: int = 0

    # Save interval
    save_interval: int = 10

    # Shift
    shift_frame_idx: int = 1
    shift_xy: tuple = (0, 0)

    # Max distance
    max_distance: float = 30.0

    # ROI
    roi_enabled: bool = True
    radius: int = 2000
    y_shift: int = 0
    x_shift: int = 0

    # Duplicate / trajectories
    grace_period: int = 3

    # Post-analysis
    f0_frame: int = 1
    stim_frames: str = ""

    # Web GUI guidance / validation
    last_preview_frame: int = -1
    preview_mask_source: str = ""
    segmentation_reviewed: bool = False
    last_roi_count: int = 0
    last_run_mode: str = "full"
    completed_stages: list = field(default_factory=list)
    validation_warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shift_xy"] = list(self.shift_xy)
        return d

    def update(self, patch: dict) -> None:
        """Coerce every value to its field's type, then apply. Raises
        ValueError, applying nothing, if any value does not fit."""
        types = {f.name: f.type for f in fields(self)}
        clean = {}
        for k, v in patch.items():
            if k not in types or v is None:
                continue
            clean[k] = _coerce(k, types[k], v)
        for k, v in clean.items():
            setattr(self, k, v)


class SessionStore:
    """Holds the single PipelineState and persists it to session.json."""

    def __init__(self, path: str, experiments_root: str):
        self.path = path
        self.experiments_root = os.path.realpath(experiments_root)
        self.lock = threading.Lock()
        self.state = PipelineState()
        # Non-serialised runtime caches
        self.temp_segmentation = None  # numpy array from last Cellpose run
        self.temp_segmentation_version = 0
        self.load()

    def load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        # Validate field by field like a client patch; drop what does not fit.
        for k, v in data.items():
            try:
                if k in PATH_FIELDS:
                    _check_path_field(k, v, self.experiments_root)
                self.state.update({k: v})
            except ValueError as e:
                print(f"[session] dropped {k} from {self.path}: {e}", flush=True)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        os.rename(tmp, self.path)

    def apply_patch(self, patch: dict) -> dict:
        with self.lock:
            self.state.update(patch)
            self.save()
            return self.state.to_dict()

    def snapshot(self) -> dict:
        with self.lock:
            return self.state.to_dict()


def patch_from_config(values: dict, frames: list | None = None) -> dict:
    """Map the validated sections of a pipeline_config.yaml
    (pipeline_config.validate) onto PipelineState fields.

    shift_frame and f0_frame in the file are timepoint_NNNNN tokens; they are
    mapped back to positions in `frames` (the sorted frame list). A token with
    no matching frame is kept as-is, e.g. the "never shift" sentinel 1000.
    radius 0 means the ROI is off."""
    positions = {timepoint_token(f): i for i, f in enumerate(frames or [])}
    patch: dict[str, Any] = {}
    seg = values.get("segmentation")
    if seg:
        patch.update({k: seg[k] for k in
                      ("flow_threshold", "cellprob_threshold", "niter", "diameter")})
    trk = values.get("tracking")
    if trk:
        patch.update({
            "max_distance": trk["max_distance"],
            "grace_period": trk["grace_period"],
            "roi_enabled": trk["radius"] > 0,
            "y_shift": trk["radius_y"],
            "x_shift": trk["radius_x"],
            "shift_frame_idx": positions.get(trk["shift_frame"], trk["shift_frame"]),
            "shift_xy": trk["shift_xy"],
            "save_interval": trk["save_interval"],
        })
        if trk["radius"] > 0:
            patch["radius"] = trk["radius"]
    post = values.get("post_analysis")
    if post:
        patch["f0_frame"] = positions.get(post["f0_frame"], post["f0_frame"])
        patch["stim_frames"] = ",".join(str(x) for x in post["stim_frames"])
    return patch


def parse_config_txt(path: str, frames: list | None = None) -> dict:
    """Parse the analysis/config.txt that run_processes.sh wrote before
    pipeline_config.yaml existed into a patch dict compatible with
    PipelineState. Unknown keys are ignored.

    SHIFT_FRAME in config.txt is a timepoint_NNNNN token; it is mapped back to
    a position in `frames` (the sorted frame list). A token with no matching
    frame is kept as-is: it can only be the "never shift" sentinel (e.g. 1000).
    RADIUS=0 (or the legacy 999999999) means the ROI was off."""
    if not os.path.isfile(path):
        return {}
    key_map = {
        "FLOW_THRESHOLD": ("flow_threshold", float),
        "CELLPROB_THRESHOLD": ("cellprob_threshold", float),
        "NITER": ("niter", int),
        "DIAMETER": ("diameter", int),
        "MAX_DISTANCE": ("max_distance", float),
        "GRACE_PERIOD": ("grace_period", int),
        "RADIUS_Y": ("y_shift", int),
        "RADIUS_X": ("x_shift", int),
        "SAVE_INTERVAL": ("save_interval", int),
    }
    positions = {timepoint_token(f): i for i, f in enumerate(frames or [])}
    patch: dict[str, Any] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if "=" not in line or line.startswith("["):
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k == "SHIFT_XY":
                    try:
                        parts = v.split()
                        if len(parts) == 2:
                            patch["shift_xy"] = (int(parts[0]), int(parts[1]))
                    except ValueError:
                        pass
                elif k == "RADIUS":
                    try:
                        r = int(v)
                    except ValueError:
                        continue
                    if r <= 0 or r >= 999999999:
                        patch["roi_enabled"] = False
                    else:
                        patch["roi_enabled"] = True
                        patch["radius"] = r
                elif k == "SHIFT_FRAME":
                    try:
                        token = int(v)
                    except ValueError:
                        continue
                    patch["shift_frame_idx"] = positions.get(token, token)
                elif k in key_map:
                    name, cast = key_map[k]
                    try:
                        patch[name] = _coerce(name, cast, v)
                    except ValueError:
                        pass
    except Exception:
        pass
    return patch
