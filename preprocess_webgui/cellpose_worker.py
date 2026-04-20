"""Cached Cellpose model + background segmentation thread.

Pattern mirrors PE_Pipeline/V6/LaunchWebGUI.py: load the model once on first
use, run model.eval in a daemon thread, expose status via a module-level dict
that the Flask endpoints poll.
"""

import os
import threading
import time
from typing import Optional

import numpy as np


_model_lock = threading.Lock()
_cached_model = None


def get_model():
    global _cached_model
    with _model_lock:
        if _cached_model is None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
            from cellpose import models
            print("[cellpose_worker] Loading Cellpose model (first use)...", flush=True)
            _cached_model = models.CellposeModel(gpu=True)
            print("[cellpose_worker] Cellpose model ready.", flush=True)
        return _cached_model


class CellposeJob:
    """Singleton-style job holder. Only one segmentation runs at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.status = {
            "state": "idle",
            "message": "Ready",
            "frame_idx": 0,
            "n_cells": 0,
            "started_at": None,
            "finished_at": None,
        }
        self.result: Optional[np.ndarray] = None  # last masks array

    def is_running(self) -> bool:
        with self.lock:
            return self.thread is not None and self.thread.is_alive()

    def start(self, img: np.ndarray, frame_idx: int,
              flow_threshold: float, cellprob_threshold: float,
              niter: int, diameter: int, on_done=None):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return False
            self.status = {
                "state": "running",
                "message": "Loading Cellpose model..." if _cached_model is None
                           else "Segmenting...",
                "frame_idx": frame_idx,
                "n_cells": 0,
                "started_at": time.time(),
                "finished_at": None,
            }
            self.thread = threading.Thread(
                target=self._run,
                args=(img, frame_idx, flow_threshold, cellprob_threshold,
                      niter, diameter, on_done),
                daemon=True,
            )
            self.thread.start()
            return True

    def _run(self, img, frame_idx, flow_threshold, cellprob_threshold,
             niter, diameter, on_done):
        try:
            model = get_model()
            with self.lock:
                self.status["message"] = "Segmenting..."
            masks, _, _ = model.eval(
                [img],
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                niter=niter,
                diameter=diameter if diameter > 0 else None,
            )
            masks = masks[0]
            n = int(masks.max()) if masks.max() > 0 else 0
            with self.lock:
                self.result = masks
                self.status = {
                    "state": "done",
                    "message": f"Detected {n} cells",
                    "frame_idx": frame_idx,
                    "n_cells": n,
                    "started_at": self.status["started_at"],
                    "finished_at": time.time(),
                }
            if on_done is not None:
                try:
                    on_done(masks)
                except Exception as e:
                    print(f"[cellpose_worker] on_done error: {e}", flush=True)
        except Exception as e:
            with self.lock:
                self.status = {
                    "state": "error",
                    "message": f"Error: {e}",
                    "frame_idx": frame_idx,
                    "n_cells": 0,
                    "started_at": self.status.get("started_at"),
                    "finished_at": time.time(),
                }
            print(f"[cellpose_worker] error: {e}", flush=True)
