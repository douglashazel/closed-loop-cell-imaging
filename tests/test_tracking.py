"""Stage 1 tracking: each detection is assigned to at most one track."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "SCRIPTS", "core_pipeline"))

from trajectories import update_trajectories_inplace

NO_SHIFT = {10_000: (0.0, 0.0)}


def _track(traj, frame, centers, max_distance=20.0):
    update_trajectories_inplace(traj, frame, centers, NO_SHIFT,
                                grace_period=3, max_distance=max_distance)


def test_two_tracks_converging_on_one_detection():
    traj = {"0": {"x0": 0.0, "y0": 0.0}, "1": {"x0": 10.0, "y0": 0.0}}
    # One detection, within reach of both tracks but nearer to track 1
    _track(traj, 1, [(6, 0)])
    assert traj["1"]["x1"] == 6.0
    assert "x1" not in traj["0"]  # track 0 misses this frame
    assert set(traj) == {"0", "1"}  # and no new track is started


def test_losing_track_takes_its_next_nearest_detection():
    traj = {"0": {"x0": 0.0, "y0": 0.0}, "1": {"x0": 10.0, "y0": 0.0}}
    # (6, 0) is nearest for both; track 1 is closer to it, track 0 falls back to (-5, 0)
    _track(traj, 1, [(6, 0), (-5, 0)])
    assert (traj["1"]["x1"], traj["0"]["x1"]) == (6.0, -5.0)
    assert set(traj) == {"0", "1"}


def test_without_conflicts_every_track_gets_its_nearest():
    traj = {"0": {"x0": 0.0, "y0": 0.0}, "1": {"x0": 100.0, "y0": 0.0}}
    _track(traj, 1, [(98, 1), (2, 1), (50, 50)])
    assert (traj["0"]["x1"], traj["0"]["y1"]) == (2.0, 1.0)
    assert (traj["1"]["x1"], traj["1"]["y1"]) == (98.0, 1.0)
    # The unmatched detection starts a new track
    assert traj["2"] == {"x1": 50.0, "y1": 50.0}


def test_detection_beyond_max_distance_starts_a_new_track():
    traj = {"0": {"x0": 0.0, "y0": 0.0}}
    _track(traj, 1, [(30, 0)], max_distance=20.0)
    assert "x1" not in traj["0"]
    assert traj["1"] == {"x1": 30.0, "y1": 0.0}
