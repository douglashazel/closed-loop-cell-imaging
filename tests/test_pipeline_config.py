"""pipeline_config.yaml: validation, the run_*.sh drivers' view of it, and the
per-run record under analysis/run_history/."""
import os
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "SCRIPTS", "core_pipeline"))

import pipeline_config as pc  # noqa: E402

CONFIG = {
    "segmentation": {"flow_threshold": 0.95, "cellprob_threshold": -4,
                     "niter": 5000, "diameter": 19},
    "tracking": {"max_distance": 230, "grace_period": 3, "radius": 1313,
                 "radius_y": -456, "radius_x": -46, "shift_frame": 1000,
                 "shift_xy": [-6, 0], "save_interval": 10},
    "post_analysis": {"f0_frame": 1, "stim_frames": [77, 112]},
}


def _experiment(tmp_path, config=CONFIG, name="exp one"):
    """An experiment folder (with a space in its name) holding the config."""
    exp = tmp_path / name
    (exp / "frames").mkdir(parents=True)
    (exp / pc.CONFIG_NAME).write_text(yaml.safe_dump(config))
    return exp


def _fake_python(tmp_path):
    """A python3 on PATH that runs pipeline_config.py for real and logs every
    other call (the pipeline scripts) to calls.txt, one tab-separated line each."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    fake = bin_dir / "python3"
    fake.write_text(
        "#!/bin/bash\n"
        f'if [[ "$1" == *pipeline_config.py ]]; then exec {sys.executable} "$@"; fi\n'
        f"printf '%s\\t' \"$@\" >> '{calls}'\n"
        f"echo >> '{calls}'\n")
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return env, calls


def _run_driver(script, args, env, cwd):
    return subprocess.run(["bash", os.path.join(ROOT, script), *args], env=env, cwd=cwd,
                          capture_output=True, text=True)


def _called(calls, script):
    """Arguments the driver passed to SCRIPTS/core_pipeline/<script>."""
    for line in calls.read_text().splitlines():
        argv = line.rstrip("\t").split("\t")
        if argv[1].endswith(script):
            return argv[2:]
    raise AssertionError(f"{script} was not called")


def test_validate_reports_every_problem():
    bad = {"segmentation": {"flow_treshold": 0.9, "niter": 2.5, "diameter": True,
                            "cellprob_threshold": -1},
           "tracking": {}}
    with pytest.raises(pc.ConfigError) as e:
        pc.validate(bad, ["segmentation", "tracking"])
    msg = str(e.value)
    for expected in ("segmentation.flow_treshold: unknown key",
                     "segmentation.flow_threshold: missing",
                     "segmentation.niter: expected a whole number",
                     "segmentation.diameter: expected a number",
                     "tracking.max_distance: missing"):
        assert expected in msg


def test_optional_keys_and_old_stim_text():
    raw = {**CONFIG, "post_analysis": {"f0_frame": 1, "stim_frames": "77, 112,148"}}
    values = pc.validate(raw, ["tracking", "post_analysis"])
    assert values["tracking"]["workers"] == pc.DEFAULTS[("tracking", "workers")]
    assert values["post_analysis"]["stim_frames"] == [77, 112, 148]
    assert pc.validate({"post_analysis": {"f0_frame": 0}}, ["post_analysis"]) == \
        {"post_analysis": {"f0_frame": 0, "stim_frames": []}}


def test_example_config_is_valid():
    path = os.path.join(ROOT, "configs", "example_c2c12_chamber_A.yaml")
    values = pc.validate(pc.read_config(path), list(pc.SECTIONS), path)
    assert values["segmentation"]["niter"] == 2250


def test_run_processes_passes_the_config_values(tmp_path):
    exp = _experiment(tmp_path)
    env, calls = _fake_python(tmp_path)
    # Relative experiment path from another directory: the driver must resolve
    # it before it changes to the project root.
    res = _run_driver("run_processes.sh", ["exp one"], env, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert _called(calls, "segmentation.py") == [
        "--image_dir", f"{exp}/frames", "--mask_dir", f"{exp}/masks",
        "--flow_threshold", "0.95", "--cellprob_threshold", "-4",
        "--niter", "5000", "--diameter", "19"]
    assert _called(calls, "trajectories.py") == [
        "--mask_dir", f"{exp}/masks", "--image_dir", f"{exp}/frames",
        "--save_path", f"{exp}/analysis", "--max_distance", "230",
        "--grace_period", "3", "--radius", "1313", "--radius_y", "-456",
        "--radius_x", "-46", "--shift_frame", "1000", "--shift_xy", "-6", "0",
        "--save_interval", "10", "--workers", "6"]
    assert _called(calls, "PreAnalysis.py") == [
        "--exp", str(exp), "--analysis_dir", f"{exp}/analysis"]

    # The run left a record of exactly the sections it used.
    (record,) = (exp / "analysis" / pc.HISTORY_DIR).iterdir()
    saved = yaml.safe_load(record.read_text())
    assert saved["run"] == "run_processes.sh"
    assert saved["segmentation"] == CONFIG["segmentation"]
    assert saved["tracking"] == {**CONFIG["tracking"], "workers": 6}
    assert "post_analysis" not in saved


def test_skip_segmentation_and_post_processes(tmp_path):
    exp = _experiment(tmp_path)
    env, calls = _fake_python(tmp_path)
    res = _run_driver("run_processes.sh", ["--skip-segmentation", str(exp / pc.CONFIG_NAME)],
                      env, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "segmentation.py" not in calls.read_text()
    res = _run_driver("run_post_processes.sh", [str(exp)], env, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert _called(calls, "PostAnalysis.py")[-4:] == [
        "--f0_frame", "1", "--stim_frames", "77,112"]
    labels = sorted(yaml.safe_load(p.read_text())["run"]
                    for p in (exp / "analysis" / pc.HISTORY_DIR).iterdir())
    assert labels == ["run_post_processes.sh", "run_processes.sh --skip-segmentation"]


def test_bad_config_stops_before_any_pipeline_script(tmp_path):
    config = {**CONFIG, "tracking": {**CONFIG["tracking"], "max_distance": None}}
    exp = _experiment(tmp_path, config)
    env, calls = _fake_python(tmp_path)
    res = _run_driver("run_trajectories.sh", [str(exp)], env, cwd=tmp_path)
    assert res.returncode != 0
    assert "tracking.max_distance: missing" in res.stderr
    assert not calls.exists()
    res = _run_driver("run_segmentation.sh", [str(tmp_path / "nowhere")], env, cwd=tmp_path)
    assert res.returncode != 0 and "No config file" in res.stderr
    assert _run_driver("run_segmentation.sh", [], env, cwd=tmp_path).returncode == 2


def test_write_config_merges_and_keeps_unmanaged_keys(tmp_path):
    path = tmp_path / pc.CONFIG_NAME
    path.write_text(yaml.safe_dump({"experiment_dir": "elsewhere",
                                    "tracking": {**CONFIG["tracking"], "workers": 3},
                                    "post_analysis": {"f0_frame": 5}}))
    pc.write_config(str(path), {"segmentation": CONFIG["segmentation"],
                                "tracking": {"max_distance": 99.5}}, "a test")
    text = path.read_text()
    assert text.startswith("# Cell Trainer pipeline parameters") and "a test" in text
    assert "shift_xy: [-6, 0]" in text
    saved = yaml.safe_load(text)
    assert saved["experiment_dir"] == "elsewhere"
    assert saved["tracking"]["max_distance"] == 99.5
    assert saved["tracking"]["workers"] == 3
    assert saved["post_analysis"] == {"f0_frame": 5, "stim_frames": []}
    assert saved["segmentation"] == CONFIG["segmentation"]


def test_experiment_dir_is_relative_to_the_config_file(tmp_path):
    exp = tmp_path / "data" / "exp"
    (exp / "frames").mkdir(parents=True)
    cfg = tmp_path / "configs" / "mine.yaml"
    cfg.parent.mkdir()
    cfg.write_text(yaml.safe_dump({"experiment_dir": "../data/exp",
                                   "segmentation": CONFIG["segmentation"]}))
    out = pc.prepare(str(cfg), "run_segmentation.sh", ["segmentation"])
    assert f"IMAGE_DIR={exp}/frames" in out
    assert (exp / "analysis" / pc.HISTORY_DIR).is_dir()
