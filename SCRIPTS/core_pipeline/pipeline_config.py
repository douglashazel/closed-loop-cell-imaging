"""Per-experiment pipeline parameters: <experiment>/pipeline_config.yaml.

The run_*.sh drivers get their parameters from this file through `prepare`,
which validates it, saves the values it hands out under
<experiment>/analysis/run_history/, and prints them as shell assignments.
TUNE_GUI and preprocess_gui.py write the file with `write_config`.
configs/example_c2c12_chamber_A.yaml documents every key.

Check a config without running anything:
    python3 SCRIPTS/core_pipeline/pipeline_config.py check <experiment_dir | config.yaml>
"""

import argparse
import datetime
import getpass
import math
import os
import shlex
import subprocess
import sys

try:
    import yaml
except ImportError:
    sys.exit("pipeline_config.py needs PyYAML: pip install PyYAML==6.0.1 (see requirements.txt)")

CONFIG_NAME = "pipeline_config.yaml"
HISTORY_DIR = "run_history"  # under the experiment's analysis/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# section -> key -> (shell variable, type, minimum). Types: int, float,
# "pair" (two numbers) and "frames" (a list of frame numbers).
SECTIONS = {
    "segmentation": {
        "flow_threshold":     ("FLOW_THRESHOLD", float, None),
        "cellprob_threshold": ("CELLPROB_THRESHOLD", float, None),
        "niter":              ("NITER", int, 0),
        "diameter":           ("DIAMETER", int, 0),
    },
    "tracking": {
        "max_distance":  ("MAX_DISTANCE", float, 0),
        "grace_period":  ("GRACE_PERIOD", int, 0),
        "radius":        ("RADIUS", int, 0),
        "radius_y":      ("RADIUS_Y", int, None),
        "radius_x":      ("RADIUS_X", int, None),
        "shift_frame":   ("SHIFT_FRAME", int, 0),
        "shift_xy":      ("SHIFT_XY", "pair", None),
        "save_interval": ("SAVE_INTERVAL", int, 1),
        "workers":       ("WORKERS", int, 1),
    },
    "post_analysis": {
        "f0_frame":    ("F0_FRAME", int, 0),
        "stim_frames": ("STIM_FRAMES", "frames", 0),
    },
}
# Keys that may be left out, with the value used then.
DEFAULTS = {
    ("tracking", "workers"): 6,
    ("post_analysis", "stim_frames"): [],
}


class ConfigError(Exception):
    pass


class _Dumper(yaml.SafeDumper):
    """Block-style sections, but short lists inline: shift_xy: [-6, 0]."""


_Dumper.add_representer(
    list, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))


def _dump(data):
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, width=1000)


# ---------------- reading + validation ---------------- #
def find_config(target):
    """<experiment_dir | config.yaml> -> absolute path of the config file."""
    path = os.path.abspath(target)
    if os.path.isdir(path):
        path = os.path.join(path, CONFIG_NAME)
    if not os.path.isfile(path):
        raise ConfigError(
            f"No config file at {path}. Save one from TUNE_GUI (Review & Run), "
            f"or copy configs/example_c2c12_chamber_A.yaml into the experiment "
            f"folder as {CONFIG_NAME} and edit it.")
    return path


def read_config(path):
    """Load a config file as a dict without validating it."""
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must hold a mapping of sections, not {type(raw).__name__}")
    return raw


def _check_value(kind, minimum, v):
    """Return `v` as `kind`, or raise ValueError saying what was expected."""
    if kind == "pair":
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("expected [dx, dy]")
        return [_check_value(float, None, x) for x in v]
    if kind == "frames":
        # A list, or the comma-separated text the old drivers used ("77,112").
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        elif isinstance(v, int) and not isinstance(v, bool):
            v = [v]
        if not isinstance(v, list):
            raise ValueError("expected a list of frame numbers")
        return [_check_value(int, minimum, x) for x in v]
    if isinstance(v, str):
        # YAML reads e.g. 1e-3 (no decimal point) as text.
        try:
            v = float(v)
        except ValueError:
            raise ValueError(f"expected a number, got {v!r:.40}") from None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"expected a number, got {v!r:.40}")
    if not math.isfinite(v):
        raise ValueError(f"expected a finite number, got {v}")
    if kind is int:
        if isinstance(v, float) and not v.is_integer():
            raise ValueError(f"expected a whole number, got {v}")
        v = int(v)
    if minimum is not None and v < minimum:
        raise ValueError(f"must be >= {minimum}, got {v}")
    return v


def validate(raw, sections, path="config"):
    """Check `raw` and return {section: {key: value}} for `sections`, with
    defaults filled in. Raises ConfigError listing every problem found.
    Sections not asked for are checked only if present."""
    problems = []
    unknown = sorted(set(raw) - set(SECTIONS) - {"experiment_dir"})
    if unknown:
        problems.append(f"unknown top-level key(s) {unknown}; expected experiment_dir, "
                        f"{', '.join(SECTIONS)}")
    out = {}
    for section, spec in SECTIONS.items():
        if section not in raw and section not in sections:
            continue
        values = raw.get(section)
        if values is None:
            problems.append(f"{section}: section is missing")
            continue
        if not isinstance(values, dict):
            problems.append(f"{section}: expected key: value lines")
            continue
        for key in sorted(set(values) - set(spec)):
            problems.append(f"{section}.{key}: unknown key; expected one of {', '.join(spec)}")
        clean = {}
        for key, (_, kind, minimum) in spec.items():
            if key not in values or values[key] is None:
                if (section, key) in DEFAULTS:
                    clean[key] = DEFAULTS[(section, key)]
                else:
                    problems.append(f"{section}.{key}: missing")
                continue
            try:
                clean[key] = _check_value(kind, minimum, values[key])
            except ValueError as e:
                problems.append(f"{section}.{key}: {e}")
        out[section] = clean
    if problems:
        raise ConfigError(f"{path}:\n  " + "\n  ".join(problems))
    return {s: out[s] for s in SECTIONS if s in out}


def experiment_dir(config_path, raw):
    """The experiment folder: `experiment_dir` (relative to the config
    file's folder) if given, else the folder holding the config file."""
    base = os.path.dirname(config_path)
    exp = raw.get("experiment_dir")
    if exp is None:
        return base
    if not isinstance(exp, str) or not exp:
        raise ConfigError(f"{config_path}: experiment_dir must be a folder path")
    return os.path.normpath(os.path.join(base, os.path.expanduser(exp)))


def display_path(path):
    """`path` relative to the project root when it is inside it, else absolute."""
    rel = os.path.relpath(os.path.abspath(path), PROJECT_ROOT)
    return os.path.abspath(path) if rel == ".." or rel.startswith(".." + os.sep) else rel


# ---------------- shell export + run record ---------------- #
def _shell_value(v):
    if isinstance(v, list):  # stim_frames
        return ",".join(str(x) for x in v)
    return str(v)


def shell_assignments(values, paths):
    """Bash `NAME=value` lines for the drivers to eval."""
    lines = [f"{name}={shlex.quote(str(p))}" for name, p in paths.items()]
    for section, clean in values.items():
        for key, v in clean.items():
            name = SECTIONS[section][key][0]
            if key == "shift_xy":
                lines.append(f"SHIFT_X={shlex.quote(_shell_value(_plain(v[0])))}")
                lines.append(f"SHIFT_Y={shlex.quote(_shell_value(_plain(v[1])))}")
            else:
                lines.append(f"{name}={shlex.quote(_shell_value(v))}")
    return "\n".join(lines)


def _plain(x):
    """3.0 -> 3, so whole-pixel shifts print the way they were written."""
    return int(x) if isinstance(x, float) and x.is_integer() else x


def _git_version():
    """HEAD commit, with "-dirty" if tracked files differ from it."""
    def git(*args):
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", PROJECT_ROOT, *args],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def write_run_record(analysis_dir, run_label, config_path, exp_dir, values):
    """Save the values a driver is about to use to
    analysis/run_history/<timestamp>_<script>.yaml and return its path."""
    history = os.path.join(analysis_dir, HISTORY_DIR)
    os.makedirs(history, exist_ok=True)
    now = datetime.datetime.now()
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    record = {
        "run": run_label,
        "started": now.isoformat(timespec="seconds"),
        "user": user,
        "code_version": _git_version(),
        "config_file": display_path(config_path),
        "experiment_dir": display_path(exp_dir),
        **values,
    }
    header = (f"# Parameters `{run_label}` used for this experiment. Written when the\n"
              f"# run started, so the run may have failed or been stopped. Do not edit.\n")
    stem = os.path.join(history, f"{now:%Y%m%d-%H%M%S}_{run_label.split()[0].removesuffix('.sh')}")
    for n in range(1, 100):
        path = stem + (f"_{n}" if n > 1 else "") + ".yaml"
        try:
            with open(path, "x") as f:
                f.write(header + _dump(record))
            return path
        except FileExistsError:
            continue
    raise ConfigError(f"could not create a run record under {history}")


def prepare(target, run_label, sections):
    """Validate the config for `sections`, save the run record, and return
    the shell assignments for the driver."""
    config_path = find_config(target)
    raw = read_config(config_path)
    values = validate(raw, sections, config_path)
    values = {s: values[s] for s in sections}  # record only what this run uses
    exp_dir = experiment_dir(config_path, raw)
    frames_dir = os.path.join(exp_dir, "frames")
    if not os.path.isdir(frames_dir):
        raise ConfigError(f"{config_path}: no frames/ folder in the experiment folder {exp_dir}")
    analysis_dir = os.path.join(exp_dir, "analysis")
    record = write_run_record(analysis_dir, run_label, config_path, exp_dir, values)
    paths = {
        "CONFIG_FILE": display_path(config_path),
        "EXP_DIR": display_path(exp_dir),
        "IMAGE_DIR": display_path(frames_dir),
        "MASK_DIR": display_path(os.path.join(exp_dir, "masks")),
        "ANALYSIS_DIR": display_path(analysis_dir),
        "RUN_RECORD": display_path(record),
    }
    return shell_assignments(values, paths)


# ---------------- writing (GUIs) ---------------- #
_HEADER = """\
# Cell Trainer pipeline parameters for the experiment in this folder.
# Written by {source}. You can edit it by hand, but comments are not kept when
# {source} saves it again. Every key is documented in
# configs/example_c2c12_chamber_A.yaml.
#
# Run from the project root (or give the full path to the scripts):
#   bash run_processes.sh      "<this folder>"   # segmentation + tracking + pre-analysis
#   bash run_post_processes.sh "<this folder>"   # post-analysis
# Each run saves the values it used under analysis/{history}/.
#
# shift_frame, f0_frame and stim_frames are timepoint_NNNNN numbers from the
# frame file names. radius: 0 turns the ROI off. diameter: 0 lets Cellpose
# estimate it.
"""


def render_config(path, updates, source):
    """The text `write_config` would write, without writing it."""
    raw = read_config(path) if os.path.isfile(path) else {}
    for section, values in updates.items():
        if not isinstance(raw.get(section), dict):
            raw[section] = {}
        raw[section].update(values)
    values = validate(raw, list(updates), path)
    out = {"experiment_dir": raw["experiment_dir"]} if "experiment_dir" in raw else {}
    for section, clean in values.items():
        if section == "tracking":
            clean = {**clean, "shift_xy": [_plain(x) for x in clean["shift_xy"]]}
        out[section] = clean
    return _HEADER.format(source=source, history=HISTORY_DIR) + "\n" + _dump(out)


def write_config(path, updates, source):
    """Merge `updates` ({section: {key: value}}) into the config at `path`
    (creating it if needed), validate the result, and write it. Keys the
    caller does not manage (e.g. tracking.workers, experiment_dir) are kept."""
    text = render_config(path, updates, source)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return text


# ---------------- CLI ---------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="used by the run_*.sh drivers")
    p.add_argument("target", help="experiment folder or config file")
    p.add_argument("run_label", help='e.g. "run_processes.sh"')
    p.add_argument("sections", nargs="+", choices=list(SECTIONS))
    c = sub.add_parser("check", help="validate a config and print the values it holds")
    c.add_argument("target", help="experiment folder or config file")
    args = parser.parse_args(argv)

    try:
        if args.cmd == "prepare":
            print(prepare(args.target, args.run_label, args.sections))
        else:
            config_path = find_config(args.target)
            raw = read_config(config_path)
            values = validate(raw, [], config_path)
            print(f"# {display_path(config_path)}  (experiment: "
                  f"{display_path(experiment_dir(config_path, raw))})")
            print(_dump(values), end="")
    except ConfigError as e:
        sys.exit(f"Config error: {e}")


if __name__ == "__main__":
    main()
