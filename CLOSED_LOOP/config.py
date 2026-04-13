import os
import json

DEFAULT_GLOBAL_PATH = "/mnt/data/Close_Loop_Data"

_DIR_KEYS = [
    "watch_dir", "mask_dir", "temp_overlays", "curr_mask_dir",
    "decision_dir", "final_dir", "flags_dir",
]


def build_config(global_path=None, **overrides):
    """Build the full config dict, deriving all paths from *global_path*.

    Any key passed as a keyword argument overrides the default value.
    """
    if global_path is None:
        global_path = DEFAULT_GLOBAL_PATH

    config = {
        "global_path": global_path,
        "watch_dir": f"{global_path}/images/nrk_acid_feedback_experiment_chCD_13APR26_take2",
        "mask_dir": f"{global_path}/processed_masks",
        "temp_overlays": f"{global_path}/temp_overlays",
        "curr_mask_dir": f"{global_path}/current_masks",
        "decision_dir": f"{global_path}/temp_decisions",
        "final_dir": f"{global_path}/final_decisions",
        "flags_dir": f"{global_path}/flags",
        "setpoint_file": f"{global_path}/setpoints.txt",
        "luminosity_file": f"{global_path}/luminosity_log.json",
        "threshold_ratio": 0.05,
        "num_channels": 2,
        "num_tries": 30,
        "sleep_time": 0.2,
        "decision_key": {'add neutral media': 1,
                         'add acidic media': 2,
                         'add basic media': 3},
        "onix_server_ip": "192.0.2.10",
        "onix_server_port": 8881,
        "experiment_templates": {
            "NN": "C:\\ONIX2\\Experiments\\NN.OnixExp",
            "AN": "C:\\ONIX2\\Experiments\\AN.OnixExp",
            "NA": "C:\\ONIX2\\Experiments\\NA.OnixExp",
            "AA": "C:\\ONIX2\\Experiments\\AA.OnixExp",
        },
        "retention_time_hours": 3,
        "cleanup_interval_sec": 1800,
        "directories_to_clean": [
            f"{global_path}/processed_masks",
            f"{global_path}/temp_overlays"
        ],
        "run_duration_sec": 86400,
        "acidic_pulse_sec": 30,
        "neutral_experiment": "NN",
        "continuous_segmentation": False,
    }
    config.update(overrides)
    return config


def save_config(config, save_dir=None):
    """Ensure all data directories exist and write config.json.

    Returns the path to the saved file.
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    for key in _DIR_KEYS:
        if key in config:
            os.makedirs(config[key], exist_ok=True)
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)
    return save_path


if __name__ == "__main__":
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(save_path):
        # Config already exists (e.g. saved by Napari UI) -- just ensure directories
        with open(save_path) as f:
            cfg = json.load(f)
        for key in _DIR_KEYS:
            if key in cfg:
                os.makedirs(cfg[key], exist_ok=True)
        print(f"config.json exists -- ensured directories at {save_path}")
    else:
        cfg = build_config()
        path = save_config(cfg)
        print(f"Configuration saved to {path}")
