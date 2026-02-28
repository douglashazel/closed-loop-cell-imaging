import os
import json

global_path = "/mnt/data/Close_Loop_Data"

config = {
    "gobal_path": global_path,
    "watch_dir": f"{global_path}/images/feedback test",
    "mask_dir": f"{global_path}/processed_masks",
    "temp_overlays": f"{global_path}/temp_overlays",
    "curr_mask_dir": f"{global_path}/current_masks",
    "decision_dir": f"{global_path}/temp_decisions",
    "final_dir": f"{global_path}/final_decisions",
    "flags_dir": f"{global_path}/flags",
    "setpoint_file": f"{global_path}/setpoints.txt",
    "threshold_ratio": 0.05,
    "num_channels": 2,
    "num_tries": 5,
    "sleep_time": 2,
    "decision_key": {'add neutral media': 1,
                'add acidic media': 2,
                'add basic media': 3}
}

# Ensure flags directory exists
os.makedirs(config["watch_dir"], exist_ok=True)
os.makedirs(config["mask_dir"], exist_ok=True)
os.makedirs(config["temp_overlays"], exist_ok=True)
os.makedirs(config["curr_mask_dir"], exist_ok=True)
os.makedirs(config["decision_dir"], exist_ok=True)
os.makedirs(config["final_dir"], exist_ok=True)
os.makedirs(config["flags_dir"], exist_ok=True)

save_path = f"/mnt/exDisk1/douglashazel/DHcode/PE_Pipeline/DEMO/config.json"

with open(save_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"Configuration saved to {os.path.abspath(save_path)}")