import os
import time
import json
import tomlkit
import numpy as np
from PIL import Image

from io_utils import log, load_config, parse_filename

cfg = load_config()
num_channels    = cfg["num_channels"]
watch_dir       = cfg["watch_dir"]
curr_mask_dir   = cfg["curr_mask_dir"]
decision_dir    = cfg["decision_dir"]
final_dir       = cfg["final_dir"]
setpoint_file   = cfg["setpoint_file"]
luminosity_file = cfg["luminosity_file"]

decision_key    = cfg['decision_key']
decision_rev    = {v: k for k, v in decision_key.items()}
continuous_seg  = cfg.get("continuous_segmentation", True)

# Tracks the active mask filename per channel so we can log hot-swaps exactly once
_active_mask_name = {}

def _luminosity_path(channel):
    """Per-channel luminosity log path.
    luminosity_log.json -> luminosity_log_channel1.json, luminosity_log_channel2.json, ...
    """
    base, ext = os.path.splitext(luminosity_file)
    return f"{base}_channel{channel}{ext}"

def append_luminosity(frame, channel, mean_val, setpoint, decision_label):
    """Append a luminosity record to the per-channel JSON log file.

    The file stays one JSON array (rewritten atomically each frame) because the
    GUI and the Stage 2 analysis / published supplement read it in that format."""
    record = {
        "frame": frame,
        "channel": channel,
        "mean_luminosity": round(float(mean_val), 4),
        "setpoint": round(float(setpoint), 4),
        "decision": decision_label,
    }
    path = _luminosity_path(channel)
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(record)
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)

def get_latest_complete_frame():
    """Return the highest frame number that has all channels present, or -1."""
    frame_channels = {}
    for f in os.listdir(watch_dir):
        ch, frame_num = parse_filename(f)
        if ch is None:
            continue
        frame_channels.setdefault(frame_num, set()).add(ch)
    complete = [fr for fr, chs in frame_channels.items() if len(chs) >= num_channels]
    return max(complete) if complete else -1

def wait_for_frame(frame):
    """Block until the given frame has all channels present."""
    while True:
        chs = set()
        for f in os.listdir(watch_dir):
            ch, frame_num = parse_filename(f)
            if ch is not None and frame_num == frame:
                chs.add(ch)
        if len(chs) >= num_channels:
            return
        time.sleep(cfg["sleep_time"])

def compute_setpoint(initial_masks):
    """Return a per-channel dict of setpoints at 200% of initial brightness."""
    setpoints = {}
    for ch in range(1, num_channels + 1):
        img_file = next(
            (f for f in os.listdir(watch_dir) if parse_filename(f) == (ch, 0)), None
        )
        if img_file is None:
            log(f"compute_setpoint: no frame-0 image for channel {ch} in {watch_dir}")
            continue
        img_path = os.path.join(watch_dir, img_file)
        for attempt in range(cfg["num_tries"]):
            try:
                img = np.array(Image.open(img_path), dtype=np.float32)
                mask = initial_masks[ch]
                if mask.shape != img.shape[:2]:
                    log(f"compute_setpoint: channel {ch} mask shape {mask.shape} does not "
                        f"match image shape {img.shape[:2]} ({img_file}); no setpoint")
                    break
                setpoints[ch] = float(img[mask].mean()) * 2.0
                break
            except (OSError, ValueError, EOFError, AttributeError, SyntaxError,
                    KeyError, IndexError) as e:
                log(f"compute_setpoint retry {attempt + 1}/{cfg['num_tries']} "
                    f"for channel {ch}: {e}")
                time.sleep(1)
        else:
            log(f"compute_setpoint: failed channel {ch} after {cfg['num_tries']} retries")
    return setpoints

def save_setpoints(setpoints):
    tmp = setpoint_file + ".tmp"
    with open(tmp, 'w') as f:
        for ch, val in setpoints.items():
            f.write(f"setpoint_channel{ch}={val:.6f}\n")
    os.rename(tmp, setpoint_file)

def load_setpoints(default_setpoints):
    if not os.path.exists(setpoint_file):
        return dict(default_setpoints)
    result = dict(default_setpoints)
    try:
        with open(setpoint_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('setpoint_channel'):
                    key, val = line.split('=', 1)
                    ch = int(key[len('setpoint_channel'):])
                    result[ch] = float(val)
    except Exception as e:
        log(f"Warning: failed to parse {setpoint_file}: {e}")
    return result

def process_frame(frame, initial_masks, default_setpoints):
    """
    Evaluate a single frame.
    - continuous_segmentation=False: uses fixed in-memory frame-0 masks (no disk reads).
    - continuous_segmentation=True:  re-reads curr_mask_dir each frame, so any mask the user
      pushes with the web GUI's Push mask action mid-experiment is picked up
      automatically (hot-swap).
    """
    wait_for_frame(frame)
    setpoints = load_setpoints(default_setpoints)

    for ch in range(1, num_channels + 1):
        setpoint = setpoints.get(ch)
        if setpoint is None:
            log(f"Frame {frame} channel {ch}: no setpoint in {setpoint_file} "
                f"(set one in the web GUI). Skipping.")
            continue
        img_file = next(
            (f for f in os.listdir(watch_dir) if parse_filename(f) == (ch, frame)), None
        )
        if img_file is None:
            continue

        img_path = os.path.join(watch_dir, img_file)

        for attempt in range(cfg["num_tries"]):
            try:
                img = np.array(Image.open(img_path), dtype=np.float32)

                if continuous_seg:
                    mask_files = [f for f in os.listdir(curr_mask_dir)
                                  if f.endswith(f"_channel{ch}.npy")]
                    if mask_files:
                        mask_fname = mask_files[0]
                        if _active_mask_name.get(ch) != mask_fname:
                            log(f"Channel {ch}: active mask -> '{mask_fname}'")
                            _active_mask_name[ch] = mask_fname
                        mask = np.load(os.path.join(curr_mask_dir, mask_fname)) > 0
                    else:
                        mask = initial_masks[ch]
                else:
                    mask = initial_masks[ch]

                if mask.shape != img.shape[:2]:
                    log(f"Frame {frame} channel {ch}: mask shape {mask.shape} does not match "
                        f"image shape {img.shape[:2]} ({img_file}). Skipping.")
                    break

                mean_val = img[mask].mean()

                if mean_val >= setpoint:
                    decision = decision_key['add acidic media']
                else:
                    decision = decision_key['add neutral media']

                with open(os.path.join(decision_dir, f"{frame:05d}_channel{ch}.txt"), 'w') as f:
                    f.write(str(decision))

                append_luminosity(frame, ch, mean_val, setpoint, decision_rev[decision])
                log(f"{frame:05d}_channel{ch}: {mean_val:.3f} -> {decision_rev[decision]} "
                    f"(setpoint={setpoint:.3f})")
                break

            except (OSError, ValueError, EOFError, AttributeError, SyntaxError,
                    KeyError, IndexError) as e:
                log(f"Retry {attempt + 1}/{cfg['num_tries']} for frame {frame} channel {ch}: {e}")
                time.sleep(1)
        else:
            log(f"Failed frame {frame} channel {ch} after {cfg['num_tries']} retries. Skipping.")

def finalize_decisions(frame):
    """Read per-channel decisions and write a crossing signal for SendDecisions.

    The file contains per-channel threshold state (acid / neutral) so that
    SendDecisions can manage independent 30-second pulse timers.

    process_frame() writes the decision files synchronously before this runs,
    so a channel it skipped (no setpoint, unreadable image, mask mismatch) is
    left out here rather than waited for.
    """
    channel_states = {}
    for ch in range(1, num_channels + 1):
        dec_path = os.path.join(decision_dir, f"{frame:05d}_channel{ch}.txt")
        if not os.path.exists(dec_path):
            log(f"Frame {frame}: no decision for channel {ch}; leaving it out of actions.toml")
            continue
        with open(dec_path, 'r') as f:
            dec = int(f.read().strip())
        channel_states[str(ch)] = decision_rev[dec]
    if not channel_states:
        log(f"Frame {frame}: no channel decisions; actions.toml not written")
        return

    # Write atomically via lock file then rename
    lock_path    = os.path.join(final_dir, f"{frame:05d}.lock")
    actions_path = os.path.join(final_dir, "actions.toml")
    with open(lock_path, "w") as f:
        tomlkit.dump({"frame": frame, "channels": channel_states}, f)
    os.rename(lock_path, actions_path)

    log(f"Frame {frame}: wrote actions.toml -> {channel_states}")

    # Clean up per-channel decision files
    for ch in range(1, num_channels + 1):
        dec_path = os.path.join(decision_dir, f"{frame:05d}_channel{ch}.txt")
        try:
            os.remove(dec_path)
        except FileNotFoundError:
            pass

# ---- INITIAL SETUP ----
# Stall here until the user has pushed the approved frame-0 segmentation
# into curr_mask_dir with the web GUI's Push mask action.  Nothing starts
# until the user is happy.
log("Waiting for user to push approved frame-0 masks (web GUI: Segmentation tab, Push mask)...")

for ch in range(1, num_channels + 1):
    mask_path = os.path.join(curr_mask_dir, f"00000_channel{ch}.npy")
    waited = 0
    while not os.path.exists(mask_path):
        time.sleep(cfg["sleep_time"])
        waited += cfg["sleep_time"]
        if waited % 60 < cfg["sleep_time"]:
            log(f"Still waiting for channel {ch} mask at {mask_path} ({waited:.0f}s elapsed)")

initial_masks = {
    ch: np.load(os.path.join(curr_mask_dir, f"00000_channel{ch}.npy")) > 0
    for ch in range(1, num_channels + 1)
}

setpoints = compute_setpoint(initial_masks)

save_setpoints(setpoints)
for ch, val in setpoints.items():
    log(f"Setpoint channel{ch}: {val:.3f} (200% of initial brightness) — saved to {setpoint_file}")

# ---- MAIN LOOP ----
# Always jump to the newest complete frame and make a decision on it.
# Older frames are dropped — we don't need to process a backlog, only the
# current state of the dish matters. Per-channel pulses are managed
# independently downstream by SendDecisions' PulseManager.
last_processed = -1
while True:
    try:
        latest = get_latest_complete_frame()
        if latest < 0 or latest == last_processed:
            time.sleep(cfg["sleep_time"])
            continue

        process_frame(latest, initial_masks, setpoints)
        finalize_decisions(latest)
        last_processed = latest
    except (OSError, ValueError, EOFError, AttributeError, SyntaxError,
            KeyError, IndexError) as e:
        log(f"Error on frame {last_processed + 1}: {e}")
        time.sleep(cfg["sleep_time"])