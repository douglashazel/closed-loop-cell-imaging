import os
import time
import numpy as np
import json
import zipfile
from collections import defaultdict

from io_utils import log, load_config

cfg = load_config()
mask_dir        = cfg["mask_dir"]
flags_dir       = cfg["flags_dir"]
num_channels    = cfg["num_channels"]
threshold_ratio = cfg["threshold_ratio"]

# Track last frame ID and ROI count per channel
last_frame = defaultdict(lambda: None)
last_count = defaultdict(lambda: None)

def count_rois(meta_file):
    """Count ROIs from a precomputed metadata file (.json)."""
    try:
        with open(meta_file, 'r') as f:
            data = json.load(f)
        return data.get("roi_count", 0)
    except Exception:
        return 0

def get_frame_and_channel(filename):
    """Extract frame and channel from filename like 00000_channel1_meta.json."""
    parts = filename.split('_channel')
    if len(parts) != 2:
        return None, None
    try:
        frame = int(parts[0])
        channel = int(parts[1].split('_meta.json')[0])
        return frame, channel
    except ValueError:
        return None, None

def create_flag_file(frame, channel, message):
    """Create a text file in flags directory with the message."""
    filename = f"{frame:05d}_channel{channel}.txt"
    filepath = os.path.join(flags_dir, filename)
    with open(filepath, 'w') as f:
        f.write(message)

def cleanup_old_files():
    now = time.time()
    retention_sec = cfg.get("retention_time_hours", 24) * 3600
    dirs = cfg.get("directories_to_clean", [])
    if not dirs:
        return
        
    archive_name = os.path.join(cfg["global_path"], f"archive_{time.strftime('%Y%m%d')}.zip")
    files_to_compress = []
    
    for d in dirs:
        if not os.path.exists(d): continue
        for root, _, filenames in os.walk(d):
            for f in filenames:
                file_path = os.path.join(root, f)
                if file_path == archive_name: continue
                if now - os.path.getmtime(file_path) > retention_sec:
                    files_to_compress.append(file_path)
    
    if files_to_compress:
        log(f"Compressing {len(files_to_compress)} old files into {archive_name}")
        try:
            with zipfile.ZipFile(archive_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                for fpath in files_to_compress:
                    zf.write(fpath, os.path.relpath(fpath, cfg["global_path"]))
            for fpath in files_to_compress:
                try:
                    os.remove(fpath)
                except OSError as e:
                    log(f"Failed to remove {fpath}: {e}")
        except Exception as e:
            log(f"Error during compression: {e}")

log("Performance monitor started...")

retries = 0
last_cleanup = 0
cleanup_interval = cfg.get("cleanup_interval_sec", 3600)

while True:
    try:
        if time.time() - last_cleanup > cleanup_interval:
            cleanup_old_files()
            last_cleanup = time.time()

        files = [f for f in os.listdir(mask_dir) if f.endswith('_meta.json')]
        files.sort()  # ensure chronological order

        for f in files:
            frame, channel = get_frame_and_channel(f)
            if frame is None or channel is None or channel < 1 or channel > num_channels:
                continue

            # Skip files for frames we've already processed
            if last_frame[channel] is not None and frame <= last_frame[channel]:
                continue

            filepath = os.path.join(mask_dir, f)
            current_count = count_rois(filepath)

            if last_count[channel] is not None:
                prev_count = last_count[channel]
                diff = current_count - prev_count
                if diff != 0:
                    sign = '+' if diff > 0 else ''
                    msg = f"channel {channel} {'gained' if diff > 0 else 'lost'} {sign}{diff} ROI's on frame {frame}"
                    log(msg)

                    # Check for significant change (5% of previous count)
                    if prev_count > 0 and abs(diff) / prev_count >= threshold_ratio:
                        flag_msg = (
                            f"Frame: {frame}\n"
                            f"Channel: {channel}\n"
                            f"Previous: {prev_count} cells\n"
                            f"Current: {current_count} cells\n"
                            f"Change: {sign}{diff} cells\n"
                        )
                        create_flag_file(frame, channel, flag_msg)

            last_frame[channel] = frame
            last_count[channel] = current_count

        retries = 0  # reset retries if successful
        time.sleep(cfg["sleep_time"])

    except (OSError, ValueError, EOFError, AttributeError) as e:
        retries += 1
        log(f"Error accessing {mask_dir} (retry {retries}/{cfg['num_tries']}): {e}")
        if retries >= cfg["num_tries"]:
            log("Max retries reached. Stopping monitor.")
            break
        time.sleep(cfg["sleep_time"])