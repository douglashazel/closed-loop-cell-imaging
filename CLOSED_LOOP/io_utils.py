"""
Shared utilities for the closed-loop pipeline.
All scripts import log(), load_config(), and parse_filename() from here.
"""
import os
import re
import json
import time
from datetime import datetime

# Matches filenames like: channel_1_image_0_a_timepoint_00000.png
_FILENAME_PATTERN = re.compile(r"channel_(\d+).*timepoint_(\d+)\.png$", re.IGNORECASE)

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

def parse_filename(fname):
    """Return (channel, frame) ints from an image filename, or (None, None) on no match."""
    m = _FILENAME_PATTERN.search(os.path.basename(fname))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def wait_for_file(path, sleep_time=2):
    """Block until path exists on disk."""
    while not os.path.exists(path):
        time.sleep(sleep_time)