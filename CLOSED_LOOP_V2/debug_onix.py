#!/usr/bin/env python3
"""Standalone ONIX2 server probe.

Usage:
    python3 debug_onix.py                 # run full probe sequence
    python3 debug_onix.py status          # just /Status
    python3 debug_onix.py open            # just /IsExperimentOpen
    python3 debug_onix.py close           # /CloseExperiment save=false
    python3 debug_onix.py create          # /CreateExperiment NN (default params)
    python3 debug_onix.py create-variants # try CreateExperiment with several param shapes
    python3 debug_onix.py raw <cmd> k=v   # arbitrary GET, e.g.  raw CreateExperiment filename=... templatename=...

Reads host/port/templates from V5/config.json so it matches the real pipeline.
"""
import json
import os
import sys
from datetime import datetime

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

HOST = cfg.get("onix_server_ip", "192.0.2.10")
PORT = cfg.get("onix_server_port", 8881)
BASE = f"http://{HOST}:{PORT}/onixserver"
TEMPLATES = cfg.get("experiment_templates", {})
NEUTRAL = cfg.get("neutral_experiment", "NN")

session = requests.Session()


def _pp(label, resp):
    print(f"--- {label} ---")
    print(f"  URL:    {resp.url}")
    print(f"  Status: {resp.status_code}")
    try:
        print(f"  JSON:   {json.dumps(resp.json(), indent=2)}")
    except Exception:
        print(f"  Body:   {resp.text[:500]}")
    print()


def get(cmd, params=None, label=None):
    url = f"{BASE}/{cmd}"
    try:
        r = session.get(url, params=params or {}, timeout=10)
    except requests.RequestException as e:
        print(f"--- {label or cmd} ---")
        print(f"  URL:    {url}")
        print(f"  ERROR:  {e}\n")
        return None
    _pp(label or cmd, r)
    return r


def cmd_status():
    get("Status")


def cmd_open():
    get("IsExperimentOpen")


def cmd_close():
    get("CloseExperiment", {"save": "false"})


def _new_filename(template_path):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if "\\" in template_path:
        template_dir = template_path.rsplit("\\", 1)[0]
        return f"{template_dir}\\DebugRun_{ts}.OnixExp"
    return f"DebugRun_{ts}.OnixExp"


def cmd_create(experiment_name=None):
    name = experiment_name or NEUTRAL
    template = TEMPLATES.get(name)
    if not template:
        print(f"No template for '{name}' in config.json. Available: {list(TEMPLATES)}")
        return
    new_file = _new_filename(template)
    get(
        "CreateExperiment",
        {"filename": new_file, "templatename": template},
        label=f"CreateExperiment ({name}, default params)",
    )


def cmd_create_variants():
    """Try several plausible param shapes to see which the server accepts."""
    name = NEUTRAL
    template = TEMPLATES.get(name)
    if not template:
        print(f"No template for '{name}' in config.json.")
        return
    new_file = _new_filename(template)

    variants = [
        ("filename+templatename", {"filename": new_file, "templatename": template}),
        ("filename+templateName (camelCase)", {"filename": new_file, "templateName": template}),
        ("filename+template", {"filename": new_file, "template": template}),
        ("fileName+templateName (both camel)", {"fileName": new_file, "templateName": template}),
        ("+ empty actions", {"filename": new_file, "templatename": template, "actions": "[]"}),
        ("+ empty actions object", {"filename": new_file, "templatename": template, "actions": "{}"}),
        ("just templatename", {"templatename": template}),
        ("just template path as filename", {"filename": template}),
    ]

    for label, params in variants:
        get("CreateExperiment", params, label=f"CreateExperiment :: {label}")
        # If any variant worked, clean up so the next one has a blank slate.
        get("CloseExperiment", {"save": "false"}, label="  cleanup: CloseExperiment")


def cmd_raw(argv):
    if not argv:
        print("raw usage: debug_onix.py raw <command> [key=value ...]")
        return
    cmd = argv[0]
    params = {}
    for kv in argv[1:]:
        if "=" not in kv:
            print(f"bad param: {kv}")
            return
        k, v = kv.split("=", 1)
        params[k] = v
    get(cmd, params)


def cmd_full():
    print(f"ONIX2 probe against {BASE}")
    print(f"Templates: {TEMPLATES}\n")
    get("Status", label="1. Status")
    get("IsExperimentOpen", label="2. IsExperimentOpen")
    get("CloseExperiment", {"save": "false"}, label="3. CloseExperiment (safety)")
    cmd_create()
    get("IsExperimentOpen", label="5. IsExperimentOpen (after Create)")
    get("CloseExperiment", {"save": "false"}, label="6. CloseExperiment (cleanup)")


def main():
    argv = sys.argv[1:]
    if not argv:
        cmd_full()
        return
    sub = argv[0]
    if sub == "status":
        cmd_status()
    elif sub == "open":
        cmd_open()
    elif sub == "close":
        cmd_close()
    elif sub == "create":
        cmd_create(argv[1] if len(argv) > 1 else None)
    elif sub == "create-variants":
        cmd_create_variants()
    elif sub == "raw":
        cmd_raw(argv[1:])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
