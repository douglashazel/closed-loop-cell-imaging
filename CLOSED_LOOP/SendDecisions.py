import os
import json as _json
import time
import threading
import tomlkit
import requests
from datetime import datetime

from io_utils import log, load_config

cfg = load_config()
global_dir   = cfg["global_path"]
final_dir    = cfg["final_dir"]
sleep_time   = cfg.get("sleep_time", 0.5)
run_duration = cfg.get("run_duration_sec", 6000)

# ONIX Configuration
ONIX_SERVER_IP   = cfg.get("onix_server_ip", "192.0.2.10")
ONIX_SERVER_PORT = cfg.get("onix_server_port", 8881)

# Experiment name to template path mapping
EXPERIMENT_TEMPLATES = cfg.get("experiment_templates", {})

if not EXPERIMENT_TEMPLATES:
    log("WARNING: No experiment templates found in config.json!")
    log("Please ensure 'experiment_templates' is properly configured.")
class OnixController:
    """Minimal ONIX2 controller for executing experiments."""
    
    def __init__(self, host_ip, port):
        self.base_url = f"http://{host_ip}:{port}/onixserver"
        self.session = requests.Session()
        self.abort_current = False  # NEW: flag to abort current experiment
        self.current_experiment = None  # NEW: track current experiment name
        log(f"Connecting to ONIX2 Server at: {self.base_url}")
        self.init_logging()

    def init_logging(self):
        """Creates a CSV file to track hardware flags over time."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filename = f"{global_dir}/ONIX_Hardware_Log_{timestamp}.csv"
        
        header = "PC_Time,Context,RunState,Flags0(Sys),Flags1(Gas/Leak),Flags2(Env),PressureX,PressureY,Temp_C\n"
        
        with open(self.log_filename, "w") as f:
            f.write(header)
        log(f"Telemetry will be saved to: {self.log_filename}")

    def log_telemetry(self, context_label, status=None):
        """Appends a row to the CSV log with current hardware state."""
        if not status:
            try:
                status = self._send_request("Status")
            except:
                return

        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        run_state = status.get("RunState", "ERR")
        f0 = status.get("Flags0", "0"*16)
        f1 = status.get("Flags1", "0"*16)
        f2 = status.get("Flags2", "0"*16)
        pX = status.get("X", 0)
        pY = status.get("Y", 0)
        temp = status.get("Temperature", 0)

        row = f"{now},{context_label},{run_state},{f0},{f1},{f2},{pX},{pY},{temp}\n"
        
        try:
            with open(self.log_filename, "a") as f:
                f.write(row)
        except Exception as e:
            log(f"Log Error: Could not write to file: {e}")

    def _send_request(self, command, params=None):
        url = f"{self.base_url}/{command}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            log(f"Network error on command '{command}': {e}")
            raise

    def get_status(self):
        return self._send_request("Status")

    def _poll_for_state(self, target_states, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status()
            self.log_telemetry("Polling_Wait", status)
            
            try:
                current_state = int(status.get("RunState", -999))
            except (ValueError, TypeError):
                current_state = -999
            
            if current_state in target_states:
                return True, current_state
            time.sleep(sleep_time)
        return False, current_state

    def wait_for_hardware_idle(self, timeout=15):
        """Polls Flags0 Bit 0 (System Ready). Bit 0 = 1 means READY."""
        log("Checking Hardware Ready Flag (Flags0, Bit 0)...")
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status()
            self.log_telemetry("Wait_For_Idle", status)
            
            flags0 = status.get("Flags0", "0"*16)
            is_ready = flags0[-1] == '1'
            
            if is_ready:
                run_state = int(status.get("RunState", -999))
                if run_state == 0:
                    return True
            
            time.sleep(sleep_time)
            
        log("Warning: Hardware Ready Flag did not set (System Busy).")
        return False

    def run_experiment(self, template_path, experiment_name, run_duration=run_duration):  # MODIFIED: added experiment_name
        """
        Execute a single ONIX experiment:
        Create -> Open -> Start -> Wait -> Abort -> Close(Save)
        """
        log(f"Starting experiment: {template_path}")
        self.log_telemetry(f"Start_{os.path.basename(template_path)}")
        self.abort_current = False        # clear any stale abort flag from previous experiment
        self.current_experiment = experiment_name  # NEW: set current experiment
        
        # 1. ENSURE CLEAN SLATE
        try:
            check_resp = self._send_request("IsExperimentOpen")
            if check_resp.get("experimentOpen", False):
                log("Found open experiment. Closing...")
                self._send_request("CloseExperiment", {"save": "false"})
                time.sleep(sleep_time)
        except Exception as e:
            log(f"Warning: Cleanup check failed: {e}")
        
        # 2. CREATE NEW EXPERIMENT
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if "\\" in template_path:
            template_dir = template_path.rsplit("\\", 1)[0]
            new_filename = f"{template_dir}\\Run_{timestamp}.OnixExp"
        else:
            new_filename = f"Run_{timestamp}.OnixExp"

        log(f"Creating: {os.path.basename(new_filename)}")
        params = {"filename": new_filename, "templatename": template_path}
        create_resp = self._send_request("CreateExperiment", params)
        if not create_resp.get("success"):
            log(f"Create failed: {create_resp}")
            self.current_experiment = None  # NEW: clear on failure
            return False

        time.sleep(sleep_time)

        # 3. VERIFY AND OPEN
        log("Verifying active experiment...")
        check_open = self._send_request("IsExperimentOpen")
        
        is_open = check_open.get("experimentOpen", False)
        open_file = check_open.get("experimentFile", "").replace("\\\\", "\\").lower()
        target_file = new_filename.replace("\\\\", "\\").lower()
        
        if not is_open or (is_open and target_file not in open_file):
            if is_open:
                log("Mismatch! Closing wrong file...")
                self._send_request("CloseExperiment", {"save": "false"})
                time.sleep(sleep_time)

            log("Opening experiment...")
            open_resp = self._send_request("OpenExperiment", {"filename": new_filename})
            if not open_resp.get("success"):
                log(f"Open failed: {open_resp}")
                self.current_experiment = None  # NEW: clear on failure
                return False
            time.sleep(sleep_time)
        else:
            log("Correct file is already open.")

        # 4. PRE-RUN SAFETY CHECKS
        log("Performing pre-run safety checks...")
        for attempt in range(5):
            check = self._send_request("IsExperimentOpen")
            has_data = check.get("containsRunData", True)
            if not has_data:
                log("File confirmed clean (No Run Data).")
                break
            else:
                log(f"File still registering run data... waiting {sleep_time}s...")
                time.sleep(sleep_time)
        else:
            log("Error: System insists file has run data. Cannot start.")
            self.current_experiment = None  # NEW: clear on failure
            return False

        # Wait for Hardware Ready
        if not self.wait_for_hardware_idle():
            log("System stuck BUSY. Sending Force Abort to reset...")
            self._send_request("Abort")
            time.sleep(sleep_time)
            if not self.wait_for_hardware_idle():
                log("Error: Hardware refused to go Idle.")
                self.log_telemetry("Hardware_Stuck_Busy")
                self.current_experiment = None  # NEW: clear on failure
                return False
        
        log("Hardware is IDLE. Ready to start.")

        # Clear Errors and Save
        self._send_request("ClearErrors")
        log("Enforcing pre-run save...")
        self._send_request("SaveExperiment")
        time.sleep(sleep_time)

        # 5. START RUN
        log("Starting run...")
        self.log_telemetry("Pre_Start_Attempt")
        start_resp = self._send_request("StartRun")
        
        # Retry Logic
        if not start_resp.get("success"):
            log(f"Start failed: {start_resp}. Retrying in {sleep_time}s...")
            self.log_telemetry("Start_Fail_Retry")
            time.sleep(sleep_time)
            start_resp = self._send_request("StartRun")
            
        if not start_resp.get("success"):
             log("StartRun rejected persistently.")
             self.log_telemetry("Start_Fail_Final")
             
             status = self.get_status()
             log(f"DEBUG RunState: {status.get('RunState')}")
             log(f"DEBUG Flags1: {status.get('Flags1')}")
             self.current_experiment = None  # NEW: clear on failure
             return False
        
        success, state = self._poll_for_state([1])
        if not success:
            log(f"Run started but state did not transition to 1. State: {state}")
            self.current_experiment = None  # NEW: clear on failure
            return False
        
        log(f"Run STARTED. Running for {run_duration} seconds...")
        self.log_telemetry("Run_Started")
        
        # 6. WAIT FOR RUN DURATION (MODIFIED to check for abort signal)
        start_run_time = time.time()
        while time.time() - start_run_time < run_duration:
            time.sleep(sleep_time)
            status = self.get_status()
            if int(status.get("RunState", -999)) not in [1]:
                log("Alert: Run stopped unexpectedly!")
                break
            
            # NEW: Check if we should abort for new experiment
            if self.abort_current:
                log("Aborting current experiment due to new experiment request...")
                self.abort_current = False
                break

        # 7. ABORT
        log("Aborting run...")
        self._send_request("Abort")
        
        success, state = self._poll_for_state([-2, 30, 0])
        if not success:
            log("Error: System did not stop.")
            self.current_experiment = None  # NEW: clear on failure
            return False
            
        log(f"System stopped (State: {state}). Waiting for data flush...")
        time.sleep(sleep_time)

        # 8. CLOSE AND SAVE
        log("Closing and saving...")
        self._send_request("CloseExperiment", {"save": "true"})
        
        log("Experiment complete.")
        self.log_telemetry("Experiment_Complete")
        self.current_experiment = None  # NEW: clear after completion
        return True

NEUTRAL_EXPERIMENT = cfg.get("neutral_experiment", "experiment1")
ACIDIC_PULSE_SEC   = cfg.get("acidic_pulse_sec", 30)

# Live media-status file read by the Napari GUI
_STATUS_FILE = os.path.join(final_dir, "media_status.json")


def _write_media_status(media, experiment=None, start_time=None, duration=None):
    """Atomically write the current media state for the Napari indicator."""
    tmp = _STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        _json.dump({
            "media": media,
            "experiment": experiment,
            "start_time": start_time,
            "duration": duration,
        }, f)
    os.rename(tmp, _STATUS_FILE)

def process_actions_toml_direct(experiment_name, onix_controller):
    """Execute an experiment directly by name.
    - Neutral experiments run for run_duration (effectively indefinitely until interrupted).
    - Acidic experiments run for acidic_pulse_sec, then auto-resume neutral.
    """
    template_path = EXPERIMENT_TEMPLATES[experiment_name]

    if experiment_name == NEUTRAL_EXPERIMENT:
        duration = run_duration
        log(f"Starting NEUTRAL experiment: {experiment_name} (runs until interrupted)")
        _write_media_status("neutral", experiment_name, time.time(), duration)
    else:
        duration = ACIDIC_PULSE_SEC
        log(f"Starting ACIDIC pulse: {experiment_name} ({duration}s)")
        _write_media_status("acidic", experiment_name, time.time(), duration)

    log(f"Template path: {template_path}")
    success = onix_controller.run_experiment(template_path, experiment_name, run_duration=duration)

    if success:
        log(f"Completed experiment: {experiment_name}")
    else:
        log(f"Failed experiment: {experiment_name}")

    # After acidic pulse completes, auto-resume neutral
    if experiment_name != NEUTRAL_EXPERIMENT:
        log(f"Acidic pulse done — auto-resuming neutral ({NEUTRAL_EXPERIMENT})")
        actions_path = os.path.join(final_dir, "actions.toml")
        lock_path = os.path.join(final_dir, "auto_neutral.lock")
        with open(lock_path, "w") as f:
            tomlkit.dump({"experiment": NEUTRAL_EXPERIMENT}, f)
        os.rename(lock_path, actions_path)

    return success

def watch_for_actions():
    """
    Continuously watch for actions.toml in final_dir.
    Experiments run in a background thread so this loop stays responsive.
    - Same experiment already running → delete actions.toml and ignore.
    - Different experiment requested  → set abort_current flag; the running
      thread picks it up, stops early, then this loop starts the new one.
    - No experiment running           → start a new daemon thread.
    """
    log(f"Starting TOML watcher on directory: {final_dir}")

    onix = OnixController(host_ip=ONIX_SERVER_IP, port=ONIX_SERVER_PORT)
    actions_path = os.path.join(final_dir, "actions.toml")
    experiment_thread = None

    while True:
        try:
            if os.path.exists(actions_path):
                try:
                    with open(actions_path, 'r') as f:
                        data = tomlkit.load(f)
                    requested_experiment = data.get("experiment", "unknown")
                except Exception as e:
                    log(f"Error reading {actions_path}: {e}")
                    time.sleep(sleep_time)
                    continue

                thread_running = experiment_thread is not None and experiment_thread.is_alive()

                if thread_running:
                    if requested_experiment == onix.current_experiment:
                        log(f"Experiment '{requested_experiment}' is already running. Ignoring request.")
                        os.remove(actions_path)
                    else:
                        log(f"New experiment '{requested_experiment}' requested "
                            f"(current: '{onix.current_experiment}'). Aborting current...")
                        onix.abort_current = True
                        # Leave actions.toml in place; next iteration will start the new experiment
                        # once the thread finishes.
                else:
                    if requested_experiment not in EXPERIMENT_TEMPLATES:
                        log(f"Error: Unknown experiment '{requested_experiment}'")
                        log(f"Available experiments: {list(EXPERIMENT_TEMPLATES.keys())}")
                        os.remove(actions_path)
                    else:
                        os.remove(actions_path)
                        experiment_thread = threading.Thread(
                            target=process_actions_toml_direct,
                            args=(requested_experiment, onix),
                            daemon=True,
                        )
                        experiment_thread.start()

            else:
                # No actions.toml and no running thread → idle
                thread_running = experiment_thread is not None and experiment_thread.is_alive()
                if not thread_running:
                    try:
                        with open(_STATUS_FILE) as _sf:
                            _cur = _json.load(_sf).get("media")
                    except Exception:
                        _cur = None
                    if _cur != "idle":
                        _write_media_status("idle")

            time.sleep(sleep_time)

        except KeyboardInterrupt:
            log("Watcher stopped by user")
            break
        except Exception as e:
            log(f"Unexpected error in watcher loop: {e}")
            time.sleep(sleep_time)

if __name__ == "__main__":
    watch_for_actions()