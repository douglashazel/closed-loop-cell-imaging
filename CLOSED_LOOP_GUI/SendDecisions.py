import os
import json as _json
import time
import signal
import threading
import tomlkit
import requests
from datetime import datetime

from io_utils import log, load_config

cfg = load_config()
global_dir   = cfg["global_path"]
final_dir    = cfg["final_dir"]
sleep_time   = cfg.get("sleep_time", 0.5)
run_duration = cfg.get("run_duration_sec", 86400)
DRY_RUN      = cfg.get("dry_run", False)

# ONIX Configuration
ONIX_SERVER_IP   = cfg.get("onix_server_ip", "192.0.2.10")
ONIX_SERVER_PORT = cfg.get("onix_server_port", 8881)

# Experiment name to template path mapping
EXPERIMENT_TEMPLATES = cfg.get("experiment_templates", {})

# Delay before restarting an experiment thread that ended: 5 s, doubled for
# each consecutive failure up to 5 min, back to 5 s after a successful run
RESTART_BACKOFF_MIN_SEC = 5.0
RESTART_BACKOFF_MAX_SEC = 300.0

# Set by SIGTERM/SIGINT: the watcher aborts the ONIX run, then exits
stop_event = threading.Event()

if not EXPERIMENT_TEMPLATES:
    log("WARNING: No experiment templates found in config.json!")
    log("Please ensure 'experiment_templates' is properly configured.")
class OnixController:
    """Minimal ONIX2 controller for executing experiments."""
    
    def __init__(self, host_ip, port):
        self.base_url = f"http://{host_ip}:{port}/onixserver"
        self.session = requests.Session()
        self._abort_event = threading.Event()
        self.current_experiment = None  # track current experiment name
        # (experiment_name, filename) created by an attempt that failed before
        # StartRun; the next attempt at the same state reopens it
        self._pending_file = None
        self._check_connectivity()
        self.init_logging()

    def _check_connectivity(self):
        """Quick health check — log a clear warning if the ONIX server is unreachable."""
        log(f"Connecting to ONIX2 Server at: {self.base_url}")
        try:
            status = self.session.get(f"{self.base_url}/Status", timeout=5).json()
            log(f"ONIX server reachable (RunState={status.get('RunState', '?')})")
        except Exception as e:
            log(f"WARNING: ONIX server unreachable at {self.base_url} — {e}")

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

    def _poll_for_state(self, target_states, timeout=30, abortable=True):
        """Poll RunState until it is in target_states. Returns (reached, state).

        An abortable poll gives up as soon as an abort is requested. Polls that
        confirm an Abort already sent, or a run already started, pass
        abortable=False so they run to completion."""
        start_time = time.time()
        current_state = -999
        while time.time() - start_time < timeout:
            if abortable and self._abort_event.is_set():
                return False, current_state
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
            if self._abort_event.is_set():
                return False
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

    def ensure_stopped(self):
        """Shutdown check: if the ONIX still reports a live run (RunState 1),
        Abort it and close the experiment without saving."""
        try:
            status = self.get_status()
            if int(status.get("RunState", -999)) != 1:
                return
            log("ONIX still running at shutdown. Aborting run...")
            self._send_request("Abort")
            self._poll_for_state([-2, 30, 0], timeout=10, abortable=False)
            log("Closing experiment without saving...")
            self._send_request("CloseExperiment", {"save": "false"})
        except Exception as e:
            log(f"Warning: shutdown check failed: {e}")

    def _abort_requested(self, step):
        """True (and logged) if an abort arrived before this setup step."""
        if not self._abort_event.is_set():
            return False
        log(f"Abort requested before {step}; skipping the rest of the setup.")
        self.current_experiment = None
        return True

    def run_experiment(self, template_path, experiment_name, run_duration=run_duration):
        """
        Execute a single ONIX experiment:
        Create -> Open -> Start -> Wait -> Abort -> Close(Save)

        An abort (state change or stop signal) is checked before every setup
        step. Once StartRun has been accepted, the run is always Aborted and
        Closed with save before this returns.
        """
        log(f"Starting experiment: {template_path}")
        self.log_telemetry(f"Start_{os.path.basename(template_path)}")
        self.current_experiment = experiment_name
        
        # 1. ENSURE CLEAN SLATE
        #    The server refuses CloseExperiment while RunState == 1 (running),
        #    so we must Abort first. A crashed prior run can leave the server
        #    wedged in this state, and without the Abort every subsequent
        #    CreateExperiment fails.
        if self._abort_requested("cleanup"):
            return False
        try:
            status = self._send_request("Status")
            run_state = int(status.get("RunState", 0))
            if run_state == 1:
                log(f"Prior run active (RunState={run_state}). Aborting...")
                self._send_request("Abort")
                self._poll_for_state([-2, 30, 0], timeout=10)

            check_resp = self._send_request("IsExperimentOpen")
            if check_resp.get("experimentOpen", False):
                log("Found open experiment. Closing...")
                close_resp = self._send_request("CloseExperiment", {"save": "false"})
                if not close_resp.get("success"):
                    log(f"Close refused ({close_resp}). Forcing Abort + retry...")
                    self._send_request("Abort")
                    self._poll_for_state([-2, 30, 0], timeout=10)
                    self._send_request("CloseExperiment", {"save": "false"})
                time.sleep(sleep_time)
        except Exception as e:
            log(f"Warning: Cleanup check failed: {e}")
        
        # 2. CREATE NEW EXPERIMENT
        #    A file that an earlier attempt at this same state created, but
        #    never started, holds no run data: reopen it rather than leave a
        #    new .OnixExp behind on every retry.
        if self._abort_requested("create"):
            return False
        if self._pending_file and self._pending_file[0] == experiment_name:
            new_filename = self._pending_file[1]
            log(f"Reusing: {os.path.basename(new_filename)}")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if "\\" in template_path:
                template_dir = template_path.rsplit("\\", 1)[0]
                new_filename = f"{template_dir}\\Run_{experiment_name}_{timestamp}.OnixExp"
            else:
                new_filename = f"Run_{experiment_name}_{timestamp}.OnixExp"

            log(f"Creating: {os.path.basename(new_filename)}")
            params = {"filename": new_filename, "templatename": template_path}
            create_resp = self._send_request("CreateExperiment", params)
            if not create_resp.get("success"):
                log(f"Create failed: {create_resp}")
                self.current_experiment = None
                return False
            self._pending_file = (experiment_name, new_filename)

            time.sleep(sleep_time)

        # 3. VERIFY AND OPEN
        if self._abort_requested("open"):
            return False
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
                self._pending_file = None  # create a fresh file next time
                self.current_experiment = None
                return False
            time.sleep(sleep_time)
        else:
            log("Correct file is already open.")

        # 4. PRE-RUN SAFETY CHECKS
        if self._abort_requested("pre-run checks"):
            return False
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
            self._pending_file = None  # create a fresh file next time
            self.current_experiment = None
            return False

        # Wait for Hardware Ready (the waits return False early on abort)
        if not self.wait_for_hardware_idle() and not self._abort_event.is_set():
            log("System stuck BUSY. Sending Force Abort to reset...")
            self._send_request("Abort")
            time.sleep(sleep_time)
            if not self.wait_for_hardware_idle() and not self._abort_event.is_set():
                log("Error: Hardware refused to go Idle.")
                self.log_telemetry("Hardware_Stuck_Busy")
                self.current_experiment = None
                return False
        
        if self._abort_requested("start"):
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
             self.current_experiment = None
             return False

        # The file now holds run data; a later attempt needs a new one
        self._pending_file = None
        
        success, state = self._poll_for_state([1], abortable=False)
        if not success:
            log(f"Run started but state did not transition to 1. State: {state}")
            self.current_experiment = None
            return False
        
        log(f"Run STARTED. Running for {run_duration} seconds...")
        self.log_telemetry("Run_Started")
        
        # 6. WAIT FOR RUN DURATION, OR UNTIL AN ABORT IS REQUESTED
        start_run_time = time.time()
        while time.time() - start_run_time < run_duration:
            time.sleep(sleep_time)
            status = self.get_status()
            if int(status.get("RunState", -999)) not in [1]:
                log("Alert: Run stopped unexpectedly!")
                break
            
            if self._abort_event.is_set():
                log("Aborting current experiment (state change or stop request)...")
                break

        # 7. ABORT
        log("Aborting run...")
        self._send_request("Abort")
        
        success, state = self._poll_for_state([-2, 30, 0], abortable=False)
        if not success:
            log("Error: System did not stop.")
            self.current_experiment = None
            return False
            
        log(f"System stopped (State: {state}). Waiting for data flush...")
        time.sleep(sleep_time)

        # 8. CLOSE AND SAVE
        log("Closing and saving...")
        self._send_request("CloseExperiment", {"save": "true"})
        
        log("Experiment complete.")
        self.log_telemetry("Experiment_Complete")
        self.current_experiment = None
        return True


class FakeOnixController(OnixController):
    """Dry-run stand-in for OnixController (config "dry_run": true).

    Answers every ONIX command locally, so the full Create/Open/Start/Wait/
    Abort/Close sequence runs with no network I/O; the inherited wait loop
    then just sleeps until an abort or run_duration."""

    def __init__(self):
        self._run_state = 0
        self._open_file = ""
        super().__init__(host_ip="dry-run", port=0)

    def _check_connectivity(self):
        log("DRY RUN: simulating the ONIX server; no network requests are sent")

    def _send_request(self, command, params=None):
        if command == "StartRun":
            self._run_state = 1
        elif command == "Abort":
            self._run_state = 0
        elif command in ("CreateExperiment", "OpenExperiment"):
            self._open_file = params["filename"]
        elif command == "CloseExperiment":
            self._open_file = ""
        return {
            "success": True,
            "RunState": self._run_state,
            "Flags0": "0" * 15 + "1",  # bit 0 set = system ready
            "experimentOpen": bool(self._open_file),
            "experimentFile": self._open_file,
            "containsRunData": False,
        }

NEUTRAL_EXPERIMENT = cfg.get("neutral_experiment", "NN")
ACIDIC_PULSE_SEC   = cfg.get("acidic_pulse_sec", 30)
num_channels       = cfg.get("num_channels", 2)

# Live media-status file read by the web GUI
_STATUS_FILE = os.path.join(final_dir, "media_status.json")


def _write_media_status(experiment, pulse_manager):
    """Atomically write per-channel pulse state for the web GUI's media indicator."""
    channels_data = {}
    for ch in range(1, pulse_manager.num_channels + 1):
        ps = pulse_manager.pulse_start[ch]
        channels_data[str(ch)] = {
            "state": "acidic" if ps is not None else "neutral",
            "pulse_start": ps,
        }
    tmp = _STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        _json.dump({
            "experiment": experiment,
            "channels": channels_data,
            "pulse_duration": pulse_manager.pulse_duration,
        }, f)
    os.rename(tmp, _STATUS_FILE)


class PulseManager:
    """Track independent per-channel acid pulse timers and compute the
    combined ONIX experiment state (NN / AN / NA / AA)."""

    def __init__(self, num_channels, pulse_duration):
        self.num_channels = num_channels
        self.pulse_duration = pulse_duration
        # None = neutral (idle), float = pulse start time
        self.pulse_start = {ch: None for ch in range(1, num_channels + 1)}

    def handle_crossing(self, ch, label):
        """Process a threshold crossing for a single channel.
        Returns True if a NEW pulse was started (state changed)."""
        if label == "add acidic media":
            if self.pulse_start[ch] is None:
                self.pulse_start[ch] = time.time()
                log(f"Channel {ch}: acid pulse STARTED ({self.pulse_duration}s)")
                return True
            # Already pulsing — ignore
            return False
        # Neutral / basic — no action (pulses expire on timer, not on signal)
        return False

    def check_expirations(self):
        """Expire any pulses that have exceeded their duration.
        Returns True if any pulse expired (state changed)."""
        changed = False
        now = time.time()
        for ch in range(1, self.num_channels + 1):
            if self.pulse_start[ch] is not None:
                elapsed = now - self.pulse_start[ch]
                if elapsed >= self.pulse_duration:
                    log(f"Channel {ch}: acid pulse EXPIRED after {elapsed:.1f}s")
                    self.pulse_start[ch] = None
                    changed = True
        return changed

    def get_experiment_name(self):
        """Map per-channel pulse state to experiment name.
        A = acid (pulsing), N = neutral (idle).
        Ordering: channel1 letter first, channel2 letter second."""
        letters = []
        for ch in range(1, self.num_channels + 1):
            letters.append("A" if self.pulse_start[ch] is not None else "N")
        return "".join(letters)


def run_experiment_blocking(experiment_name, onix_controller, result):
    """Run an ONIX experiment until aborted (state change / stop) or completion.

    Every state runs for run_duration; a state change aborts it sooner.
    result["success"] records the outcome for the watcher's restart backoff."""
    success = False
    try:
        template_path = EXPERIMENT_TEMPLATES[experiment_name]
        kind = "NEUTRAL" if experiment_name == NEUTRAL_EXPERIMENT else "ACIDIC"
        log(f"Starting {kind} experiment: {experiment_name} (runs until state change)")
        success = onix_controller.run_experiment(template_path, experiment_name,
                                                 run_duration=run_duration)
    except Exception as e:
        log(f"Experiment {experiment_name} error: {e}")
    if success:
        log(f"Completed experiment: {experiment_name}")
    elif onix_controller._abort_event.is_set():
        log(f"Experiment {experiment_name} stopped early (abort requested)")
    else:
        log(f"Failed experiment: {experiment_name}")
    result["success"] = success
    return success


def _start_experiment(experiment_name, onix_controller):
    """Start run_experiment_blocking in a non-daemon thread.

    Called only once the previous experiment thread has returned, so clearing
    the abort flag here cannot cancel an abort that thread still needs."""
    onix_controller._abort_event.clear()
    result = {}
    thread = threading.Thread(
        target=run_experiment_blocking,
        args=(experiment_name, onix_controller, result),
    )
    thread.start()
    return thread, result


def watch_for_actions():
    """Main loop: watch for per-channel threshold crossings from
    CreateDecisions and manage independent acid pulse timers.

    The combined pulse state (NN/AN/NA/AA) determines which ONIX
    experiment is active. When the state changes — either because a new
    channel crosses the threshold or because a pulse timer expires — the
    current experiment is aborted and the new one is started.

    SIGTERM / SIGINT end the loop; the running experiment is then Aborted
    and Closed on the ONIX before the process exits.
    """
    log(f"Starting pulse-state watcher on directory: {final_dir}")

    if DRY_RUN:
        onix = FakeOnixController()
    else:
        onix = OnixController(host_ip=ONIX_SERVER_IP, port=ONIX_SERVER_PORT)

    def _request_stop(signum, frame):
        # Only set flags here: logging from a signal handler can re-enter print()
        stop_event.set()
        onix._abort_event.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    actions_path = os.path.join(final_dir, "actions.toml")
    claimed_path = actions_path + ".processing"
    pulses = PulseManager(num_channels, ACIDIC_PULSE_SEC)
    current_experiment = NEUTRAL_EXPERIMENT
    # Backoff so a failing ONIX CreateExperiment doesn't hot-loop the server
    consecutive_failures = 0
    next_restart_at = None  # set once an ended experiment thread is noticed

    # Start with neutral experiment
    _write_media_status(NEUTRAL_EXPERIMENT, pulses)
    experiment_thread, last_result = _start_experiment(NEUTRAL_EXPERIMENT, onix)

    while not stop_event.is_set():
        try:
            state_changed = False

            # 1. Check for new threshold crossings from CreateDecisions.
            #    Claim the file by renaming it first, so a newer actions.toml
            #    written meanwhile is neither read half-written nor deleted unread.
            if os.path.exists(actions_path):
                try:
                    os.replace(actions_path, claimed_path)
                    with open(claimed_path, 'r') as f:
                        data = tomlkit.load(f)
                    os.remove(claimed_path)

                    channels = data.get("channels", {})
                    frame = data.get("frame", "?")
                    for ch_str, label in channels.items():
                        ch = int(ch_str)
                        if pulses.handle_crossing(ch, label):
                            state_changed = True
                    log(f"Frame {frame}: processed crossings {dict(channels)}")

                except Exception as e:
                    log(f"Error reading {actions_path}: {e}")

            # 2. Check for pulse expirations
            if pulses.check_expirations():
                state_changed = True

            # 3. If state changed, switch experiment
            new_experiment = pulses.get_experiment_name()
            if new_experiment != current_experiment:
                state_changed = True

            if state_changed:
                target = pulses.get_experiment_name()
                if target not in EXPERIMENT_TEMPLATES:
                    log(f"Error: Unknown experiment '{target}' — "
                        f"available: {list(EXPERIMENT_TEMPLATES.keys())}")
                elif target != current_experiment:
                    log(f"State change: {current_experiment} -> {target}")
                    current_experiment = target

                    # Abort the current experiment and wait until its thread
                    # has returned, so only one thread ever drives the ONIX
                    if experiment_thread.is_alive():
                        onix._abort_event.set()
                        experiment_thread.join(timeout=30)
                        if experiment_thread.is_alive():
                            log("Warning: previous experiment still shutting down "
                                "after 30 s; waiting for it before starting "
                                f"{current_experiment}...")
                            experiment_thread.join()
                    if stop_event.is_set():
                        break

                    # Update status for the web GUI
                    _write_media_status(current_experiment, pulses)

                    # Start new experiment
                    experiment_thread, last_result = _start_experiment(current_experiment, onix)
                    next_restart_at = None

            # 4. If no experiment running (run_duration elapsed or the thread
            #    failed), restart the current state after a backoff delay so a
            #    persistent ONIX failure doesn't hot-loop.
            if not experiment_thread.is_alive() and not state_changed:
                now = time.time()
                if next_restart_at is None:
                    if last_result.get("success"):
                        consecutive_failures = 0
                        delay = RESTART_BACKOFF_MIN_SEC
                    else:
                        consecutive_failures += 1
                        delay = min(RESTART_BACKOFF_MIN_SEC * 2 ** (consecutive_failures - 1),
                                    RESTART_BACKOFF_MAX_SEC)
                    next_restart_at = now + delay
                    log(f"Experiment thread ended; restarting "
                        f"{pulses.get_experiment_name()} in {delay:.0f} s")
                elif now >= next_restart_at:
                    next_restart_at = None
                    current_experiment = pulses.get_experiment_name()
                    experiment_thread, last_result = _start_experiment(current_experiment, onix)

            time.sleep(sleep_time)

        except Exception as e:
            log(f"Unexpected error in watcher loop: {e}")
            time.sleep(sleep_time)

    # Stop requested: let the experiment thread Abort and Close its run,
    # then make sure the ONIX is not left running.
    log("Stop requested -- aborting the ONIX run before exit...")
    onix._abort_event.set()
    experiment_thread.join()
    onix.ensure_stopped()
    log("SendDecisions stopped.")

if __name__ == "__main__":
    watch_for_actions()