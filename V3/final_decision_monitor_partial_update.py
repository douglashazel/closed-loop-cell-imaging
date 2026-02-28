import os
import time
import json
import tomlkit
import requests
from datetime import datetime

# Load configuration
with open("config.json", "r") as f:
    cfg = json.load(f)

global_dir = cfg["global_path"]
final_dir = cfg["final_dir"]
sleep_time = cfg.get("sleep_time", 0.5)
run_duration = cfg.get("run_duration_sec", 300)

# ONIX Configuration
ONIX_SERVER_IP = cfg.get("onix_server_ip", "192.0.2.10")
ONIX_SERVER_PORT = cfg.get("onix_server_port", 8881)

# Experiment name to template path mapping
EXPERIMENT_TEMPLATES = cfg.get("experiment_templates", {})

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

if not EXPERIMENT_TEMPLATES:
    log("WARNING: No experiment templates found in config.json!")
    log("Please ensure 'experiment_templates' is properly configured.")
class OnixController:
    """Minimal ONIX2 controller for executing experiments."""
    
    def __init__(self, host_ip, port):
        self.base_url = f"http://{host_ip}:{port}/onixserver"
        self.session = requests.Session()
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
            
            time.sleep(sleep_time) # originally 1s
            
        log("Warning: Hardware Ready Flag did not set (System Busy).")
        return False

    def run_experiment(self, template_path, run_duration=run_duration):
        """
        Execute a single ONIX experiment:
        Create -> Open -> Start -> Wait -> Abort -> Close(Save)
        """
        log(f"Starting experiment: {template_path}")
        self.log_telemetry(f"Start_{os.path.basename(template_path)}")
        
        # 1. ENSURE CLEAN SLATE
        try:
            check_resp = self._send_request("IsExperimentOpen")
            if check_resp.get("experimentOpen", False):
                log("Found open experiment. Closing...")
                self._send_request("CloseExperiment", {"save": "false"})
                time.sleep(sleep_time) # originally 3s
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
            return False

        time.sleep(sleep_time) # originally 5s

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
                time.sleep(sleep_time) # originally 3s

            log("Opening experiment...")
            open_resp = self._send_request("OpenExperiment", {"filename": new_filename})
            if not open_resp.get("success"):
                log(f"Open failed: {open_resp}")
                return False
            time.sleep(sleep_time) # originally 3s
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
                log("File still registering run data... waiting 2s...")
                time.sleep(sleep_time) # originally 2s
        else:
            log("Error: System insists file has run data. Cannot start.")
            return False

        # Wait for Hardware Ready
        if not self.wait_for_hardware_idle():
            log("System stuck BUSY. Sending Force Abort to reset...")
            self._send_request("Abort")
            time.sleep(sleep_time) # originally 3s
            if not self.wait_for_hardware_idle():
                log("Error: Hardware refused to go Idle.")
                self.log_telemetry("Hardware_Stuck_Busy")
                return False
        
        log("Hardware is IDLE. Ready to start.")

        # Clear Errors and Save
        self._send_request("ClearErrors")
        log("Enforcing pre-run save...")
        self._send_request("SaveExperiment")
        time.sleep(sleep_time) # originally 3s

        # 5. START RUN
        log("Starting run...")
        self.log_telemetry("Pre_Start_Attempt")
        start_resp = self._send_request("StartRun")
        
        # Retry Logic
        if not start_resp.get("success"):
            log(f"Start failed: {start_resp}. Retrying in 3s...")
            self.log_telemetry("Start_Fail_Retry")
            time.sleep(sleep_time) # originally 3s
            start_resp = self._send_request("StartRun")
            
        if not start_resp.get("success"):
             log("StartRun rejected persistently.")
             self.log_telemetry("Start_Fail_Final")
             
             status = self.get_status()
             log(f"DEBUG RunState: {status.get('RunState')}")
             log(f"DEBUG Flags1: {status.get('Flags1')}")
             return False
        
        success, state = self._poll_for_state([1])
        if not success:
            log(f"Run started but state did not transition to 1. State: {state}")
            return False
        
        log(f"Run STARTED. Waiting {run_duration} seconds...")
        self.log_telemetry("Run_Started")
        
        # 6. WAIT FOR RUN DURATION
        for _ in range(run_duration // 5):
            time.sleep(sleep_time)  # originally 5s
            status = self.get_status()
            if int(status.get("RunState", -999)) not in [1]:
                log("Alert: Run stopped unexpectedly!")
                break

        # 7. ABORT
        log("Aborting run...")
        self._send_request("Abort")
        
        success, state = self._poll_for_state([-2, 30, 0])
        if not success:
            log("Error: System did not stop.")
            return False
            
        log(f"System stopped (State: {state}). Waiting for data flush...")
        time.sleep(sleep_time) # originally 5s

        # 8. CLOSE AND SAVE
        log("Closing and saving...")
        self._send_request("CloseExperiment", {"save": "true"})
        
        log("Experiment complete.")
        self.log_telemetry("Experiment_Complete")
        return True


def process_actions_toml(toml_path, onix_controller):
    """
    Read the actions.toml file, extract the experiment name,
    and execute it on the ONIX device.
    """
    try:
        with open(toml_path, 'r') as f:
            data = tomlkit.load(f)
        
        experiment_name = data.get("experiment", "unknown")
        log(f"Received action: {experiment_name}")
        
        # Get the template path for this experiment
        if experiment_name not in EXPERIMENT_TEMPLATES:
            log(f"Error: Unknown experiment '{experiment_name}'")
            log(f"Available experiments: {list(EXPERIMENT_TEMPLATES.keys())}")
            return False
        
        template_path = EXPERIMENT_TEMPLATES[experiment_name]
        log(f"Executing experiment: {experiment_name}")
        log(f"Template path: {template_path}")
        
        # Execute the experiment on ONIX
        success = onix_controller.run_experiment(template_path, run_duration=run_duration)
        
        if success:
            log(f"Successfully completed experiment: {experiment_name}")
        else:
            log(f"Failed to complete experiment: {experiment_name}")
        
        return success
        
    except Exception as e:
        log(f"Error processing {toml_path}: {e}")
        return False

def watch_for_actions():
    """
    Continuously watch for actions.toml in final_dir.
    When found, read it, execute the experiment on ONIX, and delete it.
    Retry failed experiments up to num_tries times before discarding.
    """
    log(f"Starting TOML watcher on directory: {final_dir}")
    
    # Initialize ONIX controller
    onix = OnixController(host_ip=ONIX_SERVER_IP, port=ONIX_SERVER_PORT)
    
    actions_path = os.path.join(final_dir, "actions.toml")
    
    current_file_retries = 0
    max_retries = cfg.get("num_tries", 5)
    
    while True:
        try:
            # Check if actions.toml exists
            if os.path.exists(actions_path):
                log(f"Found {actions_path}")
                
                # Process the TOML file and execute on ONIX
                if process_actions_toml(actions_path, onix):
                    # Delete the file after successful processing
                    os.remove(actions_path)
                    log(f"Deleted {actions_path}")
                    current_file_retries = 0  # Reset for next file
                else:
                    current_file_retries += 1
                    log(f"Failed to process {actions_path}, attempt {current_file_retries}/{max_retries}")
                    
                    if current_file_retries >= max_retries:
                        log(f"Max retries reached. Deleting problematic {actions_path}")
                        os.remove(actions_path)
                        current_file_retries = 0  # Reset for next file
            else:
                # No file present, reset retry counter
                current_file_retries = 0
            
            # Sleep before checking again
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            log("Watcher stopped by user")
            break
        except Exception as e:
            log(f"Unexpected error in watcher loop: {e}")
            time.sleep(sleep_time)

if __name__ == "__main__":
    watch_for_actions()