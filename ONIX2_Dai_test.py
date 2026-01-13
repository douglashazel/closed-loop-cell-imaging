import requests
import time
import sys
import os
from datetime import datetime

import requests
import time
import sys
import os
from datetime import datetime

class OnixTester:
    """
    Wrapper for CellASIC ONIX2 Web Services API.
    Includes CSV Logging for Hardware Flags.
    """
    
    def __init__(self, host_ip, port):
        self.base_url = f"http://{host_ip}:{port}/onixserver"
        self.session = requests.Session()
        print(f"[Init] Connecting to ONIX2 Server at: {self.base_url}")
        
        # Initialize the CSV Log
        self.init_logging()

    def init_logging(self):
        """Creates a CSV file to track hardware flags over time."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filename = f"ONIX_Hardware_Log_{timestamp}.csv"
        
        # [cite_start]Write CSV Header [cite: 90-176]
        header = "PC_Time,Context,RunState,Flags0(Sys),Flags1(Gas/Leak),Flags2(Env),PressureX,PressureY,Temp_C\n"
        
        with open(self.log_filename, "w") as f:
            f.write(header)
        print(f"[Log] Telemetry will be saved to: {self.log_filename}")

    def log_telemetry(self, context_label, status=None):
        """
        Appends a row to the CSV log with current hardware state.
        If status is None, it fetches a fresh one.
        """
        if not status:
            try:
                status = self._send_request("Status")
            except:
                return # Don't crash logging if network fails

        # [cite_start]Extract relevant fields [cite: 123-176]
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
            print(f"[Log Error] Could not write to file: {e}")

    def _send_request(self, command, params=None):
        url = f"{self.base_url}/{command}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"[Error] Network error on command '{command}': {e}")
            raise

    def get_status(self):
        return self._send_request("Status")

    def _poll_for_state(self, target_states, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status()
            
            # Log the state while we wait (helps debug transitions)
            self.log_telemetry("Polling_Wait", status)
            
            try:
                current_state = int(status.get("RunState", -999))
            except (ValueError, TypeError):
                current_state = -999
            
            if current_state in target_states:
                return True, current_state
            time.sleep(0.5)
        return False, current_state

    # --- TELEMETRY & SAFETY ---

    def report_instrument_telemetry(self):
        print("\n--- Instrument Telemetry Report ---")
        try:
            status = self.get_status()
            self.log_telemetry("Initial_Report", status) # Log it
            
            flags0 = status.get("Flags0", "0"*16)
            is_sealed = flags0[::-1][2] == '1'
            print(f"  > Manifold Sealed: {'YES' if is_sealed else 'NO'}")
            print(f"  > Flags0 (Debug):  {flags0}")

            state_map = {0: "IDLE", 1: "RUNNING", 2: "PAUSED", -2: "ABORTED", 30: "COMPLETE"}
            try:
                run_state = int(status.get("RunState", -999))
            except:
                run_state = -999
            print(f"  > System State:    {state_map.get(run_state, str(run_state))}")
            return True
        except Exception as e:
            print(f"[Error] Telemetry failed: {e}")
            return False

    def check_critical_errors(self):
        print("\n--- Checking Hardware Health ---")
        status = self.get_status()
        self.log_telemetry("Health_Check", status) # Log it
        
        flags0 = status.get("Flags0", "0"*16)
        flags1 = status.get("Flags1", "0"*16)
        errors = {
            "Pump Error": flags0[::-1][15] == '1', # [cite: 142]
            "Leak X": flags1[::-1][11] == '1',     # [cite: 155]
            "Leak Y": flags1[::-1][15] == '1'      # [cite: 157]
        }
        if any(errors.values()):
            print(f"[CRITICAL FAIL] Hardware Errors: {errors}")
            return False
        print("[Pass] No critical hardware errors detected.")
        return True

    def is_sealed(self):
        status = self.get_status()
        flags0 = status.get("Flags0", "0"*16)
        return flags0[::-1][2] == '1' # [cite: 136]

    def seal_manifold(self, timeout=20):
        print("  > Attempting to SEAL manifold...")
        self._send_request("Seal") 
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_sealed():
                print("[Pass] Manifold seal confirmed.")
                return True
            time.sleep(1)
        return False

    def wait_for_hardware_idle(self, timeout=15):
        """
        Polls Flags0 Bit 0 (System Ready).
        Bit 0 = 1 means READY. Bit 0 = 0 means BUSY.
        """
        print("   > Checking Hardware Ready Flag (Flags0, Bit 0)...")
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status()
            
            # LOG THIS: We want to see this bit flip in the CSV
            self.log_telemetry("Wait_For_Idle", status)
            
            flags0 = status.get("Flags0", "0"*16)
            
            # Bit 0 is the last character in the string. [cite_start]1 = Ready. [cite: 132]
            is_ready = flags0[-1] == '1'
            
            if is_ready:
                run_state = int(status.get("RunState", -999))
                if run_state == 0:
                    return True
            
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(1)
            
        print("\n[Warning] Hardware Ready Flag did not set (System Busy).")
        return False

    # --- LIFECYCLE LOGIC ---

    def run_lifecycle_sequence(self, template_path):
        print(f"\n>>> Starting Test using Template: {template_path}")
        self.log_telemetry(f"Start_Cycle_{os.path.basename(template_path)}")

        # 0. Ensure Clean Slate
        try:
            check_resp = self._send_request("IsExperimentOpen")
            if check_resp.get("experimentOpen", False):
                print("   > An experiment is currently open. Closing it now...")
                self._send_request("CloseExperiment", {"save": "false"})
                print("   > Experiment closed. Waiting 3s...")
                time.sleep(1) 
        except Exception as e:
            print(f"[Warning] Could not verify/close existing experiment: {e}")
        
        # 1. GENERATE NEW FILENAME
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if "\\" in template_path:
            template_dir = template_path.rsplit("\\", 1)[0]
            new_filename = f"{template_dir}\\AutoTest_{timestamp}.OnixExp"
        else:
            new_filename = f"AutoTest_{timestamp}.OnixExp"

        print(f"1. Creating New Experiment...")
        params = {"filename": new_filename, "templatename": template_path}
        create_resp = self._send_request("CreateExperiment", params)
        if not create_resp.get("success"):
            raise ValueError(f"Create failed: {create_resp}")
        print(f"   > Create Response: {create_resp}")

        print("   > Waiting 5s for file creation...")
        time.sleep(1)

        # 1.5. VERIFY AND OPEN
        print(f"1.5. Verifying Active Experiment...")
        check_open = self._send_request("IsExperimentOpen")
        
        is_open = check_open.get("experimentOpen", False)
        open_file = check_open.get("experimentFile", "").replace("\\\\", "\\").lower()
        target_file = new_filename.replace("\\\\", "\\").lower()
        
        if not is_open or (is_open and target_file not in open_file):
            if is_open:
                print(f"   > Mismatch! Closing wrong file...")
                self._send_request("CloseExperiment", {"save": "false"})
                time.sleep(1)
            
            print(f"   > Opening Correct File: {os.path.basename(new_filename)}...")
            open_resp = self._send_request("OpenExperiment", {"filename": new_filename})
            if not open_resp.get("success"):
                raise ValueError(f"Open failed: {open_resp}")
            print(f"   > Open Response: {open_resp}")
            time.sleep(1)
        else:
            print("   > Correct file is already open.")

        print("1.6. Performing Pre-Run Safety Checks...")
        for attempt in range(5):
            check = self._send_request("IsExperimentOpen")
            has_data = check.get("containsRunData", True) 
            if not has_data:
                print("   > [Pass] File confirmed clean (No Run Data).")
                break
            else:
                print("   > [Wait] File still registering run data... waiting 2s...")
                time.sleep(1)
        else:
            print("[Error] Timeout: System insists file has run data. Cannot start.")
            return False

        # === CHECK HARDWARE IDLE ===
        if not self.wait_for_hardware_idle():
            print("\n   > System stuck BUSY. Sending Force Abort to reset...")
            self._send_request("Abort")
            time.sleep(1)
            if not self.wait_for_hardware_idle():
                print("[Error] Hardware refused to go Idle.")
                return False
        
        print("\n   > [Pass] Hardware is IDLE (Bit 0 cleared). Ready to start.")

        # === FIX 1: CLEAR ERROR FLAGS ===
        self._send_request("ClearErrors")

        # === FIX 2: EXPLICIT SAVE ===
        print("1.7. Enforcing Pre-Run Save...")
        self._send_request("SaveExperiment")
        time.sleep(1)

        # 2. Start Run
        print("2. Starting Run...")
        self.log_telemetry("Pre_Start_Attempt") # Log strictly before command
        
        start_resp = self._send_request("StartRun")
        
        # RETRY LOGIC (Attempt 2)
        if not start_resp.get("success"):
            print(f"   > Start failed: {start_resp}. Retrying in 3s...")
            self.log_telemetry("Start_Fail_1") # Log the failure state
            time.sleep(1)
            start_resp = self._send_request("StartRun")
            
        print(f"   > Start Response: {start_resp}")
        
        # FINAL FAILURE CHECK & DEBUG LOGGING
        if not start_resp.get("success"):
             print(f"   > StartRun Failed persistently.")
             
             # Log final failed state to CSV
             self.log_telemetry("Start_Fail_Final")
             
             status = self.get_status()
             print(f"   > [DEBUG] RunState: {status.get('RunState')}")
             print(f"   > [DEBUG] Flags0 (System): {status.get('Flags0')} (Bit0=Ready, Bit2=Sealed)")
             print(f"   > [DEBUG] Flags1 (Gas/Leak): {status.get('Flags1')} (Check Bit0/1 for Gas, Bit11/15 for Leak)")
             print(f"   > [DEBUG] Flags2 (Env):    {status.get('Flags2')}")
             return False
        
        self.log_telemetry("Start_Success")
        
        success, state = self._poll_for_state([1])
        if not success:
            print(f"[Fail] Could not start run. State: {state}")
            return False
        print("[Pass] System is RUNNING.")
        time.sleep(1) 

        # 3. Pause
        print("3. Pausing...")
        pause_resp = self._send_request("Pause", {"keepflow": "true"})
        success, state = self._poll_for_state([2, 11, 12])
        if not success: return False
        print("[Pass] System is PAUSED.")
        time.sleep(1)

        # 4. Resume
        print("4. Resuming...")
        resume_resp = self._send_request("Resume")
        success, state = self._poll_for_state([1])
        if not success: return False
        print("[Pass] System is RUNNING again.")
        time.sleep(1)

        # 5. Abort
        print("5. Aborting Run...")
        abort_resp = self._send_request("Abort")
        success, state = self._poll_for_state([-2, 30, 0])
        if not success: return False
        print(f"[Pass] System Stopped (State: {state}).")
        
        print("   > Waiting 5s for data to write to disk...")
        time.sleep(1)
        
        # 6. Close AND Save
        print(f"6. Saving and Closing...")
        save_params = {"save": "true"}
        save_resp = self._send_request("CloseExperiment", save_params)
        print(f"   > Close Response: {save_resp}")
        print("[Pass] Cycle Complete.")
        
        self.log_telemetry("Cycle_Complete")
        return True
    

    def run_short_test_cycle(self, template_path):
        """
        Executes the full lifecycle test:
        Create -> Open -> Start -> Pause -> Resume -> Abort -> Close(Save)
        Includes strict safety checks and CSV telemetry logging.
        """
        print(f"\n>>> Starting Full Lifecycle Sequence: {template_path}")
        self.log_telemetry(f"Start_Cycle_Full_{os.path.basename(template_path)}")
        
        # 1. ENSURE CLEAN SLATE
        try:
            check_resp = self._send_request("IsExperimentOpen")
            if check_resp.get("experimentOpen", False):
                print("   > Found open experiment. Closing...")
                self._send_request("CloseExperiment", {"save": "false"})
                time.sleep(3) 
        except Exception as e:
            print(f"[Warning] Cleanup check failed: {e}")
        
        # 2. CREATE NEW EXPERIMENT
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if "\\" in template_path:
            template_dir = template_path.rsplit("\\", 1)[0]
            new_filename = f"{template_dir}\\Lifecycle_{timestamp}.OnixExp"
        else:
            new_filename = f"Lifecycle_{timestamp}.OnixExp"

        print(f"   > Creating: {os.path.basename(new_filename)}")
        params = {"filename": new_filename, "templatename": template_path}
        create_resp = self._send_request("CreateExperiment", params)
        if not create_resp.get("success"):
            print(f"[Fail] Create failed: {create_resp}")
            return False

        # Wait for file system I/O
        time.sleep(1)

        # 3. VERIFY AND OPEN
        print(f"1.5. Verifying Active Experiment...")
        check_open = self._send_request("IsExperimentOpen")
        
        is_open = check_open.get("experimentOpen", False)
        open_file = check_open.get("experimentFile", "").replace("\\\\", "\\").lower()
        target_file = new_filename.replace("\\\\", "\\").lower()
        
        if not is_open or (is_open and target_file not in open_file):
            if is_open:
                print(f"   > Mismatch! Closing wrong file...")
                self._send_request("CloseExperiment", {"save": "false"})
                time.sleep(1)

            print("   > Opening experiment...")
            open_resp = self._send_request("OpenExperiment", {"filename": new_filename})
            if not open_resp.get("success"):
                print(f"[Fail] Open failed: {open_resp}")
                return False
            time.sleep(1)
        else:
            print("   > Correct file is already open.")


        # 4. PRE-RUN SAFETY CHECKS (CRITICAL FIXES)
        print("1.6. Performing Pre-Run Safety Checks...")
        for attempt in range(5):
            check = self._send_request("IsExperimentOpen")
            has_data = check.get("containsRunData", True) 
            if not has_data:
                print("   > [Pass] File confirmed clean (No Run Data).")
                break
            else:
                print("   > [Wait] File still registering run data... waiting 2s...")
                time.sleep(1)
        else:
            print("[Error] Timeout: System insists file has run data. Cannot start.")
            return False

        # A. Wait for Hardware Ready (Flags0 Bit 0 == 1)
        if not self.wait_for_hardware_idle():
            print("\n   > System stuck BUSY. Sending Force Abort to reset...")
            self._send_request("Abort")
            time.sleep(1)
            if not self.wait_for_hardware_idle():
                print("[Error] Hardware refused to go Idle.")
                self.log_telemetry("Hardware_Stuck_Busy")
                return False
        
        print("\n   > [Pass] Hardware is IDLE. Ready to start.")

        # [cite_start]B. Clear Errors [cite: 75]
        self._send_request("ClearErrors")

        # [cite_start]C. Explicit Save [cite: 75]
        # API requires experiment to be saved before StartRun
        print("   > Enforcing Pre-Run Save...")
        self._send_request("SaveExperiment")
        time.sleep(1)

        # 5. START RUN
        print("5. Starting Run...")
        self.log_telemetry("Pre_Start_Attempt")
        start_resp = self._send_request("StartRun")
        
        # Retry Logic
        if not start_resp.get("success"):
            print(f"   > Start failed: {start_resp}. Retrying in 3s...")
            self.log_telemetry("Start_Fail_Retry")
            time.sleep(1)
            start_resp = self._send_request("StartRun")
            
        if not start_resp.get("success"):
             print(f"   > [Fail] StartRun rejected persistently.")
             self.log_telemetry("Start_Fail_Final")
             
             # Debug Output
             status = self.get_status()
             print(f"   > [DEBUG] RunState: {status.get('RunState')}")
             print(f"   > [DEBUG] Flags1 (Gas/Leak): {status.get('Flags1')}")
             return False
        
        success, state = self._poll_for_state([1])
        if not success:
            print(f"[Fail] Run started but state did not transition to 1. State: {state}")
            return False
        
        print("   > Run STARTED. Waiting 30 seconds...")
        self.log_telemetry("Run_Started")
        
        # 5. WAIT 30 SECONDS
        # We poll periodically just to log telemetry, but we wait for the full duration
        for _ in range(6): # 6 * 5s = 30s
            time.sleep(5)
            # Optional: check if run crashed/aborted early
            status = self.get_status()
            if int(status.get("RunState", -999)) not in [1]: # 1 = Running [cite: 176]
                print("   > [Alert] Run stopped unexpectedly!")
                break

        # 6. ABORT
        print("   > Aborting Run...")
        self._send_request("Abort")
        
        # Wait for system to confirm it stopped (RunState 0, -2, or 30)
        success, state = self._poll_for_state([-2, 30, 0])
        if not success:
            print("[Error] System did not stop.")
            return False
            
        print(f"   > System Stopped (State: {state}). Waiting for data flush...")
        time.sleep(5)

        # 7. CLOSE AND SAVE
        print("   > Closing and Saving...")
        # 'save=true' ensures the 30s of data is written to disk [cite: 80]
        self._send_request("CloseExperiment", {"save": "true"})
        
        print("[Pass] Short cycle complete.")
        self.log_telemetry("Cycle_Complete")
        return True


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    
    # ==========================================
    # CONFIGURATION - UPDATE THESE VALUES
    # ==========================================
    
    ONIX_SERVER_IP = "192.0.2.10"   # <--- Server IP Address
    ONIX_SERVER_PORT = 8881            # <--- Server Port Number (Default is 8888)
    
    # Absolute paths on the ONIX Computer (Server)
    TEMPLATE_PATH_1 = r"C:\ONIX2\Experiments\AutoTest_Run_1.OnixExp"
    TEMPLATE_PATH_2 = r"C:\ONIX2\Experiments\AutoTest_Run_2.OnixExp"
    
    # ==========================================
    
    tester = OnixTester(host_ip=ONIX_SERVER_IP, port=ONIX_SERVER_PORT)

    if not tester.report_instrument_telemetry(): sys.exit(1)
    if not tester.check_critical_errors(): sys.exit(1)

    if not tester.is_sealed():
        print("\n[!] Manifold Not Sealed. Attempting Auto-Seal...")
        if not tester.seal_manifold(): sys.exit(1)
    else:
        print("\n[Pass] Manifold is sealed.")

    
    if tester.run_short_test_cycle(TEMPLATE_PATH_1):
        print("=== SHORT TEST CYCLE 1 SUCCESSFUL ===")
    else:
        print("=== SHORT TEST CYCLE 1 FAILED ===")
        sys.exit(1)
    
    if tester.run_short_test_cycle(TEMPLATE_PATH_2):
        print("=== SHORT TEST CYCLE 1 SUCCESSFUL ===")
    else:
        print("=== SHORT TEST CYCLE 1 FAILED ===")
        sys.exit(1)

    if tester.run_short_test_cycle(TEMPLATE_PATH_1):
        print("=== SHORT TEST CYCLE 1 SUCCESSFUL ===")
    else:
        print("=== SHORT TEST CYCLE 1 FAILED ===")
        sys.exit(1)


    # for counter in range(5):    
    #   print(f"\n========== Round {counter + 1} =====================")
    #   # Run Test 1
    #   if tester.run_lifecycle_sequence(TEMPLATE_PATH_1):
    #      print("=== TEST 1 SUCCESSFUL ===")
    #   else:
    #      print("=== TEST 1 FAILED ===")
    #      sys.exit(1) 

    #   print("\n[Cooldown] Waiting 10 seconds before starting Test 2...")
    #   time.sleep(1)

    #   # Run Test 2
    #   if tester.run_lifecycle_sequence(TEMPLATE_PATH_2):
    #      print("=== TEST 2 SUCCESSFUL ===")
    #   else:
    #      print("=== TEST 2 FAILED ===")
      
    #   time.sleep(1)
    #   print("\n[Cooldown] Waiting 1 seconds before next iteration...")

    print("\nDONE.")



# start ex 1, load and start wait 30s, abort, close and save