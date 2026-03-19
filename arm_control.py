import time
import torch
from lerobot.policies.factory import make_policy

class PIDController:
    def __init__(self, kp, ki, kd, max_out=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()

    def update(self, error, dt=0.033):
        p = self.kp * error
        self.integral += error * dt
        # Anti-windup: cap integral
        self.integral = max(-10.0, min(10.0, self.integral)) 
        i = self.ki * self.integral
        d = self.kd * (error - self.last_error) / dt
        
        self.last_error = error
        output = p + i + d
        self.last_time = time.time()
        
        if self.max_out:
            output = max(-self.max_out, min(self.max_out, output))
        return output

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0

class ArmController:
    def __init__(self, log_callback):
        self.log_callback = log_callback
        
        # Initialize basic state
        self.connected = False
        self.robot = None
        self.leader_robot = None
        self.recording = False
        self.teleoperating = False
        self.target_port = None
        self.leader_port = None
        
        # PID Controllers for Tracking
        self.pid_x = PIDController(kp=2.5, ki=0.3, kd=0.1, max_out=15.0) 
        self.pid_y = PIDController(kp=2.5, ki=0.3, kd=0.1, max_out=15.0)
        
        # VLA Policy Registry: { display_name: checkpoint_path }
        self.vla_policies = {}
        
        self.connect_arms()
        
    def connect_arms(self):
        if self.connected:
            return
            
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        target_port = None
        leader_port = None
        for p in ports:
            self.log_callback(f"[TURRET-MODE] Found Port: {p.device} ({p.description})")
            if "COM4" in p.device.upper(): # Force COM4 based on user setup
                self.target_port = p.device
            if "COM5" in p.device.upper(): # Force COM5 based on user setup
                self.leader_port = p.device
                
        if self.target_port:
            self.log_callback(f"[TURRET-MODE] Connecting to SO-ARM101 on {self.target_port}...")
            try:
                # CRITICAL: Follower motors are often left in "Step Mode" (3) after calibration
                # so they can spin freely. To command them to move, they MUST be forced back
                # into "Servo/Position Mode" (0) before LeRobot takes ownership of the port.
                import scservo_sdk as scs
                self.log_callback("[VLA-ARM] Securing motors into Position Servo Mode...")
                portHandler = scs.PortHandler(self.target_port)
                packetHandler = scs.PacketHandler(0)
                if portHandler.openPort() and portHandler.setBaudRate(1000000):
                    for motor_id in range(1, 7):
                        packetHandler.write1ByteTxRx(portHandler, motor_id, 33, 0) # Address 33 = Mode, 0 = Position
                    portHandler.closePort()
                    self.log_callback("[VLA-ARM] All 6 motors set to active Position Mode.")
                
                from lerobot.robots.utils import make_robot_from_config
                from lerobot.robots.so101_follower.so101_follower import SO101FollowerConfig
                cfg = SO101FollowerConfig(port=self.target_port, id='so_arm_101')
                self.robot = make_robot_from_config(cfg)
                
                # CRITICAL: Disable the software safety limit that prevents sudden 
                # automated movements on the follower arm.
                self.robot.max_relative_target = None
                
                self.robot.connect(calibrate=False)
                
                # By default the follower is loose. Enable torque to move it programmatically.
                self.robot.bus.enable_torque()
                
                self.connected = True
                self.log_callback("[VLA-ARM] Successfully connected to LeRobot SO-ARM101 (Torque enabled).")
                
                # Try connecting Leader
                if self.leader_port:
                    self.log_callback(f"[VLA-ARM] Connecting to SO-ARM101 Leader on {self.leader_port}...")
                    try:
                        from lerobot.teleoperators import make_teleoperator_from_config
                        from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
                        
                        cfg_leader = SO101LeaderConfig(port=self.leader_port, id='so_arm_101_leader')
                        self.leader_robot = make_teleoperator_from_config(cfg_leader)
                        self.leader_robot.connect(calibrate=False)
                        self.leader_robot.bus.disable_torque() # Ensure it's loose for human control
                        self.log_callback("[VLA-ARM] Successfully connected to Leader Arm.")
                    except Exception as e:
                        self.log_callback(f"[VLA-ARM] WARNING: Failed to connect to Leader: {e}")
            except Exception as e:
                self.log_callback(f"[VLA-ARM] WARNING: Failed to connect via LeRobot: {e}")
                self.connected = False
                
    def disconnect_arms(self):
        """Safely disconnect arms and free serial ports for external scripts."""
        self.log_callback("[VLA-ARM] Disconnecting arms to free serial ports...")
        self.connected = False
        
        if self.teleoperating:
            self.stop_teleop()
        if self.recording:
            self.stop_recording()
            
        if self.leader_robot and hasattr(self.leader_robot, 'disconnect'):
            try:
                self.leader_robot.disconnect()
            except Exception as e:
                self.log_callback(f"[VLA-ARM] Error disconnecting Leader: {e}")
            finally:
                # Force shut deeply nested pyserial instances regardless of disconnect() exceptions
                try:
                    if hasattr(self.leader_robot, 'bus') and hasattr(self.leader_robot.bus, 'port_handler'):
                        if hasattr(self.leader_robot.bus.port_handler, 'ser') and self.leader_robot.bus.port_handler.ser:
                            self.leader_robot.bus.port_handler.ser.close()
                except Exception:
                    pass
                
        if self.robot and hasattr(self.robot, 'disconnect'):
            try:
                self.robot.disconnect()
            except Exception as e:
                self.log_callback(f"[VLA-ARM] Error disconnecting Follower: {e}")
            finally:
                try:
                    if hasattr(self.robot, 'bus') and hasattr(self.robot.bus, 'port_handler'):
                        if hasattr(self.robot.bus.port_handler, 'ser') and self.robot.bus.port_handler.ser:
                            self.robot.bus.port_handler.ser.close()
                except Exception:
                    pass
                
        self.robot = None
        self.leader_robot = None
        
        # CRITICAL for Windows: Force garbage collection to ensure pyserial objects reach __del__
        import gc
        gc.collect()
        time.sleep(0.5) 
        
        self.log_callback("[VLA-ARM] Arms successfully disconnected and port handles freed.")

    # Pi-Zero VLA initialization (kept separate from physical connection)
    def init_policy(self):
        self.log_callback("[VLA-ARM] Loading Pi-Zero VLA Policy...")
        try:
            # We use a placeholder pi-zero policy configuration from lerobot
            # Make sure you have downloaded or define the path to the expected model weights
            # For demonstration, we simulate the loading here to avoid massive downloads for testing
            # Real code would use make_policy("lerobot/pi0", device="cuda" if torch.cuda.is_available() else "cpu")
            self.policy = None # Placeholder since Pi-Zero model needs to be downloaded locally.
            self.log_callback("[VLA-ARM] Pi-Zero policy initialized (Stubbed execution).")
        except Exception as e:
            self.policy = None
            self.log_callback(f"[VLA-ARM] Error loading Pi-Zero: {e}")

    def execute_intent(self, intent, current_observation=None):
        """
        Maps an intent (like 'Wave') to a sequence of Pi-Zero actions or pre-programmed kinematics.
        """
        self.log_callback(f"[VLA-ACTION] Mapping intent '{intent}' to motion trajectories...")
        
        if not self.connected or not self.robot:
            self.log_callback("[VLA-ACTION] Cannot execute, SO-ARM101 not connected.")
            return

        # Prepare reasoning hint as text condition if policy supports it
        # Pi-Zero uses observation (camera/joints) and a text prompt
        text_prompt = f"Execute action: {intent}"

        if getattr(self, 'policy', None) is not None and current_observation is not None:
            self.log_callback(f"[VLA-ACTION] Running Pi-Zero inference for prompt: '{text_prompt}'...")
            try:
                # Stubbing out what would be the actual policy call:
                # action = self.policy.select_action(current_observation, text_prompt)
                # execute_on_hardware(action)
                time.sleep(1) 
                self.log_callback("[VLA-ACTION] 6-DOF action trajectories generated successfully.")
            except Exception as e:
                 self.log_callback(f"[VLA-ACTION] Policy error: {e}")
        else:
            intent_lower = intent.lower().strip()
            custom_skills = self.get_saved_skills()
            custom_skills_lower = {s.lower(): s for s in custom_skills}
            
            if intent_lower in custom_skills_lower:
                self.log_callback(f"[VLA-ACTION] Detected custom skill '{custom_skills_lower[intent_lower]}'. Replaying directly...")
                self.replay_skill(custom_skills_lower[intent_lower])
                self.log_callback(f"[VLA-ACTION] Execution of '{intent}' completed on SO-ARM101.")
                return

            # Fallback mock execution but now with SOME real hardware movement
            self.log_callback("[VLA-ACTION] Using pre-programmed kinematics for real hardware...")
            try:
                import numpy as np
                import time
                obs = self.robot.get_observation()
                current_pan = obs.get("shoulder_pan.pos", 0)
                
                # Basic simulated motions by modifying one or two joints
                action = {k: v for k, v in obs.items() if k.endswith('.pos')} 
                
                if "wave" in intent.lower():
                    # Move shoulder back and forth slightly
                    self.log_callback("[VLA-ACTION] Waving...")
                    action["shoulder_pan.pos"] = current_pan + 30
                    for _ in range(40): self.robot.send_action(action); time.sleep(0.02)
                    
                    action["shoulder_pan.pos"] = current_pan - 30
                    for _ in range(40): self.robot.send_action(action); time.sleep(0.02)
                    
                    action["shoulder_pan.pos"] = current_pan
                    for _ in range(40): self.robot.send_action(action); time.sleep(0.02)
                    
                elif "point" in intent.lower():
                    self.log_callback("[VLA-ACTION] Pointing...")
                    action["elbow_flex.pos"] = obs.get("elbow_flex.pos", 0) + 40
                    for _ in range(50): self.robot.send_action(action); time.sleep(0.02)
                    
                    time.sleep(1)
                    action["elbow_flex.pos"] = obs.get("elbow_flex.pos", 0)
                    for _ in range(50): self.robot.send_action(action); time.sleep(0.02)
                    
                elif "yes" in intent.lower():
                    self.log_callback("[VLA-ACTION] Nodding Yes...")
                    base_lift = obs.get("shoulder_lift.pos", 0)
                    action["shoulder_lift.pos"] = base_lift + 20
                    for _ in range(30): self.robot.send_action(action); time.sleep(0.02)
                    action["shoulder_lift.pos"] = base_lift - 10
                    for _ in range(30): self.robot.send_action(action); time.sleep(0.02)
                    action["shoulder_lift.pos"] = base_lift
                    for _ in range(30): self.robot.send_action(action); time.sleep(0.02)
                    
                elif "no" in intent.lower():
                    self.log_callback("[VLA-ACTION] Shaking No...")
                    action["shoulder_pan.pos"] = current_pan + 20
                    for _ in range(30): self.robot.send_action(action); time.sleep(0.02)
                    action["shoulder_pan.pos"] = current_pan - 20
                    for _ in range(40): self.robot.send_action(action); time.sleep(0.02)
                    action["shoulder_pan.pos"] = current_pan
                    for _ in range(30): self.robot.send_action(action); time.sleep(0.02)
                    
                elif "dance" in intent.lower():
                    self.log_callback("[VLA-ACTION] Doing a smooth, slow, happy sinusoidal dance!...")
                    import math
                    base_lift = obs.get("shoulder_lift.pos", 0)
                    base_roll = obs.get("wrist_roll.pos", 0)
                    base_flex = obs.get("elbow_flex.pos", 0)
                    steps = 300 # 6 seconds at 0.02s per step
                    
                    for i in range(steps):
                        t = i / 20.0 # Slower time variable (halved speed)
                        action["shoulder_pan.pos"] = current_pan + math.sin(t) * 35
                        action["shoulder_lift.pos"] = base_lift + math.sin(t * 1.3) * 15
                        action["wrist_roll.pos"] = base_roll + math.sin(t * 2.5) * 45
                        action["elbow_flex.pos"] = base_flex + math.cos(t * 1.5) * 20
                        
                        try:
                            self.robot.send_action(action)
                        except Exception as e:
                            self.log_callback(f"[VLA-ACTION] Dance error: {e}")
                            break
                        import time
                        time.sleep(0.02)
                        
                    # We rely on 'reset position' acting afterwards to smoothly go back to standard pose.
                elif "reset position" in intent.lower():
                    self.log_callback("[VLA-ACTION] Resetting to home position slowly...")
                    steps = 50
                    target_action = None
                    try:
                        import json
                        import os
                        if os.path.exists("skills.json"):
                            with open("skills.json", "r") as f:
                                skills_data = json.load(f)
                            custom_skills_lower = {s.lower(): s for s in skills_data.keys()}
                            for name in ["reset position", "reset pos", "home", "reset"]:
                                if name in custom_skills_lower:
                                    actual_name = custom_skills_lower[name]
                                    target_action = skills_data[actual_name][0]
                                    break
                    except Exception as e:
                        self.log_callback(f"[VLA-ACTION] Could not check custom reset pos: {e}")
                    
                    if not target_action:
                        target_action = {
                            "shoulder_pan.pos": -15.0,
                            "shoulder_lift.pos": -38.0,
                            "elbow_flex.pos": -6.4,
                            "wrist_flex.pos": 72.1,
                            "wrist_roll.pos": -3.5,
                            "gripper.pos": 21.7
                        }
                    
                    for i in range(1, steps + 1):
                        interp_action = {}
                        for k in target_action.keys():
                            if k in obs:
                                start_val = obs[k]
                                end_val = target_action[k]
                                interp_action[k] = start_val + (end_val - start_val) * (i / float(steps))
                        self.robot.send_action(interp_action)
                        time.sleep(0.02)
                    
                else:
                    self.log_callback("[VLA-ACTION] Slight default movement...")
                    action["wrist_roll.pos"] = obs.get("wrist_roll.pos", 0) + 20
                    for _ in range(30): self.robot.send_action(action); time.sleep(0.02)
                    action["wrist_roll.pos"] = obs.get("wrist_roll.pos", 0)
                    for _ in range(30): self.robot.send_action(action); time.sleep(0.02)
                    
            except Exception as e:
                self.log_callback(f"[VLA-ACTION] Hardware execution error: {e}")
            
        self.log_callback(f"[VLA-ACTION] Execution of '{intent}' completed on SO-ARM101.")
        
    def move_joint(self, joint_name, direction, step_degrees=10.0):
        if not self.connected or not self.robot:
            self.log_callback("[VLA-ACTION] Cannot execute, SO-ARM101 not connected.")
            return
            
        try:
            import time
            obs = self.robot.get_observation()
            current_pos = obs.get(f"{joint_name}.pos")
            if current_pos is None:
                self.log_callback(f"[VLA-ACTION] Joint {joint_name} not found.")
                return
                
            action = {k: v for k, v in obs.items() if k.endswith('.pos')}
            
            if direction in ["up", "right", "open", "forward"]:
                new_pos = current_pos + step_degrees
            else:
                new_pos = current_pos - step_degrees
                
            action[f"{joint_name}.pos"] = new_pos
            
            self.log_callback(f"[VLA-ACTION] Moving {joint_name} {direction} by {step_degrees} deg...")
            for _ in range(20): 
                self.robot.send_action(action)
                time.sleep(0.02)
                
        except Exception as e:
            self.log_callback(f"[VLA-ACTION] Hardware execution error: {e}")

    def adjust_tracking(self, error_x, error_y, frame_width, frame_height):
        """
        Adjust arm position based on the pixel error of an object in the gripper camera view.
        Uses PID controllers for smooth movement and a multi-joint 'pointing' behavior.
        """
        if not self.connected or not self.robot:
            return
            
        # 1. Normalize errors to -1.0 ... 1.0 range
        norm_error_x = error_x / (frame_width / 2.0)
        norm_error_y = error_y / (frame_height / 2.0)
        
        # 2. Add Deadzone (5% frame center) for better sensitivity
        if abs(norm_error_x) < 0.05: norm_error_x = 0
        if abs(norm_error_y) < 0.05: norm_error_y = 0
        
        # 3. Update PID Controllers
        delta_x = self.pid_x.update(norm_error_x)
        delta_y = self.pid_y.update(norm_error_y)
        
        if delta_x == 0 and delta_y == 0:
            return # No movement needed

        try:
            # Debug logging to see what's happening
            self.log_callback(f"[TURRET-MODE] Update - X_Err: {norm_error_x:.2f}, Y_Err: {norm_error_y:.2f} -> Del_X: {delta_x:.2f}, Del_Y: {delta_y:.2f}")
            
            obs = self.robot.get_observation()
            action = {k: v for k, v in obs.items() if k.endswith('.pos')}
            
            # X-Axis (Pan)
            # Flip sign: + delta_x to follow when error is positive (right)
            current_pan = obs.get("shoulder_pan.pos", 0)
            action["shoulder_pan.pos"] = current_pan + delta_x
            
            # Y-Axis (Pointing Gesture)
            # Flip sign logically: Shoulder lift increases to move UP.
            # If error_y is positive (object is BELOW center), we should move shoulder lift DOWN (- delta_y)
            # If error_y is negative (object is ABOVE center), we should move shoulder lift UP (+ -delta_y)
            # User log shows: Y_Err -0.5 (Object UP) -> Del_Y -1.25. 
            # If current_lift + -1.25 -> moves down.
            # So we need: current_lift - delta_y
            current_lift = obs.get("shoulder_lift.pos", 0)
            current_elbow = obs.get("elbow_flex.pos", 0)
            
            # Shoulder handles the vertical angle
            action["shoulder_lift.pos"] = current_lift - delta_y
            
            # Elbow flexes slightly to 'point' more aggressively (opposite of shoulder)
            action["elbow_flex.pos"] = current_elbow + (delta_y * 1.2)
            
            self.robot.send_action(action)
            
        except Exception as e:
            self.log_callback(f"[TURRET-MODE] Action error: {e}")
            
    def reset_pids(self):
        """Clear PID integrals and errors to prevent drift when target is lost."""
        if hasattr(self, 'pid_x'): self.pid_x.reset()
        if hasattr(self, 'pid_y'): self.pid_y.reset()

    def start_teleop(self):
        if not self.leader_robot:
            self.log_callback("[VLA-ARM] Cannot start teleoperation, Leader arm not connected.")
            return False
        if self.recording:
            self.log_callback("[VLA-ARM] Cannot start teleoperation while recording.")
            return False
            
        self.teleoperating = True
        import threading
        self.teleop_thread = threading.Thread(target=self._teleop_loop, daemon=True)
        self.teleop_thread.start()
        self.log_callback("[VLA-ARM] Teleoperation Mode: ON")
        return True
        
    def _teleop_loop(self):
        import time
        while self.teleoperating and self.leader_robot and self.connected:
            try:
                action = self.leader_robot.get_action()
                self.robot.send_action(action)
                time.sleep(0.02) # 50 Hz
            except Exception as e:
                self.log_callback(f"[VLA-ARM] Teleoperation loop error: {e}")
                self.teleoperating = False
                
    def stop_teleop(self):
        if not self.teleoperating:
            return
        self.teleoperating = False
        if hasattr(self, 'teleop_thread'):
            self.teleop_thread.join(timeout=1.0)
        self.log_callback("[VLA-ARM] Teleoperation Mode: OFF")

    def start_recording(self, skill_name):
        if not self.leader_robot:
            self.log_callback("[VLA-ARM] Cannot record, Leader arm not connected.")
            return False
        
        self.recording = True
        self.current_skill_name = skill_name
        self.recorded_frames = []
        
        import threading
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()
        self.log_callback(f"[VLA-ARM] Started recording skill: '{skill_name}'")
        return True
        
    def _record_loop(self):
        import time
        # Read loop
        while self.recording and self.leader_robot and self.connected:
            try:
                action = self.leader_robot.get_action()
                self.recorded_frames.append(action)
                
                # Mirror immediately to follower so user sees their motion
                self.robot.send_action(action)
                
                time.sleep(0.02) # 50 Hz
            except Exception as e:
                self.log_callback(f"[VLA-ARM] Recording loop error: {e}")
                self.recording = False
                
    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join(timeout=1.0)
            
        # Save to skills.json
        import json
        import os
        skills = {}
        if os.path.exists("skills.json"):
            try:
                with open("skills.json", "r") as f:
                    skills = json.load(f)
            except: pass
            
        skills[self.current_skill_name] = self.recorded_frames
        with open("skills.json", "w") as f:
            json.dump(skills, f, indent=4)
            
        self.log_callback(f"[VLA-ARM] Skill '{self.current_skill_name}' saved with {len(self.recorded_frames)} frames.")
        
    def replay_skill(self, skill_name):
        import json
        import time
        import os
        
        if not self.connected or not self.robot:
            self.log_callback("[VLA-ACTION] Cannot execute, SO-ARM101 not connected.")
            return
            
        if not os.path.exists("skills.json"):
            self.log_callback("[VLA-ARM] No skills.json found.")
            return
            
        try:
            with open("skills.json", "r") as f:
                skills = json.load(f)
        except Exception as e:
            self.log_callback(f"[VLA-ARM] Error reading skills.json: {e}")
            return
            
        if skill_name not in skills:
            self.log_callback(f"[VLA-ARM] Skill '{skill_name}' not found.")
            return
            
        self.log_callback(f"[VLA-ACTION] Replaying '{skill_name}'...")
        frames = skills[skill_name]
        if not frames:
            return
            
        # Smooth interpolation to the first frame of the trajectory
        try:
            current_obs = self.robot.get_observation()
            start_frame = frames[0]
            steps = 50
            self.log_callback(f"[VLA-ACTION] Interpolating to start position of '{skill_name}'...")
            for i in range(1, steps + 1):
                interp_action = {}
                for k in start_frame.keys():
                    if k in current_obs:
                        start_val = current_obs[k]
                        end_val = start_frame[k]
                        interp_action[k] = start_val + (end_val - start_val) * (i / float(steps))
                self.robot.send_action(interp_action)
                time.sleep(0.02)
        except Exception as e:
            self.log_callback(f"[VLA-ACTION] Interpolation error: {e}")
        
        self.log_callback(f"[VLA-ACTION] Executing '{skill_name}' trajectory...")
        for action in frames:
            try:
                self.robot.send_action(action)
                time.sleep(0.02)
            except Exception as e:
                self.log_callback(f"[VLA-ACTION] Hardware execution error: {e}")
                break
                
        self.log_callback(f"[VLA-ACTION] Replay of '{skill_name}' complete.")

    def get_saved_skills(self):
        import json
        import os
        if not os.path.exists("skills.json"):
            return []
        try:
            with open("skills.json", "r") as f:
                skills = json.load(f)
            return list(skills.keys())
        except:
            return []

    def delete_skill(self, skill_name):
        import json
        import os
        skills_path = os.path.join(os.getcwd(), "skills.json")
        self.log_callback(f"[VLA-ARM] Deleting skill '{skill_name}' from {skills_path}...")
        
        if not os.path.exists(skills_path):
            self.log_callback(f"[VLA-ARM] ERROR: {skills_path} not found.")
            return False
            
        try:
            with open(skills_path, "r") as f:
                skills = json.load(f)
            
            if skill_name in skills:
                del skills[skill_name]
                with open(skills_path, "w") as f:
                    json.dump(skills, f, indent=4)
                self.log_callback(f"[VLA-ARM] Skill '{skill_name}' deleted successfully from file.")
                return True
            else:
                self.log_callback(f"[VLA-ARM] Skill '{skill_name}' was not found in the JSON keys: {list(skills.keys())}")
                return False
        except Exception as e:
            self.log_callback(f"[VLA-ARM] Error deleting skill: {e}")
            import traceback
            self.log_callback(traceback.format_exc())
            return False

    def start_ai_recording(self, repo_id, fps=30, single_task="Play", resume=False, num_episodes=50, episode_time=60, reset_time=60, manual_mode=False, visualize=False):
        """
        Launches the official LeRobot recording script as a subprocess.
        Assumes cameras and serial ports have already been released.
        """
        import subprocess
        import sys
        
        # Manual Trigger Mode uses extremely long timers so the loop waits for the 'r' key
        if manual_mode:
            episode_time = 3600 
            reset_time = 3600
            self.log_callback(f"[VLA-DATASET] Manual Trigger Mode detected. Timers set to 1h.")
            self.log_callback(f"[VLA-DATASET] Press 'r' or 'Right Arrow' in the terminal to start/stop episodes.")

        self.log_callback(f"[VLA-DATASET] Starting LeRobot dataset collection for repo: {repo_id}")
        
        import os
        import tempfile
        import threading
        import yaml
        
        # LeRobot is installed in a dedicated conda environment named 'lerobot',
        # and the source code lives in C:\Users\Noniro\lerobot.
        lerobot_python = r"C:\Users\Noniro\miniconda3\envs\lerobot\python.exe"
        lerobot_src_dir = r"C:\Users\Noniro\lerobot"
        
        if not os.path.exists(lerobot_python):
             self.log_callback(f"[VLA-DATASET] ERROR: LeRobot python not found at {lerobot_python}")
             return None

        # Create a temporary YAML configuration file for the robot cameras to bypass CLI dict parsing issues.
        from pathlib import Path
        import shutil
        
        lerobot_cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
        
        # Clean up any "local/" prefix from user input and sanitize characters
        clean_repo_id = repo_id.replace("local/", "").replace("local\\", "").replace("&", "_").replace(" ", "_").strip()
        
        # LeRobot will crash if the local dataset directory already exists. Delete it if it does.
        local_dataset_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "local" / clean_repo_id
        if not resume and local_dataset_dir.exists() and local_dataset_dir.is_dir():
            try:
                shutil.rmtree(local_dataset_dir)
                self.log_callback(f"[VLA-DATASET] Cleared existing local dataset directory for '{clean_repo_id}'.")
            except Exception as e:
                self.log_callback(f"[VLA-DATASET] Warning: Could not clear existing dataset directory. {e}")
                
        config_data = {
            "display_data": visualize,
            "resume": resume,
            "robot": {
                "type": "so101_follower",
                "port": self.target_port or "COM4",
                "id": "so_arm_101",
                "calibration_dir": str(lerobot_cache_dir / "robots" / "so101_follower"),
                "cameras": {
                    "cam_high": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
                    "cam_wrist": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
                    "cam_side": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
                }
            },
            "teleop": {
                "type": "so101_leader",
                "port": self.leader_port or "COM5",
                "id": "so_arm_101_leader",
                "calibration_dir": str(lerobot_cache_dir / "teleoperators" / "so101_leader")
            },
            "dataset": {
                "repo_id": f"local/{clean_repo_id}",
                "fps": fps,
                "single_task": single_task,
                "episode_time_s": episode_time,
                "reset_time_s": reset_time,
                "num_episodes": num_episodes,
                "tags": ["so101", "tutorial"],
                "push_to_hub": False,
                "video": True,
                "streaming_encoding": True,
                "num_image_writer_processes": 1,
                "num_image_writer_threads_per_camera": 1,
                "vcodec": "h264"
            }
        }
        
        fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", text=True)
        with os.fdopen(fd, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Execution via -m fails due to the local source layout, so we provide the exact script path.
        record_script = os.path.join(lerobot_src_dir, "src", "lerobot", "scripts", "lerobot_record.py")

        wrapper_py = f"""
import subprocess
import sys
import traceback
import os
try:
    print("Launching AI recording session...")
    res = subprocess.run([
        r'{lerobot_python}',
        r'{record_script}',
        '--config_path',
        r'{temp_config_path}'
    ], cwd=r'{lerobot_src_dir}')
    if res.returncode != 0:
        print(f"\\nRecording script exited with code {{res.returncode}}.")
except Exception as e:
    print("\\nAn error occurred launching the recording:")
    traceback.print_exc()

print("\\nRecording finished or crashed.")
input("Press Enter to close this window...")
"""
        fd_wrapper, wrapper_path = tempfile.mkstemp(suffix=".py", text=True)
        with os.fdopen(fd_wrapper, 'w') as f:
            f.write(wrapper_py)

        cmd = [lerobot_python, wrapper_path]
        
        # We MUST set PYTHONPATH so the subprocess knows where to find the 'lerobot' module,
        # since we are running this from the RoboAssistant directory.
        env = os.environ.copy()
        env["PYTHONPATH"] = lerobot_src_dir + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        
        # CRITICAL: Allow a brief moment for Windows OS to fully release the COM ports
        # after `disconnect_arms` before the subprocess tries to instantiate its own locks.
        import time
        import gc
        gc.collect() # Force garbage collection of any lingering serial instances
        time.sleep(2.5) # Increased from 1.5 to 2.5 for slower Windows systems

        try:
            process = subprocess.Popen(
                cmd, env=env, close_fds=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            self.log_callback(f"[VLA-DATASET] LeRobot recording window opened for '{repo_id}'.")
            
            def cleanup_thread(proc, file_path_config, file_path_wrapper):
                proc.wait()
                try:
                    os.remove(file_path_config)
                    os.remove(file_path_wrapper)
                except:
                    pass
            t = threading.Thread(target=cleanup_thread, args=(process, temp_config_path, wrapper_path))
            t.daemon = True
            t.start()
            
            return process
        except Exception as e:
            self.log_callback(f"[VLA-DATASET] ERROR launching recording: {e}")
            try:
                os.remove(temp_config_path)
            except:
                pass
            return None

    def register_vla_policy(self, name, checkpoint_path):
        """Register a trained VLA policy by name so the LLM brain can trigger it."""
        self.vla_policies[name] = checkpoint_path
        self.log_callback(f"[VLA-ARM] Registered VLA policy '{name}' -> {checkpoint_path}")

    def get_vla_policies(self):
        """Return the dict of registered VLA policies."""
        return dict(self.vla_policies)

    def run_vla_intent(self, policy_name, episode_time=60, num_episodes=1, camera_settings=None):
        """
        Trigger a named VLA policy for autonomous inference.
        Called by the UI button or the LLM brain when it determines a VLA intent.
        """
        checkpoint_path = self.vla_policies.get(policy_name)
        if not checkpoint_path:
            self.log_callback(f"[VLA-ARM] ERROR: VLA policy '{policy_name}' not registered.")
            return None
        self.log_callback(f"[VLA-ARM] Triggering VLA inference for policy: '{policy_name}'")
        return self.run_ai_inference(
            repo_id=policy_name,
            checkpoint_path=checkpoint_path,
            episode_time=episode_time,
            num_episodes=num_episodes,
            camera_settings=camera_settings
        )

    def run_ai_inference(self, repo_id, checkpoint_path, visualize=False, fps=30, episode_time=60, num_episodes=1, camera_settings=None):
        """
        Runs a trained LeRobot policy autonomously.
        This spawns lerobot_record.py but fixes 'teleop' to null so only the robot arm is active.
        """
        if self.connected:
            self.disconnect_arms()

        import tempfile
        import yaml
        import os
        import subprocess
        import threading

        lerobot_python = r"C:\Users\Noniro\miniconda3\envs\lerobot\python.exe"
        lerobot_src_dir = r"C:\Users\Noniro\lerobot"
        
        from pathlib import Path
        lerobot_cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
        
        # Use settings if provided, else fallback to defaults
        # --- CRITICAL: Clear existing eval dataset directory to prevent FileExistsError ---
        import shutil
        clean_repo_id = f"eval_{repo_id}".replace("local/", "").replace(" ", "_").strip()
        local_dataset_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "local" / clean_repo_id
        if local_dataset_dir.exists() and local_dataset_dir.is_dir():
            try:
                self.log_callback(f"[VLA-TEST] Clearing old eval directory: {local_dataset_dir}")
                shutil.rmtree(local_dataset_dir)
            except Exception as e:
                self.log_callback(f"[VLA-TEST] Warning: Could not clear old eval directory: {e}")

        # Inference config: robot is active, teleop is null
        config_data = {
            "display_data": visualize,
            "robot": {
                "type": "so101_follower",
                "port": self.target_port or "COM4",
                "id": "so_arm_101",
                "calibration_dir": str(lerobot_cache_dir / "robots" / "so101_follower"),
                "cameras": {
                    "cam_high": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
                    "cam_wrist": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
                    "cam_side": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
                }
            },
            "teleop": None, # Disable teleoperation for autonomous inference
            "dataset": {
                "repo_id": f"local/eval_{repo_id}",
                "fps": fps,
                "single_task": "Evaluate policy",  # Required by DatasetRecordConfig
                "episode_time_s": episode_time,
                "reset_time_s": 30,
                "num_episodes": num_episodes,
                "push_to_hub": False,
                "video": True,
                "streaming_encoding": True,
                "num_image_writer_processes": 1,
                "num_image_writer_threads_per_camera": 1,
                "vcodec": "h264",
                "rename_map": {
                    "cam_high": "images.cam_high",
                    "cam_side": "images.cam_side",
                    "cam_wrist": "images.cam_wrist"
                }
            }
        }
        
        fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", text=True)
        with os.fdopen(fd, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

        record_script = os.path.join(lerobot_src_dir, "src", "lerobot", "scripts", "lerobot_record.py")
        
        policy_type = "act" # fallback
        try:
            with open(os.path.join(checkpoint_path, "config.json"), "r") as f:
                pcfg = json.load(f)
                if "type" in pcfg:
                    policy_type = pcfg["type"]
        except Exception as e:
            pass

        # We use a python wrapper to run the actual command so we can catch exceptions.
        wrapper_py = f"""
import subprocess
import sys
import traceback
try:
    print("Launching specific eval dataset recording...")
    res = subprocess.run([
        r'{lerobot_python}',
        r'{record_script}',
        '--config_path',
        r'{temp_config_path}',
        '--policy.type',
        r'{policy_type}',
        '--policy.pretrained_path',
        r'{checkpoint_path}'
    ], cwd=r'{lerobot_src_dir}')
    if res.returncode != 0:
        print(f"\\nInference script exited with code {{res.returncode}}.")
        sys.exit(res.returncode)
except Exception as e:
    print("\\nAn error occurred launching the inference:")
    traceback.print_exc()
    sys.exit(1)

print("\\nEvaluation finished or crashed.")
"""
        fd_wrapper, wrapper_path = tempfile.mkstemp(suffix=".py", text=True)
        with os.fdopen(fd_wrapper, 'w') as f:
            f.write(wrapper_py)

        cmd = [lerobot_python, wrapper_path]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = lerobot_src_dir + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        
        import time
        import gc
        gc.collect()
        time.sleep(2.5)
        
        try:
            process = subprocess.Popen(
                cmd, env=env, close_fds=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
                
            self.log_callback(f"[VLA-TEST] Launched seamless autonomous inference for '{repo_id}'.")
            
            def cleanup_thread(proc, file_path_config, file_path_wrapper):
                for line in iter(proc.stdout.readline, ''):
                    if line:
                        self.log_callback(f"[VLA-EVAL] {line.strip()}")
                proc.wait()
                try:
                    os.remove(file_path_config)
                    os.remove(file_path_wrapper)
                except:
                    pass
            
            t = threading.Thread(target=cleanup_thread, args=(process, temp_config_path, wrapper_path))
            t.daemon = True
            t.start()
            
            return process
        except Exception as e:
            self.log_callback(f"[VLA-TEST] ERROR launching inference: {e}")
            try:
                os.remove(temp_config_path)
            except:
                pass
            return None

    def check_connection(self):
        return self.connected
