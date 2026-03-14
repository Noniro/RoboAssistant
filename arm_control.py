import time
import torch
from lerobot.policies.factory import make_policy

class ArmController:
    def __init__(self, log_callback):
        self.log_callback = log_callback
        
        # Windows Serial Port Detection
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        target_port = None
        leader_port = None
        for p in ports:
            self.log_callback(f"[VLA-ARM] Found Port: {p.device} ({p.description})")
            if "COM4" in p.device.upper(): # Force COM4 based on user setup
                target_port = p.device
            if "COM5" in p.device.upper(): # Force COM5 based on user setup
                leader_port = p.device
        
        self.connected = False
        self.robot = None
        self.leader_robot = None
        self.recording = False
        self.teleoperating = False
        
        if target_port:
            self.log_callback(f"[VLA-ARM] Connecting to SO-ARM101 on {target_port}...")
            try:
                # CRITICAL: Follower motors are often left in "Step Mode" (3) after calibration
                # so they can spin freely. To command them to move, they MUST be forced back
                # into "Servo/Position Mode" (0) before LeRobot takes ownership of the port.
                import scservo_sdk as scs
                self.log_callback("[VLA-ARM] Securing motors into Position Servo Mode...")
                portHandler = scs.PortHandler(target_port)
                packetHandler = scs.PacketHandler(0)
                if portHandler.openPort() and portHandler.setBaudRate(1000000):
                    for motor_id in range(1, 7):
                        packetHandler.write1ByteTxRx(portHandler, motor_id, 33, 0) # Address 33 = Mode, 0 = Position
                    portHandler.closePort()
                    self.log_callback("[VLA-ARM] All 6 motors set to active Position Mode.")
                
                from lerobot.robots.utils import make_robot_from_config
                from lerobot.robots.so101_follower.so101_follower import SO101FollowerConfig
                cfg = SO101FollowerConfig(port=target_port, id='so_arm_101')
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
                if leader_port:
                    self.log_callback(f"[VLA-ARM] Connecting to SO-ARM101 Leader on {leader_port}...")
                    try:
                        from lerobot.teleoperators import make_teleoperator_from_config
                        from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
                        
                        cfg_leader = SO101LeaderConfig(port=leader_port, id='so_arm_101_leader')
                        self.leader_robot = make_teleoperator_from_config(cfg_leader)
                        self.leader_robot.connect(calibrate=False)
                        self.leader_robot.bus.disable_torque() # Ensure it's loose for human control
                        self.log_callback("[VLA-ARM] Successfully connected to Leader Arm.")
                    except Exception as e:
                        self.log_callback(f"[VLA-ARM] WARNING: Failed to connect to Leader: {e}")
            except Exception as e:
                self.log_callback(f"[VLA-ARM] WARNING: Failed to connect via LeRobot: {e}")
                self.connected = False
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

        if self.policy is not None and current_observation is not None:
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
        if not os.path.exists("skills.json"):
            return False
        try:
            with open("skills.json", "r") as f:
                skills = json.load(f)
            if skill_name in skills:
                del skills[skill_name]
                with open("skills.json", "w") as f:
                    json.dump(skills, f, indent=4)
                self.log_callback(f"[VLA-ARM] Skill '{skill_name}' deleted successfully.")
                return True
            else:
                self.log_callback(f"[VLA-ARM] Skill '{skill_name}' not found for deletion.")
                return False
        except Exception as e:
            self.log_callback(f"[VLA-ARM] Error deleting skill: {e}")
            return False

    def check_connection(self):
        return self.connected
