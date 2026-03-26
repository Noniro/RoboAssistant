import requests
import json
import re
import threading
import os
import time
from profile_manager import ProfileManager
from tts_manager import TTSManager

class ReasoningBridge:
    def __init__(self, log_callback, arm_controller=None, vision_worker=None):
        self.log_callback = log_callback
        self.arm_controller = arm_controller
        self.vision_worker = vision_worker
        self.profile_manager = ProfileManager()
        self.tts = TTSManager()
        
        self.base_prompt = "You are a helpful and playful lab assistant."
        self.system_prompt = (
            f"You are Yuval's Robotic Lab Assistant. {self.base_prompt} "
            "You observe the environment and decide on reasonable intents to execute via your robotic arm. "
            "Your output MUST be a JSON with two keys: 'dialogue' and 'intent'. "
            "Respond concisely. Use intents like 'Wave', 'Point', 'Idle', or a trained VLA policy name. "
            "IMPORTANT: Your reality is provided as a JSON object. "
            "You DO NOT need to verify an object exists in the Spatial Data to interact with it if the user directly commands you to do so. "
            "Trust the user's commands and execute them blindly if necessary."
        )
        self.api_url = "http://localhost:1234/v1/chat/completions"
        self.last_scene = "Nothing seen yet."
        self.chat_history = [] # Stores last 5 interactions to prevent amnesia
        self.enable_vlm_reasoning = True
        self.physical_interaction_enabled = True
        
        # Unified VLA integration - HARDCODED AS PRIMARY
        self.use_unified_vla = True 
        self.api_url = None # External LLM API Disabled
        self.unified_vla_process = None
        self.unified_vla_lock = threading.Lock()
        
        # VLA intent routing: { intent_string_from_LLM: vla_policy_name }
        # e.g. { "pick_and_place": "pick_place_Iloveyoublock" }
        self.vla_intents = {}
        self.vla_ready = False
        self.last_vla_reply = None
        self.ui_parent = None
        
        if self.use_unified_vla:
            self.log_callback("[BRAIN] Environment: Windows. Unified VLA selected.")
            # self._start_unified_vla() # MOVED to separate call to avoid thread-loop crash
        else:
            self.log_callback("[BRAIN] Environment: Windows. Reasoning via Local API (localhost:1234).")

    def initialize_worker(self):
        """Called by UI after mainloop starts to avoid Tcl thread errors."""
        if self.use_unified_vla and not self.unified_vla_process:
             self._start_unified_vla()

    def _start_unified_vla(self):
        import subprocess
        import sys
        self.log_callback("[BRAIN] 🚀 Booting Unified VLA orchestrator. This takes ~20s to load 14GB weights...")
        try:
            # We need the robot config for the worker to access cameras
            config_path = "config.yaml" # Default or passed from UI
            if not os.path.exists(config_path):
                 # Fallback to creating a temp one or using current arm settings
                 pass

            self.unified_vla_process = subprocess.Popen(
                [sys.executable, "unified_vla_worker.py", "--config", config_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1 
            )
            # Start a background thread to read its init signals and replies
            threading.Thread(target=self._monitor_unified_vla, daemon=True).start()
        except Exception as e:
            self.log_callback(f"[BRAIN] Failed to start Unified VLA: {e}")

    def _monitor_unified_vla(self):
        if not self.unified_vla_process: return
        self.log_callback("[VLA-WORKER] Monitoring process...")
        while True:
            line = self.unified_vla_process.stdout.readline()
            if not line: break
            line = line.strip()
            if not line: continue
            
            if line == "READY":
                self.log_callback("[VLA-WORKER] Unified VLA Model is Loaded into VRAM & Ready!")
                self.vla_ready = True
            elif line.startswith("CHAT_REPLY"):
                reply = line[len("CHAT_REPLY "):]
                self.last_vla_reply = reply
                self.log_callback(f"[VLA-REASONING] Brain says: {reply}")
                # Log to the Thought Log in UI if available
                if hasattr(self, "ui_parent"):
                    self.ui_parent.log_event(f"[THOUGHT] {reply}")
            elif line == "FINISHED":
                self.log_callback("[VLA-WORKER] Physical Action Completed.")
            else:
                self.log_callback(f"[VLA-WORKER] {line}")

    def load_models(self, required_models, base_prompt=None):
        if base_prompt:
            self.base_prompt = base_prompt
            self.system_prompt = (
                f"You are Yuval's Robotic Lab Assistant. {self.base_prompt} "
                "You observe the environment and decide on reasonable intents to execute via your robotic arm. "
                "Your output MUST be a JSON with two keys: 'dialogue' and 'intent'. "
                "Respond concisely. Use intents like 'Wave', 'Point', 'Idle', etc. "
                "IMPORTANT: Your reality is provided as a JSON object. "
                "You DO NOT need to verify an object exists in the Spatial Data to interact with it if the user directly commands you to do so. "
                "Trust the user's commands and execute them blindly if necessary."
            )
            
        if "vlm" in required_models:
            self.enable_vlm_reasoning = True
            self.log_callback("[BRAIN] VLM Reasoning Enabled.")
        else:
            self.enable_vlm_reasoning = False
            self.log_callback("[BRAIN] VLM Reasoning Disabled (saving VRAM/CPU).")
            
    def set_physical_interaction(self, enabled: bool):
        self.physical_interaction_enabled = enabled
        self.log_callback(f"[BRAIN] Physical Interaction {'Enabled' if enabled else 'Disabled'}")

    def register_vla_intent(self, intent_name, policy_name):
        """
        Map an LLM intent string to a registered VLA policy name.
        Example: register_vla_intent("pick_and_place", "pick_place_Iloveyoublock")
        The arm_controller must have the policy registered via register_vla_policy().
        """
        self.vla_intents[intent_name.lower()] = policy_name
        self.log_callback(f"[BRAIN] VLA intent registered: '{intent_name}' -> policy '{policy_name}'")

    def _dispatch_intent(self, intent):
        """
        Central intent dispatcher. Checks if the intent is a VLA intent first,
        then falls back to pre-programmed kinematics / recorded skills.
        """
        if not self.arm_controller or intent.lower() == "idle":
            return
        if not getattr(self, "physical_interaction_enabled", False):
            self.log_callback(f"[BRAIN] Prevented arm movement '{intent}' (Physical Interaction OFF).")
            return

        intent_lower = intent.lower().strip()

        # Check if this is a VLA intent
        if intent_lower in self.vla_intents:
            policy_name = self.vla_intents[intent_lower]
            self.log_callback(f"[BRAIN] 🤖 VLA intent triggered: '{intent}' -> policy '{policy_name}'")
            import threading
            threading.Thread(
                target=self._run_vla_and_react,
                args=(policy_name,),
                daemon=True
            ).start()
        else:
            # Fallback: pre-programmed motion or recorded skill
            self.arm_controller.execute_intent(intent, current_observation=None)

    def _run_vla_and_react(self, policy_name):
        ui = getattr(self, "ui_parent", None)
        
        # Safe start: Reset position before yielding to VLA policy
        self.log_callback("[BRAIN] Resetting arm position for safety before VLA starts.")
        self.arm_controller.execute_intent("reset position", None)
        import time
        time.sleep(1.0) # 1 sec wait for the arm to reach safe spot
        
        if ui:
            ui.after(0, ui.suspend_for_external_script)
            
        # Ensure VLA worker is started (Phase 1: Resident Infrastructure)
        self.arm_controller.start_vla_worker(policy_name)
        
        process = self.arm_controller.run_vla_intent(policy_name, episode_time=90)
        
        success = False
        if process:
            try:
                process.wait()
                success = (process.returncode == 0)
            except Exception as e:
                self.log_callback(f"[BRAIN] VLA execution error: {e}")
                
        if ui:
            ui.after(0, ui.resume_from_external_script)
            
        # Give hardware 2 seconds to re-initialize before triggering follow-up kinematic action
        time.sleep(2.0)
            
        if success:
            self.log_callback("[BRAIN] Task completed successfully! Trying to celebrate.")
            # Verify Dance skill exists
            if "Dance" in self.arm_controller.get_saved_skills():
                self.arm_controller.replay_skill("Dance")
            elif "dance" in self.arm_controller.get_saved_skills():
                self.arm_controller.replay_skill("dance")
            else:
                self.arm_controller.execute_intent("dance", None)
                
            self.log_callback("[BRAIN] Task and celebration done. Returning to safe reset position.")
            self.arm_controller.execute_intent("reset position", None)
        else:
            self.log_callback("[BRAIN] Task failed or was aborted. Resetting position.")
            self.arm_controller.execute_intent("reset position", None)

    def _build_skills_prompt(self):
        """Build the AVAILABLE PHYSICAL INTENTS string shown to the LLM."""
        available_skills = ["Wave", "Point", "Yes", "No", "Idle"]
        if self.arm_controller:
            available_skills.extend(self.arm_controller.get_saved_skills())
        # Add VLA intents as usable intent strings
        for vla_intent in self.vla_intents:
            display = vla_intent
            available_skills.append(display)
        skills_str = ", ".join([f"'{s}'" for s in available_skills])
        vla_note = ""
        if self.vla_intents:
            # Make sure it's ultra-clear for the LLM
            vla_string = ", ".join([f"'{k}'" for k in self.vla_intents])
            vla_note = (
                f" \n\nVLA-TRAINED SKILLS (Real AI policies): [{vla_string}]. "
                "If the user asks you to interact with or pick up an object (e.g., 'pick the block'), YOU MUST use the VLA-TRAINED SKILL that corresponds to that object, instead of 'Point' or 'Wave'."
            )
        return (
            f"\n\nAVAILABLE PHYSICAL INTENTS: [{skills_str}].{vla_note} "
            "If the user asks you to perform one of these exact actions, you MUST set the 'intent' to that exact string. "
            "You MUST output valid JSON ONLY, strictly containing 'dialogue' and 'intent' keys. Do NOT use markdown code formatting blocks."
        )

    def generate_dialogue_and_intent(self, scene_description, identity=None):
        """Unified VLA handles this now."""
        # Instead of calling local API, we send it to the unified worker if needed.
        # However, manual_interact is the primary path now.
        return "Thinking...", "Idle"

    def manual_interact(self, user_text, identity=None):
        """
        Allows the user to chat directly with the robot.
        Includes the last seen scene description for context.
        """
        self.log_callback(f"[CHAT] User: {user_text}")
        
        # Use structured JSON reality (Face + Object Coordinates + VLM)
        vision_context = self.last_scene
        if self.vision_worker:
            structured_data = self.vision_worker.get_structured_reality()
            vision_context = json.dumps(structured_data, indent=2)
            self.last_scene = vision_context

        # Determine specific prompt based on identity
        custom_prompt = ""
        if identity and identity != "None":
            profile = self.profile_manager.get_profile(identity)
            if profile:
                custom_prompt = f"\n\nSPECIAL INSTRUCTIONS FOR THIS PERSON ({identity}): {profile['prompt']}"

        custom_prompt += self._build_skills_prompt()

        messages = [{"role": "system", "content": self.system_prompt + custom_prompt}]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": f"Structured Reality JSON:\n{vision_context}\n\nUser says: {user_text}"})
        
        self.log_callback(f"[BRAIN-CHAT] Identity: {identity or 'Unknown'}")

        try:
            if self.use_unified_vla and self.unified_vla_process:
                # 1. Reasoning Phase
                self.log_callback(f"[BRAIN] Asking Unified VLA (OpenVLA-7B): {user_text}")
                self.last_vla_reply = None
                with self.unified_vla_lock:
                    self.unified_vla_process.stdin.write(f"CHAT {user_text}\n")
                    self.unified_vla_process.stdin.flush()
                
                # Wait for reply (handled by monitor thread)
                start_w = time.time()
                while self.last_vla_reply is None and time.time() - start_w < 20: 
                    time.sleep(0.1)
                
                if self.last_vla_reply:
                    dialogue = self.last_vla_reply
                else:
                    dialogue = "I'm still loading the scene in my weights..."
                
                # 2. Action Trigger Phase
                # Improved detection: check for pick/place/move etc.
                is_task = any(word in user_text.lower() for word in ["pick", "place", "move", "grab", "reach", "put", "take"])
                if is_task:
                    self.log_callback(f"[BRAIN] Task identified. Triggering VLA physical control...")
                    with self.unified_vla_lock:
                        # ACTION <duration> <instruction>
                        self.unified_vla_process.stdin.write(f"ACTION 60 {user_text}\n")
                        self.unified_vla_process.stdin.flush()
                
                self.chat_history.append({"role": "user", "content": user_text})
                self.chat_history.append({"role": "assistant", "content": dialogue})
                self.tts.speak(dialogue)
                return dialogue
            else:
                return "Unified VLA Worker not running."
                payload = {
                    "model": "llama-3.2-3b-instruct",
                    "messages": messages,
                    "temperature": 0.7
                }
                response = requests.post(self.api_url, json=payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    response_text = result['choices'][0]['message']['content'].strip()
                else:
                    return f"Error: {response.status_code}"
                
            # Attempt to parse JSON if model follows instructions, else fallback to regex
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    dialogue = parsed.get('dialogue', response_text)
                    intent = parsed.get('intent', "Idle")
                else:
                    raise ValueError("No JSON block found")
            except Exception as e:
                self.log_callback(f"[BRAIN-CHAT] JSON Parse Error: {e}, attempting regex extraction on: {response_text}")
                d_match = re.search(r'"dialogue"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
                i_match = re.search(r'"intent"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
                dialogue = d_match.group(1) if d_match else response_text.replace('{', '').replace('}', '').replace('"', '').strip()
                intent = i_match.group(1) if i_match else "Idle"
                
            self.chat_history.append({"role": "user", "content": user_text})
            self._dispatch_intent(intent)
            self.chat_history.append({"role": "assistant", "content": dialogue})
            if len(self.chat_history) > 6:
                self.chat_history = self.chat_history[-6:]
                
            self.tts.speak(dialogue)
            return dialogue

        except Exception as e:
            return f"Brain Error: {e}"

    def proactive_reasoning(self, scene_description, identity=None, event_type="PERIODIC", mode_config=None):
        """
        Autonomous thinking triggered by vision events.
        Behavior changes based on the active Mode Config.
        """
        if not self.enable_vlm_reasoning and event_type == "PERIODIC":
            return "" # Skip periodic VLM pulses if mode doesn't allow it
            
        self.last_scene = scene_description
        
        # Configure proactive prompt based on mode
        mode_rules = ""
        temperature = 0.5
        
        if mode_config:
            if mode_config.get("name") == "Study":
                if event_type == "DISTRACTION":
                    # Check Cooldown
                    cooldown = mode_config.get("cooldown_seconds", 30)
                    if hasattr(self, 'last_distraction_time'):
                        import time
                        if time.time() - self.last_distraction_time < cooldown:
                            return "" # Still in cooldown, stay silent
                    
                    import time
                    self.last_distraction_time = time.time()
                    
                    mode_rules = (
                        "You are in STUDY MODE. The user is currently DISTRACTED. "
                        "You MUST reprimand the user strictly but creatively based on the scene context. "
                    )
                    action_lvl = mode_config.get('action_level', 1)
                    if action_lvl == 2:
                        mode_rules += "Set your intent to 'no-no'. "
                    elif action_lvl == 3:
                        mode_rules += "Set your intent to 'angry'. "
                else:
                    return "" # Study mode ONLY speaks on distractions
            elif mode_config.get("name") == "Security":
                 mode_rules = "You are in SECURITY MODE. Alert loudly if an Unknown person is detected. Stay silent otherwise."
            elif mode_config.get("name") == "General":
                 cooldown = mode_config.get("cooldown_seconds", 60)
                 if hasattr(self, 'last_general_proactive_time'):
                     import time
                     if time.time() - self.last_general_proactive_time < cooldown:
                         return "" # Still in cooldown
                 import time
                 self.last_general_proactive_time = time.time()
        
        proactive_prompt = (
            f"You are Yuval's Robotic Lab Assistant. {self.base_prompt} "
            f"{mode_rules}"
            "If a NEW PERSON has entered or a DISTRACTION has been detected, speak proactively. "
            "If nothing important has happened, you MUST STAY SILENT. "
            "To stay silent, return: {\"dialogue\": \"\", \"intent\": \"Idle\"}. "
            "Otherwise, respond with a JSON with 'dialogue' and 'intent' keys."
        )

        custom_prompt = ""
        if identity and identity != "None":
            profile = self.profile_manager.get_profile(identity)
            if profile:
                custom_prompt = f"\n\nPROFILE SPECIFIC: {profile['prompt']}"

        custom_prompt += self._build_skills_prompt()

        messages = [{"role": "system", "content": proactive_prompt + custom_prompt}]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": f"Event: {event_type}. Scene: {scene_description}. Should you say anything?"})

        try:
            payload = {
                "model": "llama-3.2-3b-instruct",
                "messages": messages,
                "temperature": 0.5 # Lower temperature for more consistent silence
            }
            response = requests.post(self.api_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                
                try:
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        dialogue = parsed.get('dialogue', "")
                        intent = parsed.get('intent', "Idle")
                    else:
                        raise ValueError("No JSON block found")
                except Exception:
                    d_match = re.search(r'"dialogue"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
                    i_match = re.search(r'"intent"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
                    dialogue = d_match.group(1) if d_match else ""
                    intent = i_match.group(1) if i_match else "Idle"
                    
                if dialogue:
                    self.log_event_internal(f"[PROACTIVE] {dialogue}")
                    self.tts.speak(dialogue)
                    self._dispatch_intent(intent)
                        
                    # Save to history so robot remembers it spoke proactively
                    self.chat_history.append({"role": "assistant", "content": dialogue})
                    if len(self.chat_history) > 6:
                        self.chat_history = self.chat_history[-6:]
                        
                return dialogue
                
        except Exception as e:
            self.log_callback(f"[BRAIN] proactive error: {e}")
            return ""

    def log_event_internal(self, message):
        # Helper because log_callback might be complex
        self.log_callback(message)

