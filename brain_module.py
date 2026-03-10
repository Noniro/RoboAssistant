import requests
import json
import re
import threading
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
            "Respond concisely. Use intents like 'Wave', 'Point', 'Idle', etc. "
            "IMPORTANT: Your reality is provided as a JSON object containing User Identity, Contextual Description, and Spatial Data. "
            "If commanded to interact with an object (e.g., 'touch', 'take'), look up the EXACT [x, y] coordinates in the Spatial Data. "
            "If there is a conflict between the VLM description and YOLO Spatial Data, ask the user for clarification."
        )
        self.api_url = "http://localhost:1234/v1/chat/completions"
        self.last_scene = "Nothing seen yet."
        self.chat_history = [] # Stores last 5 interactions to prevent amnesia
        self.enable_vlm_reasoning = True
        self.physical_interaction_enabled = False
        self.log_callback("[BRAIN] Environment: Windows. Reasoning via Local API (localhost:1234).")

    def load_models(self, required_models, base_prompt=None):
        if base_prompt:
            self.base_prompt = base_prompt
            self.system_prompt = (
                f"You are Yuval's Robotic Lab Assistant. {self.base_prompt} "
                "You observe the environment and decide on reasonable intents to execute via your robotic arm. "
                "Your output MUST be a JSON with two keys: 'dialogue' and 'intent'. "
                "Respond concisely. Use intents like 'Wave', 'Point', 'Idle', etc. "
                "IMPORTANT: Your reality is provided as a JSON object containing User Identity, Contextual Description, and Spatial Data. "
                "If commanded to interact with an object (e.g., 'touch', 'take'), look up the EXACT [x, y] coordinates in the Spatial Data. "
                "If there is a conflict between the VLM description and YOLO Spatial Data, ask the user for clarification."
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

    def generate_dialogue_and_intent(self, scene_description, identity=None):
        self.last_scene = scene_description
        self.log_callback(f"[BRAIN] Reasoning over scene: {scene_description} (Seen: {identity or 'Unknown'})")
        
        # Determine specific prompt based on identity
        custom_prompt = ""
        if identity:
            profile = self.profile_manager.get_profile(identity)
            if profile:
                custom_prompt = f"\n\nSPECIAL INSTRUCTIONS FOR THIS PERSON ({identity}): {profile['prompt']}"

        messages = [
            {"role": "system", "content": self.system_prompt + custom_prompt},
            {"role": "user", "content": f"Scene Description: {scene_description}. Determine my next intent."}
        ]

        try:
            payload = {
                "model": "llama-3.2-3b-instruct", # Updated to match loaded model
                "messages": messages,
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                self.log_callback(f"[BRAIN] Raw Model Output: {response_text}")
                
                try:
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        dialogue = parsed.get('dialogue', response_text)
                        intent = parsed.get('intent', "Idle")
                    else:
                        dialogue = response_text
                        intent = "Idle"
                except json.JSONDecodeError:
                    self.log_callback("[BRAIN] JSON Parse Error. Falling back to raw text.")
                    dialogue = response_text
                    intent = "Idle"
            else:
                self.log_callback(f"[BRAIN] API Error: {response.status_code}")
                dialogue = "I'm having trouble thinking."
                intent = "Idle"
                
        except Exception as e:
            self.log_callback(f"[BRAIN] Inference error: {e}")
            dialogue = "Brain Error."
            intent = "Idle"
        
        self.log_callback(f"[BRAIN] Generated Dialogue: {dialogue}")
        self.log_callback(f"[BRAIN] Intent determined: {intent}")
        
        self.tts.speak(dialogue)
        
        if self.arm_controller and intent.lower() != "idle":
            self.arm_controller.execute_intent(intent, current_observation=None)
        
        return dialogue, intent

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

        messages = [{"role": "system", "content": self.system_prompt + custom_prompt}]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": f"Structured Reality JSON:\n{vision_context}\n\nUser says: {user_text}"})
        
        self.log_callback(f"[BRAIN-CHAT] Identity: {identity or 'Unknown'}")

        try:
            payload = {
                "model": "llama-3.2-3b-instruct",
                "messages": messages,
                "temperature": 0.7
            }
            response = requests.post(self.api_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                
                # Attempt to parse JSON if model follows instructions, else just use text
                try:
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        dialogue = parsed.get('dialogue', response_text)
                        intent = parsed.get('intent', "Idle")
                    else:
                        dialogue = response_text
                        intent = "Idle"
                except json.JSONDecodeError as e:
                    self.log_callback(f"[BRAIN-CHAT] JSON Parse Error: {e}")
                    dialogue = response_text
                    intent = "Idle"
                    
                self.chat_history.append({"role": "user", "content": user_text})
        
                if self.arm_controller and intent.lower() != "idle":
                    if getattr(self, "physical_interaction_enabled", False):
                        self.arm_controller.execute_intent(intent)
                    else:
                        self.log_callback(f"[BRAIN] Prevented Arm Movement '{intent}' because Physical Interaction is OFF.")
                    
                self.chat_history.append({"role": "assistant", "content": dialogue})
                if len(self.chat_history) > 6:
                    self.chat_history = self.chat_history[-6:]
                    
                self.tts.speak(dialogue)
                return dialogue
            else:
                return f"Error: {response.status_code}"
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
                        dialogue = "" # Default to silence if malformed
                        intent = "Idle"
                except json.JSONDecodeError:
                    dialogue = ""
                    intent = "Idle"
                    
                if dialogue:
                    self.log_event_internal(f"[PROACTIVE] {dialogue}")
                    self.tts.speak(dialogue)
                    if self.arm_controller and intent.lower() != "idle":
                        self.arm_controller.execute_intent(intent)
                        
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

