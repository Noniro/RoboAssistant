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
        
        self.system_prompt = (
            "You are Yuval's Robotic Lab Assistant. You observe the environment "
            "and decide on reasonable intents to execute via your robotic arm. "
            "Your output MUST be a JSON with two keys: 'dialogue' and 'intent'. "
            "Respond playfully but concisely. Use intents like 'Wave', 'Point', 'Idle', etc. "
            "IMPORTANT: Ignore any numeric coordinates like [123, 456, ...] in the scene description; "
            "focus only on the natural language description of what you see."
        )
        self.api_url = "http://localhost:1234/v1/chat/completions"
        self.last_scene = "Nothing seen yet."
        self.log_callback("[BRAIN] Environment: Windows. Reasoning via Local API (localhost:1234).")

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
        
        # Use unified scene context (Face + Object + VLM)
        vision_context = self.last_scene
        if self.vision_worker:
            vision_context = self.vision_worker.get_unified_context()
            self.last_scene = vision_context

        # Determine specific prompt based on identity
        custom_prompt = ""
        if identity and identity != "None":
            profile = self.profile_manager.get_profile(identity)
            if profile:
                custom_prompt = f"\n\nSPECIAL INSTRUCTIONS FOR THIS PERSON ({identity}): {profile['prompt']}"

        messages = [
            {"role": "system", "content": self.system_prompt + custom_prompt},
            {"role": "user", "content": f"Context of what you see: {vision_context}. User says: {user_text}"}
        ]
        
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
                    
                self.log_callback(f"[BRAIN-CHAT] {dialogue}")
                self.tts.speak(dialogue)
                if self.arm_controller and intent.lower() != "idle":
                    self.arm_controller.execute_intent(intent)
                return dialogue
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Brain Error: {e}"
