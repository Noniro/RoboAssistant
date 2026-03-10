import json
import os

class ModeManager:
    MODES = {
        "General": {
            "required_models": ["yolo", "face_rec", "vlm"],
            "base_prompt": "You are a helpful and playful lab assistant.",
            "allow_physical_interaction": True,
            "proactive_pulse": False # Only speaks when asked
        },
        "Study": {
            "required_models": ["yolo"], # No VLM needed, saves VRAM
            "distraction_objects": ["cell phone", "remote"],
            "action_level": 1, # 1: Sarcastic remark, 2: Arm "no-no", 3: Script
            "cooldown_seconds": 30, # Time to stay silent after a reprimand
            "proactive_pulse": True # Needs to monitor for distractions
        },
        "Security": {
            "required_models": ["yolo", "face_rec"], # No VLM needed
            "enable_tracking": True,
            "threat_target": "unknown_faces", # "unknown_faces", "everyone", "specific_profiles"
            "patrol_sweep": True,
            "proactive_pulse": True # Needs to monitor for threats
        }
    }

    def __init__(self, config_file="mode_config.json"):
        self.config_file = config_file
        self.current_mode = "General"
        self.config = self._load_config()

    def _load_config(self):
        """Loads custom configurations, falling back to defaults."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    saved_config = json.load(f)
                    
                    # Merge saved with defaults to ensure all keys exist
                    merged = self.MODES.copy()
                    for mode, settings in saved_config.items():
                        if mode in merged:
                            merged[mode].update(settings)
                    return merged
            except Exception as e:
                print(f"Error loading mode config: {e}")
        return self.MODES.copy()

    def save_config(self):
        """Saves current configurations to disk."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving mode config: {e}")

    def set_mode(self, mode_name):
        if mode_name in self.config:
            self.current_mode = mode_name
            return True
        return False

    def get_current_settings(self):
        return self.config[self.current_mode]
