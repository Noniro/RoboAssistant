import json
import os
import numpy as np

class ProfileManager:
    def __init__(self, data_path="profiles.json"):
        self.data_path = data_path
        self.profiles = {}
        self.load_profiles()

    def load_profiles(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, "r") as f:
                raw_data = json.load(f)
                # Convert list back to numpy arrays for face recognition
                for name, data in raw_data.items():
                    encoding = data.get("encoding")
                    self.profiles[name] = {
                        "name": data.get("name", name),
                        "prompt": data.get("prompt", ""),
                        "encoding": np.array(encoding) if encoding is not None else None
                    }
        else:
            # Default profiles
            self.profiles = {
                "Yuval": {
                    "name": "Yuval",
                    "prompt": "You are greeting Yuval, your creator. Be extremely helpful, respectful, and slightly informal.",
                    "encoding": None
                }
            }
            self.save_profiles()

    def save_profiles(self):
        serializable_data = {}
        for name, data in self.profiles.items():
            enc = data["encoding"]
            serializable_data[name] = {
                "name": data["name"],
                "prompt": data["prompt"],
                "encoding": enc.tolist() if enc is not None else None
            }
        with open(self.data_path, "w") as f:
            json.dump(serializable_data, f, indent=4)

    def get_profile(self, name):
        return self.profiles.get(name)

    def add_profile(self, name, prompt, encoding=None):
        self.profiles[name] = {
            "name": name,
            "prompt": prompt,
            "encoding": encoding
        }
        self.save_profiles()

    def delete_profile(self, name):
        if name in self.profiles:
            self.profiles.pop(name, None)
            self.save_profiles()

    def get_all_names(self):
        return list(self.profiles.keys())
