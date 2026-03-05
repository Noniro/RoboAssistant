import sys

def check_dependencies():
    required = ["customtkinter", "cv2", "PIL", "requests"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    
    if missing:
        print("ERROR: Missing required dependencies:", ", ".join(missing))
        print("Please ensure you have activated your virtual environment:")
        print("  .\\venv\\Scripts\\Activate.ps1")
        print("Then run: pip install -r requirements.txt")
        sys.exit(1)

check_dependencies()

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading

try:
    from vision_module import VisionWorker
    from brain_module import ReasoningBridge
    from arm_control import ArmController
    from hardware_utils import list_cameras, list_audio_devices
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import internal module. {e}")
    print("Ensure you are running from the project root directory.")
    sys.exit(1)

import json
import os

class RobotSupervisorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Yuval's Robotic Lab Assistant Supervisor")
        self.geometry("1200x800")
        ctk.set_appearance_mode("System")
        
        self.settings_file = "config.json"
        
        from typing import Dict, Any
        self.settings: Dict[str, Any] = self.load_settings()
        
        # UI Layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=3) # Camera feed
        self.grid_columnconfigure(1, weight=1) # Log/Controls

        # Camera frame
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.camera_frame, text="")
        self.video_label.pack(expand=True, fill="both")

        # Sidebar frame
        self.sidebar_frame = ctk.CTkFrame(self)
        self.sidebar_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Thought Log
        self.log_label = ctk.CTkLabel(self.sidebar_frame, text="Thought Log", font=ctk.CTkFont(size=20, weight="bold"))
        self.log_label.pack(pady=10)
        
        self.thought_log = ctk.CTkTextbox(self.sidebar_frame, state="disabled", height=150)
        self.thought_log.pack(expand=False, fill="x", padx=10, pady=5)
        
        # Chat History Window
        self.chat_label = ctk.CTkLabel(self.sidebar_frame, text="Chat with Brain", font=ctk.CTkFont(size=16, weight="bold"))
        self.chat_label.pack(pady=(10, 0))
        
        self.chat_history = ctk.CTkTextbox(self.sidebar_frame, state="disabled")
        self.chat_history.pack(expand=True, fill="both", padx=10, pady=5)
        
        # Chat Input Area
        self.chat_input_frame = ctk.CTkFrame(self.sidebar_frame)
        self.chat_input_frame.pack(fill="x", padx=10, pady=5)
        
        self.user_input = ctk.CTkEntry(self.chat_input_frame, placeholder_text="Type to chat...")
        self.user_input.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.user_input.bind("<Return>", lambda e: self.send_chat())
        
        self.send_btn = ctk.CTkButton(self.chat_input_frame, text="Send", width=60, command=self.send_chat)
        self.send_btn.pack(side="right")

        # Settings Toggle Button
        self.settings_btn = ctk.CTkButton(self.sidebar_frame, text="⚙ Hardware Settings", command=self.toggle_settings)
        self.settings_btn.pack(pady=5, padx=20, fill="x")

        # Profiles Button
        self.profiles_btn = ctk.CTkButton(self.sidebar_frame, text="👤 Manage Profiles", command=self.open_profiles)
        self.profiles_btn.pack(pady=5, padx=20, fill="x")

        # Modules initialization (Need brain for TTS voices)
        from arm_control import ArmController
        from brain_module import ReasoningBridge
        from vision_module import VisionWorker
        
        self.arm = ArmController(self.log_event)
        self.brain = ReasoningBridge(self.log_event, self.arm)
        self.vision = VisionWorker(self.log_event, self.brain)
        self.brain.vision_worker = self.vision # Link them back

        # Collapsible Settings Frame
        self.settings_visible = False
        self.settings_frame = ctk.CTkFrame(self.sidebar_frame)
        
        # Camera Selection
        ctk.CTkLabel(self.settings_frame, text="Camera").pack(pady=(5,0))
        self.cam_list = list_cameras()
        self.cam_names = [c["name"] for c in self.cam_list] if self.cam_list else ["No Camera Found"]
        self.cam_dropdown = ctk.CTkOptionMenu(self.settings_frame, values=self.cam_names, command=self.change_camera)
        if self.settings.get("camera_name") in self.cam_names:
            self.cam_dropdown.set(self.settings["camera_name"])
        self.cam_dropdown.pack(pady=5, padx=10, fill="x")
        
        # Audio Selection (Placeholders)
        mics, speakers = list_audio_devices()
        
        ctk.CTkLabel(self.settings_frame, text="Microphone").pack(pady=(5,0))
        self.mic_names = [m["name"] for m in mics] if mics else ["None"]
        self.mic_dropdown = ctk.CTkOptionMenu(self.settings_frame, values=self.mic_names, command=lambda v: self.save_settings())
        if self.settings.get("mic_name") in self.mic_names:
            self.mic_dropdown.set(self.settings["mic_name"])
        self.mic_dropdown.pack(pady=5, padx=10, fill="x")
        
        ctk.CTkLabel(self.settings_frame, text="Speaker").pack(pady=(5,0))
        self.spk_names = [s["name"] for s in speakers] if speakers else ["None"]
        self.spk_dropdown = ctk.CTkOptionMenu(self.settings_frame, values=self.spk_names, command=lambda v: self.save_settings())
        if self.settings.get("spk_name") in self.spk_names:
            self.spk_dropdown.set(self.settings["spk_name"])
        self.spk_dropdown.pack(pady=5, padx=10, fill="x")

        # Voice Selection
        ctk.CTkLabel(self.settings_frame, text="Robot Voice").pack(pady=(5,0))
        self.voices = self.brain.tts.get_available_voices()
        self.voice_names = [v["name"] for v in self.voices] if self.voices else ["Default"]
        self.voice_dropdown = ctk.CTkOptionMenu(self.settings_frame, values=self.voice_names, command=self.change_voice)
        saved_voice_name = self.settings.get("voice_name")
        if saved_voice_name in self.voice_names:
            self.voice_dropdown.set(saved_voice_name)
            # Apply saved voice immediately
            for v in self.voices:
                if v["name"] == saved_voice_name:
                    self.brain.tts.set_voice(v["id"])
                    break
        elif self.voice_names:
            self.voice_dropdown.set(self.voice_names[0])
            
        self.voice_dropdown.pack(pady=5, padx=10, fill="x")

        # Recognized Person Label
        self.person_label = ctk.CTkLabel(self.sidebar_frame, text="Recognized: None", font=ctk.CTkFont(size=14, slant="italic"))
        self.person_label.pack(pady=5)

        # Start/Stop Button
        self.control_btn = ctk.CTkButton(self.sidebar_frame, text="Start System", command=self.toggle_system, fg_color="green")
        self.control_btn.pack(pady=20, padx=20, fill="x")
        
        self.log_event("[SYSTEM] Initializing camera...")
        self.cap = self._init_camera()
        
        self.system_running = False
        
        self.update_camera_feed()

    def _init_camera(self):
        # On Windows, DirectShow (CAP_DSHOW) is often much faster and more reliable
        import platform
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        
        # Try saved index first
        saved_index = self.settings.get("camera_index")
        indices_to_try = [saved_index] + [i for i in range(5) if i != saved_index] if saved_index is not None else range(5)

        for index in indices_to_try:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                # Microsoft LifeCam settings
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                ret, _ = cap.read()
                if ret:
                    self.log_event(f"[SYSTEM] Camera connected on index {index}.")
                    return cap
                cap.release()
                
        self.log_event("[SYSTEM] ERROR: No working camera found.")
        return cv2.VideoCapture(0)

    def log_event(self, message):
        self.thought_log.configure(state="normal")
        self.thought_log.insert("end", message + "\n")
        self.thought_log.configure(state="disabled")
        self.thought_log.see("end")

    def toggle_settings(self):
        if self.settings_visible:
            self.settings_frame.pack_forget()
            # Reinstate the log to take up space
            self.thought_log.pack(expand=True, fill="both", padx=10, pady=10)
            self.settings_btn.configure(text="⚙ Hardware Settings")
        else:
            # Hide log temporarily to show settings
            self.thought_log.pack_forget()
            self.settings_frame.pack(expand=True, fill="both", padx=10, pady=10, before=self.control_btn)
            self.settings_btn.configure(text="✖ Close Settings")
        self.settings_visible = not self.settings_visible

    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading settings: {e}")
        return {}

    def save_settings(self):
        self.settings["camera_name"] = self.cam_dropdown.get()
        # Find camera index for name
        for c in self.cam_list:
            if c["name"] == self.settings["camera_name"]:
                self.settings["camera_index"] = c["index"]
                break
        
        self.settings["mic_name"] = self.mic_dropdown.get()
        self.settings["spk_name"] = self.spk_dropdown.get()
        self.settings["voice_name"] = self.voice_dropdown.get()

        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            self.log_event(f"[SYSTEM] ERROR: Could not save settings: {e}")

    def change_camera(self, choice):
        # Find index
        new_index = 0
        for c in self.cam_list:
            if c["name"] == choice:
                new_index = c["index"]
                break
        
        self.log_event(f"[SYSTEM] Switching to camera index {new_index}...")
        self.save_settings()
        
        # Release old
        if self.cap:
            self.cap.release()
            
        # Init new
        import platform
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(new_index, backend)
        
        if not self.cap.isOpened():
             self.log_event(f"[SYSTEM] ERROR: Could not open camera {new_index}.")
        else:
             self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
             self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
             self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def change_voice(self, choice):
        self.save_settings()
        for v in self.voices:
            if v["name"] == choice:
                self.log_event(f"[SYSTEM] Switched TTS voice to {choice}.")
                self.brain.tts.set_voice(v["id"])
                break

    def send_chat(self):
        user_text = self.user_input.get()
        if not user_text.strip():
            return
        
        self.user_input.delete(0, "end")
        self.append_to_chat(f"You: {user_text}")
        
        # Run inference in a thread to keep UI responsive
        identity = self.vision.last_identity
        threading.Thread(target=self._manual_chat_thread, args=(user_text, identity), daemon=True).start()

    def _manual_chat_thread(self, text, identity):
        response = self.brain.manual_interact(text, identity=identity)
        self.append_to_chat(f"Brain: {response}")

    def append_to_chat(self, message):
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", message + "\n\n")
        self.chat_history.configure(state="disabled")
        self.chat_history.see("end")

    def toggle_system(self):
        self.system_running = not self.system_running
        if self.system_running:
            self.control_btn.configure(text="Stop System")
            self.log_event("[SYSTEM] Starting AI routines...")
            self.vision.start()
        else:
            self.control_btn.configure(text="Start System")
            self.log_event("[SYSTEM] Stopping AI routines...")
            self.vision.stop()

    def update_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # To prevent freezing, the vision module will process the frame asynchronously
            if self.system_running:
                self.vision.process_frame(frame)
                
            # Draw Visual Overlays (Faces)
            with self.vision.lock:
                faces = list(self.vision.face_metadata)
                objects = list(self.vision.object_metadata)
                vlm_desc = self.vision.latest_scene_description

            for (top, right, bottom, left), name in faces:
                # Draw Box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw Label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

            for (box, label) in objects:
                ymin, xmin, ymax, xmax = box
                # YOLOv4-tiny returns actual pixel coordinates based on the frame size (640x480)
                # No scaling out of 1000 needed anymore.
                top, left, bottom, right = int(ymin), int(xmin), int(ymax), int(xmax)
                
                # Ensure within frame bounds
                top, left = max(0, top), max(0, left)
                bottom, right = min(480, bottom), min(640, right)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 2) # Orange for objects
                cv2.putText(frame, label, (left + 5, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # Draw VLM Description Overlay
            cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1) # Black bar at top
            cv2.putText(frame, f"VLM Analysis: {vlm_desc}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Convert to CTK image for display
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ctk.CTkImage(light_image=img, size=(640, 480))
            self.video_label.configure(image=imgtk, text="")
            self.video_label.image = imgtk

            # Update recognized person
            identity = self.vision.last_identity or "None"
            self.person_label.configure(text=f"Recognized: {identity}")
        else:
            self.video_label.configure(text="CAMERA FEED OFFLINE\n(Check WSL Connection)", text_color="red")
            
        # 30 fps -> ~33 ms delay
        self.after(33, self.update_camera_feed)

    def open_profiles(self):
        ProfileManagerWindow(self)

    def on_closing(self):
        self.cap.release()
        self.vision.stop()
        self.destroy()

class ProfileManagerWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("User Profile Manager")
        self.geometry("600x500")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self, text="Manage User Profiles", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, pady=10)

        # Profile List
        self.profile_frame = ctk.CTkScrollableFrame(self)
        self.profile_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.refresh_profile_list()

        # Add Profile Button
        self.add_btn = ctk.CTkButton(self, text="+ Create New Profile", command=self.create_profile_dialog)
        self.add_btn.grid(row=2, column=0, pady=10)

    def refresh_profile_list(self):
        for widget in self.profile_frame.winfo_children():
            widget.destroy()
            
        profiles = self.parent.brain.profile_manager.profiles
        for name, data in profiles.items():
            f = ctk.CTkFrame(self.profile_frame)
            f.pack(fill="x", pady=5, padx=5)
            
            ctk.CTkLabel(f, text=name, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
            
            has_enc = "✅ Face Set" if data["encoding"] is not None else "❌ No Face"
            ctk.CTkLabel(f, text=has_enc).pack(side="left", padx=10)
            
            ctk.CTkButton(f, text="Register Face", width=100, command=lambda n=name: self.register_face(n)).pack(side="right", padx=5)
            ctk.CTkButton(f, text="Edit Prompt", width=80, command=lambda n=name: self.edit_prompt(n)).pack(side="right", padx=5)
            ctk.CTkButton(f, text="Del", width=40, fg_color="red", command=lambda n=name: self.delete_profile(n)).pack(side="right", padx=5)

    def create_profile_dialog(self):
        name = ctk.CTkInputDialog(text="Enter profile name:", title="New Profile").get_input()
        if name:
            self.parent.brain.profile_manager.add_profile(name, f"You are talking to {name}.")
            self.refresh_profile_list()

    def edit_prompt(self, name):
        profile = self.parent.brain.profile_manager.get_profile(name)
        new_prompt = ctk.CTkInputDialog(text=f"Edit behavior prompt for {name}:", title="Edit Prompt").get_input()
        if new_prompt:
            self.parent.brain.profile_manager.add_profile(name, new_prompt, profile["encoding"])
            self.refresh_profile_list()

    def delete_profile(self, name):
        self.parent.brain.profile_manager.delete_profile(name)
        self.refresh_profile_list()

    def register_face(self, name):
        """
        Captures a frame from the live camera and saves the encoding.
        """
        import face_recognition
        import numpy as np
        
        try:
            ret, frame = self.parent.cap.read()
            if not ret or frame is None:
                self.parent.log_event("[PROFILES] ERROR: Could not access camera.")
                return
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Find all faces in the image
            face_locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if encodings:
                encoding = encodings[0]
                profile = self.parent.brain.profile_manager.get_profile(name)
                if profile is None:
                    # Fallback if dialog just added it but it didn't save right
                    prompt = f"You are talking to {name}."
                else:
                    prompt = profile["prompt"]
                    
                self.parent.brain.profile_manager.add_profile(name, prompt, encoding)
                self.parent.log_event(f"[PROFILES] Successfully registered face for {name} ({len(encodings)} detected).")
                # Tell vision worker to refresh
                self.parent.vision._refresh_profiles()
                self.refresh_profile_list()
            else:
                self.parent.log_event("[PROFILES] ERROR: No face detected in frame. Please ensure you are visible and well-lit.")
        except Exception as e:
            import traceback
            error_str = traceback.format_exc()
            self.parent.log_event(f"[PROFILES] FATAL ERROR during capture:\n{error_str}")
            print(f"Face Capture Crash:\n{error_str}")

if __name__ == "__main__":
    app = RobotSupervisorApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

