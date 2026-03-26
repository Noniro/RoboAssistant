try:
    import cv2
except ImportError:
    print("ERROR: OpenCV (cv2) not found. Please install it with 'pip install opencv-python'.")
    raise

import time
import requests
import base64
import io
import threading
import numpy as np
from PIL import Image

# [DEBUG] Temporarily disabled face_recognition to allow VLA-only testing
HAS_FACE_REC = False
class MockFaceRec:
    def face_locations(self, *args, **kwargs): return []
    def face_encodings(self, *args, **kwargs): return []
    def compare_faces(self, *args, **kwargs): return []
    def face_distance(self, *args, **kwargs): return []
face_recognition = MockFaceRec()
print("[INFO] Face recognition manually DISABLED to skip model-loading dependencies.")

from profile_manager import ProfileManager

class VisionWorker:
    def __init__(self, log_callback, brain=None, event_callback=None):
        self.log_callback = log_callback
        self.brain = brain
        self.event_callback = event_callback
        self.running = False
        self.current_frame = None
        self.current_gripper_frame = None
        self.target_objects_to_track = []
        self.last_identity = "None"
        self._identity_timeout = 0
        self.face_metadata = [] # List of (location, name)
        self.object_metadata = [] # List of (box, label) for main camera
        self.gripper_object_metadata = [] # List of (box, label) for gripper camera
        self.latest_scene_description = "Waiting for analysis..."
        self.lock = threading.RLock()
        
        # Tracking smoothing (EMA)
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.ema_alpha = 0.6 # Faster response (0.1 = smooth/slow, 0.9 = fast/jittery)
        
        # LM Studio / Local API Settings
        self.api_url = None # Disabled - Using LOCAL UNIFIED VLA 
        
        # Low power motion/face detection setup
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Models will be loaded dynamically by the ModeManager
        self.net = None
        self.output_layers = []
        self.coco_classes = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.profile_manager = ProfileManager()

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.last_motion_time = 0.0
        self.cooldown = 5.0 # Seconds between VLM triggers
        
        self.thread = None
        self._identity_timeout = 0.0
        self.frame_count = 0
        self.last_known_face_locations = []
        self.last_known_face_names = []
        
        # Proactive Interaction State
        self.last_stable_identity = "None"
        self.identity_stability_counter = 0
        self.potential_identity = "None"
        self.last_pulse_time = time.time()
        self.pulse_interval = 15.0 # Seconds between context refreshes when person present
        
        # Distraction Tracking
        self.last_distraction_time = 0.0
        self.active_distractions = []
        
        self.proactive_pulse_enabled = True # Configured by main_ui
        
        self.log_callback("[VISION] Environment: Windows/Native. Using API-based inference.")
        
        self.log_callback("[VISION] Environment: Windows/Native. Using API-based inference.")

    def load_models(self, required_models):
        self.log_callback(f"[VISION] Loading required models: {required_models}")
        with self.lock:
            # Load YOLO if needed
            if "yolo" in required_models:
                if self.net is None:
                    try:
                        self.log_callback("[VISION] Loading YOLOv4-tiny...")
                        self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
                        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        else:
                            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                            
                        with open("coco.names", "r") as f:
                            self.coco_classes = [line.strip() for line in f.readlines()]
                            
                        try:
                            layer_names = self.net.getLayerNames()
                            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                        except:
                            self.output_layers = self.net.getUnconnectedOutLayersNames()
                    except Exception as e:
                        self.log_callback(f"[VISION] YOLO loading failed (missing weights?): {e}")
                        self.net = None
            else:
                self.net = None # Unload to save VRAM

            # Load Face Recognition if needed
            if "face_rec" in required_models:
                self.log_callback("[VISION] Loading Face Recognition Profiles...")
                self._refresh_profiles()
            else:
                self.known_face_encodings = []
                self.known_face_names = []

    def _refresh_profiles(self):
        """Reloads profiles and their encodings."""
        with self.lock:
            self.profile_manager.load_profiles()
            self.known_face_encodings = []
            self.known_face_names = []
            for name, data in self.profile_manager.profiles.items():
                if data["encoding"] is not None:
                    self.known_face_encodings.append(data["encoding"])
                    self.known_face_names.append(name)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def process_frame(self, frame):
        with self.lock:
            self.current_frame = frame.copy()

    def process_gripper_frame(self, frame, targets):
        with self.lock:
            self.current_gripper_frame = frame.copy()
            self.target_objects_to_track = targets

    def _worker_loop(self):
        self.log_callback("[VISION] Background worker started. Waiting for motion/faces...")
        while self.running:
            frame_to_process = None
            gripper_frame_to_process = None
            targets = []
            
            with self.lock:
                if self.current_frame is not None:
                    frame_to_process = self.current_frame
                    self.current_frame = None
                if self.current_gripper_frame is not None:
                    gripper_frame_to_process = self.current_gripper_frame
                    self.current_gripper_frame = None
                    targets = tuple(self.target_objects_to_track)
                    
            if frame_to_process is not None:
                self._detect_triggers(frame_to_process)
                
            if gripper_frame_to_process is not None:
                self._track_gripper_objects(gripper_frame_to_process, targets)
                
            if frame_to_process is not None or gripper_frame_to_process is not None:
                time.sleep(0.02) # Yield significantly to main UI thread (stops camera jitter)
            else:
                time.sleep(0.01) # Allow thread to yield but run at high FPS

    def _track_gripper_objects(self, frame, targets):
        """Detect all objects in gripper frame. Store all for display, track only objects in 'targets'."""
        if self.net is None:
            with self.lock:
                self.gripper_object_metadata = []
            return
            
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        
        # Store ALL detected objects for display on the gripper feed
        detected = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = self.coco_classes[class_ids[i]]
                detected.append(((int(y), int(x), int(y + h), int(x + w)), label))
        
        with self.lock:
            self.gripper_object_metadata = detected
            
        # Use ONLY target-matching detections to drive turret movement
        arm_ctrl = getattr(self.brain, 'arm_controller', None)
        if arm_ctrl and hasattr(arm_ctrl, 'adjust_tracking') and len(indices) > 0:
            target_matches = [(boxes[i], i) for i in indices.flatten()
                              if self.coco_classes[class_ids[i]] in targets]
            
            if target_matches:
                best_box, _ = target_matches[0]
                x, y, w, h = best_box
                center_x = x + w / 2
                center_y = y + h / 2
                error_x = center_x - (width / 2)
                error_y = center_y - (height / 2)
                
                # Apply EMA smoothing to the errors to reduce YOLO jitter
                self.last_error_x = (self.ema_alpha * error_x) + ((1.0 - self.ema_alpha) * self.last_error_x)
                self.last_error_y = (self.ema_alpha * error_y) + ((1.0 - self.ema_alpha) * self.last_error_y)
                
                arm_ctrl.adjust_tracking(self.last_error_x, self.last_error_y, width, height)
            else:
                # Target lost - reset PID to prevent windup drift
                if hasattr(arm_ctrl, 'reset_pids'):
                    arm_ctrl.reset_pids()
        elif arm_ctrl and hasattr(arm_ctrl, 'reset_pids'):
            # No detections at all - reset PID
            arm_ctrl.reset_pids()
                
    def _detect_triggers(self, frame):
        # 1. ALWAYS perform Face Recognition for UI Overlays
        metadata = []
        current_frame_identity = "None"
        self.frame_count += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Fast Haar face tracking
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        has_face = len(faces) > 0

        if has_face and HAS_FACE_REC and getattr(self, "known_face_encodings", []):
            try:
                # Only run heavy face encoding once every 10 frames
                if self.frame_count % 10 == 0:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    self.last_known_face_locations = face_locations
                    self.last_known_face_names = []
                    
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = "Unknown"
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = self.known_face_names[best_match_index]
                        self.last_known_face_names.append(name)
            except Exception as e:
                # Catch "Please install face_recognition_models" or other errors
                # and disable face rec for the rest of the session
                self.log_callback(f"[VISION] Face recognition failed/uninstalled models: {e}")
                globals()["HAS_FACE_REC"] = False
            
            # Draw boxes based on the heavy face_recognition if available
            if self.last_known_face_locations:
                for (top, right, bottom, left), name in zip(self.last_known_face_locations, self.last_known_face_names):
                    metadata.append(((top*4, right*4, bottom*4, left*4), name))
                    if current_frame_identity == "None":
                        current_frame_identity = name
            else:
                # Fallback to fast Haar boxes if no recognition has run yet
                for (x, y, w, h) in faces:
                    metadata.append(((int(y), int(x+w), int(y+h), int(x)), "Detecting..."))

        # 2. FAST Object Detection (YOLOv4-tiny)
        fast_objects = []
        if self.net is not None:
            height, width = frame.shape[:2]
            
            # Fast DNN trigger (skip frames to save CPU if needed, but YOLOv4-tiny is fast)
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
        
            class_ids = []
            confidences = []
            boxes = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.4: # Faster threshold
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        
            # Apply Non-Max Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    label = self.coco_classes[class_ids[i]]
                    # Map back to [ymin, xmin, ymax, xmax] format the UI expects
                    fast_objects.append(((int(y), int(x), int(y + h), int(x + w)), label))

        # Update real-time metadata for UI
        with self.lock:
            self.face_metadata = metadata
            self.object_metadata = fast_objects
            self.last_identity = current_frame_identity
        
        # 3. PROACTIVE EVENT DETECTION
        # Identity Stability Check (requires 15 consistent frames)
        if current_frame_identity == self.potential_identity and current_frame_identity != "None":
            self.identity_stability_counter += 1
        else:
            self.potential_identity = current_frame_identity
            self.identity_stability_counter = 0

        if self.identity_stability_counter >= 15:
            if current_frame_identity != self.last_stable_identity:
                self.log_callback(f"[VISION] Identity stabilized: {current_frame_identity}")
                self.last_stable_identity = current_frame_identity
                if self.event_callback:
                    self.event_callback("NEW_PERSON", current_frame_identity)
            self.identity_stability_counter = 0 # Reset but keep stable state

        # Clear stable identity if no face seen for a while
        if current_frame_identity == "None":
            self._identity_timeout += 0.03
            if self._identity_timeout > 5.0: # 5 seconds of absence
                self.last_stable_identity = "None"
        else:
            self._identity_timeout = 0.0

        # Periodic Scene Pulse (VLM/Brain Refresh)
        sees_person_or_face = current_frame_identity != "None" or len(faces) > 0 or any("person" in label.lower() for _, label in fast_objects)
        
        if sees_person_or_face and self.proactive_pulse_enabled:
            if time.time() - self.last_pulse_time > self.pulse_interval:
                self.log_callback("[VISION] Triggering periodic scene pulse...")
                self.last_pulse_time = time.time()
                identity_to_pass = self.last_stable_identity if self.last_stable_identity != "None" else "Unknown"
                if self.event_callback:
                    self.event_callback("PERIODIC_SCAN", identity_to_pass)

        # Distraction Detection
        if self.event_callback:
            # Check if we should even look for distractions (we need a way to pass down active config, 
            # but for now we'll just emit the objects and let the brain/UI filter them)
            current_objects = [label for _, label in fast_objects]
            if current_objects:
                # We emit the event and the receiver checks if it's a known distraction and handles cooldowns.
                # To prevent flooding, we'll only emit if the set of detected objects has changed, 
                # or if it's been a few seconds.
                current_time = time.time()
                if current_objects != self.active_distractions or (current_time - self.last_distraction_time) > 2.0:
                    self.active_distractions = current_objects.copy()
                    self.last_distraction_time = current_time
                    self.event_callback("OBJECTS_DETECTED", {"objects": current_objects, "identity": current_frame_identity})

        # Note: Background VLM polling has been completely removed.
        # VLM analysis is now 100% on-demand via `get_immediate_description`.

    def get_unified_context(self):
        """
        Aggregates all current knowledge into a single scene description.
        Returns Foreground (Faces+Objects) + Background (VLM).
        """
        with self.lock:
            faces = [f[1] for f in self.face_metadata if f[1] != "Unknown"]
            objs = [o[1] for o in self.object_metadata]
            bg = self.latest_scene_description
            identity = self.last_identity
        
        context = []
        if identity and identity != "None":
            context.append(f"I am looking at {identity}.")
        elif faces:
            context.append("I see an unknown person.")
        else:
            context.append("No one is clearly visible.")
            
        if objs:
            context.append(f"Nearby objects include: {', '.join(set(objs))}.")
            
            if bg and "waiting" not in bg.lower():
                context.append(f"Background context: {bg}")
                
            return " ".join(context)

    def get_structured_reality(self):
        """
        Synchronously captures a frame, gets a VLM description, and packages 
        the unified reality (Faces + YOLO Coordinates + VLM Statement) into a JSON dictionary.
        """
        frame = None
        identity = "Unknown"
        faces = []
        ui_objects = []
        
        with self.lock:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
            identity = self.last_identity if self.last_identity != "None" else "Unknown"
            faces = [(f[0], f[1]) for f in self.face_metadata if f[1] != "Unknown"]
            ui_objects = list(self.object_metadata) # ((ymin, xmin, ymax, xmax), label)

        vlm_desc = "No frame available."
        if frame is not None:
            self.log_callback("[VISION] Capturing frame for Structured Sensor Fusion...")
            vlm_desc = self._vlm_inference(frame) or "Failed to generate visual description."

        spatial_data = []
        for box, label in ui_objects:
            ymin, xmin, ymax, xmax = box
            # Calculate pixel center points
            x_center = int((xmin + xmax) / 2)
            y_center = int((ymin + ymax) / 2)
            spatial_data.append({
                "label": label,
                "coordinates": [x_center, y_center]
            })

        # Optionally add known faces to spatial data
        for face_box, face_name in faces:
            ymin, xmax, ymax, xmin = face_box
            x_center = int((xmin + xmax) / 2)
            y_center = int((ymin + ymax) / 2)
            spatial_data.append({
                "label": f"Face: {face_name}",
                "coordinates": [x_center, y_center]
            })

        reality_dict = {
            "User Identity": identity,
            "Contextual Description": vlm_desc,
            "Spatial Data": spatial_data
        }
        
        return reality_dict

    def get_immediate_description(self):
        """
        Synchronously captures a frame and gets a VLM description.
        Useful for manual chat interactions.
        """
        frame = None
        with self.lock:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
        
        if frame is None:
            return "No frame available."
            
        return self._vlm_inference(frame)

    def _trigger_vlm(self, frame, identity=None):
        # Background scene analysis is handled by the Unified VLA on request
        pass

    def _vlm_inference(self, frame):
        """LOCAL UNIFIED VLA handles this now via brain_module."""
        return "Using Local Unified OpenVLA Brain."

    def log_event_for_vlm(self, description):
        # Helper to log and store description
        self.log_callback(f"[VISION-VLM] Scene Description: '{description}'")
