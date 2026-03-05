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

try:
    import face_recognition
    HAS_FACE_REC = True
except ImportError:
    HAS_FACE_REC = False

from profile_manager import ProfileManager

class VisionWorker:
    def __init__(self, log_callback, brain):
        self.log_callback = log_callback
        self.brain = brain
        self.running = False
        self.current_frame = None
        self.last_identity = "None"
        self._identity_timeout = 0
        self.face_metadata = [] # List of (location, name)
        self.object_metadata = [] # List of (box, label)
        self.latest_scene_description = "Waiting for analysis..."
        self.lock = threading.Lock()
        
        # LM Studio / Local API Settings
        self.api_url = "http://localhost:1234/v1/chat/completions" # Default LM Studio Port
        
        # Low power motion/face detection setup
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Real-Time Object Detection (YOLOv4-tiny)
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
            # Handle different OpenCV versions
            self.output_layers = self.net.getUnconnectedOutLayersNames()

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.last_motion_time = 0.0
        self.cooldown = 5.0 # Seconds between VLM triggers
        
        self.thread = None
        self._identity_timeout = 0.0
        self.frame_count = 0
        self.last_known_face_locations = []
        self.last_known_face_names = []
        
        self.log_callback("[VISION] Environment: Windows/Native. Using API-based inference.")
        
        self.profile_manager = ProfileManager()
        self.known_face_encodings = []
        self.known_face_names = []
        self._refresh_profiles()

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

    def _worker_loop(self):
        self.log_callback("[VISION] Background worker started. Waiting for motion/faces...")
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.current_frame is not None:
                    frame_to_process = self.current_frame
                    self.current_frame = None
                    
            if frame_to_process is not None:
                self._detect_triggers(frame_to_process)
                time.sleep(0.03) # Yield significantly to main UI thread (stops camera jitter)
            else:
                time.sleep(0.01) # Allow thread to yield but run at high FPS

    def _detect_triggers(self, frame):
        # 1. ALWAYS perform Face Recognition for UI Overlays
        metadata = []
        current_frame_identity = "None"
        self.frame_count += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Fast Haar face tracking
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        has_face = len(faces) > 0

        if has_face and HAS_FACE_REC and self.known_face_encodings:
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
            
            # Draw boxes based on the heavy face_recognition if available
            if self.last_known_face_locations:
                for (top, right, bottom, left), name in zip(self.last_known_face_locations, self.last_known_face_names):
                    metadata.append(((top*4, right*4, bottom*4, left*4), name))
                    if current_frame_identity == "None":
                        current_frame_identity = name
            else:
                # Fallback to fast Haar boxes if no recognition has run yet
                for (x, y, w, h) in faces:
                    metadata.append(((y, x+w, y+h, x), "Detecting..."))

        # 2. FAST Object Detection (YOLOv4-tiny)
        fast_objects = []
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
                fast_objects.append(((y, x, y + h, x + w), label))

        # Update real-time metadata for UI
        with self.lock:
            self.face_metadata = metadata
            self.object_metadata = fast_objects
            # Sticky identity
            if current_frame_identity != "None":
                self.last_identity = current_frame_identity
                self._identity_timeout = time.time()
            elif time.time() - self._identity_timeout > 2.0:
                self.last_identity = "None"

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
        self.log_callback("[VISION-VLM] Triggering background scene analysis...")
        scene_description = self._vlm_inference(frame)
        if scene_description:
            with self.lock:
                self.latest_scene_description = scene_description
            self.log_event_for_vlm(scene_description)
            if self.brain:
                self.brain.generate_dialogue_and_intent(scene_description, identity)

    def _vlm_inference(self, frame):
        """
        Core VLM API call with coordinate parsing.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        prompt = (
            "Describe the overall scene, the environment, and the main action happening. "
            "Keep it to 1-2 short sentences. Do not list standard objects or bounding boxes."
        )

        payload = {
            "model": "qwen2-vl-2b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 100
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                with self.lock:
                    self.latest_scene_description = content
                
                return content
            else:
                return None
        except Exception as e:
            self.log_callback(f"[VISION-VLM] Inference error: {e}")
            return None

    def log_event_for_vlm(self, description):
        # Helper to log and store description
        self.log_callback(f"[VISION-VLM] Scene Description: '{description}'")
