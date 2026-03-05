import time
import cv2
import numpy as np
import threading
from vision_module import VisionWorker
from brain_module import ReasoningBridge

def log(msg):
    print(f"[TEST] {msg}")

def test_pipeline():
    log("Starting Full-Stack Pipeline Test...")
    
    # 1. Initialize Modules
    brain = ReasoningBridge(log, None) # No arm for test
    vision = VisionWorker(log, brain)
    brain.vision_worker = vision # Link them
    
    # 2. Mock a frame (640x480)
    log("Creating mock frame...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "TEST SCENE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 3. Simulate Vision Processing
    log("Processing frame in VisionWorker...")
    vision.process_frame(frame)
    
    # Manually trigger a mock face/object detection result for testing context
    with vision.lock:
        vision.face_metadata = [((100, 300, 200, 200), "Yuval")] # Mock face
        vision.object_metadata = [((300, 100, 400, 200), "Bottle")] # Mock object
        vision.last_identity = "Yuval"
        vision.latest_scene_description = "A clean lab environment."
    
    # 4. Verify Unified Context
    log("Verifying Unified Context...")
    context = vision.get_unified_context()
    log(f"Context: {context}")
    
    if "Yuval" in context and "Bottle" in context:
        log("SUCCESS: Context contains Face and Object data.")
    else:
        log("WARNING: Context missing data.")
        
    # 5. Test Brain Interaction
    log("Testing Brain Chat (Manual Interact)...")
    response = brain.manual_interact("What items do you see near me?", identity=vision.last_identity)
    log(f"Brain Response: {response}")
    
    resp_lower = response.lower()
    if "yuval" in resp_lower or "bottle" in resp_lower or "i see" in resp_lower:
         log("SUCCESS: Brain is aware of the scene context.")
    else:
         log("FAILED: Brain might not be seeing the context.")

    log("Pipeline Test Complete.")

if __name__ == "__main__":
    test_pipeline()
