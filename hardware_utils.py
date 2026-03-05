import cv2
import sounddevice as sd
import platform

def list_cameras():
    """
    Returns a list of available camera indices and names.
    """
    camera_list = []
    # On Windows, DirectShow is better for enumeration
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    
    # Check first 5 indices
    for i in range(5):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                camera_list.append({"index": i, "name": f"Camera {i}"})
            cap.release()
    return camera_list

def list_audio_devices():
    """
    Returns lists of available microphones and speakers.
    """
    devices = sd.query_devices()
    mics = []
    speakers = []
    
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            mics.append({"index": i, "name": dev['name']})
        if dev['max_output_channels'] > 0:
            speakers.append({"index": i, "name": dev['name']})
            
    return mics, speakers

if __name__ == "__main__":
    print("Cameras:", list_cameras())
    mics, speakers = list_audio_devices()
    print("Mics:", mics)
    print("Speakers:", speakers)
