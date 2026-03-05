import cv2
import platform

def test_camera():
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    found = False
    for index in range(5):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"SUCCESS: Working camera found on index {index}")
                found = True
                cap.release()
                break
            cap.release()
    if not found:
        print("FAILURE: No working camera found.")

if __name__ == "__main__":
    test_camera()
