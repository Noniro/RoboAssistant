import cv2
import numpy as np

class CameraStitcher:
    """
    Utility to take multiple camera frames (High, Side, Wrist) and stitch them 
    into a single 'Panorama' or Grid image for Unified VLAs (like OpenVLA) 
    that prefer a single image input embedding.
    """
    def __init__(self, target_resolution=(640, 480)):
        # target_resolution is for EACH individual frame before stitching
        self.target_width, self.target_height = target_resolution
        
    def resize_with_pad(self, image):
        """Resizes an image and pads it to match the exact target resolution."""
        # Advanced VLAs are strict about aspect ratios.
        h, w = image.shape[:2]
        scale = min(self.target_width / w, self.target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a black canvas of the target size
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Center the resized image on the canvas
        y_offset = (self.target_height - new_h) // 2
        x_offset = (self.target_width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas

    def stitch_triple(self, frame_high, frame_side, frame_wrist):
        """
        Stitches 3 frames horizontally (High | Side | Wrist).
        This gives the VLM a complete panoramic view of the workspace.
        """
        # Ensure all frames are the exact same size
        f1 = self.resize_with_pad(frame_high)
        f2 = self.resize_with_pad(frame_side)
        f3 = self.resize_with_pad(frame_wrist)
        
        # Concatenate horizontally
        # Resulting shape: (480, 1920, 3) 
        stiched_image = np.hstack((f1, f2, f3))
        
        return stiched_image

    def stitch_grid(self, frame_high, frame_side, frame_wrist):
        """
        Stitches 3 frames into a 2x2 grid (Bottom right is black).
        This keeps the image closer to a square, which some Vision Transformers prefer.
        """
        f1 = self.resize_with_pad(frame_high)
        f2 = self.resize_with_pad(frame_side)
        f3 = self.resize_with_pad(frame_wrist)
        blank = np.zeros_like(f1)
        
        top_row = np.hstack((f1, f2))
        bottom_row = np.hstack((f3, blank))
        
        # Resulting shape: (960, 1280, 3)
        grid_image = np.vstack((top_row, bottom_row))
        return grid_image

# --- Example Usage ---
if __name__ == "__main__":
    print("Testing Camera Stitcher...")
    # Mocking frames
    high = np.ones((480, 640, 3), dtype=np.uint8) * 255 # White
    side = np.ones((480, 640, 3), dtype=np.uint8) * 128 # Gray
    wrist = np.zeros((480, 640, 3), dtype=np.uint8)     # Black
    
    stitcher = CameraStitcher()
    pano = stitcher.stitch_triple(high, side, wrist)
    grid = stitcher.stitch_grid(high, side, wrist)
    
    print(f"Panorama Shape: {pano.shape}")
    print(f"Grid Shape: {grid.shape}")
    print("Ready to feed into OpenVLA Vision Tower!")
