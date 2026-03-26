"""
camera_bridge.py
─────────────────────────────────────────────────────────────────────────────
Shared-memory camera bridge.

  VLA worker  (CameraWriter)  →  named shared-memory  →  UI / Vision (CameraReader)

Every camera slot occupies one fixed-size shared-memory segment:

  Bytes [0:4]    uint32  frame counter   – incremented on every write
  Bytes [4:8]    uint32  frame width     – always BRIDGE_W after first write
  Bytes [8:12]   uint32  frame height    – always BRIDGE_H after first write
  Bytes [12:]    uint8[] BGR pixel data  – width × height × 3 bytes

Research note (OpenVLA):
  OpenVLA-7B natively processes 224×224 images (SigLIP + DINOv2 encoders).
  Higher source resolution gives no accuracy benefit because the processor
  always resizes to 224×224 internally.  We capture at 640×480 for a sharp
  display, then let the processor handle the downscale at inference time.
  (Source: arXiv:2406.09246 ablations – 384px vs 224px → no difference)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import cv2
from multiprocessing.shared_memory import SharedMemory

# ── Dimensions ────────────────────────────────────────────────────────────
# 640×480 is a good balance: crisp for display, LeRobot-compatible,
# and the VLA processor will crop/resize to 224×224 for its own inference.
BRIDGE_W  = 640
BRIDGE_H  = 480
BRIDGE_C  = 3
HEADER_B  = 12                                # 3 × uint32
FRAME_B   = BRIDGE_W * BRIDGE_H * BRIDGE_C   # 921 600 bytes
SEG_B     = HEADER_B + FRAME_B               # 921 612 bytes per slot

# ── Slot names (shared between worker and UI process) ─────────────────────
CAM_MAIN  = "vla_cam_main"   # top-down / primary inference camera
CAM_SIDE  = "vla_cam_side"   # side-view camera
CAM_GRIP  = "vla_cam_grip"   # gripper / wrist camera
ALL_CAMS  = (CAM_MAIN, CAM_SIDE, CAM_GRIP)


# ══════════════════════════════════════════════════════════════════════════
#  WRITER  –  lives inside  unified_vla_worker.py  only
# ══════════════════════════════════════════════════════════════════════════

class CameraWriter:
    """Creates shared-memory segments and writes BGR frames into them."""

    def __init__(self) -> None:
        self._shm: dict[str, SharedMemory] = {}
        self._cnt: dict[str, int]          = {}

    # ── segment lifecycle ─────────────────────────────────────────────────
    def create_segment(self, cam_name: str) -> bool:
        """Create (or re-attach to existing) a named segment. Returns True on success."""
        try:
            shm = SharedMemory(name=cam_name, create=True, size=SEG_B)
        except FileExistsError:
            try:
                shm = SharedMemory(name=cam_name, create=False, size=SEG_B)
                # Zero the counter so readers wait for the first real frame
                np.frombuffer(shm.buf, dtype=np.uint32, count=1, offset=0)[0] = 0
            except Exception as exc:
                print(f"[BRIDGE] Cannot attach to existing segment '{cam_name}': {exc}", flush=True)
                return False
        except Exception as exc:
            print(f"[BRIDGE] Cannot create segment '{cam_name}': {exc}", flush=True)
            return False

        self._shm[cam_name] = shm
        self._cnt[cam_name] = 0
        return True

    # ── write ─────────────────────────────────────────────────────────────
    def write(self, cam_name: str, frame: np.ndarray) -> None:
        """Write one BGR frame into the named segment (resizes if needed)."""
        if cam_name not in self._shm:
            return
        shm = self._shm[cam_name]

        # Normalise to bridge dimensions
        if frame.shape[1] != BRIDGE_W or frame.shape[0] != BRIDGE_H:
            frame = cv2.resize(frame, (BRIDGE_W, BRIDGE_H),
                               interpolation=cv2.INTER_LINEAR)

        self._cnt[cam_name] += 1
        arr = np.frombuffer(shm.buf, dtype=np.uint8, count=SEG_B)
        # Write header (counter, width, height)
        hdr = arr.view(np.uint32)
        hdr[0] = self._cnt[cam_name]
        hdr[1] = BRIDGE_W
        hdr[2] = BRIDGE_H
        # Write pixel data
        arr[HEADER_B:] = frame.flatten()

    # ── cleanup ───────────────────────────────────────────────────────────
    def cleanup(self) -> None:
        for name, shm in list(self._shm.items()):
            shm.close()
            try:
                shm.unlink()
            except Exception:
                pass  # Windows releases the segment when all handles close
        self._shm.clear()


# ══════════════════════════════════════════════════════════════════════════
#  READER  –  lives inside  main_ui.py  and  vision_module.py
# ══════════════════════════════════════════════════════════════════════════

class CameraReader:
    """Attaches to shared-memory segments and reads BGR frames from them."""

    def __init__(self) -> None:
        self._shm:  dict[str, SharedMemory] = {}
        self._prev: dict[str, int]           = {}

    # ── internal attach ───────────────────────────────────────────────────
    def _attach(self, cam_name: str) -> bool:
        if cam_name in self._shm:
            return True
        try:
            shm = SharedMemory(name=cam_name, create=False)
            self._shm[cam_name]  = shm
            self._prev[cam_name] = 0
            return True
        except FileNotFoundError:
            return False

    # ── public API ────────────────────────────────────────────────────────
    def is_available(self, cam_name: str) -> bool:
        """True once the VLA worker has created this segment."""
        return self._attach(cam_name)

    def read(self, cam_name: str) -> tuple[np.ndarray | None, bool]:
        """
        Return *(frame, is_new)*.

        - ``frame``  – latest BGR ndarray (H×W×3) or ``None`` if not ready.
        - ``is_new`` – True only when the counter advanced since last call.

        Always returns the latest frame even when ``is_new`` is False, so
        the UI can keep displaying the last received image without tearing.
        """
        if not self._attach(cam_name):
            return None, False

        shm  = self._shm[cam_name]
        arr  = np.frombuffer(shm.buf, dtype=np.uint8, count=SEG_B)
        hdr  = arr.view(np.uint32)
        cnt  = int(hdr[0])

        if cnt == 0:
            return None, False

        w     = int(hdr[1])
        h     = int(hdr[2])
        frame = arr[HEADER_B: HEADER_B + w * h * 3].copy().reshape(h, w, 3)
        is_new = (cnt != self._prev.get(cam_name, -1))
        self._prev[cam_name] = cnt
        return frame, is_new

    def cleanup(self) -> None:
        for shm in self._shm.values():
            shm.close()
        self._shm.clear()
