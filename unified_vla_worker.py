"""
unified_vla_worker.py
─────────────────────────────────────────────────────────────────────────────
The single brain + body of the RoboAssistant.

Startup sequence
────────────────
1. Open all configured cameras  (fast, < 2 s)
2. Create shared-memory segments via camera_bridge  →  emits CAMERAS_READY
3. Start per-camera capture threads               →  UI can show live feed now
4. Derive dataset normalisation stats
5. Load OpenVLA-7B base + LoRA adapter  (~20 s)   →  emits READY
6. Enter command loop (stdin → stdout)

Stdin protocol
──────────────
  CHAT <user_text>               classify + reply [+ auto-trigger action]
  ACTION <duration_s> <task>     force-trigger action loop (legacy path)
  STOP                           abort active action
  CLEAR_HISTORY                  wipe conversation history
  QUIT                           clean shutdown

Stdout protocol
───────────────
  CAMERAS_READY                  cameras open, shared memory live
  READY                          model loaded, fully operational
  CHAT_REPLY <text>              conversational / reasoning response
  ACTION_START <text>            physical action beginning
  FINISHED                       action loop completed
  STOPPED                        action aborted by STOP command
  ERROR <message>                non-fatal error

Intent classification
─────────────────────
The model is prompted to prefix its reply with [CHAT] or [ACTION].
A keyword scan of the user text acts as a reliable fallback so the
robot never misses a pick-and-place request even if the LLM omits
the prefix (possible with 8-bit quantisation).

Camera → inference mapping
──────────────────────────
• All 3 cameras are captured at 640×480 and written to shared memory.
• Only CAM_MAIN (top/overhead view) is fed to the VLA for inference –
  this matches the training setup where cam_high was the primary view.
• OpenVLA's processor resizes any input to 224×224 internally so the
  640×480 capture resolution has no negative effect on accuracy.
  (arXiv:2406.09246 – 384px vs 224px training ablation: no difference)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging
import threading
import platform

import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

from camera_bridge import CameraWriter, CAM_MAIN, CAM_SIDE, CAM_GRIP

try:
    from lerobot.robots.utils import make_robot_from_config
    import yaml
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    print("WARNING: LeRobot not found – hardware control disabled.", flush=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VLA] %(message)s",
    stream=sys.stderr,   # keep stdout clean for the protocol
)
log = logging.getLogger(__name__)

# ── Action trigger keywords ────────────────────────────────────────────────
# The LLM is prompted to self-classify, but this keyword scan acts as a
# reliable fallback so we never miss a physical task request.
ACTION_KEYWORDS: frozenset[str] = frozenset({
    "pick", "place", "move", "grab", "reach", "put", "take",
    "lift", "drop", "push", "pull", "grasp", "carry", "bring",
})


# ══════════════════════════════════════════════════════════════════════════
#  Worker class
# ══════════════════════════════════════════════════════════════════════════

class UnifiedVLAWorker:

    def __init__(self, args: argparse.Namespace) -> None:
        self.base_model_id = args.base_model
        self.adapter_path  = args.adapter
        self.robot_config  = args.config
        self.main_cam_idx  = args.main_cam
        self.side_cam_idx  = args.side_cam
        self.grip_cam_idx  = args.grip_cam
        self.dataset_ids   = [
            "local/pick_place_finetune",
            "local/pick_place_Iloveyoublock",
        ]

        # Model components
        self.processor  = None
        self.model      = None
        self.robot      = None
        self.vla_ready  = False

        # Normalisation stats (derived from training data)
        self.local_min    : np.ndarray | None = None
        self.local_max    : np.ndarray | None = None
        self.action_range : np.ndarray | None = None

        # Runtime flags
        self.running       = True
        self.active_action = False
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"

        # Camera bridge (write side)
        self.bridge        = CameraWriter()
        self._frames: dict[str, np.ndarray | None]  = {}
        self._flocks: dict[str, threading.Lock]      = {}

        # Conversation history – rolling window (≤ MAX_HISTORY exchanges)
        self.conv_history: list[dict[str, str]] = []
        self.MAX_HISTORY  = 10

    # ══════════════════════════════════════════════════════════════════════
    #  CAMERA MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════

    def _open_cap(self, index: int) -> cv2.VideoCapture | None:
        if index < 0:
            return None
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            log.warning(f"Camera {index} could not be opened.")
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        log.info(f"Camera {index} opened.")
        return cap

    def _capture_thread(self, cap: cv2.VideoCapture | None, cam_name: str) -> None:
        """Continuously capture frames → write to bridge + local buffer."""
        while self.running:
            if cap is None:
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if ret and frame is not None:
                self.bridge.write(cam_name, frame)
                with self._flocks[cam_name]:
                    self._frames[cam_name] = frame
            else:
                time.sleep(0.01)

    def start_cameras(self) -> None:
        """
        Open cameras and start streaming to shared memory.
        Emits CAMERAS_READY once segments are live so the UI can
        switch to reading from the bridge immediately.
        """
        cam_slots = {
            CAM_MAIN: self.main_cam_idx,
            CAM_SIDE: self.side_cam_idx,
            CAM_GRIP: self.grip_cam_idx,
        }
        for cam_name, idx in cam_slots.items():
            self._frames[cam_name] = None
            self._flocks[cam_name] = threading.Lock()
            if idx < 0:
                log.info(f"Slot '{cam_name}' disabled (index={idx}).")
                continue
            if not self.bridge.create_segment(cam_name):
                log.warning(f"Could not create bridge segment for '{cam_name}'.")
                continue
            cap = self._open_cap(idx)
            t = threading.Thread(
                target=self._capture_thread,
                args=(cap, cam_name),
                daemon=True,
                name=f"cam-{cam_name}",
            )
            t.start()
            log.info(f"Camera slot '{cam_name}' streaming (index={idx}).")

        print("CAMERAS_READY", flush=True)

    def get_pil_frame(self, cam_name: str = CAM_MAIN) -> Image.Image:
        """Return latest frame from the named slot as a PIL RGB image."""
        with self._flocks.get(cam_name, threading.Lock()):
            frame = self._frames.get(cam_name)
        if frame is None:
            # Blank placeholder while camera warms up
            return Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # ══════════════════════════════════════════════════════════════════════
    #  DATASET STATS
    # ══════════════════════════════════════════════════════════════════════

    def derive_dataset_stats(self) -> None:
        log.info("Deriving action normalisation stats from training data…")
        all_actions: list[np.ndarray] = []

        for ds_id in self.dataset_ids:
            ds_name = ds_id.split("/")[-1]
            base    = (
                f"C:/Users/Noniro/.cache/huggingface/lerobot/local/{ds_name}"
            )
            candidates = [
                f"{base}/data/chunk-000/dataset.parquet",
                f"{base}/data/chunk-000/chunk-000.parquet",
                f"{base}/data/chunk-000.parquet",
            ]
            for path in candidates:
                if os.path.exists(path):
                    try:
                        df   = pd.read_parquet(path)
                        acts = np.vstack(df["action"].values)
                        all_actions.append(acts)
                        log.info(f"  {ds_name}: {len(acts)} frames loaded.")
                    except Exception as exc:
                        log.warning(f"  Could not load {path}: {exc}")
                    break

        if not all_actions:
            log.error("No dataset parquets found – using fallback [-1, 1] scale.")
            self.local_min = np.array([-1.0] * 6 + [0.0],   dtype=np.float32)
            self.local_max = np.array([ 1.0] * 6 + [100.0], dtype=np.float32)
        else:
            combined   = np.vstack(all_actions)
            raw_min    = np.min(combined, axis=0).astype(np.float32)
            raw_max    = np.max(combined, axis=0).astype(np.float32)
            margin     = (raw_max - raw_min) * 0.05   # 5 % expansion matches training
            self.local_min = raw_min - margin
            self.local_max = raw_max + margin

        self.action_range = self.local_max - self.local_min
        self.action_range[self.action_range == 0] = 1.0   # guard div-by-zero
        log.info(
            f"  Stats → min: {np.round(self.local_min, 2)}"
            f"  max: {np.round(self.local_max, 2)}"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  MODEL LOADING
    # ══════════════════════════════════════════════════════════════════════

    def load_model(self) -> None:
        self.derive_dataset_stats()
        log.info(f"Loading base: {self.base_model_id}")
        log.info(f"       LoRA: {self.adapter_path}")

        # Patch bitsandbytes frozenset validation bug
        import transformers.integrations.bitsandbytes as _bnb
        if hasattr(_bnb, "_validate_bnb_multi_backend_availability"):
            _bnb._validate_bnb_multi_backend_availability = lambda *a, **kw: None

        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

        self.processor = AutoProcessor.from_pretrained(
            self.base_model_id, trust_remote_code=True
        )
        base = AutoModelForVision2Seq.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_cfg,
            device_map={"": 0},
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()
        log.info("Model weights loaded.")

        # Robot hardware
        if self.robot_config and os.path.exists(self.robot_config) and HAS_LEROBOT:
            with open(self.robot_config) as f:
                cfg = yaml.safe_load(f)
            self.robot = make_robot_from_config(cfg["robot"])
            if not self.robot.is_connected:
                self.robot.connect()
            log.info("Robot hardware connected.")

        self.vla_ready = True
        print("READY", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    #  INFERENCE HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _build_history_str(self) -> str:
        """Format the rolling conversation history for the prompt."""
        lines = []
        for turn in self.conv_history[-(self.MAX_HISTORY * 2):]:
            role = "USER" if turn["role"] == "user" else "ASSISTANT"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def _vision_dtype(self) -> torch.dtype:
        """Dtype of the vision backbone (float16 when loaded with 8-bit quant)."""
        try:
            return next(
                self.model.base_model.model.vision_backbone.parameters()
            ).dtype
        except Exception:
            return torch.float16

    def _gen_tokens(
        self,
        prompt: str,
        image: Image.Image,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)
        # Processor returns float32 pixel_values but the 8-bit-quantized vision
        # backbone keeps its weights in float16 → cast to match before forward pass.
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._vision_dtype())
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        return out

    # ── Chat / classification ──────────────────────────────────────────────
    def chat_mode(self, user_text: str, image: Image.Image) -> str:
        """
        Generate a response and embed a self-classification prefix.

        The model is instructed to output:
          [CHAT]   <reply>   – pure conversation
          [ACTION] <reply>   – user wants physical manipulation

        The prefix is used by _needs_action() before being stripped
        from the displayed reply.
        """
        history_block = ""
        history = self._build_history_str()
        if history:
            history_block = f"\nPrevious conversation:\n{history}\n"

        prompt = (
            "A chat between a user and a robotic lab assistant. "
            "The assistant sees the workspace through its camera: <image>. "
            "It is equipped with a 6-DOF arm and gripper.\n"
            f"{history_block}"
            "RULES — prefix every reply with one of these tags:\n"
            "  [CHAT]   → casual conversation, questions, status, greetings\n"
            "  [ACTION] → user asks to physically pick, place, move, grab,\n"
            "             lift, push, carry, or otherwise manipulate objects\n"
            f"USER: {user_text}\n"
            "ASSISTANT:"
        )

        out  = self._gen_tokens(prompt, image, max_new_tokens=80)
        text = self.processor.decode(out[0], skip_special_tokens=True)

        # Extract only the new ASSISTANT content
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1]
        # Trim any leaked USER turn
        text = text.split("USER:")[0].strip()
        return text

    # ── Action loop ────────────────────────────────────────────────────────
    def action_loop(self, task_instruction: str, duration_s: float) -> None:
        """
        Run VLA inference loop for up to duration_s seconds.
        Feeds frames from CAM_MAIN (top/overhead camera) – the same view
        used during LoRA fine-tuning.
        """
        self.active_action = True
        # Prompt format MUST match the training script (train_openvla_lora.py)
        prompt   = f"In: What action should the robot take to {task_instruction}?\nOut:"
        deadline = time.perf_counter() + duration_s
        log.info(f"Action started: '{task_instruction}' ({duration_s}s)")

        while time.perf_counter() < deadline and self.active_action:
            image = self.get_pil_frame(CAM_MAIN)

            if self.robot is None:
                time.sleep(0.5)
                continue

            try:
                inputs = self.processor(
                    text=prompt, images=image, return_tensors="pt"
                ).to(self.device)
                if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self._vision_dtype())
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs, max_new_tokens=7, do_sample=False
                    )

                # Decode 7 action tokens → joint positions
                pred_ids = out[0][-7:].cpu().numpy()
                norm_acts = []
                for tid in pred_ids:
                    bin_idx = int(np.clip(31999 - int(tid), 0, 255))
                    norm_acts.append(bin_idx / 255.0)

                target = (
                    self.local_min
                    + np.array(norm_acts, dtype=np.float32) * self.action_range
                )

                joints = [
                    "shoulder_pan", "shoulder_lift", "elbow_flex",
                    "wrist_flex", "wrist_roll", "gripper",
                ]
                action_dict = {
                    f"{j}.pos": float(target[i])
                    for i, j in enumerate(joints)
                    if i < len(target)
                }
                self.robot.send_action(action_dict)

            except Exception as exc:
                log.error(f"Action step failed: {exc}")
                break

            time.sleep(0.01)   # yield – ~100 Hz upper bound

        self.active_action = False
        log.info("Action loop finished.")
        print("FINISHED", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    #  CLASSIFICATION HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _needs_action(self, user_text: str, raw_reply: str) -> bool:
        """
        True when a physical action should follow this exchange.

        Priority order:
        1. Model output contains [ACTION] tag  (model self-classified)
        2. User text contains a known action keyword  (reliable fallback)
        """
        if "[ACTION]" in raw_reply:
            return True
        lower = user_text.lower()
        return any(kw in lower for kw in ACTION_KEYWORDS)

    @staticmethod
    def _safe_str(text: str) -> str:
        """
        Make a string safe to print through a Windows pipe.
        Replaces anything that can't round-trip through ASCII with '?'.
        This prevents UnicodeEncodeError in colorama/wandb stdout wrappers
        when the model generates Unicode arrows or special characters.
        """
        return text.encode("ascii", errors="replace").decode("ascii")

    @staticmethod
    def _clean_reply(raw: str) -> str:
        """Strip classification prefix from model output before display."""
        for tag in ("[CHAT]", "[ACTION]"):
            raw = raw.replace(tag, "")
        return raw.strip()

    # ══════════════════════════════════════════════════════════════════════
    #  COMMAND LOOP
    # ══════════════════════════════════════════════════════════════════════

    def command_loop(self) -> None:
        """Read commands from stdin, emit signals to stdout."""
        while self.running:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            cmd   = parts[0].upper()
            arg   = parts[1] if len(parts) > 1 else ""

            # ── CHAT ──────────────────────────────────────────────────────
            if cmd == "CHAT":
                image   = self.get_pil_frame(CAM_MAIN)
                raw     = self.chat_mode(arg, image)
                reply   = self._clean_reply(raw) or "I'm ready and watching."

                # Persist to rolling history
                self.conv_history.append({"role": "user",      "content": arg})
                self.conv_history.append({"role": "assistant", "content": reply})
                if len(self.conv_history) > self.MAX_HISTORY * 2:
                    self.conv_history = self.conv_history[-(self.MAX_HISTORY * 2):]

                print(self._safe_str(f"CHAT_REPLY {reply}"), flush=True)

                # Decide whether a physical action is also needed
                if self.vla_ready and self._needs_action(arg, raw):
                    print(self._safe_str(f"ACTION_START {reply}"), flush=True)
                    threading.Thread(
                        target=self.action_loop,
                        args=(arg, 60.0),
                        daemon=True,
                        name="action-loop",
                    ).start()

            # ── ACTION (legacy / direct path) ─────────────────────────────
            elif cmd == "ACTION":
                sub = arg.split(maxsplit=1)
                if len(sub) == 2:
                    try:
                        dur   = float(sub[0])
                        instr = sub[1]
                        print(f"ACTION_START {instr}", flush=True)
                        threading.Thread(
                            target=self.action_loop,
                            args=(instr, dur),
                            daemon=True,
                            name="action-loop",
                        ).start()
                    except ValueError:
                        print("ERROR Invalid duration for ACTION command", flush=True)
                else:
                    print("ERROR Usage: ACTION <duration_s> <instruction>", flush=True)

            # ── STOP ──────────────────────────────────────────────────────
            elif cmd == "STOP":
                self.active_action = False
                print("STOPPED", flush=True)

            # ── CLEAR_HISTORY ─────────────────────────────────────────────
            elif cmd == "CLEAR_HISTORY":
                self.conv_history.clear()
                print("HISTORY_CLEARED", flush=True)

            # ── QUIT ──────────────────────────────────────────────────────
            elif cmd == "QUIT":
                self.running       = False
                self.active_action = False
                if self.robot:
                    try:
                        self.robot.disconnect()
                    except Exception:
                        pass
                self.bridge.cleanup()
                print("BYE", flush=True)
                break

            else:
                print(f"ERROR Unknown command: {cmd}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified VLA Worker – brain + body")
    p.add_argument("--config",     default="config.yaml",
                   help="Robot YAML config path")
    p.add_argument("--base-model", default="openvla/openvla-7b",
                   help="HuggingFace base model ID")
    p.add_argument("--adapter",    default="outputs/train/vla_lora_adapter",
                   help="LoRA adapter checkpoint path")
    p.add_argument("--main-cam",   type=int, default=0,
                   help="Main (top/overhead) camera index  [default: 0]")
    p.add_argument("--side-cam",   type=int, default=-1,
                   help="Side-view camera index  (-1 = disabled)")
    p.add_argument("--grip-cam",   type=int, default=1,
                   help="Gripper/wrist camera index  [default: 1]")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    worker = UnifiedVLAWorker(args)
    try:
        # ── Phase 1: cameras up immediately so UI gets a live feed
        #             while the model is still loading (~20 s)
        worker.start_cameras()

        # ── Phase 2: load model weights + connect hardware
        worker.load_model()

        # ── Phase 3: serve commands
        worker.command_loop()

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        log.error(f"Worker crash: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        worker.bridge.cleanup()
