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

from transformers import (
    AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer,
)
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
        self.base_model_id  = args.base_model
        self.adapter_path   = args.adapter
        self.chat_model_id  = args.chat_model
        self.robot_config   = args.config
        self.main_cam_idx   = args.main_cam
        self.side_cam_idx   = args.side_cam
        self.grip_cam_idx   = args.grip_cam
        self.dataset_ids   = [
            "local/pick_place_finetune",
            "local/pick_place_Iloveyoublock",
        ]

        # VLA model components (GPU – action inference only)
        self.processor  = None
        self.model      = None
        self.robot      = None
        self.vla_ready  = False

        # Chat model components (CPU – English conversation)
        self.chat_tokenizer = None
        self.chat_model     = None

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
            chunk_dir = f"{base}/data/chunk-000"
            # Collect all parquet files in the chunk directory
            parquet_files = []
            if os.path.isdir(chunk_dir):
                parquet_files = [
                    os.path.join(chunk_dir, f)
                    for f in os.listdir(chunk_dir)
                    if f.endswith(".parquet")
                ]
            if not parquet_files:
                # Fallback: any parquet anywhere under data/
                data_dir = f"{base}/data"
                if os.path.isdir(data_dir):
                    for root, _, files in os.walk(data_dir):
                        for f in files:
                            if f.endswith(".parquet"):
                                parquet_files.append(os.path.join(root, f))
            if parquet_files:
                try:
                    dfs  = [pd.read_parquet(p) for p in parquet_files]
                    df   = pd.concat(dfs, ignore_index=True)
                    acts = np.vstack(df["action"].values)
                    all_actions.append(acts)
                    log.info(f"  {ds_name}: {len(acts)} frames from {len(parquet_files)} file(s).")
                except Exception as exc:
                    log.warning(f"  Could not load parquets for {ds_name}: {exc}")

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

        # ── Chat model (separate small model, runs on CPU) ────────────────
        # OpenVLA cannot generate English text — its LLaMA-2 backbone was
        # overwritten by robot action pretraining. We load a tiny dedicated
        # chat model (Qwen2.5-0.5B-Instruct, ~500 MB) on CPU for conversation.
        self._load_chat_model()

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
    ) -> tuple[torch.Tensor, int]:
        """
        Returns (full_output_tensor, prompt_token_length).

        Always use output[prompt_len:] when decoding to get ONLY the newly
        generated tokens. Decoding out[0] directly includes 256 image-patch
        tokens which produce non-ASCII garbage and break string parsing.
        """
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)
        # Processor returns float32 pixel_values; vision backbone is float16
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._vision_dtype())
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        return out, prompt_len

    # ── Chat model (CPU) ──────────────────────────────────────────────────
    def _load_chat_model(self) -> None:
        """
        Load a small instruction-tuned LLM for English conversation on CPU.
        This runs completely separately from the VLA (which is GPU-only for
        action inference).  Default: Qwen/Qwen2.5-0.5B-Instruct (~500 MB).
        """
        model_id = self.chat_model_id
        log.info(f"Loading chat model: {model_id}")
        try:
            self.chat_tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.chat_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,   # CPU – no bfloat16 needed
                device_map="cpu",
                trust_remote_code=True,
            )
            self.chat_model.eval()
            log.info(f"Chat model ready on CPU.")
        except Exception as exc:
            log.warning(f"Chat model load failed: {exc}. Chat will use fallback responses.")
            self.chat_model     = None
            self.chat_tokenizer = None

    # ── Chat / classification ──────────────────────────────────────────────
    def chat_mode(self, user_text: str, image: Image.Image) -> str:
        """
        Generate an English reply using the dedicated chat model (CPU).

        WHY NOT THE VLA FOR CHAT
        ─────────────────────────
        OpenVLA's LLaMA-2 backbone was overwritten by continual pretraining
        on robot manipulation data.  It can no longer generate English text —
        it only produces action tokens (IDs 31743-31999) which decode as
        Chinese/garbage characters.  disable_adapter() doesn't fix this
        because the damage is in the base weights, not the LoRA delta.

        ARCHITECTURE
        ─────────────
          Conversation  →  Qwen2.5-0.5B on CPU  (~500 MB, fast English)
          Actions       →  OpenVLA-7B on GPU   (trained pick-and-place)
        """
        if self.chat_model is None or self.chat_tokenizer is None:
            # Fallback if chat model failed to load
            return "[CHAT] I'm ready! (chat model not loaded — check logs)"

        history = self._build_history_str()
        system_msg = (
            "You are a helpful robotic lab assistant named RoboAssistant. "
            "You have a 6-DOF robot arm with a gripper. "
            "Keep replies concise (1-2 sentences). "
            "Prefix your reply with [CHAT] for general conversation, "
            "or [ACTION] if the user wants you to physically pick, place, "
            "move, grab, lift, or manipulate any object."
        )
        messages = [{"role": "system", "content": system_msg}]
        if history:
            for turn in self.conv_history[-(self.MAX_HISTORY * 2):]:
                messages.append(turn)
        messages.append({"role": "user", "content": user_text})

        # Use the chat template built into the tokenizer
        text_input = self.chat_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs     = self.chat_tokenizer(text_input, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = self.chat_model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.chat_tokenizer.eos_token_id,
            )

        new_tokens = out[0][prompt_len:]
        return self.chat_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ── Task completion ─────────────────────────────────────────────────────
    def check_task_complete(self, task: str, image: Image.Image) -> bool:
        """
        Ask the BASE model (LoRA disabled) to look at the current frame and
        decide whether the task is visually complete.

        Prompt is kept very short (max_new_tokens=5) so this adds only
        ~200ms overhead per check (run every ~3 s during the action loop).
        """
        # Text-only prompt — vision check via the full processor (with image)
        # so the model can actually see the workspace.
        prompt = (
            f"[INST] <<SYS>>\nYou are a robot vision system. "
            "The camera image shows the workspace: <image>. "
            "Answer in English with YES or NO only.\n<</SYS>>\n\n"
            f"Has the task '{task}' been completed? [/INST]"
        )
        try:
            with self.model.disable_adapter():
                out, prompt_len = self._gen_tokens(
                    prompt, image,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=None,
                )
            new_tokens = out[0][prompt_len:]
            text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).upper()
            result = "YES" in text
            log.info(f"Task completion check: {'DONE' if result else 'still running'} | raw='{text[-20:]}'")
            return result
        except Exception as exc:
            log.warning(f"Completion check failed: {exc}")
            return False

    # ── Action loop ────────────────────────────────────────────────────────
    def action_loop(self, task_instruction: str, duration_s: float) -> None:
        """
        Run VLA inference loop until one of three stop conditions is met:

        1. VISION CHECK  – every CHECK_EVERY steps the base model (LoRA off)
                           looks at the camera and answers "task done? YES/NO".
                           Stops on YES.  Best signal but adds ~200ms overhead
                           every 3 seconds.

        2. STILLNESS     – if the last STILL_WINDOW action vectors are all
                           within STILL_THRESHOLD of each other the arm has
                           stopped moving.  Robust fallback when the model
                           keeps repeating the same home-position tokens.

        3. TIMEOUT       – hard ceiling at duration_s seconds.

        Uses CAM_MAIN (top/overhead) – the same view used during training.
        """
        self.active_action = True
        prompt   = f"In: What action should the robot take to {task_instruction}?\nOut:"
        deadline = time.perf_counter() + duration_s

        # Completion-check tuning
        CHECK_EVERY   = 30    # run vision check every N action steps (~3 s at 10 Hz)
        STILL_WINDOW  = 15    # consecutive steps to consider arm "still"
        STILL_THRESH  = 0.02  # normalised joint change below this = not moving

        step           = 0
        recent_targets : list[np.ndarray] = []
        completion_reason = "timeout"

        log.info(f"Action started: '{task_instruction}' ({duration_s}s)")

        while time.perf_counter() < deadline and self.active_action:
            image = self.get_pil_frame(CAM_MAIN)

            # ── Vision-based completion check (every CHECK_EVERY steps) ──
            if step > 0 and step % CHECK_EVERY == 0:
                if self.check_task_complete(task_instruction, image):
                    completion_reason = "vision_check"
                    break

            if self.robot is None:
                time.sleep(0.5)
                step += 1
                continue

            try:
                inputs = self.processor(
                    text=prompt, images=image, return_tensors="pt"
                ).to(self.device)
                if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self._vision_dtype())
                prompt_len = inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs, max_new_tokens=7, do_sample=False
                    )

                # Decode exactly the 7 newly generated action tokens
                pred_ids = out[0][prompt_len: prompt_len + 7].cpu().numpy()
                norm_acts = [
                    int(np.clip(31999 - int(tid), 0, 255)) / 255.0
                    for tid in pred_ids
                ]
                target = (
                    self.local_min
                    + np.array(norm_acts, dtype=np.float32) * self.action_range
                )

                # ── Stillness detection ───────────────────────────────────
                recent_targets.append(target.copy())
                if len(recent_targets) > STILL_WINDOW:
                    recent_targets.pop(0)
                if len(recent_targets) == STILL_WINDOW:
                    spread = np.max(recent_targets, axis=0) - np.min(recent_targets, axis=0)
                    if np.all(spread < STILL_THRESH):
                        completion_reason = "stillness"
                        break

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

            step     += 1
            time.sleep(0.01)   # yield – ~100 Hz upper bound

        self.active_action = False
        log.info(f"Action loop finished. Reason: {completion_reason} after {step} steps.")
        print(self._safe_str(f"TASK_DONE {completion_reason} after {step} steps"), flush=True)
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
        Make a string safe to print through the Windows UTF-8 pipe.
        The subprocess is launched with PYTHONIOENCODING=utf-8 so valid UTF-8
        (quotes, em-dashes, etc.) passes through fine.
        Only truly unrepresentable codepoints are replaced with '?'.
        Strips raw action-token garbage (codepoints in the Private Use Area
        that the tokenizer maps high token IDs onto).
        """
        # Remove characters that fall in Unicode Private Use Areas —
        # these are what action-token IDs decode to when accidentally decoded
        import re
        text = re.sub(r'[\uE000-\uF8FF\U000F0000-\U000FFFFF\U00100000-\U0010FFFF]',
                      '', text)
        # Encode as UTF-8, replace any remaining unrepresentable bytes
        return text.encode("utf-8", errors="replace").decode("utf-8")

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
    p.add_argument("--base-model",  default="openvla/openvla-7b",
                   help="HuggingFace base model ID")
    p.add_argument("--adapter",     default="outputs/train/vla_lora_adapter",
                   help="LoRA adapter checkpoint path")
    p.add_argument("--chat-model",  default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Small chat model for English conversation (runs on CPU)")
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
