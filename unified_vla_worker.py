import os
import sys
import time
import json
import logging
import threading
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import cv2

# VLA / Transformers imports
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# LeRobot imports for hardware
try:
    from lerobot.robots.utils import make_robot_from_config
    import yaml
except ImportError:
    print("WARNING: LeRobot libs not found in this env.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [UNIFIED-VLA] %(message)s')

class UnifiedVLAWorker:
    def __init__(self, base_model_id="openvla/openvla-7b", adapter_path="outputs/train/vla_lora_adapter", dataset_ids=None):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.dataset_ids = dataset_ids or ["local/pick_place_finetune", "local/pick_place_Iloveyoublock"]
        
        self.processor = None
        self.model = None
        self.robot = None
        
        # Scaling stats (Must match training!)
        self.local_min = None
        self.local_max = None
        self.action_range = None
        
        self.running = True
        self.active_action_task = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def derive_dataset_stats(self):
        """derives normalization stats from the same datasets used in training."""
        logging.info("Scanning datasets to derive action normalization stats...")
        all_actions = []
        for ds_id in self.dataset_ids:
            ds_name = ds_id.split("/")[-1]
            ds_path = f"C:/Users/Noniro/.cache/huggingface/lerobot/local/{ds_name}"
            parquet_path = f"{ds_path}/data/chunk-000/dataset.parquet"
            if not os.path.exists(parquet_path):
                parquet_path = f"{ds_path}/data/chunk-000/chunk-000.parquet"
            if not os.path.exists(parquet_path):
                 parquet_path = f"{ds_path}/data/chunk-000.parquet"
            
            if os.path.exists(parquet_path):
                try:
                    df = pd.read_parquet(parquet_path)
                    batch_actions = np.vstack(df['action'].values)
                    all_actions.append(batch_actions)
                    logging.info(f"Loaded {len(batch_actions)} frames from {ds_name} for stats.")
                except Exception as e:
                    logging.warning(f"Failed to read parquet {parquet_path}: {e}")
        
        if not all_actions:
            logging.error("No dataset parquet found! Using default -1 to 1 scales (Risk of inaccuracy).")
            self.local_min = np.array([-1.0]*6 + [0.0]) 
            self.local_max = np.array([1.0]*6 + [100.0])
        else:
            combined = np.vstack(all_actions)
            self.local_min = np.min(combined, axis=0).astype(np.float32)
            self.local_max = np.max(combined, axis=0).astype(np.float32)
            
            # Match the 5% expansion from the training script
            action_margin = (self.local_max - self.local_min) * 0.05
            self.local_min -= action_margin
            self.local_max += action_margin
            
        self.action_range = self.local_max - self.local_min
        self.action_range[self.action_range == 0] = 1.0 # Protect against div/0
        logging.info(f"Stats Derived -> Min: {np.round(self.local_min, 2)} | Max: {np.round(self.local_max, 2)}")

    def load(self, robot_config_path=None):
        self.derive_dataset_stats()
        
        logging.info(f"Loading Base: {self.base_model_id} + Adapter: {self.adapter_path}")
        
        # Monkey-patch transformers bitsandbytes integration for frozenset bug as seen in training script
        import transformers.integrations.bitsandbytes as bnb_int
        def _patched_validate_bnb_multi_backend_availability(*args, **kwargs):
             return
        if hasattr(bnb_int, "_validate_bnb_multi_backend_availability"):
            bnb_int._validate_bnb_multi_backend_availability = _patched_validate_bnb_multi_backend_availability

        bnb_config = BitsAndBytesConfig(load_in_8bit=True) 
        
        self.processor = AutoProcessor.from_pretrained(self.base_model_id, trust_remote_code=True)
        base_model = AutoModelForVision2Seq.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        logging.info("Unified VLA Model is READY.")

        if robot_config_path and os.path.exists(robot_config_path):
            with open(robot_config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            self.robot = make_robot_from_config(cfg['robot'])
            if not self.robot.is_connected:
                self.robot.connect()
            logging.info("Hardware connected.")

        print("READY", flush=True)

    def generate_tokens(self, prompt, image, max_new_tokens=10):
        """Low-level helper to get model tokens."""
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7 
            )
        return output

    def chat_mode(self, user_text, image):
        """Unified Reasoning: Act as the LLM inside OpenVLA."""
        prompt = (
            f"A chat between a curious user and a robotic assistant. "
            f"The assistant sees the world through a camera: <image>. "
            f"The robot is equipped with a 6-DOF arm and gripper. "
            f"USER: {user_text}\n"
            f"ASSISTANT: Reasoning:"
        )
        
        output = self.generate_tokens(prompt, image, max_new_tokens=60)
        full_text = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract everything after 'Reasoning:'
        if "Reasoning:" in full_text:
            reasoning = full_text.split("Reasoning:")[-1].split("USER:")[0].split("\n\n")[0].strip()
        else:
            reasoning = "I see the scene. Transitioning to action."
            
        if not reasoning:
             reasoning = "I understand. I am preparing to assist you."
             
        return reasoning

    def action_loop(self, task_name, duration_s):
        self.active_action_task = True
        start_t = time.perf_counter()
        
        # The prompt MUST match what was used during fine-tuning!
        # According to train_openvla_lora: "In: What action should the robot take to {task}?\nOut:"
        prompt = f"In: What action should the robot take to {task_name}?\nOut:"
        logging.info(f"Action Started: {task_name} for {duration_s}s")

        while (time.perf_counter() - start_t) < duration_s and self.active_action_task:
            if not self.robot: 
                time.sleep(1)
                continue
            
            try:
                obs = self.robot.get_observation()
                img_bgr = obs.get('images.cam_high', obs.get('observation.images.cam_high'))
                if img_bgr is None:
                    logging.error("Could not find cam_high in observation keys!")
                    break

                img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                
                # For actions, we want greedy/deterministic
                inputs = self.processor(text=prompt, images=img_pil, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    output = self.model.generate(**inputs, max_new_tokens=7, do_sample=False)
                
                pred_ids = output[0][-7:].cpu().numpy()
                
                normalized_actions = []
                for tid in pred_ids:
                    bin_idx = 31999 - tid
                    bin_idx = np.clip(bin_idx, 0, 255)
                    norm = bin_idx / 255.0
                    normalized_actions.append(norm)
                
                target_pose = self.local_min + (np.array(normalized_actions) * self.action_range)
                
                joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                action_dict = {}
                for i, jname in enumerate(joints):
                    if i < len(target_pose):
                        action_dict[f"{jname}.pos"] = float(target_pose[i])
                
                self.robot.send_action(action_dict)
            except Exception as e:
                logging.error(f"Action prediction/execution failed: {e}")
                break
                
            time.sleep(0.01) # Yield
            
        logging.info("Action finished.")
        print("FINISHED", flush=True)
        self.active_action_task = False

    def command_loop(self):
        while self.running:
            line = sys.stdin.readline()
            if not line: break
            line = line.strip()
            if not line: continue
            
            parts = line.split(maxsplit=2)
            cmd = parts[0].upper()
            
            if cmd == "CHAT":
                if self.robot:
                    obs = self.robot.get_observation()
                    img_data = obs.get('images.cam_high', obs.get('observation.images.cam_high'))
                    if img_data is not None:
                        img_pil = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
                    else:
                        img_pil = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
                else: 
                    img_pil = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
                
                user_msg = line[len("CHAT "):].strip()
                resp = self.chat_mode(user_msg, img_pil)
                print(f"CHAT_REPLY {resp}", flush=True)
                
            elif cmd == "ACTION":
                if len(parts) >= 3:
                    try:
                        dur = float(parts[1])
                        instr = parts[2]
                        threading.Thread(target=self.action_loop, args=(instr, dur), daemon=True).start()
                    except ValueError:
                         print("ERROR: Invalid Duration", flush=True)
                else:
                    print("ERROR: ACTION <duration> <instruction>", flush=True)
            elif cmd == "STOP":
                self.active_action_task = False
                print("STOPPED", flush=True)
            elif cmd == "QUIT":
                self.running = False
                if self.robot: self.robot.disconnect()
                print("BYE", flush=True)
                break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    worker = UnifiedVLAWorker()
    try:
        worker.load(robot_config_path=args.config)
        worker.command_loop()
    except Exception as e:
        logging.error(f"Worker Crash: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
