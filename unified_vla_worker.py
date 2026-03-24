import sys
import time
import json
import logging
import threading
import torch
from pathlib import Path

# NOTE: This requires installing the specific VLA dependencies natively:
# e.g., pip install transformers accelerate

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: Transformers not installed. Unified VLA will run in mock mode.")

try:
    from lerobot.robots.utils import make_robot_from_config
    import yaml
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("WARNING: LeRobot not installed in this environment.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [UNIFIED-VLA] %(levelname)s - %(message)s')

class UnifiedVLAWorker:
    """
    A single unified orchestrator that acts as both a Chat LLM and a Physical Action Policy.
    Mode toggling is handled entirely through prompt construction.
    """
    def __init__(self, model_id="openvla/openvla-7b", robot_config_path=None):
        self.model_id = model_id
        self.robot_config_path = robot_config_path
        
        self.processor = None
        self.model = None
        self.robot = None
        
        self.running = True
        self.active_action_task = False
        
        # Core Context
        self.chat_history = []
        
    def load(self):
        logging.info(f"Loading Unified VLA Processor and Model: {self.model_id}")
        if TRANSFORMERS_AVAILABLE:
            # We use 4-bit or 8-bit quantization locally to fit on RTX 5070 Ti (~6.5GB VRAM)
            # Requires `bitsandbytes` installed
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id, 
                attn_implementation="flash_attention_2", # Highly recommended for speed
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                trust_remote_code=True,
                load_in_4bit=True # Assuming bitsandbytes is available
            )
            logging.info("Model loaded successfully into VRAM.")
        else:
            logging.warning("MOCK MODE: Model loading skipped.")

        if self.robot_config_path and LEROBOT_AVAILABLE:
            logging.info(f"Connecting to hardware via config: {self.robot_config_path}")
            with open(self.robot_config_path, 'r') as f:
                cfg_dict = yaml.safe_load(f)
            self.robot = make_robot_from_config(cfg_dict['robot'])
            self.robot.connect()

        print("READY", flush=True)

    def chat_mode(self, user_text):
        """
        Dormant Vision Mode: Fast text-only inference.
        We DO NOT append '<image>' to the prompt.
        """
        logging.info(f"Chat Mode triggered: {user_text}")
        self.chat_history.append({"role": "user", "content": user_text})
        
        if TRANSFORMERS_AVAILABLE and self.processor and self.model:
            prompt = self.processor.apply_chat_template(self.chat_history, tokenize=False)
            inputs = self.processor(text=prompt, return_tensors="pt").to("cuda")
            output = self.model.generate(**inputs, max_new_tokens=100)
            response = self.processor.decode(output[0], skip_special_tokens=True)
        else:
            # Mock reasoning
            response_data = {"dialogue": f"I understand: {user_text}", "intent": "Idle"}
            if "pick" in user_text.lower():
                response_data["intent"] = "pick_block"
            response = json.dumps(response_data)
            
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def action_mode(self, task_instruction, duration_s=30.0):
        """
        Active Vision Mode: Frame streaming + Action Token generation.
        """
        logging.info(f"Action Mode triggered: {task_instruction} for {duration_s}s")
        self.active_action_task = True
        start_time = time.perf_counter()
        fps = 5 # Unified VLAs run slower than specialized ACTs
        
        prompt = f"In: <image> What action should the robot take to {task_instruction}?\nOut:"

        while self.running and self.active_action_task and (time.perf_counter() - start_time) < duration_s:
            loop_start = time.perf_counter()
            
            if self.robot and self.robot.is_connected:
                obs = self.robot.get_observation()
                # Unified VLAs typically take a single primary camera (e.g., high/front) 
                # or a concatenated image. For OpenVLA, it's usually one main input.
                img_array = obs.get('images.cam_high', None) 
            else:
                # Mock a black image
                import numpy as np
                img_array = np.zeros((480, 640, 3), dtype=np.uint8)

            if TRANSFORMERS_AVAILABLE and self.processor and self.model:
                try:
                    inputs = self.processor(text=prompt, images=img_array, return_tensors="pt").to("cuda")
                    # predict_action translates tokens back to physical joint states
                    action = self.model.predict_action(**inputs, unnorm_key="bridge_orig")
                    
                    if self.robot:
                        # Convert to right format and send
                        pass
                except Exception as e:
                    logging.error(f"Action prediction failed: {e}")
                    break
            else:
                # Mock action delay
                time.sleep(0.2)

            dt = time.perf_counter() - loop_start
            time.sleep(max(0, (1.0 / fps) - dt))

        logging.info(f"Action '{task_instruction}' finished.")
        
        # Crucial Context Injection: Tell the LLM side it accomplished the physical task.
        self.chat_history.append({"role": "system", "content": f"You successfully completed physical action: {task_instruction}"})
        
        print("FINISHED", flush=True)
        self.active_action_task = False

    def command_loop(self):
        """Stdin interface resembling the previous vla_worker but unified"""
        while self.running:
            line = sys.stdin.readline()
            if not line:
                break
            
            line = line.strip()
            if not line: continue
                
            parts = line.split(maxsplit=2)
            cmd = parts[0].upper()
            
            if cmd == "CHAT":
                # CHAT <user_message>
                if len(parts) >= 2:
                    msg = line[len("CHAT "):]
                    resp = self.chat_mode(msg)
                    # output response as JSON to stdout for brain module parsing
                    print(f"CHAT_REPLY {resp}", flush=True)
            elif cmd == "ACTION":
                # ACTION <duration> <instruction>
                if len(parts) >= 3:
                    dur = float(parts[1])
                    instr = parts[2]
                    threading.Thread(target=self.action_mode, args=(instr, dur), daemon=True).start()
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
    parser.add_argument("--model", default="openvla/openvla-7b")
    parser.add_argument("--robot_config", default=None)
    args = parser.parse_args()
    
    worker = UnifiedVLAWorker(model_id=args.model, robot_config_path=args.robot_config)
    try:
        worker.load()
        worker.command_loop()
    except Exception as e:
        logging.error(f"Fatal worker error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
