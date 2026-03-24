import os
import sys
import time
import json
import yaml
import torch
import logging
import threading
from pathlib import Path

try:
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device, init_logging
    from lerobot.configs.policies import PreTrainedConfig
except ImportError:
    print("ERROR: Could not import LeRobot. Ensure PYTHONPATH includes lerobot src.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VLAWorker:
    def __init__(self, config_path, policy_path):
        self.config_path = config_path
        self.policy_path = policy_path
        
        self.robot = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.device = None
        
        self.running = True
        self.active_task = False
        self.task_command = None
        self.task_duration = 0
        self.cfg_dict = {}
        
    def load(self):
        logging.info(f"VLA Worker: Pre-loading policy from {self.policy_path}...")
        
        with open(self.config_path, 'r') as f:
            self.cfg_dict = yaml.safe_load(f)

        # Mock metadata for policy initialization
        class MockMeta:
            def __init__(self, path):
                self.stats = {}
                stats_path = os.path.join(path, "stats.json")
                if os.path.exists(stats_path):
                    with open(stats_path, "r") as f:
                        self.stats = json.load(f)
        
        class MockDS:
            def __init__(self, path):
                self.meta = MockMeta(path)
        
        ds = MockDS(self.policy_path)
        policy_cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        policy_cfg.pretrained_path = self.policy_path
        
        self.device = get_safe_torch_device(getattr(policy_cfg, "device", "cpu"))
        logging.info(f"Using device: {self.device}")
        
        self.policy = make_policy(policy_cfg, ds_meta=ds.meta)
        self.policy.to(self.device)
        self.policy.eval()
        
        rename_map = self.cfg_dict.get('dataset', {}).get('rename_map', {})
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=self.policy_path,
            dataset_stats=rename_stats(ds.meta.stats, rename_map),
            preprocessor_overrides={
                "device_processor": {"device": self.device},
                "rename_observations_processor": {"rename_map": rename_map},
            },
        )
        
        logging.info("VLA Worker: Policy loaded and warm.")
        print("READY", flush=True)

    def run_task_loop(self, task_name, duration_s):
        logging.info(f"Worker: Connecting to hardware for task: {task_name}")
        try:
            if not self.robot:
                self.robot = make_robot_from_config(self.cfg_dict['robot'])
            
            if not self.robot.is_connected:
                self.robot.connect()
                
            start_time = time.perf_counter()
            fps = 30
            
            self.policy.reset()
            self.preprocessor.reset()
            self.postprocessor.reset()
            
            logging.info("Worker: Inference loop started.")
            while (time.perf_counter() - start_time) < duration_s:
                if not self.active_task:
                    break
                    
                loop_start = time.perf_counter()
                obs = self.robot.get_observation()
                
                # Format obs for policy (mimicking lerobot_record.py)
                formatted_obs = {}
                for k, v in obs.items():
                    formatted_obs[f"observation.{k}"] = v

                try:
                    action_values = predict_action(
                        observation=formatted_obs,
                        policy=self.policy,
                        device=self.device,
                        preprocessor=self.preprocessor,
                        postprocessor=self.postprocessor,
                        task=task_name,
                        robot_type=self.robot.robot_type,
                    )
                    
                    if torch.is_tensor(action_values):
                        joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                        action_dict = {}
                        vals = action_values.cpu().numpy().flatten()
                        for i, j in enumerate(joints):
                            if i < len(vals):
                                action_dict[f"{j}.pos"] = float(vals[i])
                        self.robot.send_action(action_dict)
                    else:
                        self.robot.send_action(action_values)
                        
                except Exception as e:
                    logging.error(f"Inference error: {e}")
                    break
                    
                dt = time.perf_counter() - loop_start
                sleep_time = max(0, (1.0 / fps) - dt)
                time.sleep(sleep_time)
                
        except Exception as e:
            logging.error(f"Hardware error: {e}")
        finally:
            self.cleanup_hardware()
            logging.info(f"Worker: Task {task_name} finished and hardware released.")
            print("FINISHED", flush=True)
            self.active_task = False

    def cleanup_hardware(self):
        if self.robot and self.robot.is_connected:
            try:
                self.robot.disconnect()
            except:
                pass

    def command_loop(self):
        while self.running:
            line = sys.stdin.readline()
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(maxsplit=2)
            cmd = parts[0].upper()
            
            if cmd == "START":
                if len(parts) >= 3:
                    self.task_duration = float(parts[1])
                    self.task_command = parts[2]
                    self.active_task = True
                    threading.Thread(target=self.run_task_loop, args=(self.task_command, self.task_duration), daemon=True).start()
                else:
                    print("ERROR: START <duration> <task_name>", flush=True)
            elif cmd == "STOP":
                self.active_task = False
            elif cmd == "QUIT":
                self.running = False
                self.cleanup_hardware()
                print("BYE", flush=True)
                break
            else:
                print(f"UNKNOWN COMMAND: {cmd}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vla_worker.py <config_path> <policy_path>")
        sys.exit(1)
        
    worker = VLAWorker(sys.argv[1], sys.argv[2])
    try:
        worker.load()
        worker.command_loop()
    except Exception as e:
        logging.error(f"Fatal worker error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
