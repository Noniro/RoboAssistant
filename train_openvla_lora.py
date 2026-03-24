import os
import torch
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms.functional as F

# LeRobot Libraries
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("WARNING: Please ensure lerobot is installed.")

# Monkey-patch transformers bitsandbytes integration for frozenset bug
import transformers.integrations.bitsandbytes as bnb_int
def _patched_validate_bnb_multi_backend_availability(*args, **kwargs):
    # Bypass the buggy internal check entirely
    return
bnb_int._validate_bnb_multi_backend_availability = _patched_validate_bnb_multi_backend_availability

# Monkey-patch lerobot to stop hitting the network for local datasets
try:
    import lerobot.datasets.lerobot_dataset as lr_dataset
    lr_dataset.get_safe_version = lambda repo_id, rev: rev if rev else "main"
except Exception as e:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [VLA-TRAIN] %(message)s')

@dataclass
class TrainConfig:
    model_id: str = "openvla/openvla-7b"
    dataset_ids: List[str] = field(default_factory=lambda: [
        "local/pick_place_finetune", 
        "local/pick_place_Iloveyoublock"
    ])
    output_dir: str = "outputs/train/vla_lora_adapter"
    batch_size: int = 2
    grad_accum_steps: int = 16
    epochs: int = 3
    learning_rate: float = 1e-5
    task_instruction: str = "Pick and place the I love you block."
    save_steps: int = 1000  # Save every 1000 steps
    max_checkpoints: int = 2 # Keep only the latest 2

import pandas as pd
import cv2
from PIL import Image
import numpy as np

class OpenVLALeRobotDataset(Dataset):
    def __init__(self, df, video_path, processor, instruction, frame_skip=1):
        self.df = df
        self.video_path = video_path
        self.processor = processor
        self.instruction = instruction
        
        # Calculate local bounds accurately for this specific dataset (to combat Degrees vs Radians collision)
        all_actions = np.vstack(self.df['action'].values)
        self.local_min = np.min(all_actions, axis=0).astype(np.float32)
        self.local_max = np.max(all_actions, axis=0).astype(np.float32)
        
        # Expand safely by 5% to prevent edge-case clipping during inference
        action_margin = (self.local_max - self.local_min) * 0.05
        self.local_min -= action_margin
        self.local_max += action_margin
        
        self.action_range = self.local_max - self.local_min
        self.action_range[self.action_range == 0] = 1.0  # safe division
        
        self.frame_skip = frame_skip
        self.cap = None
        
        logging.info(f"Loaded {len(self.df)} frames from {video_path}. Local Bounds -> Min: {np.round(self.local_min, 2)} | Max: {np.round(self.local_max, 2)}")

    def __len__(self):
        return len(self.df) // self.frame_skip

    def __getitem__(self, idx):
        real_idx = idx * self.frame_skip
        row = self.df.iloc[real_idx]
        
        # 1. Image
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
            
        frame_idx = row['frame_index']
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx-1))
            ret, frame = self.cap.read()
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        # 2. Action parsing and text construction
        action_vec = row['action']

        prompt = f"In: What action should the robot take to {self.instruction}?\nOut:"
        inputs = self.processor(text=prompt, images=img_pil, return_tensors="pt")
        
        for k in inputs:
            inputs[k] = inputs[k].squeeze(0)

        # 3. Discretize Map continuous actions
        # Normalize to [0, 1] using strict independent local bounds
        norm_actions = (action_vec - self.local_min) / self.action_range
        norm_actions = np.clip(norm_actions, 0.0, 1.0)

        BINS = 256
        discretized_actions = []
        for norm in norm_actions:
            bin_idx = int(norm * (BINS - 1))
            bin_idx = np.clip(bin_idx, 0, BINS - 1)
            discretized_actions.append(bin_idx)
            
        action_token_ids = []
        for bin_value in discretized_actions:
            mapped_id = 31999 - bin_value 
            action_token_ids.append(mapped_id)

        # 4. Inject Action Tokens
        action_ids_tensor = torch.tensor(action_token_ids, dtype=torch.long)
        inputs['input_ids'] = torch.cat([inputs['input_ids'], action_ids_tensor])
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones(len(action_token_ids), dtype=torch.long)])

        # 5. Create labels for manual loss (whole sequence, but shifted)
        # Sequence: [BOS, prompt_tokens, action_tokens]
        # We'll calculate loss only on action_tokens
        inputs['target_ids'] = action_ids_tensor
        
        return inputs

def collate_fn(batch):
    collated = {}
    for k in batch[0].keys():
        if k == 'pixel_values':
            collated[k] = torch.stack([item[k] for item in batch])
        elif k == 'target_ids':
            collated[k] = torch.stack([item[k] for item in batch])
        else:
            collated[k] = pad_sequence([item[k] for item in batch], batch_first=True, padding_value=0)
    return collated

def cleanup_checkpoints(output_dir, max_to_keep=2):
    import shutil
    import re
    
    checkpoints = []
    if not os.path.exists(output_dir):
        return
        
    for d in os.listdir(output_dir):
        if d.startswith("checkpoint-"):
            path = os.path.join(output_dir, d)
            if os.path.isdir(path):
                # Try to extract step number for sorting
                match = re.search(r"checkpoint-(\d+)", d)
                step = int(match.group(1)) if match else 0
                checkpoints.append((step, path))
    
    # Sort by step number (descending)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    
    # Delete older ones
    if len(checkpoints) > max_to_keep:
        for step, path in checkpoints[max_to_keep:]:
            logging.info(f"Deleting old checkpoint: {path}")
            shutil.rmtree(path)

def train():
    os.environ["HF_HUB_OFFLINE"] = "1"
    cfg = TrainConfig()
    logging.info(f"Final training attempt for {cfg.model_id} via Manual Loss (OFFLINE MODE)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable() 
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load dataframes independently to maintain isolated action scales
    train_datasets = []
    for ds_id in cfg.dataset_ids:
        ds_name = ds_id.split("/")[-1]
        ds_path = f"C:/Users/Noniro/.cache/huggingface/lerobot/local/{ds_name}"
        
        parquet_path = f"{ds_path}/data/chunk-000/dataset.parquet"
        if not os.path.exists(parquet_path):
            parquet_path = f"{ds_path}/data/chunk-000/chunk-000.parquet"
            if not os.path.exists(parquet_path):
                parquet_path = f"{ds_path}/data/chunk-000"
                if not os.path.exists(parquet_path):
                    parquet_path = f"{ds_path}/data/chunk-000.parquet"
        
        df = pd.read_parquet(parquet_path)
        video_path = f"{ds_path}/videos/observation.images.cam_high/chunk-000/file-000.mp4"
        
        ds = OpenVLALeRobotDataset(
            df, video_path, processor, cfg.task_instruction, frame_skip=5
        )
        train_datasets.append(ds)
        
    from torch.utils.data import ConcatDataset
    dataloader = DataLoader(ConcatDataset(train_datasets), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    
    model.train()
    logging.info("Starting Training Loop (Manual Loss on Actions only)...")
    
    ce_loss = torch.nn.CrossEntropyLoss()
    
    for epoch in range(cfg.epochs):
        for step, batch in enumerate(dataloader):
            target_ids = batch.pop('target_ids').to(model.device)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass (WITHOUT internal labels)
            outputs = model(**batch)
            logits = outputs.logits # [batch, seq_len, vocab_size]
            
            # Action tokens are the last N tokens in the sequence
            N = target_ids.shape[1]
            action_logits = logits[:, -N-1:-1, :] 
            
            loss = ce_loss(action_logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
            
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            
            if (step + 1) % cfg.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            if step % 5 == 0:
                logging.info(f"Epoch {epoch} | Step {step} | Loss: {loss.item() * cfg.grad_accum_steps:.4f}")
            
            # Save periodic checkpoints
            if (step + 1) % cfg.save_steps == 0:
                checkpoint_path = os.path.join(cfg.output_dir, f"checkpoint-{step + 1}")
                logging.info(f"Intermediary save to {checkpoint_path}")
                model.save_pretrained(checkpoint_path)
                processor.save_pretrained(checkpoint_path)
                cleanup_checkpoints(cfg.output_dir, cfg.max_checkpoints)
                
    logging.info(f"Done! Saving to {cfg.output_dir}")
    model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    train()
