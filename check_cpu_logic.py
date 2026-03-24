import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
import numpy as np

model_id = "openvla/openvla-7b"
print("Loading model for CPU check...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 1. Add tokens to tokenizer
action_tokens = [f"<action_{i:03d}>" for i in range(256)]
processor.tokenizer.add_tokens(action_tokens)

# 2. Load model (No quantization for CPU check, just to verify logic)
model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
model.resize_token_embeddings(len(processor.tokenizer))
model = prepare_model_for_kbit_training(model)

# 3. Create dummy multimodal sample
img = Image.new('RGB', (224, 224), color='red')
action_str = " ".join(action_tokens[:7])
prompt = f"In: Pick up the block.\nOut: {action_str}"
inputs = processor(text=prompt, images=img, return_tensors="pt")
inputs['labels'] = inputs['input_ids'].clone()

# 4. Try forward
print("Running forward...")
outputs = model(**inputs)
print(f"Loss: {outputs.loss}")

# 5. Try backward
print("Running backward...")
outputs.loss.backward()
print("Success! Logic is sound on CPU.")
