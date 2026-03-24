from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

model_id = "openvla/openvla-7b"
print(f"Testing {model_id}...")

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
tokenizer = processor.tokenizer

# 1. Check for action tokens
test_action = "<action_000>"
tokens = tokenizer.tokenize(test_action)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Tokenizing '{test_action}': {tokens} -> {ids}")

# 2. Check vocab size vs model
print(f"Tokenizer vocab size: {len(tokenizer)}")
# Load config only to see expected vocab
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print(f"Model config vocab size: {config.vocab_size}")

if 32000 in ids: # Default Llama-2-7B unknown or start of added tokens
    print("WARNING: action tokens might be treated as unknown or broken into sub-words.")

# 3. Check for specific token IDs
for i in range(5):
    t = f"<action_{i:03d}>"
    idx = tokenizer.convert_tokens_to_ids(t)
    print(f"Token {t} ID: {idx}")
