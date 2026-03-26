from huggingface_hub import snapshot_download
import sys

model_id = "openvla/openvla-7b"

print(f"==================================================")
print(f"Starting download of {model_id}")
print(f"This model is roughly 14GB. Depending on your internet speed, this will take some time.")
print(f"==================================================")

try:
    # snapshot_download only downloads the files and puts them in your HF cache.
    # It does NOT attempt to load them all into your RAM at once.
    path = snapshot_download(repo_id=model_id)
    print(f"\nSUCCESS! Model downloaded to: {path}")
except Exception as e:
    print(f"\nERROR downloading model: {e}")
    sys.exit(1)

