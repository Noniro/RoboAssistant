import sys
import argparse

# Monkey-patch lerobot to stop hitting the network for local datasets
try:
    import lerobot.datasets.lerobot_dataset as lr_dataset
    lr_dataset.get_safe_version = lambda repo_id, rev: rev if rev else "main"
except Exception as e:
    pass

try:
    import lerobot.datasets.utils as lr_utils
    lr_utils.get_repo_versions = lambda repo_id: ["main"]
    lr_utils.get_safe_version = lambda repo_id, rev: rev if rev else "main"
except Exception as e:
    pass

from lerobot.datasets.v21.convert_dataset_v20_to_v21 import convert_dataset

if __name__ == "__main__":
    convert_dataset(
        repo_id="local/pick_place_finetune",
    )
    print("Success 1")
    convert_dataset(
        repo_id="local/pick_place_Iloveyoublock",
    )
    print("Success 2")
