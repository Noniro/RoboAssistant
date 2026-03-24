import os
import pandas as pd
import json
import numpy as np

base_path = r"C:\Users\Noniro\.cache\huggingface\lerobot\local"
datasets = ["pick_place_finetune", "pick_place_Iloveyoublock"]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
        
for ds in datasets:
    meta_dir = os.path.join(base_path, ds, "meta")
    episodes_dir = os.path.join(meta_dir, "episodes")
    
    # In newer lerobot versions, episodes might be stored in a parquet inside a chunk folder
    chunk_dir = os.path.join(episodes_dir, "chunk-000")
    if os.path.exists(chunk_dir):
        # Could be an episodes.parquet inside chunk-000 or the chunk-000 is the parquet file
        parquet_path = os.path.join(chunk_dir, "episodes.parquet")
        if not os.path.exists(parquet_path):
            parquet_path = chunk_dir # sometimes it's directly named chunk-000 but it's a file
            
        print(f"Reading {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        jsonl_path = os.path.join(meta_dir, "episodes.jsonl")
        print(f"Writing {jsonl_path}")
        with open(jsonl_path, 'w') as f:
            for record in df.to_dict(orient="records"):
                if "data/chunk_index" in record:
                    record["chunk_index"] = record.pop("data/chunk_index")
                if "data/file_index" in record:
                    record["file_index"] = record.pop("data/file_index")
                f.write(json.dumps(record, cls=NumpyEncoder) + '\n')
        print(f"Success for {ds}")
