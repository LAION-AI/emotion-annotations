import os
import json
from datasets import Dataset, Audio, Features, Value
from tqdm import tqdm
import pandas as pd
from huggingface_hub import HfApi

# Configuration
root_dir = os.path.expanduser("~/emilia-yodas/EN")
dataset_repo = "laion/Emilia-Annotated"

# Step 1: Collect valid .mp3 and .json file pairs
file_pairs = []
for foldername in tqdm(os.listdir(root_dir), desc="Scanning folders"):
    folder_path = os.path.join(root_dir, foldername)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            base_name = filename[:-4]
            mp3_path = os.path.join(folder_path, filename)
            json_path = os.path.join(folder_path, f"{base_name}.json")
            if os.path.exists(json_path):
                file_pairs.append((mp3_path, json_path))

# Step 2: Process files and extract metadata
data = {"audio": [], "caption": [], "emotions": []}
skipped = 0

for mp3_path, json_path in tqdm(file_pairs, desc="Processing files"):
    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)
    except Exception:
        skipped += 1
        continue

    # Validate required fields
    if not isinstance(metadata, dict) or "caption" not in metadata or "emotions" not in metadata:
        skipped += 1
        continue

    # Normalize caption: if it starts with "AA", reduce to single "A"
    caption = metadata["caption"]
    if caption.startswith("AA"):
        caption = "A" + caption[2:]

    # Store valid entry
    data["audio"].append(mp3_path)
    data["caption"].append(caption)
    data["emotions"].append(json.dumps(metadata["emotions"]))

print(f"Processed {len(data['audio'])} files | Skipped {skipped} invalid files")

# Step 3: Create Hugging Face Dataset object
features = Features({
    "audio": Audio(),
    "caption": Value("string"),
    "emotions": Value("string")
})

dataset = Dataset.from_pandas(pd.DataFrame(data), features=features)

# Step 4: Push dataset to Hugging Face Hub
dataset.push_to_hub(dataset_repo, max_shard_size="500MB")
