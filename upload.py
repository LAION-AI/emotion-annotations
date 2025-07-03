import os
import json
from datasets import Dataset, Audio, Features, Value
from tqdm import tqdm
import pandas as pd
from huggingface_hub import HfApi
from joblib import Parallel, delayed

# --- Configuration ---
# The root directory containing the folders with audio/json pairs.
root_dir = os.path.expanduser("~/emilia-yodas/EN")
# The Hugging Face Hub repository to push the dataset to.
dataset_repo = "laion/Emilia-Annotated-WIP"
# Number of parallel jobs to run for scanning folders. -1 uses all available cores.
N_JOBS = -1

# --- Helper Function for Parallel Processing ---
def scan_folder(foldername, root_dir):
    """
    Scans a single folder for valid .mp3 and .json file pairs.

    Args:
        foldername (str): The name of the folder to scan.
        root_dir (str): The root directory containing the folder.

    Returns:
        list: A list of tuples, where each tuple contains the path to an .mp3 file
              and its corresponding .json file.
    """
    folder_path = os.path.join(root_dir, foldername)
    if not os.path.isdir(folder_path):
        return []

    pairs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            base_name = filename[:-4]
            mp3_path = os.path.join(folder_path, filename)
            json_path = os.path.join(folder_path, f"{base_name}.json")
            if os.path.exists(json_path):
                pairs.append((mp3_path, json_path))
    return pairs

# --- Main Script ---

# Step 1: Collect valid .mp3 and .json file pairs using parallel processing
print("Starting parallel scan of folders...")
folder_list = os.listdir(root_dir)

# Use joblib to parallelize the folder scanning
# tqdm is wrapped around the execution to show progress.
parallel_results = Parallel(n_jobs=N_JOBS)(
    delayed(scan_folder)(foldername, root_dir) for foldername in tqdm(folder_list, desc="Scanning folders")
)

# Flatten the list of lists into a single list of file pairs
file_pairs = [pair for sublist in parallel_results for pair in sublist]
print(f"Found {len(file_pairs)} potential file pairs.")


# Step 2: Process files, extract metadata, and include raw JSON
data = {"audio": [], "caption": [], "emotions": [], "raw_json": []}
skipped_files = 0

for mp3_path, json_path in tqdm(file_pairs, desc="Processing files"):
    try:
        # Read the entire JSON file content first
        with open(json_path, "r", encoding="utf-8") as f:
            raw_json_content = f.read()
            metadata = json.loads(raw_json_content)

    except (json.JSONDecodeError, IOError) as e:
        # Handle cases where JSON is malformed or file can't be read
        # print(f"Warning: Skipping {json_path} due to error: {e}") # Uncomment for debugging
        skipped_files += 1
        continue

    # Validate that the metadata is a dictionary and has the required fields
    if (
        not isinstance(metadata, dict)
        or "caption" not in metadata
        or "emotions" not in metadata
    ):
        skipped_files += 1
        continue

    # Normalize the caption: if it starts with "AA", reduce it to a single "A"
    caption = metadata["caption"]
    if isinstance(caption, str) and caption.startswith("AA"):
        caption = "A" + caption[2:]

    # Store the valid entry, including the raw JSON string
    data["audio"].append(mp3_path)
    data["caption"].append(caption)
    data["emotions"].append(json.dumps(metadata["emotions"])) # Store emotions as a JSON string
    data["raw_json"].append(raw_json_content)


print(f"\nSuccessfully processed {len(data['audio'])} files.")
print(f"Skipped {skipped_files} invalid or unreadable files.")

# Step 3: Create the Hugging Face Dataset object with the new schema
features = Features({
    "audio": Audio(sampling_rate=16000),
    "caption": Value("string"),
    "emotions": Value("string"),
    "raw_json": Value("string") # Add the new raw_json field
})

# Convert the dictionary to a pandas DataFrame and then to a Dataset
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df, features=features)

print("\nDataset object created successfully.")
print(dataset)

# Step 4: Push the dataset to the Hugging Face Hub
print(f"\nPushing dataset to Hugging Face Hub repository: {dataset_repo}")
dataset.push_to_hub(dataset_repo, max_shard_size="500MB")

print("\nScript finished.")
# Note: The push_to_hub command is commented out.
# Uncomment it when you are ready to upload the data.
