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
# Number of parallel jobs to run. -1 uses all available CPU cores.
N_JOBS = -1

# --- Helper Functions for Parallel Processing ---

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

def process_file_pair(mp3_path, json_path):
    """
    Processes a single mp3/json pair, validates it, and extracts data.

    Args:
        mp3_path (str): The file path to the MP3 audio file.
        json_path (str): The file path to the corresponding JSON metadata file.

    Returns:
        dict: A dictionary containing the extracted data ('audio', 'caption',
              'emotions', 'raw_json') if the pair is valid.
        None: Returns None if the file pair is invalid or cannot be read.
    """
    try:
        # Read the entire JSON file content first
        with open(json_path, "r", encoding="utf-8") as f:
            raw_json_content = f.read()
            metadata = json.loads(raw_json_content)
    except (json.JSONDecodeError, IOError):
        # Return None if JSON is malformed or the file can't be read
        return None

    # Validate that the metadata is a dictionary and has the required fields
    if (
        not isinstance(metadata, dict)
        or "caption" not in metadata
        or "emotions" not in metadata
    ):
        return None

    # Normalize the caption: if it starts with "AA", reduce it to a single "A"
    caption = metadata["caption"]
    if isinstance(caption, str) and caption.startswith("AA"):
        caption = "A" + caption[2:]

    # Return the processed data as a dictionary
    return {
        "audio": mp3_path,
        "caption": caption,
        "emotions": json.dumps(metadata["emotions"]),
        "raw_json": raw_json_content,
    }

# --- Main Script ---

# Step 1: Collect valid .mp3 and .json file pairs using parallel folder scanning
print("Starting parallel scan of folders...")
folder_list = os.listdir(root_dir)
parallel_results = Parallel(n_jobs=N_JOBS)(
    delayed(scan_folder)(foldername, root_dir) for foldername in tqdm(folder_list, desc="Scanning folders")
)
# Flatten the list of lists into a single list of file pairs
file_pairs = [pair for sublist in parallel_results for pair in sublist]
print(f"Found {len(file_pairs)} potential file pairs.")


# Step 2: Process files in parallel to extract metadata
print("\nStarting parallel processing of files...")
processed_results = Parallel(n_jobs=N_JOBS)(
    delayed(process_file_pair)(mp3_path, json_path) for mp3_path, json_path in tqdm(file_pairs, desc="Processing files")
)

# Filter out the None values from invalid/skipped files
valid_data = [item for item in processed_results if item is not None]
skipped_files = len(file_pairs) - len(valid_data)

print(f"\nSuccessfully processed {len(valid_data)} files.")
print(f"Skipped {skipped_files} invalid or unreadable files.")


# Step 3: Create the Hugging Face Dataset object with the new schema
features = Features({
    "audio": Audio(sampling_rate=16000),
    "caption": Value("string"),
    "emotions": Value("string"),
    "raw_json": Value("string")
})

# Convert the list of dictionaries directly to a pandas DataFrame
df = pd.DataFrame(valid_data)
dataset = Dataset.from_pandas(df, features=features)

print("\nDataset object created successfully.")
print(dataset)

# Step 4: Push the dataset to the Hugging Face Hub
print(f"\nPushing dataset to Hugging Face Hub repository: {dataset_repo}")
dataset.push_to_hub(dataset_repo, max_shard_size="500MB")

print("\nScript finished.")
# Note: The push_to_hub command is commented out.
# Uncomment it when you are ready to upload the data.
