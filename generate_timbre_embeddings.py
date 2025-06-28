# generate_timbre_embeddings.py
import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
import multiprocessing
from tqdm import tqdm

# --- Configuration ---
# !!! SET YOUR INPUT FOLDER HERE !!!
INPUT_FOLDER = "/mnt/raid/spirit/Speaker-wavLM-tbr/laion_audio_output_data/"
# The Hugging Face model name for timbre embeddings
MODEL_NAME = "Orange/Speaker-wavLM-tbr"
# Audio file extensions to search for
AUDIO_EXTENSIONS = ('.flac', '.mp3', '.wav', '.m4a', '.ogg', '.opus')
# Batch size for GPU inference (how many audio files are processed by the model in one go on a single GPU)
BATCH_SIZE = 16
# Target sample rate for the model
TARGET_SAMPLE_RATE = 16000
# Maximum audio duration in seconds. Files longer than this will be truncated.
# Set to None for no truncation. The model authors used 30s for some evaluations.
MAX_AUDIO_DURATION_SEC = 30
MAX_AUDIO_SAMPLES = int(TARGET_SAMPLE_RATE * MAX_AUDIO_DURATION_SEC) if MAX_AUDIO_DURATION_SEC else None

# --- Ensure spk_embeddings.py can be imported ---
# This adds the script's directory to Python's path, helping to find spk_embeddings.py
# and ensuring that spk_embeddings.py can import its own dependencies (like transformers).
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from spk_embeddings import EmbeddingsModel
except ImportError as e:
    print(f"Detailed ImportError when trying to import 'EmbeddingsModel' from 'spk_embeddings.py': {e}")
    print("\nThis could be because:")
    print("1. spk_embeddings.py is not in the current path (less likely if it's in the same directory).")
    print("2. A module that spk_embeddings.py itself tries to import is missing or cannot be imported.")
    print("   Common dependencies for spk_embeddings.py are: torch, torchaudio, transformers, huggingface_hub.")
    print("   Please ensure these are installed in your Python environment.")
    print(f"\nAttempted to load spk_embeddings.py from directory: {script_dir}")
    print("Current sys.path includes:")
    for p_item in sys.path:
        print(f"  - {p_item}")
    print("\nIf you haven't, try: pip install torch torchaudio transformers huggingface_hub")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

# --- Helper Functions ---

def find_audio_files(folder_path: Path, extensions: tuple) -> list:
    """Recursively finds all audio files with given extensions in a folder."""
    audio_files = []
    print(f"Scanning for audio files with extensions {extensions} in '{folder_path}'...")
    for ext in extensions:
        audio_files.extend(list(folder_path.rglob(f"*{ext.lower()}")))
        audio_files.extend(list(folder_path.rglob(f"*{ext.upper()}"))) # Case-insensitive
    
    # Sort and remove duplicates
    unique_audio_files = sorted(list(set(audio_files)))
    print(f"Found {len(unique_audio_files)} unique audio file paths.")
    return unique_audio_files

def load_and_resample_audio(file_path: Path, target_sr: int):
    """
    Loads an audio file, resamples it to target_sr, converts to mono,
    and returns it as a Tensor of shape (1, num_samples).
    """
    try:
        waveform, sample_rate = torchaudio.load(file_path)  # (num_channels, num_frames)
        
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            
        # Convert to mono if stereo, ensure shape is (1, num_samples)
        if waveform.shape[0] > 1:  # If more than 1 channel
            waveform = torch.mean(waveform, dim=0, keepdim=True) # Average channels
        elif waveform.shape[0] == 0: # Should not happen with valid audio
            print(f"Warning: Audio file {file_path} has 0 channels after loading. Skipping.")
            return None, None
        # If waveform.shape[0] == 1, it's already mono and correctly shaped (1, num_samples)
            
        return waveform, target_sr
    except Exception as e:
        print(f"Error loading or resampling {file_path}: {e}")
        return None, None

def worker_process_files(
    process_id: int,
    gpu_id: int,
    assigned_file_paths: list,  # List of all file paths assigned to this worker process
    model_name: str,
    progress_queue: multiprocessing.Queue,
    gpu_batch_size: int, # Actual batch size for GPU inference
    max_samples: int # Max samples for truncation
):
    """
    Worker function to process a list of audio files on a specific GPU.
    It loads the model once and processes its assigned files in batches.
    """
    # Set CUDA device for this specific process
    # This ensures each process uses its designated GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{0}") # After CUDA_VISIBLE_DEVICES, cuda:0 is the target GPU

    try:
        # Each process loads its own instance of the model
        model = EmbeddingsModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[Process {process_id} on GPU {gpu_id}] Error loading model '{model_name}': {e}")
        # If model loading fails, signal failure for all assigned files for progress tracking
        for _ in assigned_file_paths:
            progress_queue.put(1)
        return

    files_processed_by_worker = 0
    # Process assigned files in chunks of gpu_batch_size
    for i in range(0, len(assigned_file_paths), gpu_batch_size):
        current_chunk_paths = assigned_file_paths[i : i + gpu_batch_size]
        if not current_chunk_paths:
            continue

        waveforms_for_batch = []
        valid_paths_in_chunk = [] # To keep track of which files loaded successfully for this chunk
        max_len_in_chunk = 0

        for audio_file_path in current_chunk_paths:
            output_npy_path = audio_file_path.with_suffix('.npy')
            if output_npy_path.exists():
                # print(f"[Process {process_id}] Skipping {output_npy_path}, already exists.")
                progress_queue.put(1) # Signal progress for this skipped file
                continue # Skip this file for batch processing

            waveform, sr = load_and_resample_audio(audio_file_path, TARGET_SAMPLE_RATE)
            if waveform is None: # Loading or resampling failed
                progress_queue.put(1) # Signal failure for this file
                continue # Skip this file for batch processing

            # Truncate if MAX_AUDIO_SAMPLES is set and waveform is longer
            if max_samples is not None and waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples] # waveform is (1, num_samples)
            
            waveforms_for_batch.append(waveform.squeeze(0)) # Squeeze to (num_samples) for padding
            valid_paths_in_chunk.append(audio_file_path)
            if waveform.shape[1] > max_len_in_chunk:
                max_len_in_chunk = waveform.shape[1]
        
        if not waveforms_for_batch: # All files in this chunk were skipped or failed loading
            continue

        # Pad waveforms in the current batch to the max_len_in_chunk
        padded_waveforms_list = []
        for wf in waveforms_for_batch:
            padding_needed = max_len_in_chunk - wf.shape[0]
            # Pad at the end of the audio signal (dim 0 after squeeze)
            padded_wf = torch.nn.functional.pad(wf, (0, padding_needed))
            padded_waveforms_list.append(padded_wf)
        
        # Stack into a batch tensor: (current_chunk_actual_size, max_len_in_chunk)
        batch_input_tensor = torch.stack(padded_waveforms_list).to(device)

        try:
            with torch.no_grad():
                # The EmbeddingsModel.forward expects (batch_size, num_samples)
                embeddings_batch_output = model(batch_input_tensor) 
            
            # Save embeddings for each valid file in the processed chunk
            for idx, original_path in enumerate(valid_paths_in_chunk):
                output_npy_path = original_path.with_suffix('.npy')
                # embeddings_batch_output[idx] is a 1D tensor for the embedding
                np.save(output_npy_path, embeddings_batch_output[idx].cpu().numpy())
                files_processed_by_worker += 1
        except Exception as e:
            print(f"[Process {process_id} on GPU {gpu_id}] Error processing batch (starts with {valid_paths_in_chunk[0].name if valid_paths_in_chunk else 'N/A'}): {e}")
            # If batch processing fails, signal progress for all files intended for this batch as failures
            for _ in valid_paths_in_chunk: # These were files that loaded but failed during model inference/saving
                progress_queue.put(1)
            continue # to the next chunk of files for this worker
        
        # Signal progress for all successfully processed files in this batch
        for _ in valid_paths_in_chunk:
            progress_queue.put(1)
            
    # print(f"[Process {process_id} on GPU {gpu_id}] Finished. Processed {files_processed_by_worker} new files.")

# --- Main Logic ---
if __name__ == "__main__":
    # 'spawn' is recommended for CUDA with multiprocessing for safety
    multiprocessing.set_start_method("spawn", force=True)

    input_path = Path(INPUT_FOLDER)
    if not input_path.is_dir():
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist or is not a directory.")
        sys.exit(1)

    all_audio_files = find_audio_files(input_path, AUDIO_EXTENSIONS)

    if not all_audio_files:
        print("No audio files found in the specified folder and subfolders. Exiting.")
        sys.exit(0)
    
    # Filter out files for which embeddings (.npy) already exist
    files_to_process = []
    for f_path in all_audio_files:
        if not f_path.with_suffix('.npy').exists():
            files_to_process.append(f_path)
    
    if not files_to_process:
        print("All audio files seem to have corresponding .npy embeddings already. Exiting.")
        sys.exit(0)
        
    print(f"Found {len(files_to_process)} audio files that need new embeddings.")

    # Determine number of GPUs to use
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA device(s). Using all available GPUs.")
    else:
        print("CUDA not available. This script is optimized for GPU acceleration.")
        print("It will attempt to run on CPU using 1 process, which will be very slow.")
        num_gpus = 1 # Fallback to 1 CPU process (GPU ID will be 0, but device will be CPU in worker)
                     # The worker will still try to use 'cuda:0'. This needs a CPU fallback in worker.
                     # For simplicity, let's assume CUDA is the primary target.
                     # A robust CPU version would explicitly set device to 'cpu'.
        # Let's enforce CUDA for this version as per typical high-performance audio processing.
        if not torch.cuda.is_available(): # Re-check for clarity
            print("ERROR: No CUDA GPUs found. This script requires CUDA for efficient operation.")
            print("Please install PyTorch with CUDA support or adapt the script for CPU use.")
            sys.exit(1)


    # Distribute files_to_process among worker processes (one worker per GPU)
    worker_assignments = [[] for _ in range(num_gpus)]
    for idx, file_path in enumerate(files_to_process):
        # Simple round-robin assignment to GPUs
        worker_assignments[idx % num_gpus].append(file_path)

    processes = []
    progress_queue = multiprocessing.Queue() # For TQDM progress updates

    print(f"\nStarting {num_gpus} worker process(es). Each will handle its assigned files.")
    print(f"GPU batch size for model inference within each worker: {BATCH_SIZE}")
    print(f"Max audio samples per file (after {TARGET_SAMPLE_RATE}Hz resample): {MAX_AUDIO_SAMPLES or 'No limit'}\n")

    for i in range(num_gpus):
        if not worker_assignments[i]: # If a GPU has no files assigned (e.g., fewer files than GPUs)
            continue
        
        # process_id is 'i', gpu_id is also 'i' (0-indexed)
        # worker_assignments[i] is the list of file paths for this worker
        p = multiprocessing.Process(
            target=worker_process_files,
            args=(i, i, worker_assignments[i], MODEL_NAME, progress_queue, BATCH_SIZE, MAX_AUDIO_SAMPLES)
        )
        processes.append(p)
        p.start()

    # Progress bar handling using TQDM
    # total=len(files_to_process) because progress_queue.put(1) is called for each file attempt
    with tqdm(total=len(files_to_process), desc="Generating Embeddings", unit="file") as pbar:
        for _ in range(len(files_to_process)):
            progress_queue.get() # Wait for a signal that one file attempt is complete
            pbar.update(1)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\nAll embeddings generated successfully.")
    print(f"Embeddings (.npy files) are saved alongside their original audio files in '{INPUT_FOLDER}'.")