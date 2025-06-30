import argparse
import json
import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import OrderedDict
import multiprocessing as mp

# --- Core ML/AI Libraries ---
import torch
import torch.distributed as dist
import torch.nn as nn
import librosa
from transformers import AutoProcessor, WhisperForConditionalGeneration
from huggingface_hub import snapshot_download
from tqdm import tqdm

# --- Configuration Section ---
# Set up logging to be clean and informative
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Performance & Model Configuration ---
# Multi-GPU configuration
NUM_GPUS = torch.cuda.device_count()
WORLD_SIZE = NUM_GPUS if NUM_GPUS > 0 else 1
USE_DDP = NUM_GPUS > 1

# Enable optimizations
USE_FLASH_ATTENTION_2 = True  # Enable Flash Attention 2
USE_TORCH_COMPILE = True       # Enable model compilation
USE_FP16 = True                # Use half-precision
BASE_BATCH_SIZE = 16           # Base batch size per GPU
MAX_AUDIO_LENGTH = 30          # Max audio length in seconds

# Model and Audio Parameters
WHISPER_MODEL_ID = "mkrausio/EmoWhisper-AnS-Small-v0.1"
HF_MLP_REPO_ID = "laion/Empathic-Insight-Voice-Small"
LOCAL_MODELS_DIR = Path("./models_cache")  # Dedicated directory for downloaded models
SAMPLING_RATE = 16000

# Parameters required for the MLP architecture
WHISPER_SEQ_LEN = 1500
WHISPER_EMBED_DIM = 768
PROJECTION_DIM = 64
MLP_HIDDEN_DIMS = [64, 32, 16]
MLP_DROPOUTS = [0.0, 0.1, 0.1, 0.1]

# List of supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}

class FullEmbeddingMLP(nn.Module):
    """ The defined architecture for the emotion/attribute MLP classifiers. """
    def __init__(self, seq_len: int, embed_dim: int, projection_dim: int, 
                 mlp_hidden_dims: List[int], mlp_dropout_rates: List[float]):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(seq_len * embed_dim, projection_dim)
        layers = [nn.ReLU(), nn.Dropout(mlp_dropout_rates[0])]
        current_dim = projection_dim
        for i, h_dim in enumerate(mlp_hidden_dims):
            layers.extend([
                nn.Linear(current_dim, h_dim), 
                nn.ReLU(), 
                nn.Dropout(mlp_dropout_rates[i+1])
            ])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1: 
            x = x.squeeze(1)
        projected = self.proj(self.flatten(x))
        return self.mlp(projected)

def load_models_and_processor(device: torch.device, use_fp16: bool = True):
    """
    Loads all required models (Whisper and MLPs) from Hugging Face Hub into memory.
    This function is called once per process in DDP mode.
    """
    rank = dist.get_rank() if USE_DDP else 0
    if rank == 0:
        logging.info("Starting model loading process...")
    
    LOCAL_MODELS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Synchronize processes before model loading
    if USE_DDP:
        dist.barrier()
    
    # --- Robust Whisper Model Loading with Flash Attention Fallback ---
    if rank == 0:
        logging.info(f"Loading Whisper model: {WHISPER_MODEL_ID}")
    
    whisper_dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32
    whisper_model = None

    if USE_FLASH_ATTENTION_2 and device.type == "cuda":
        try:
            if rank == 0:
                logging.info("Attempting to load Whisper model with Flash Attention 2...")
            whisper_model = WhisperForConditionalGeneration.from_pretrained(
                WHISPER_MODEL_ID, torch_dtype=whisper_dtype, cache_dir=LOCAL_MODELS_DIR,
                use_safetensors=True, attn_implementation="flash_attention_2"
            )
            if rank == 0:
                logging.info("Successfully loaded Whisper model with Flash Attention 2.")
        except (ValueError, ImportError) as e:
            if rank == 0:
                logging.warning(f"Flash Attention 2 failed: {e}")
                logging.warning("Falling back to default attention mechanism.")
            
    if whisper_model is None:
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL_ID, torch_dtype=whisper_dtype, cache_dir=LOCAL_MODELS_DIR,
            use_safetensors=True, attn_implementation="sdpa"
        )
        if rank == 0:
            logging.info("Whisper model loaded with default attention.")

    whisper_model.to(device)
    whisper_model.eval()
    
    # Compile model for better performance
    if USE_TORCH_COMPILE and device.type == "cuda":
        try:
            whisper_model = torch.compile(whisper_model, mode="reduce-overhead")
            if rank == 0:
                logging.info("Whisper model compiled with torch.compile")
        except Exception as e:
            if rank == 0:
                logging.warning(f"Model compilation failed: {e}")
    
    # Load processor
    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID, cache_dir=LOCAL_MODELS_DIR)
    
    # --- Load MLP Models ---
    mlp_models_dir = LOCAL_MODELS_DIR / "empathic_insight_models"
    if rank == 0:
        logging.info(f"Downloading MLP models from {HF_MLP_REPO_ID} to {mlp_models_dir}...")
        snapshot_download(
            repo_id=HF_MLP_REPO_ID, 
            local_dir=mlp_models_dir, 
            local_dir_use_symlinks=False, 
            ignore_patterns=["*.mp3", "*.md", ".gitattributes"]
        )
    
    # Synchronize after download
    if USE_DDP:
        dist.barrier()
    
    mlp_models = {}
    mlp_files = list(mlp_models_dir.glob("*.pth"))
    
    if rank == 0:
        logging.info(f"Found {len(mlp_files)} MLP model files.")
    
    for model_path in mlp_files:
        filename = model_path.stem
        parts = filename.split('_')
        dimension_name = '_'.join(parts[1:-1]) if "best" in parts[-1] else '_'.join(parts[1:])
        
        mlp_model = FullEmbeddingMLP(
            seq_len=WHISPER_SEQ_LEN, embed_dim=WHISPER_EMBED_DIM, projection_dim=PROJECTION_DIM,
            mlp_hidden_dims=MLP_HIDDEN_DIMS, mlp_dropout_rates=MLP_DROPOUTS
        ).to(device)
        
        state_dict = torch.load(model_path, map_location=device)
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in state_dict.items())
        
        mlp_model.load_state_dict(state_dict)
        mlp_model.eval()
        if use_fp16 and device.type == "cuda":
            mlp_model.half()
            
        # Compile MLP model
        if USE_TORCH_COMPILE and device.type == "cuda":
            try:
                mlp_model = torch.compile(mlp_model)
            except Exception as e:
                logging.warning(f"MLP compilation failed: {e}")
                
        mlp_models[dimension_name] = mlp_model
    
    if rank == 0:
        logging.info("All models loaded successfully.")
        
    return whisper_processor, whisper_model, mlp_models

class InferenceProcessor:
    """ A class to manage the loaded models and perform efficient batch inference. """
    def __init__(self, whisper_processor, whisper_model, mlp_models, device):
        self.whisper_processor = whisper_processor
        self.whisper_model = whisper_model
        self.mlp_models = mlp_models
        self.device = device

    def process_batch(self, audio_paths: List[Path]) -> List[Dict[str, Any]]:
        """ Processes a batch of audio files, generating transcription and emotion scores. """
        if not audio_paths: 
            return []
        
        # Load audio waveforms for the current batch
        processed_audios = [self._load_audio(p) for p in audio_paths]
        valid_audios = [a for a in processed_audios if a is not None]
        if not valid_audios: 
            return [{"error": "Invalid audio"} for _ in audio_paths]
        
        with torch.no_grad():
            # Pre-process the entire batch of audio
            inputs = self.whisper_processor(
                [a['waveform'] for a in valid_audios], 
                sampling_rate=SAMPLING_RATE,
                return_tensors="pt", 
                padding="max_length", 
                truncation=True
            ).to(self.device, non_blocking=True)
            
            if USE_FP16 and self.device.type == "cuda":
                inputs['input_features'] = inputs['input_features'].to(dtype=self.whisper_model.dtype)
            
            # --- EFFICIENT DUAL INFERENCE ---
            # 1. Run the encoder ONLY ONCE
            encoder_outputs = self.whisper_model.get_encoder()(inputs.input_features, return_dict=True)
            
            # 2. Use the encoder output for transcription (decoding)
            predicted_ids = self.whisper_model.generate(encoder_outputs=encoder_outputs)
            captions = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            # 3. Use the SAME encoder output's hidden state for emotion analysis
            embeddings = encoder_outputs.last_hidden_state
            
            batch_results = []
            for i in range(embeddings.size(0)):
                single_embedding = embeddings[i:i+1]
                emotion_predictions = {}
                for dim, mlp in self.mlp_models.items():
                    # Ensure consistent precision
                    if USE_FP16 and self.device.type == "cuda":
                        single_embedding = single_embedding.half()
                    emotion_predictions[dim] = mlp(single_embedding).item()
                
                # Assemble the final result with the caption and emotion scores
                batch_results.append({
                    "audio_file": str(valid_audios[i]['path']),  # Store full path for reference
                    "caption": captions[i].strip(),
                    "emotions": emotion_predictions
                })
        
        # Map results back to the original input paths to handle any loading errors
        final_results_map = {res['audio_file']: res for res in batch_results}
        return [final_results_map.get(str(p), {"audio_file": str(p), "error": "Processing failed"}) for p in audio_paths]

    def _load_audio(self, audio_path: Path):
        try:
            waveform, _ = librosa.load(
                audio_path, 
                sr=SAMPLING_RATE, 
                mono=True, 
                duration=MAX_AUDIO_LENGTH
            )
            return {'waveform': waveform, 'path': audio_path}
        except Exception as e:
            logging.error(f"Failed to load audio file {audio_path}: {e}")
            return None

def setup_ddp(rank, world_size):
    """ Initialize distributed processing with proper device assignment """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Explicitly set the CUDA device before initializing DDP
    torch.cuda.set_device(rank)
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="env://"
    )
    logging.info(f"Rank {rank} initialized on device cuda:{rank}")

def cleanup_ddp():
    dist.destroy_process_group()

def process_worker(rank, world_size, input_folder, file_chunk):
    """ Worker process for DDP execution """
    # Set device explicitly at the start
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)
    
    # Initialize DDP if using multi-GPU
    if USE_DDP and world_size > 1:
        try:
            setup_ddp(rank, world_size)
        except RuntimeError as e:
            logging.error(f"Rank {rank} failed to initialize DDP: {e}")
            return
    
    try:
        # Calculate dynamic batch size based on GPU memory
        if device.type == "cuda":
            total_mem = torch.cuda.get_device_properties(device).total_memory
            available_mem = total_mem * 0.8  # Use 80% of available memory
            batch_size = max(1, int(available_mem / (3 * 1024**3)))  # ~3GB per batch
        else:
            batch_size = BASE_BATCH_SIZE
        
        if rank == 0:
            logging.info(f"Using batch size: {batch_size} per GPU")
        
        # Load models
        whisper_processor, whisper_model, mlp_models = load_models_and_processor(
            device, USE_FP16
        )
        inference_processor = InferenceProcessor(
            whisper_processor, whisper_model, mlp_models, device
        )
        
        # Process file chunk
        with tqdm(total=len(file_chunk), desc=f"GPU {rank}", position=rank) as pbar:
            for i in range(0, len(file_chunk), batch_size):
                batch_paths = file_chunk[i:i + batch_size]
                batch_results = inference_processor.process_batch(batch_paths)
                
                # Save results
                for idx, result in enumerate(batch_results):
                    audio_path = batch_paths[idx]  # Original audio file path
                    json_path = audio_path.with_suffix('.json')  # JSON in same directory
                    
                    # Skip saving if error occurred
                    if "error" in result:
                        continue
                    
                    # Load existing data for merging
                    existing_data = {}
                    if json_path.exists():
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                existing_data = json.load(f)
                        except Exception as e:
                            logging.warning(f"Error reading {json_path}: {e}")
                    
                    # Update with new results
                    existing_data.update({
                        'caption': result.get('caption'),
                        'emotions': result.get('emotions'),
                        'source_audio_file': os.path.basename(result.get('audio_file'))
                    })
                    
                    # Ensure directory exists
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write with JSON
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, indent=4)
                    
                pbar.update(len(batch_paths))
    except Exception as e:
        logging.error(f"Error in worker process {rank}: {e}")
    finally:
        # Cleanup DDP
        if USE_DDP and world_size > 1:
            cleanup_ddp()

def process_folder(input_folder: Path):
    """
    Main orchestration function. Scans a folder, splits files across GPUs,
    and spawns worker processes for parallel processing.
    """
    start_time = time.time()
    logging.info(f"Available GPUs: {NUM_GPUS}")
    logging.info(f"Using DDP: {'Yes' if USE_DDP else 'No'}")
    logging.info(f"Using optimizations: FlashAttention2={USE_FLASH_ATTENTION_2}, TorchCompile={USE_TORCH_COMPILE}")
    
    # 1. Scan for all supported audio files recursively.
    logging.info(f"Scanning for audio files in '{input_folder}'...")
    all_audio_files = [p for p in input_folder.rglob('*') if p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
    
    if not all_audio_files:
        logging.warning("No supported audio files found.")
        return

    logging.info(f"Found {len(all_audio_files)} audio files.")
    
    # 2. Filter files that need processing
    files_to_process = []
    for audio_path in all_audio_files:
        json_path = audio_path.with_suffix('.json')
        if not json_path.exists():
            files_to_process.append(audio_path)
            continue
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'caption' not in data or 'emotions' not in data or len(data.get('emotions', {})) < 55:
                files_to_process.append(audio_path)
        except Exception:
            files_to_process.append(audio_path)
    
    if not files_to_process:
        logging.info("All files already processed.")
        return
        
    logging.info(f"Processing {len(files_to_process)} files...")
    
    # 3. Split files across GPUS
    chunk_size = len(files_to_process) // WORLD_SIZE
    file_chunks = [
        files_to_process[i:i + chunk_size] 
        for i in range(0, len(files_to_process), chunk_size)
    ]
    # Ensure we have exactly WORLD_SIZE chunks
    while len(file_chunks) < WORLD_SIZE:
        file_chunks.append([])
    
    # 4. Spawn worker processes
    processes = []
    for rank in range(WORLD_SIZE):
        if file_chunks[rank]:
            p = mp.Process(
                target=process_worker,
                args=(rank, WORLD_SIZE, input_folder, file_chunks[rank])
            )
            p.start()
            processes.append(p)
    
    for p in processes:
        p.join()
    
    duration = time.time() - start_time
    processed_count = len(files_to_process)
    logging.info(f"Processing complete. Total time: {duration:.2f} seconds")
    logging.info(f"Throughput: {processed_count/duration:.2f} files/sec")
    logging.info(f"Total files processed: {processed_count}")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="Multi-GPU audio processing for transcription and emotion analysis",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to folder containing audio files"
    )
    args = parser.parse_args()

    input_path = Path(args.input_folder)
    if not input_path.is_dir():
        logging.error(f"Invalid directory: {input_path}")
    else:
        process_folder(input_path)
