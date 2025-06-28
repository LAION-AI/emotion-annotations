# annotate_audio.py
# A self-contained, high-performance script for batch audio transcription and emotion analysis.

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from collections import OrderedDict

# --- Core ML/AI Libraries ---
import torch
import torch.nn as nn
import librosa
from transformers import AutoProcessor, WhisperForConditionalGeneration
from huggingface_hub import snapshot_download
from tqdm import tqdm

# --- Configuration Section ---
# Set up logging to be clean and informative
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Performance & Model Configuration ---
# Use a GPU if available
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    DEVICE = "cuda:1"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

# This flag enables the fastest possible attention mechanism if installed and compatible.
# The script will automatically fall back if it's not available.
USE_FLASH_ATTENTION_2 = False #True
USE_FP16 = True
BATCH_SIZE = 16 # Adjust based on your GPU VRAM (16 is good for 24GB)

# Model and Audio Parameters
WHISPER_MODEL_ID = "mkrausio/EmoWhisper-AnS-Small-v0.1"
HF_MLP_REPO_ID = "laion/Empathic-Insight-Voice-Small"
LOCAL_MODELS_DIR = Path("./models_cache") # A dedicated directory for downloaded models
SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30 # All audio will be truncated to this length

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
    def __init__(self, seq_len: int, embed_dim: int, projection_dim: int, mlp_hidden_dims: List[int], mlp_dropout_rates: List[float]):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(seq_len * embed_dim, projection_dim)
        layers = [nn.ReLU(), nn.Dropout(mlp_dropout_rates[0])]
        current_dim = projection_dim
        for i, h_dim in enumerate(mlp_hidden_dims):
            layers.extend([nn.Linear(current_dim, h_dim), nn.ReLU(), nn.Dropout(mlp_dropout_rates[i+1])])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1: x = x.squeeze(1)
        projected = self.proj(self.flatten(x)); return self.mlp(projected)

def load_models_and_processor():
    """
    Loads all required models (Whisper and MLPs) from Hugging Face Hub into memory.
    This function is called only once at the start of the script.
    """
    logging.info("Starting model loading process...")
    LOCAL_MODELS_DIR.mkdir(exist_ok=True)
    
    # --- Robust Whisper Model Loading with Flash Attention Fallback ---
    logging.info(f"Loading Whisper model: {WHISPER_MODEL_ID}")
    whisper_dtype = torch.float16 if USE_FP16 and DEVICE != "cpu" else torch.float32
    whisper_model = None

    if USE_FLASH_ATTENTION_2 and DEVICE != "cpu":
        try:
            logging.info("Attempting to load Whisper model with Flash Attention 2...")
            whisper_model = WhisperForConditionalGeneration.from_pretrained(
                WHISPER_MODEL_ID, torch_dtype=whisper_dtype, cache_dir=LOCAL_MODELS_DIR,
                use_safetensors=True, attn_implementation="flash_attention_2"
            ).to(DEVICE)
            logging.info("Successfully loaded Whisper model with Flash Attention 2.")
        except (ValueError, ImportError) as e:
            logging.warning(f"Flash Attention 2 is enabled but failed to load. Error: {e}")
            logging.warning("FALLING BACK to the default 'sdpa' attention mechanism.")
            
    if whisper_model is None:
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL_ID, torch_dtype=whisper_dtype, cache_dir=LOCAL_MODELS_DIR,
            use_safetensors=True, attn_implementation="sdpa"
        ).to(DEVICE)
        logging.info("Whisper model loaded successfully with default attention.")

    whisper_model.eval()
    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID, cache_dir=LOCAL_MODELS_DIR)
    
    # --- Load MLP Models ---
    mlp_models_dir = LOCAL_MODELS_DIR / "empathic_insight_models"
    logging.info(f"Downloading MLP models from {HF_MLP_REPO_ID} to {mlp_models_dir}...")
    snapshot_download(repo_id=HF_MLP_REPO_ID, local_dir=mlp_models_dir, local_dir_use_symlinks=False, ignore_patterns=["*.mp3", "*.md", ".gitattributes"])
    
    mlp_models = {}
    mlp_files = list(mlp_models_dir.glob("*.pth"))
    logging.info(f"Found {len(mlp_files)} MLP model files.")
    for model_path in tqdm(mlp_files, desc="Loading MLP Models"):
        filename = model_path.stem
        parts = filename.split('_')
        dimension_name = '_'.join(parts[1:-1]) if "best" in parts[-1] else '_'.join(parts[1:])
        
        mlp_model = FullEmbeddingMLP(
            seq_len=WHISPER_SEQ_LEN, embed_dim=WHISPER_EMBED_DIM, projection_dim=PROJECTION_DIM,
            mlp_hidden_dims=MLP_HIDDEN_DIMS, mlp_dropout_rates=MLP_DROPOUTS
        ).to(DEVICE)
        
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in state_dict.items())
        
        mlp_model.load_state_dict(state_dict)
        mlp_model.eval()
        if USE_FP16 and DEVICE != "cpu":
            mlp_model.half()
        mlp_models[dimension_name] = mlp_model
        
    logging.info("All models loaded successfully.")
    return whisper_processor, whisper_model, mlp_models

class InferenceProcessor:
    """ A class to manage the loaded models and perform efficient batch inference. """
    def __init__(self, whisper_processor, whisper_model, mlp_models):
        self.whisper_processor = whisper_processor
        self.whisper_model = whisper_model
        self.mlp_models = mlp_models

    def process_batch(self, audio_paths: List[Path]) -> List[Dict[str, Any]]:
        """ Processes a batch of audio files, generating transcription and emotion scores. """
        if not audio_paths: return []
        # Load audio waveforms for the current batch
        processed_audios = [self._load_audio(p) for p in audio_paths]
        valid_audios = [a for a in processed_audios if a is not None]
        if not valid_audios: return [{"error": "Invalid audio"} for _ in audio_paths]
        
        with torch.no_grad():
            # Pre-process the entire batch of audio
            inputs = self.whisper_processor(
                [a['waveform'] for a in valid_audios], sampling_rate=SAMPLING_RATE,
                return_tensors="pt", padding="max_length", truncation=True
            ).to(DEVICE, non_blocking=True)
            
            if USE_FP16 and DEVICE != "cpu":
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
                emotion_predictions = {dim: mlp(single_embedding).item() for dim, mlp in self.mlp_models.items()}
                
                # Assemble the final result with the caption and emotion scores
                batch_results.append({
                    "audio_file": valid_audios[i]['path'].name,
                    "caption": captions[i].strip(),
                    "emotions": emotion_predictions
                })
        
        # Map results back to the original input paths to handle any loading errors
        final_results_map = {res['audio_file']: res for res in batch_results}
        return [final_results_map.get(p.name, {"audio_file": p.name, "error": "Processing failed"}) for p in audio_paths]

    def _load_audio(self, audio_path: Path):
        try:
            waveform, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True, duration=MAX_AUDIO_SECONDS)
            return {'waveform': waveform, 'path': audio_path}
        except Exception as e:
            logging.error(f"Failed to load audio file {audio_path}: {e}")
            return None

def process_folder(input_folder: Path):
    """
    Main orchestration function. Scans a folder, processes audio files in batches,
    and saves the results to JSON files, with intelligent skipping and updating.
    """
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Batch size set to: {BATCH_SIZE}")
    
    # 1. Load all models into memory once.
    processor, model, mlps = load_models_and_processor()
    inference_processor = InferenceProcessor(processor, model, mlps)

    # 2. Scan for all supported audio files recursively.
    logging.info(f"Scanning for audio files in '{input_folder}'...")
    all_audio_files = [p for p in input_folder.rglob('*') if p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
    
    if not all_audio_files:
        logging.warning("No supported audio files found in the specified folder.")
        return

    logging.info(f"Found {len(all_audio_files)} total audio files.")

    # 3. Create a list of files that actually need processing.
    files_to_process = []
    for audio_path in tqdm(all_audio_files, desc="Checking existing files"):
        json_path = audio_path.with_suffix('.json')
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # If the essential keys are already present, we can skip this file.
                if 'caption' in data and 'emotions' in data and len(data['emotions']) >= 55:
                    continue # Skip this file
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Malformed JSON file found: {json_path}. It will be processed and overwritten.")
        
        files_to_process.append(audio_path)

    if not files_to_process:
        logging.info("All audio files have already been processed. Nothing to do.")
        return
        
    logging.info(f"Processing {len(files_to_process)} new or incomplete files...")
    
    # 4. Process the filtered list in batches.
    with tqdm(total=len(files_to_process), desc="Annotating Batches") as pbar:
        for i in range(0, len(files_to_process), BATCH_SIZE):
            batch_paths = files_to_process[i:i + BATCH_SIZE]
            
            # Perform inference on the current batch
            batch_results = inference_processor.process_batch(batch_paths)

            # 5. Save results, handling the update/merge logic.
            for idx, result in enumerate(batch_results):
                audio_path_in_batch = batch_paths[idx]
                json_path = audio_path_in_batch.with_suffix('.json')

                if "error" in result:
                    logging.error(f"Failed to process {audio_path_in_batch.name}: {result['error']}")
                    continue

                # Load existing data if the JSON file already exists (for merging)
                existing_data = {}
                if json_path.exists():
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except (json.JSONDecodeError, TypeError):
                        logging.warning(f"Overwriting malformed JSON file: {json_path}")
                        existing_data = {}

                # Add our new results to the data
                existing_data['caption'] = result.get('caption')
                existing_data['emotions'] = result.get('emotions')
                # Keep the original audio filename for reference
                existing_data['source_audio_file'] = result.get('audio_file')

                # Save the complete (new or merged) data
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=4)
            
            pbar.update(len(batch_paths))

    logging.info("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A self-contained script to batch-process a folder of audio files for transcription and emotion analysis.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the folder containing audio files to process."
    )
    args = parser.parse_args()

    input_path = Path(args.input_folder)
    if not input_path.is_dir():
        logging.error(f"Error: The provided path '{input_path}' is not a valid directory.")
    else:
        start_time = time.time()
        process_folder(input_path)
        end_time = time.time()
        logging.info(f"Total script execution time: {end_time - start_time:.2f} seconds.")
