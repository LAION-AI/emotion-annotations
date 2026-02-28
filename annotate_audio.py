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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Performance & Model Configuration ---
# Multi-GPU configuration
NUM_GPUS = torch.cuda.device_count()
WORLD_SIZE = NUM_GPUS if NUM_GPUS > 0 else 1
USE_DDP = NUM_GPUS > 1

# Enable optimizations
USE_FLASH_ATTENTION_2 = True  # Enable Flash Attention 2
USE_TORCH_COMPILE = False  # Enable model compilation
USE_FP16 = True  # Use half-precision
BASE_BATCH_SIZE = 16  # Base batch size per GPU
MAX_AUDIO_LENGTH = 30  # Max audio length in seconds

# Model and Audio Parameters
WHISPER_MODEL_ID = "laion/BUD-E-Whisper"
HF_MLP_REPO_ID = "laion/Empathic-Insight-Voice-Plus"
LOCAL_MODELS_DIR = Path("./models_cache")  # Dedicated directory for downloaded models
SAMPLING_RATE = 16000

# Parameters required for the MLP architecture
WHISPER_SEQ_LEN = 1500
WHISPER_EMBED_DIM = 768
PROJECTION_DIM = 64
MLP_HIDDEN_DIMS = [64, 32, 16]
MLP_DROPOUTS = [0.0, 0.1, 0.1, 0.1]
POOLED_DIM = 3072  # 4 * 768 (mean + min + max + std)

# Quality expert filenames (use PooledEmbeddingMLP instead of FullEmbeddingMLP)
QUALITY_EXPERT_FILES = {
    "model_score_overall_quality_best.pth",
    "model_score_speech_quality_best.pth",
    "model_score_background_quality_best.pth",
    "model_score_content_enjoyment_best.pth",
}

# Total expected expert count: 55 emotion + 4 quality = 59
EXPECTED_TOTAL_EXPERTS = 59

# List of supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# Global flag for caption sanitization (set from CLI args)
SANITIZE_CAPTIONS = True


def sanitize_caption(text: str) -> str:
    """
    Fix a common BUD-E-Whisper decoder artifact where captions start with
    spurious capital letters before the real text begins.

    If two of the first three characters are uppercase, everything before the
    second uppercase letter is removed. This handles patterns like:
      "AThis is a sentence" -> "This is a sentence"
      "AbCdef ghi"          -> "Cdef ghi"
      "ABHello world"       -> "BHello world"

    Normal captions (e.g. "Hello world") are returned unchanged.
    """
    if not SANITIZE_CAPTIONS or len(text) < 3:
        return text

    prefix = text[:3]
    upper_positions = [i for i, c in enumerate(prefix) if c.isupper()]

    if len(upper_positions) >= 2:
        second_upper = upper_positions[1]
        return text[second_upper:]

    return text


class FullEmbeddingMLP(nn.Module):
    """The defined architecture for the emotion/attribute MLP classifiers."""

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        projection_dim: int,
        mlp_hidden_dims: List[int],
        mlp_dropout_rates: List[float],
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(seq_len * embed_dim, projection_dim)
        layers = [nn.ReLU(), nn.Dropout(mlp_dropout_rates[0])]
        current_dim = projection_dim
        for i, h_dim in enumerate(mlp_hidden_dims):
            layers.extend(
                [
                    nn.Linear(current_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout_rates[i + 1]),
                ]
            )
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        projected = self.proj(self.flatten(x))
        return self.mlp(projected)


class PooledEmbeddingMLP(nn.Module):
    """Architecture for quality expert models using pooled features (mean+min+max+std)."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        mlp_hidden_dims: List[int],
        mlp_dropout_rates: List[float],
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, projection_dim)
        layers = [nn.ReLU(), nn.Dropout(mlp_dropout_rates[0])]
        current_dim = projection_dim
        for i, h_dim in enumerate(mlp_hidden_dims):
            layers.extend(
                [
                    nn.Linear(current_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout_rates[i + 1]),
                ]
            )
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.proj(x))


def pool_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Pool encoder output [B, seq_len, 768] -> [B, 3072] using mean+min+max+std."""
    mean_pool = embedding.mean(dim=1)
    min_pool = embedding.min(dim=1).values
    max_pool = embedding.max(dim=1).values
    std_pool = embedding.std(dim=1)
    return torch.cat([mean_pool, min_pool, max_pool, std_pool], dim=1)


def load_models_and_processor(device: torch.device, use_fp16: bool = True):
    """
    Loads all required models (Whisper, emotion MLPs, and quality MLPs)
    from Hugging Face Hub into memory.
    This function is called once per process in DDP mode.
    """
    ddp_active = dist.is_initialized()
    rank = dist.get_rank() if ddp_active else 0
    if rank == 0:
        logging.info("Starting model loading process...")

    LOCAL_MODELS_DIR.mkdir(exist_ok=True, parents=True)

    # Synchronize processes before model loading
    if ddp_active:
        dist.barrier()

    # --- Robust Whisper Model Loading with Flash Attention Fallback ---
    if rank == 0:
        logging.info(f"Loading Whisper model: {WHISPER_MODEL_ID}")

    whisper_dtype = (
        torch.float16 if use_fp16 and device.type == "cuda" else torch.float32
    )
    whisper_model = None

    if USE_FLASH_ATTENTION_2 and device.type == "cuda":
        try:
            if rank == 0:
                logging.info(
                    "Attempting to load Whisper model with Flash Attention 2..."
                )
            whisper_model = WhisperForConditionalGeneration.from_pretrained(
                WHISPER_MODEL_ID,
                torch_dtype=whisper_dtype,
                cache_dir=LOCAL_MODELS_DIR,
                attn_implementation="flash_attention_2",
            )
            if rank == 0:
                logging.info(
                    "Successfully loaded Whisper model with Flash Attention 2."
                )
        except (ValueError, ImportError) as e:
            if rank == 0:
                logging.warning(f"Flash Attention 2 failed: {e}")
                logging.warning("Falling back to default attention mechanism.")

    if whisper_model is None:
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL_ID,
            torch_dtype=whisper_dtype,
            cache_dir=LOCAL_MODELS_DIR,
            attn_implementation="sdpa",
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
    whisper_processor = AutoProcessor.from_pretrained(
        WHISPER_MODEL_ID, cache_dir=LOCAL_MODELS_DIR
    )

    # --- Load MLP Models ---
    mlp_models_dir = LOCAL_MODELS_DIR / "empathic_insight_models"
    if rank == 0:
        logging.info(
            f"Downloading MLP models from {HF_MLP_REPO_ID} to {mlp_models_dir}..."
        )
        snapshot_download(
            repo_id=HF_MLP_REPO_ID,
            local_dir=mlp_models_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.mp3", "*.md", ".gitattributes"],
        )

    # Synchronize after download
    if ddp_active:
        dist.barrier()

    emotion_models = {}  # Full-sequence emotion/attribute experts
    quality_models = {}  # Pooled-feature quality experts
    mlp_files = list(mlp_models_dir.glob("*.pth"))

    if rank == 0:
        logging.info(f"Found {len(mlp_files)} MLP model files.")

    for model_path in mlp_files:
        filename = model_path.name
        stem = model_path.stem
        parts = stem.split("_")
        dimension_name = (
            "_".join(parts[1:-1]) if "best" in parts[-1] else "_".join(parts[1:])
        )

        is_quality = filename in QUALITY_EXPERT_FILES

        if is_quality:
            # Load as PooledEmbeddingMLP
            mlp_model = PooledEmbeddingMLP(
                input_dim=POOLED_DIM,
                projection_dim=PROJECTION_DIM,
                mlp_hidden_dims=MLP_HIDDEN_DIMS,
                mlp_dropout_rates=MLP_DROPOUTS,
            ).to(device)
        else:
            # Load as FullEmbeddingMLP
            mlp_model = FullEmbeddingMLP(
                seq_len=WHISPER_SEQ_LEN,
                embed_dim=WHISPER_EMBED_DIM,
                projection_dim=PROJECTION_DIM,
                mlp_hidden_dims=MLP_HIDDEN_DIMS,
                mlp_dropout_rates=MLP_DROPOUTS,
            ).to(device)

        state_dict = torch.load(model_path, map_location=device)
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = OrderedDict(
                (k.replace("_orig_mod.", ""), v) for k, v in state_dict.items()
            )

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

        if is_quality:
            quality_models[dimension_name] = mlp_model
        else:
            emotion_models[dimension_name] = mlp_model

    if rank == 0:
        logging.info(
            f"Loaded {len(emotion_models)} emotion experts + {len(quality_models)} quality experts."
        )

    return whisper_processor, whisper_model, emotion_models, quality_models


class InferenceProcessor:
    """A class to manage the loaded models and perform efficient batch inference."""

    def __init__(self, whisper_processor, whisper_model, emotion_models, quality_models, device):
        self.whisper_processor = whisper_processor
        self.whisper_model = whisper_model
        self.emotion_models = emotion_models
        self.quality_models = quality_models
        self.device = device

    def process_batch(self, audio_paths: List[Path]) -> List[Dict[str, Any]]:
        """Processes a batch of audio files, generating transcription, emotion scores, and quality scores."""
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
                [a["waveform"] for a in valid_audios],
                sampling_rate=SAMPLING_RATE,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device, non_blocking=True)

            if USE_FP16 and self.device.type == "cuda":
                inputs["input_features"] = inputs["input_features"].to(
                    dtype=self.whisper_model.dtype
                )

            # --- EFFICIENT DUAL INFERENCE ---
            # 1. Run the encoder ONLY ONCE
            encoder_outputs = self.whisper_model.get_encoder()(
                inputs.input_features, return_dict=True
            )

            # 2. Use the encoder output for transcription (decoding)
            predicted_ids = self.whisper_model.generate(encoder_outputs=encoder_outputs)
            captions = self.whisper_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            # 3. Use the SAME encoder output's hidden state for all experts
            embeddings = encoder_outputs.last_hidden_state

            # 4. Compute pooled features for quality experts (once for entire batch)
            pooled_features = pool_embedding(embeddings.float())
            if USE_FP16 and self.device.type == "cuda":
                pooled_features = pooled_features.half()

            batch_results = []
            for i in range(embeddings.size(0)):
                single_embedding = embeddings[i : i + 1]
                single_pooled = pooled_features[i : i + 1]

                # Run emotion experts (full sequence)
                all_predictions = {}
                if USE_FP16 and self.device.type == "cuda":
                    single_embedding = single_embedding.half()
                for dim, mlp in self.emotion_models.items():
                    all_predictions[dim] = round(mlp(single_embedding).item(), 4)

                # Run quality experts (pooled features)
                for dim, mlp in self.quality_models.items():
                    all_predictions[dim] = round(mlp(single_pooled).item(), 4)

                # Apply caption sanitization
                caption = sanitize_caption(captions[i].strip())

                # Assemble the final result
                batch_results.append(
                    {
                        "audio_file": str(valid_audios[i]["path"]),
                        "caption": caption,
                        "emotions": all_predictions,
                    }
                )

        # Map results back to the original input paths to handle any loading errors
        final_results_map = {res["audio_file"]: res for res in batch_results}
        return [
            final_results_map.get(
                str(p), {"audio_file": str(p), "error": "Processing failed"}
            )
            for p in audio_paths
        ]

    def _load_audio(self, audio_path: Path):
        try:
            waveform, _ = librosa.load(
                audio_path, sr=SAMPLING_RATE, mono=True, duration=MAX_AUDIO_LENGTH
            )
            return {"waveform": waveform, "path": audio_path}
        except Exception as e:
            logging.error(f"Failed to load audio file {audio_path}: {e}")
            return None


def setup_ddp(rank, world_size):
    """Initialize distributed processing with proper device assignment"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Explicitly set the CUDA device before initializing DDP
    torch.cuda.set_device(rank)

    # Initialize process group
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method="env://"
    )
    logging.info(f"Rank {rank} initialized on device cuda:{rank}")


def cleanup_ddp():
    dist.destroy_process_group()


def process_worker(rank, world_size, input_folder, file_chunk):
    """Worker process for DDP execution"""
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
        whisper_processor, whisper_model, emotion_models, quality_models = (
            load_models_and_processor(device, USE_FP16)
        )
        inference_processor = InferenceProcessor(
            whisper_processor, whisper_model, emotion_models, quality_models, device
        )

        # Process file chunk
        with tqdm(total=len(file_chunk), desc=f"GPU {rank}", position=rank) as pbar:
            for i in range(0, len(file_chunk), batch_size):
                batch_paths = file_chunk[i : i + batch_size]
                batch_results = inference_processor.process_batch(batch_paths)

                # Save results
                for idx, result in enumerate(batch_results):
                    audio_path = batch_paths[idx]  # Original audio file path
                    json_path = audio_path.with_suffix(
                        ".json"
                    )  # JSON in same directory

                    # Skip saving if error occurred
                    if "error" in result:
                        continue

                    # Load existing data for merging
                    existing_data = {}
                    if json_path.exists():
                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                existing_data = json.load(f)
                        except Exception as e:
                            logging.warning(f"Error reading {json_path}: {e}")

                    # Update with new results
                    existing_data.update(
                        {
                            "caption": result.get("caption"),
                            "emotions": result.get("emotions"),
                            "source_audio_file": os.path.basename(
                                result.get("audio_file")
                            ),
                        }
                    )

                    # Ensure directory exists
                    json_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write with JSON
                    with open(json_path, "w", encoding="utf-8") as f:
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
    logging.info(
        f"Using optimizations: FlashAttention2={USE_FLASH_ATTENTION_2}, TorchCompile={USE_TORCH_COMPILE}"
    )
    logging.info(f"Caption sanitization: {'ON' if SANITIZE_CAPTIONS else 'OFF'}")

    # 1. Scan for all supported audio files recursively.
    logging.info(f"Scanning for audio files in '{input_folder}'...")
    all_audio_files = [
        p
        for p in input_folder.rglob("*")
        if p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]

    if not all_audio_files:
        logging.warning("No supported audio files found.")
        return

    logging.info(f"Found {len(all_audio_files)} audio files. Checking which need processing...")

    # 2. Filter files that need processing, with a progress bar
    files_to_process = []
    for audio_path in tqdm(all_audio_files, desc="Checking existing files"):
        json_path = audio_path.with_suffix(".json")
        if not json_path.exists():
            files_to_process.append(audio_path)
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if (
                "caption" not in data
                or "emotions" not in data
                or len(data.get("emotions", {})) < EXPECTED_TOTAL_EXPERTS
            ):
                files_to_process.append(audio_path)
        except Exception:
            files_to_process.append(audio_path)

    if not files_to_process:
        logging.info("All files already processed.")
        return

    logging.info(f"Processing {len(files_to_process)} files...")

    # 3. Split files across GPUs
    num_workers = min(WORLD_SIZE, len(files_to_process))
    chunk_size = max(1, len(files_to_process) // num_workers)
    file_chunks = [
        files_to_process[i : i + chunk_size]
        for i in range(0, len(files_to_process), chunk_size)
    ]
    # Ensure we have exactly num_workers chunks
    while len(file_chunks) < num_workers:
        file_chunks.append([])

    # 4. Spawn worker processes
    processes = []
    for rank in range(num_workers):
        if file_chunks[rank]:
            p = mp.Process(
                target=process_worker,
                args=(rank, num_workers, input_folder, file_chunks[rank]),
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
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Multi-GPU audio processing for transcription, emotion analysis, and quality scoring",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to folder containing audio files"
    )
    parser.add_argument(
        "--no-caption-sanitize",
        action="store_true",
        help="Disable caption sanitization (removal of spurious leading capital letters)",
    )
    args = parser.parse_args()

    # Set global sanitization flag
    SANITIZE_CAPTIONS = not args.no_caption_sanitize

    input_path = Path(args.input_folder)
    if not input_path.is_dir():
        logging.error(f"Invalid directory: {input_path}")
    else:
        process_folder(input_path)
