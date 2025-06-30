# FINAL VERSION: Generates both transcription and emotion scores in one efficient pass.
# Includes Flash Attention 2 enabled by default with a robust fallback.

import os
import logging
import asyncio
import threading
import time
from typing import List, Dict, Any
from pathlib import Path
import shutil
import tempfile
import uuid
from collections import OrderedDict

# --- Core ML/AI Libraries ---
import torch
import torch.nn as nn
import librosa
from transformers import AutoProcessor, WhisperForConditionalGeneration
from huggingface_hub import snapshot_download

# --- Web Framework ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

# --- Configuration Section ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    DEVICE = "cuda:1"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
logging.info(f"Using device: {DEVICE}")

WHISPER_MODEL_ID = "mkrausio/EmoWhisper-AnS-Small-v0.1"
HF_MLP_REPO_ID = "laion/Empathic-Insight-Voice-Small"
LOCAL_MLP_MODELS_DIR = Path("./empathic_insight_models")

WHISPER_SEQ_LEN = 1500
WHISPER_EMBED_DIM = 768
PROJECTION_DIM = 64
MLP_HIDDEN_DIMS = [64, 32, 16]
MLP_DROPOUTS = [0.0, 0.1, 0.1, 0.1]
SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30

# --- PERFORMANCE OPTIMIZATION FLAGS ---
# Flash Attention 2 provides a significant speed-up on compatible GPUs.
# Set to True by default. The server will automatically fall back if it fails.
USE_FLASH_ATTENTION_2 = False #True
USE_TORCH_COMPILE_FOR_WHISPER = False # Kept False for stability
USE_TORCH_COMPILE_FOR_MLPS = True
USE_FP16 = True

class FullEmbeddingMLP(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, projection_dim: int, mlp_hidden_dims: List[int], mlp_dropout_rates: List[float]):
        super().__init__()
        if len(mlp_dropout_rates) != len(mlp_hidden_dims) + 1: raise ValueError(f"Dropout rates length error.")
        self.flatten = nn.Flatten(); self.proj = nn.Linear(seq_len * embed_dim, projection_dim)
        layers = [nn.ReLU(), nn.Dropout(mlp_dropout_rates[0])]
        current_dim = projection_dim
        for i, h_dim in enumerate(mlp_hidden_dims):
            layers.extend([nn.Linear(current_dim, h_dim), nn.ReLU(), nn.Dropout(mlp_dropout_rates[i+1])])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1)); self.mlp = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1: x = x.squeeze(1)
        projected = self.proj(self.flatten(x)); return self.mlp(projected)

def load_models_and_processor():
    logging.info("Starting model loading process...")
    logging.info(f"Loading Whisper model: {WHISPER_MODEL_ID}")
    whisper_dtype = torch.float16 if USE_FP16 and DEVICE != "cpu" else torch.float32
    whisper_model = None
    if USE_FLASH_ATTENTION_2 and DEVICE != "cpu":
        try:
            logging.info("Attempting to load Whisper model with Flash Attention 2...")
            whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID, torch_dtype=whisper_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="flash_attention_2").to(DEVICE)
            logging.info("Successfully loaded Whisper model with Flash Attention 2.")
        except Exception as e:
            logging.warning(f"Flash Attention 2 is enabled but failed to load. Error: {e}")
            logging.warning("FALLING BACK to the default 'sdpa' attention mechanism.")
    if whisper_model is None:
        whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID, torch_dtype=whisper_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa").to(DEVICE)
        logging.info("Whisper model loaded successfully with default attention.")
    whisper_model.eval()
    
    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    logging.info(f"Downloading MLP models from {HF_MLP_REPO_ID} to {LOCAL_MLP_MODELS_DIR}...")
    snapshot_download(repo_id=HF_MLP_REPO_ID, local_dir=LOCAL_MLP_MODELS_DIR, local_dir_use_symlinks=False, ignore_patterns=["*.mp3", "*.md", ".gitattributes"])
    mlp_models = {}
    mlp_files = list(LOCAL_MLP_MODELS_DIR.glob("*.pth"))
    logging.info(f"Found {len(mlp_files)} MLP model files.")
    for model_path in mlp_files:
        filename = model_path.stem; parts = filename.split('_'); dimension_name = '_'.join(parts[1:-1]) if "best" in parts[-1] else '_'.join(parts[1:])
        mlp_model = FullEmbeddingMLP(seq_len=WHISPER_SEQ_LEN, embed_dim=WHISPER_EMBED_DIM, projection_dim=PROJECTION_DIM, mlp_hidden_dims=MLP_HIDDEN_DIMS, mlp_dropout_rates=MLP_DROPOUTS).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()): state_dict = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in state_dict.items())
        mlp_model.load_state_dict(state_dict)
        mlp_model.eval()
        if USE_FP16 and DEVICE != "cpu": mlp_model.half()
        if USE_TORCH_COMPILE_FOR_MLPS:
            logging.info(f"Compiling MLP for: {dimension_name}...")
            mlp_model = torch.compile(mlp_model, mode="reduce-overhead", fullgraph=True)
        mlp_models[dimension_name] = mlp_model
    logging.info("All models loaded and optimized successfully.")
    return whisper_processor, whisper_model, mlp_models

class BatchInferenceManager:
    def __init__(self, whisper_processor, whisper_model, mlp_models):
        self.whisper_processor = whisper_processor
        self.whisper_model = whisper_model
        self.mlp_models = mlp_models

    def process_batch(self, audio_paths: List[Path]) -> List[Dict[str, Any]]:
        if not audio_paths: return []
        processed_audios = [self._load_and_process_audio(p) for p in audio_paths]
        valid_audios = [a for a in processed_audios if a is not None]
        if not valid_audios: return [{"error": "Invalid audio"} for _ in audio_paths]
        
        with torch.no_grad():
            inputs = self.whisper_processor([a['waveform'] for a in valid_audios], sampling_rate=SAMPLING_RATE, return_tensors="pt", padding="max_length", truncation=True).to(DEVICE, non_blocking=True)
            if USE_FP16 and DEVICE != "cpu": inputs['input_features'] = inputs['input_features'].to(dtype=self.whisper_model.dtype)
            
            # --- EFFICIENT DUAL INFERENCE ---
            # 1. Run the encoder ONLY ONCE
            encoder_outputs = self.whisper_model.get_encoder()(inputs.input_features, return_dict=True)
            
            # 2. Use the encoder output for transcription
            predicted_ids = self.whisper_model.generate(encoder_outputs=encoder_outputs)
            captions = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            # 3. Use the SAME encoder output for emotion analysis
            embeddings = encoder_outputs.last_hidden_state
            
            batch_results = []
            for i in range(embeddings.size(0)):
                # MLP processing
                single_embedding = embeddings[i:i+1]
                emotion_predictions = {dim: mlp(single_embedding).item() for dim, mlp in self.mlp_models.items()}
                
                # Assemble the final result with the caption
                batch_results.append({
                    "file": valid_audios[i]['path'].name,
                    "caption": captions[i].strip(),
                    "emotions": emotion_predictions
                })
        
        final_results = []
        result_map = {res['file']: res for res in batch_results}
        for path in audio_paths: final_results.append(result_map.get(path.name, {"file": path.name, "error": "Processing failed"}))
        return final_results

    def _load_and_process_audio(self, audio_path: Path):
        try:
            waveform, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True, duration=MAX_AUDIO_SECONDS)
            return {'waveform': waveform, 'path': audio_path}
        except Exception as e: logging.error(f"Failed to load audio file {audio_path}: {e}"); return None

# --- DynamicBatcher and FastAPI App (no changes needed) ---
@dataclass
class PendingRequest:
    future: asyncio.Future; temp_path: Path; enqueue_time: float = field(default_factory=time.time)
class DynamicBatcher:
    def __init__(self, inference_manager: BatchInferenceManager, batch_size: int, max_wait_time: float):
        self.inference_manager, self.batch_size, self.max_wait_time = inference_manager, batch_size, max_wait_time
        self.queue: List[PendingRequest] = []; self.lock = threading.Lock(); self.shutdown_event = threading.Event()
        self.processing_thread = threading.Thread(target=self._batching_loop, daemon=True); self.loop = None
    def start(self):
        self.loop = asyncio.get_running_loop(); self.processing_thread.start()
        logging.info(f"Dynamic batcher started with batch_size={self.batch_size}, max_wait_time={self.max_wait_time}s")
    def stop(self): self.shutdown_event.set(); self.processing_thread.join()
    async def add_request(self, temp_path: Path) -> Dict:
        future = self.loop.create_future(); request = PendingRequest(future=future, temp_path=temp_path)
        with self.lock: self.queue.append(request)
        try: return await asyncio.wait_for(future, timeout=MAX_AUDIO_SECONDS + 15)
        except asyncio.TimeoutError: raise HTTPException(status_code=504, detail="Request timed out in queue.")
    def _batching_loop(self):
        while not self.shutdown_event.is_set():
            batch_to_process = None
            with self.lock:
                if self.queue and (len(self.queue) >= self.batch_size or time.time() - self.queue[0].enqueue_time > self.max_wait_time):
                    batch_to_process = self.queue[:]; self.queue.clear()
            if batch_to_process: self._process_and_respond(batch_to_process)
            else: time.sleep(0.005)
    def _process_and_respond(self, batch: List[PendingRequest]):
        batch_paths = [req.temp_path for req in batch]; logging.info(f"Processing batch of size {len(batch_paths)}...")
        try:
            results = self.inference_manager.process_batch(batch_paths)
            result_map = {Path(res['file']).name: res for res in results}
            for request in batch:
                result = result_map.get(request.temp_path.name, {"error": "Processing failed."})
                self.loop.call_soon_threadsafe(request.future.set_result, result)
        except Exception as e:
            logging.error(f"Error processing batch: {e}", exc_info=True)
            error_result = {"error": f"An internal server error occurred: {e}"}
            for request in batch:
                if not request.future.done(): self.loop.call_soon_threadsafe(request.future.set_result, error_result)
        finally:
            for path in batch_paths:
                try: path.unlink()
                except OSError as e: logging.warning(f"Could not delete temp file {path}: {e}")
@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher; processor, whisper_model, mlps = load_models_and_processor()
    inference_manager = BatchInferenceManager(processor, whisper_model, mlps); batcher = DynamicBatcher(inference_manager, batch_size=16, max_wait_time=0.05)
    batcher.start(); yield; batcher.stop()
app = FastAPI(title="High-Performance Emotion Annotation API", lifespan=lifespan)
@app.post("/predict", summary="Analyze a single audio file for emotion and transcription")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file); temp_path = Path(temp_file.name)
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    result = await batcher.add_request(temp_path); return JSONResponse(content=result)
@app.get("/health", summary="Health check endpoint")
async def health_check(): return {"status": "ok", "device": DEVICE, "models_loaded": batcher is not None}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8022)
