# Final client with caption display.

import asyncio
import aiohttp
import argparse
from pathlib import Path
import time
import os
import shutil
import tempfile
import requests
from typing import List, Dict

# --- Configuration ---
DEFAULT_SERVER_URL = "http://127.0.0.1:8022/predict"
DEMO_FILE_URL = "https://huggingface.co/datasets/laion/School_BUD-E/resolve/main/juniper-long-en.wav"
SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.flac')
CACHE_DIR = Path("./demo_cache")

CORE_EMOTION_KEYS: List[str] = [
    'Amusement', 'Elation', 'Pleasure_Ecstasy', 'Contentment', 'Thankfulness_Gratitude',
    'Affection', 'Infatuation', 'Hope_Enthusiasm_Optimism', 'Triumph', 'Pride',
    'Interest', 'Awe', 'Astonishment_Surprise', 'Concentration', 'Contemplation',
    'Relief', 'Longing', 'Teasing', 'Impatience_and_Irritability',
    'Sexual_Lust', 'Doubt', 'Fear', 'Distress', 'Confusion', 'Embarrassment', 'Shame',
    'Disappointment', 'Sadness', 'Bitterness', 'Contempt', 'Disgust', 'Anger',
    'Malevolence_Malice', 'Sourness', 'Pain', 'Helplessness', 'Fatigue_Exhaustion',
    'Emotional_Numbness', 'Intoxication_Altered_States_of_Consciousness', 'Jealousy_&_Envy'
]
ATTRIBUTE_KEYS: List[str] = [
    'Age', 'Arousal', 'Authenticity', 'Background_Noise', 'Confident_vs._Hesitant',
    'Gender', 'High-Pitched_vs._Low-Pitched', 'Monotone_vs._Expressive',
    'Recording_Quality', 'Serious_vs._Humorous', 'Soft_vs._Harsh',
    'Submissive_vs._Dominant', 'Valence', 'Vulnerable_vs._Emotionally_Detached',
    'Warm_vs._Cold'
]

def get_demo_file() -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    dest_path = CACHE_DIR / DEMO_FILE_URL.split('/')[-1]
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"Demo file found in cache: {dest_path}")
        return dest_path
    print(f"Downloading demo file to cache: {dest_path}...")
    try:
        with requests.get(DEMO_FILE_URL, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        print("Download complete."); return dest_path
    except requests.exceptions.RequestException as e: print(f"Error downloading file: {e}"); raise

def format_and_print_results(result: dict):
    print("\n" + "="*50); print(f"RESULTS FOR: {result.get('file', 'N/A')}"); print("="*50)
    if "error" in result:
        print(f"An error occurred: {result['error']}"); print("="*50 + "\n"); return
    
    # --- ADDED: Display Transcription ---
    caption = result.get('caption', 'No caption returned.')
    print("\n--- Transcription ---\n")
    print(f'"{caption}"\n')
    # --- END OF ADDITION ---
    
    if "emotions" not in result or not result["emotions"]:
        print("No emotion data returned from server."); print("="*50 + "\n"); return

    all_scores = result["emotions"]
    emotion_scores = {k: v for k, v in all_scores.items() if k in CORE_EMOTION_KEYS}
    attribute_scores = {k: v for k, v in all_scores.items() if k in ATTRIBUTE_KEYS}
    sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
    sorted_attributes = sorted(attribute_scores.items(), key=lambda item: item[1], reverse=True)
    
    print("--- Top 5 Core Emotions ---")
    for emotion, score in sorted_emotions[:5]: print(f"- {emotion.replace('_', ' '):<35} | Score: {score:.3f}")
    print("\n--- Top 5 Attributes / Dimensions ---")
    for attr, score in sorted_attributes[:5]: print(f"- {attr.replace('_', ' '):<35} | Score: {score:.3f}")
    print(f"\n--- All {len(sorted_emotions)} Emotion Scores (descending) ---")
    for emotion, score in sorted_emotions: print(f"- {emotion.replace('_', ' '):<40} {score:7.3f}")
    print(f"\n--- All {len(sorted_attributes)} Attribute Scores (descending) ---")
    for attr, score in sorted_attributes: print(f"- {attr.replace('_', ' '):<40} {score:7.3f}")
    print("="*50 + "\n")

async def query_single_file(session: aiohttp.ClientSession, file_path: Path, url: str) -> dict:
    try:
        with open(file_path, 'rb') as f:
            data = aiohttp.FormData(); data.add_field('file', f, filename=file_path.name, content_type='audio/mpeg')
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    json_res = await response.json(); json_res['file'] = file_path.name; return json_res
                else: return {"file": file_path.name, "error": f"HTTP {response.status}: {await response.text()}"}
    except Exception as e: return {"file": file_path.name, "error": str(e)}

async def main():
    parser = argparse.ArgumentParser(description="Enhanced client for the Emotion Annotation API.", formatter_class=argparse.RawTextHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true", help="Run inference on the cached demo file and print results.")
    group.add_argument("--benchmark", action="store_true", help="Run a high-throughput benchmark with 128 concurrent requests.")
    group.add_argument("--file", type=str, help="Path to a single audio file to process.")
    group.add_argument("--folder", type=str, help="Path to a folder of audio files to process.")
    parser.add_argument("--url", type=str, default=DEFAULT_SERVER_URL, help="URL of the prediction endpoint.")
    parser.add_argument("--concurrency", type=int, default=32, help="Number of concurrent requests for folder/benchmark modes.")
    args = parser.parse_args()

    if args.demo:
        print("--- Running in DEMO mode ---")
        demo_file_path = get_demo_file()
        async with aiohttp.ClientSession() as session:
            start_time = time.time(); result = await query_single_file(session, demo_file_path, args.url); end_time = time.time()
            format_and_print_results(result); print(f"Time for single inference: {end_time - start_time:.2f} seconds.")
    elif args.benchmark:
        print("--- Running in BENCHMARK mode ---"); num_copies = 128; demo_file_path = get_demo_file()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir); print(f"Creating {num_copies-1} temporary copies for benchmark...")
            files_to_process = [demo_file_path] + [shutil.copy(demo_file_path, temp_path / f"copy_{i}_{demo_file_path.name}") for i in range(num_copies - 1)]
            print("Starting benchmark..."); start_time = time.time()
            connector = aiohttp.TCPConnector(limit=args.concurrency, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [query_single_file(session, file, args.url) for file in files_to_process]
                results = await asyncio.gather(*tasks)
            end_time = time.time(); total_time, success_count = end_time - start_time, sum(1 for r in results if "error" not in r)
            print("\n--- Benchmark Complete ---")
            print(f"Total time to process {num_copies} files: {total_time:.2f} seconds")
            print(f"Throughput: {success_count / total_time:.2f} files/second" if total_time > 0 else "N/A")
            print(f"Effective time per file: {total_time / success_count:.3f} seconds" if success_count > 0 else "N/A")
    else:
        files_to_process = [Path(args.file)] if args.file else [p for p in Path(args.folder).rglob('*') if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not files_to_process: print("No supported audio files found."); return
        print(f"Found {len(files_to_process)} file(s) to process..."); start_time = time.time()
        connector = aiohttp.TCPConnector(limit=args.concurrency, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            results = await asyncio.gather(*[query_single_file(session, file, args.url) for file in files_to_process])
        end_time = time.time(); print(f"\n--- Processing of {len(files_to_process)} file(s) complete ---")
        for res in results[:3]: format_and_print_results(res)
        if len(results) > 3: print(f"...and {len(results) - 3} more results not shown.")
        print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
