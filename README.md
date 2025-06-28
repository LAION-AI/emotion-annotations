# High-Performance Voice Annotation Toolkit

This repository contains a suite of tools for analyzing audio files. The core functionalities are generating transcriptions, detailed emotional content scores, and speaker-specific embeddings. The toolkit is optimized for GPU acceleration and designed for efficient batch processing of large datasets.

The primary goal is to generate rich, multi-faceted annotations for use in downstream tasks, such as training expressive Text-to-Speech (TTS) models. The underlying concepts are explored in more detail in LAION's research blog post, ["Do They See What We See?"](https://laion.ai/blog/do-they-see-what-we-see/).

## Features

-   **Dual-Mode Annotation**: Provides both transcription and a rich set of 55 emotion/attribute scores from a single, efficient pass over the audio using the **Empathic Insight Voice Small** models, which are based on the **EmoNet** architecture.
-   **Batch Processing Script**: A standalone script (`annotate_audio.py`) to process entire folders of audio files recursively, leveraging GPU batching for throughput.
-   **Intelligent File Handling**: The script automatically skips already processed files and can merge new annotations into existing JSON files without overwriting other data.
-   **Server/Client Architecture**: Includes a robust, asynchronous FastAPI server and a versatile client for real-time inference applications.
-   **Speaker Timbre Embeddings**: A conceptualized script to generate unique speaker embeddings using `Orange/Speaker-wavLM-tbr`, allowing for clustering of speakers based on their unique vocal characteristics (timbre).
-   **Optimizations**: Leverages FP16 (half-precision), Flash Attention 2 (with a stable fallback), and targeted `torch.compile` for performance on modern NVIDIA GPUs.

## The Annotation Workflow

The tools in this repository can be used to create a powerful data pipeline for training next-generation TTS models:

1.  **Speaker Clustering**: Use the speaker embedding script on your dataset. This generates timbre-based embeddings that can cluster speech snippets from the same (or very similar) speakers together, even if they are speaking with different emotions. This allows you to assign a consistent pseudo-identity (e.g., `speaker_001`, `speaker_002`) to each voice in a large, unlabeled dataset.
2.  **Emotion & Transcription Annotation**: Run the `annotate_audio.py` script on the same dataset. This will generate a JSON file for each audio clip containing the transcription and all 55 emotion/attribute scores from the **Empathic Insight Voice** models.
3.  **Training Data Assembly**: You can now assemble a rich training dataset. Each data point can contain:
    -   The raw audio waveform.
    -   The text transcription (the target for the TTS model to speak).
    -   The speaker identity (from the clustering step).
    -   The emotion scores (which become controllable conditioning signals).
4.  **Training a Controllable TTS Model**: With this data, one can train a TTS model to take a speaker identity, a text prompt, and a desired set of emotion scores as input. This allows the final model to generate speech for the same speaker but with different emotional expressions.

---

## 1. Standalone Folder Annotation Script (`annotate_audio.py`)

This is the primary tool for offline batch processing. It's a single, self-contained script that scans an input folder, finds all audio files, and generates detailed JSON annotations for each one.

### How it Works

-   **Input**: A folder path.
-   **Processing**:
    -   Recursively finds all supported audio files (`.wav`, `.mp3`, `.m4a`, etc.).
    -   Intelligently checks if a corresponding `.json` file already exists and contains complete annotations. If so, it skips the audio file.
    -   Groups the remaining files into batches to process efficiently on the GPU.
    -   For each audio file, it generates a transcription and the 55 emotion/attribute scores.
    -   Saves the output to a `.json` file with the same name as the audio file. If the JSON file already exists but is missing data, this script will add the new annotations to it without overwriting existing, unrelated data.
-   **Output**: A `.json` file for each processed audio file, saved in the same directory.

### Usage

1.  **Prerequisites**: Ensure you have the required libraries installed: `pip install torch transformers huggingface-hub librosa soundfile tqdm`
2.  **Run the script**:
    ```bash
    python annotate_audio.py /path/to/your/audio_dataset
    ```

### Example Output (`your_audio.json`)
```json
{
    "source_audio_file": "your_audio.wav",
    "caption": "This is the transcribed text from the audio file.",
    "emotions": {
        "Amusement": 0.038,
        "Interest": 2.822,
        "Contentment": 1.776,
        "Age": 3.010,
        "Valence": 1.950,
        "...": "..."
    }
}
```

---

## 2. High-Performance Server & Client

For real-time applications, a robust server/client architecture is provided.

### Server (`server.py`)

A high-performance FastAPI server that pre-loads all models into GPU memory and uses a dynamic batching system to handle concurrent requests with high throughput.

#### Features
-   **Pre-loading**: All models are loaded once on startup.
-   **Dynamic Batching**: Groups incoming requests into optimal batches to maximize GPU utilization.
-   **Robust Optimizations**: Attempts to use Flash Attention 2 and falls back gracefully.

#### Usage
1.  **Launch the server**:
    ```bash
    uvicorn server:app --host 0.0.0.0 --port 8022
    ```

### Client (`client.py`)

An asynchronous client for interacting with the server. It can be used for single-file analysis, batch processing of folders, or running performance benchmarks.

#### Usage
-   **Run a demo on a sample file**:
    ```bash
    python client.py --demo
    ```
-   **Analyze a single local file**:
    ```bash
    python client.py --file /path/to/your/audio.wav
    ```
-   **Run a high-throughput benchmark**:
    ```bash
    python client.py --benchmark
    ```

---

## 3. Speaker Timbre Embedding Script (Advanced)

This script is a key component for enabling advanced voice cloning and speaker identification capabilities. While not developed as part of the core Empathic Insight project, we provide and recommend this tool for its powerful and complementary functionality. It uses the **[Orange/Speaker-wavLM-tbr](https://huggingface.co/Orange/Speaker-wavLM-tbr)** model.

### The Utility of Timbre Embeddings

The core concept is to separate the *identity* of a speaker from the *emotion* of their speech.
-   **Emotion** is conveyed through prosody, pitch, and energy, which change from moment to moment.
-   **Timbre** is the unique, underlying "fingerprint" or "color" of a voice that remains constant. It's what makes a specific person's voice recognizable, regardless of whether they are whispering happily or shouting angrily.

The `Speaker-wavLM-tbr` model is specifically trained to listen to an audio clip and generate an embedding vector that represents only this timbre, effectively ignoring the emotional content.

### From Embeddings to Controllable Voice Cloning

This emotion-invariant property is particularly useful for processing large-scale, unlabeled datasets:

1.  **The Goal**: You have thousands of audio clips from many different, unknown speakers expressing various emotions. You want to group all clips from "Speaker A" together, all clips from "Speaker B" together, and so on.

2.  **The Process**:
    -   Run the speaker embedding script on your entire dataset. Each audio file now has a corresponding timbre vector.
    -   Use a clustering algorithm (like K-Means, HDBSCAN, etc.) on these vectors.
    -   The algorithm will automatically group the vectors into distinct clusters. Because the embeddings are emotion-invariant, a happy clip and a sad clip from the same person will have very similar vectors and will be placed in the same cluster.

3.  **The Result: Pseudo Speaker Identities**:
    Each resulting cluster represents a unique speaker. You can now assign a **"pseudo speaker identity"** (e.g., `speaker_001`, `speaker_002`) to every audio file in your dataset based on which cluster it belongs to.

4.  **The Application: Controllable TTS**:
    With this final piece of data, you can train a highly sophisticated TTS model. The model can be conditioned on multiple inputs: a reference audio for the speaker's voice, a target text, and a target emotion. The training data would be structured as follows:
    -   **Reference Input**: Reference audio tokens, reference transcription, reference emotion scores.
    -   **Target Input**: Target transcription, target emotion conditioning.
    -   **Prediction**: The model predicts the target audio tokens.

    This allows the model to learn the separation of identity and emotion. At inference time, you can provide a single audio file of any speaker to define the voice timbre, and then ask the model to generate **any new text with any new emotion** in that person's voice. This enables true zero-shot voice cloning with emotional control.

### Usage
*(Note: A conceptual `speaker_embedding.py` script would be provided here. It would be structured similarly to `annotate_audio.py`, loading the `Orange/Speaker-wavLM-tbr` model and saving `.pt` or `.npy` embedding files for each audio file.)*

```bash
# Example usage (conceptual)
python speaker_embedding.py /path/to/your/audio_dataset --output-folder /path/to/embeddings
```
