# EmoNet - Voice Annotation Toolkit

This repository contains a suite of tools for analyzing audio files. The core functionalities are generating transcriptions, detailed emotional content scores, audio quality scores, and speaker-specific embeddings. The toolkit is optimized for GPU acceleration and designed for efficient batch processing of large datasets.

The primary goal is to generate rich, multi-faceted annotations for use in downstream tasks, such as training expressive Text-to-Speech (TTS) models. The underlying concepts are explored in more detail in LAION's research blog post, ["Do They See What We See?"](https://laion.ai/blog/do-they-see-what-we-see/).

## Companion project: VoiceNet

This repository (EmoNet / Empathic‑Insight‑Voice) answers **which emotion is expressed** — the 40
[EmoNet](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) emotions plus Valence, Arousal and
audio‑quality scores. Its companion, **VoiceNet**, answers the complementary question — **how the
speech is delivered** — predicting **57 perceptual voice/prosody dimensions** (voice age, arousal,
valence, warmth, brightness, tempo, resonance placement, speaking style, …) on a 0–6 rubric.

Run both models on the same clip to get a complete perceptual description of a voice performance.

- **VoiceNet project & docs:** <https://github.com/LAION-AI/voicenet>
- **VoiceNet live demo:** <https://projects.laion.ai/voicenet/>
- **VoiceNet taxonomy (57 dims × 0–6):** [`taxonomy/voicenet_taxonomy.md`](https://github.com/LAION-AI/voicenet/blob/main/taxonomy/voicenet_taxonomy.md)
- **EmoNet taxonomy (40 emotions):** [`taxonomy/emonet_taxonomy.md`](https://github.com/LAION-AI/voicenet/blob/main/taxonomy/emonet_taxonomy.md)

The VoiceNet taxonomy and project come from the forthcoming (not yet published) VoiceNet paper.

## Features

-   **59 Expert Annotations**: Provides transcription, **55 emotion/attribute scores**, and **4 audio quality scores** from a single, efficient pass over the audio using the **[Empathic Insight Voice Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus)** models, which are based on the **EmoNet** architecture and the **[BUD-E-Whisper](https://huggingface.co/laion/BUD-E-Whisper)** encoder.
-   **Batch Processing Script**: A standalone script (`annotate_audio.py`) to process entire folders of audio files recursively, leveraging multi-GPU batching for throughput.
-   **Intelligent File Handling**: The script automatically skips already processed files and can merge new annotations into existing JSON files without overwriting other data.
-   **Caption Sanitization**: Automatic cleanup of a known BUD-E-Whisper decoder artifact where captions occasionally start with spurious capital letters (e.g., `"AThis is a sentence"` → `"This is a sentence"`). Enabled by default, can be disabled with `--no-caption-sanitize`.
-   **Server/Client Architecture**: Includes a robust, asynchronous FastAPI server and a versatile client for real-time inference applications.
-   **Speaker Timbre Embeddings**: A script to generate unique speaker embeddings using `Orange/Speaker-wavLM-tbr`, allowing for clustering of speakers based on their unique vocal characteristics (timbre).
-   **Optimizations**: Leverages FP16 (half-precision), Flash Attention 2 (with a stable fallback), multi-GPU DDP, and targeted `torch.compile` for performance on modern NVIDIA GPUs.

## Models

All annotations are generated from a **single forward pass** through the frozen [laion/BUD-E-Whisper](https://huggingface.co/laion/BUD-E-Whisper) encoder. The encoder embeddings are then fed into two types of lightweight MLP expert heads:

| Type | Architecture | Input | Count | Parameters |
|------|-------------|-------|-------|------------|
| **Emotion/Attribute Experts** | `FullEmbeddingMLP` | Full sequence `[1500×768]` flattened | 55 | ~73.7M each |
| **Quality Experts** | `PooledEmbeddingMLP` | Pooled features `[3072]` (mean+min+max+std) | 4 | ~203K each |

### Quality Scores

The 4 quality experts are distilled from established audio quality models:

| Expert | Source | Description |
|--------|--------|-------------|
| **Overall Quality** | [DNSMOS](https://github.com/microsoft/DNS-Challenge) | Overall perceived audio quality (MOS-like) |
| **Speech Quality** | [DNSMOS](https://github.com/microsoft/DNS-Challenge) | Clarity and naturalness of the speech signal |
| **Background Quality** | [DNSMOS](https://github.com/microsoft/DNS-Challenge) | Absence of background noise and artifacts |
| **Content Enjoyment** | [Meta AudioBox](https://ai.meta.com/research/publications/audiobox-unified-audio-generation-with-natural-language-prompts/) | How engaging/enjoyable the spoken content is |

Expert weights are downloaded from [laion/Empathic-Insight-Voice-Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) on Hugging Face.

## Caption Sanitization

The BUD-E-Whisper decoder occasionally produces captions that start with one or two spurious capital letters before the real text begins (e.g., `"AThis is a sentence"` or `"ABHello world"`). The annotation script includes an automatic fix: if two of the first three characters are uppercase, everything before the second uppercase letter is removed.

This behavior is **enabled by default**. To disable it:
```bash
python annotate_audio.py /path/to/audio --no-caption-sanitize
```

## The Annotation Workflow

The tools in this repository can be used to create a powerful data pipeline for training next-generation TTS models:

1.  **Speaker Clustering**: Use the speaker embedding script on your dataset. This generates timbre-based embeddings that can cluster speech snippets from the same (or very similar) speakers together, even if they are speaking with different emotions. This allows you to assign a consistent pseudo-identity (e.g., `speaker_001`, `speaker_002`) to each voice in a large, unlabeled dataset.
2.  **Emotion, Quality & Transcription Annotation**: Run the `annotate_audio.py` script on the same dataset. This will generate a JSON file for each audio clip containing the transcription, all 55 emotion/attribute scores, and 4 audio quality scores from the **Empathic Insight Voice Plus** models.
3.  **Training Data Assembly**: You can now assemble a rich training dataset. Each data point can contain:
    -   The raw audio waveform.
    -   The text transcription (the target for the TTS model to speak).
    -   The speaker identity (from the clustering step).
    -   The emotion scores (which become controllable conditioning signals).
    -   The quality scores (which can be used for data filtering or conditioning).
4.  **Training a Controllable TTS Model**: With this data, one can train a TTS model to take a speaker identity, a text prompt, and a desired set of emotion scores as input. This allows the final model to generate speech for the same speaker but with different emotional expressions.

---

## 1. Standalone Folder Annotation Script (`annotate_audio.py`)

This is the primary tool for offline batch processing. It's a single, self-contained script that scans an input folder, finds all audio files, and generates detailed JSON annotations for each one.

### How it Works

-   **Input**: A folder path.
-   **Processing**:
    -   Recursively finds all supported audio files (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`).
    -   Intelligently checks if a corresponding `.json` file already exists and contains complete annotations (all 59 experts). If so, it skips the audio file.
    -   Splits files across all available GPUs for parallel processing via DDP.
    -   For each audio file, the Whisper encoder runs **once**. The full-sequence embeddings are used by the 55 emotion experts, while pooled features (mean+min+max+std) are used by the 4 quality experts.
    -   Saves the output to a `.json` file with the same name as the audio file. If the JSON file already exists but is missing data, this script will add the new annotations without overwriting existing, unrelated data.
-   **Output**: A `.json` file for each processed audio file, saved in the same directory.

### Usage

1.  **Prerequisites**: Ensure you have the required libraries installed:
    ```bash
    pip install torch transformers huggingface-hub librosa soundfile tqdm
    ```
2.  **Run the script**:
    ```bash
    python annotate_audio.py /path/to/your/audio_dataset
    ```
3.  **Disable caption sanitization** (optional):
    ```bash
    python annotate_audio.py /path/to/your/audio_dataset --no-caption-sanitize
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
        "score_overall_quality": 2.541,
        "score_speech_quality": 1.892,
        "score_background_quality": 3.215,
        "score_content_enjoyment": 4.103,
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

```bash
python generate_timbre_embeddings.py /path/to/your/audio_dataset --output-folder /path/to/embeddings
```
