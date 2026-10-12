# emotion-analysis-mcp

Emotion Analysis MCP Server — Facial Expression Recognition (FER) + Speech Emotion Recognition (SER) + multimodal fusion.

## Overview

Provides 3 MCP tools:

| Tool | Description | Input |
|------|------|------|
| `analyze_face_emotion` | Facial emotion analysis in images (7 Ekman emotion classes) | Image path |
| `analyze_speech_emotion` | Speech emotion recognition in audio/video | Audio/video path + optional time range |
| `analyze_multimodal_emotion` | Multimodal fusion for video (face + speech) | Video path + timestamp |

## Models

| Module | Model | Source |
|------|------|------|
| FER (facial expressions) | `trpakov/vit-face-expression` | Hugging Face, ViT-based, 7 Ekman emotion classes |
| SER (speech emotion) | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | SpeechBrain, wav2vec2-based |
| Face detection | InsightFace (buffalo_l) | ONNX Runtime |

Model cache locations:

- FER: Standard Hugging Face cache
- SER: `~/.cache/speechbrain/speechbrain--emotion-recognition-wav2vec2-IEMOCAP/`

After installing and activating the audio environment below, run `bash scripts/download_models.sh emotion` from the repository root
to download the emotion models from the public Hugging Face repositories.

## Dependency Installation

Run from the repository root:

```bash
bash scripts/install_profiles.sh audio --torch-backend cpu
source .venvs/audio/bin/activate
```

System dependencies:
- `ffmpeg` (required for audio extraction; must be on PATH)

Dependency versions are centrally maintained in `requirements/profiles/audio.txt` at the repository root.

## Configuration

Override the default configuration (defined in `_base/config.py`) using environment variables:

| Environment Variable | Default | Description |
|----------|--------|------|
| `EMOTION_DEVICE` | `cuda` | Inference device |
| `FER_MODEL` | `trpakov/vit-face-expression` | FER model path or Hugging Face repository |
| `SER_MODEL` | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | SER model path or Hugging Face repository |
| `FER_FUSION_WEIGHT` | `0.6` | Weight of the facial modality in multimodal fusion (0-1) |

## Model Downloads

Download the models from the public Hugging Face repositories before the first run:

```bash
# FER model
hf download trpakov/vit-face-expression

# SER model
hf download speechbrain/emotion-recognition-wav2vec2-IEMOCAP \
  --local-dir "$HOME/.cache/speechbrain/speechbrain--emotion-recognition-wav2vec2-IEMOCAP"
hf download facebook/wav2vec2-base
```

## Tool Details

### analyze_face_emotion

Analyze the facial emotions of all faces in an image.

**Parameters:**
- `image_path` (string, required): Absolute path to the image
- `box_threshold` (number, default=0.5): Face detection confidence threshold

**Output:** Probabilities for 7 Ekman emotion classes (angry, disgust, fear, happy, neutral, sad, surprise), bbox, age, and gender for each face.

### analyze_speech_emotion

Analyze speech emotion in audio/video.

**Parameters:**
- `file_path` (string, required): Absolute path to the audio or video file
- `start_time` (number, optional): Start time in seconds
- `end_time` (number, optional): End time in seconds

**Supported formats:** wav, mp3, m4a, flac, ogg, wma, mp4, mov, avi, mkv, webm, m4v

**Output:** Emotion class probabilities and the dominant emotion label.

### analyze_multimodal_emotion

Perform multimodal emotion analysis by fusing facial and speech emotions at a specified timestamp in a video.

**Parameters:**
- `video_path` (string, required): Absolute path to the video file
- `time_sec` (number, required): Timestamp to analyze, in seconds
- `fusion_weight_face` (number, default=0.6): Facial emotion weight (0-1)
- `audio_window` (number, default=5.0): Duration of the speech analysis window in seconds

**Output:** Facial emotions, speech emotions, and fusion results (including the weight of each modality).

## Running the Service

```bash
.venvs/audio/bin/python tools/composite/emotion-analysis-mcp/server.py
```

The MCP server communicates over stdio and is typically started automatically by the parent agent framework.
