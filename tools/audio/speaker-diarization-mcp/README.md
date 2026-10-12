# Speaker Diarization MCP

A WhisperX + pyannote speaker diarization MCP service that identifies each speaker's speech intervals in audio and video.

## Installation

```bash
bash scripts/install_profiles.sh audio --torch-backend cpu
source .venvs/audio/bin/activate
```

Run these commands from the repository root. For CUDA 12.6 environments, use `--torch-backend cu126` instead.

### System Dependencies

- ffmpeg

```bash
sudo apt install ffmpeg
```

### Model Downloads

A Hugging Face token is required, and you must accept the pyannote model agreements:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

```bash
hf download pyannote/speaker-diarization-3.1 --token "$HF_TOKEN"
hf download pyannote/segmentation-3.0 --token "$HF_TOKEN"
# Alternatively, run from the repository root:
bash scripts/download_models.sh diarization
```

## Environment Variables

| Variable | Description |
|------|------|
| `HF_TOKEN` | Hugging Face token (required to download pyannote models) |

## Available Tools

| Tool | Description |
|--------|------|
| `speaker_diarize` | Identifies each speaker's speech intervals in audio and video |
