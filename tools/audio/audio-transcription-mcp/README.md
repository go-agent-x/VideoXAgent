# Audio Transcription MCP

A WhisperX-based audio transcription MCP service with word-level timestamps and speaker diarization.

## Features

- Supported formats: wav, mp3, m4a, flac, ogg, wma, mp4, mov, avi, mkv, webm
- Automatic language detection (99+ languages)
- Word-level timestamps through wav2vec2 alignment
- Speaker diarization using pyannote
- GPU acceleration, up to 70x real-time speed

## Installation

### 1. System Dependencies

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 2. Python Dependencies

```bash
bash scripts/install_profiles.sh audio --torch-backend cpu
source .venvs/audio/bin/activate
```

Run these commands from the repository root. For CUDA 12.6 environments, use `--torch-backend cu126` instead.

## Model Downloads

Models are normally downloaded automatically on first use. You can also download them in advance from public Hugging Face repositories for offline environments. Run from the repository root:

```bash
bash scripts/download_models.sh audio
```

### faster-whisper Models

Storage location: `~/.cache/huggingface/hub/`

```bash
# Download the base model (default, about 145MB, recommended for everyday use).
hf download Systran/faster-whisper-base

# Download the large-v3 model (about 3.1GB, for high-accuracy use cases).
hf download Systran/faster-whisper-large-v3
```

Available model sizes:

| Model | Size | VRAM Required | Use Case |
|------|------|----------|----------|
| tiny | ~75MB | ~1GB | Quick tests |
| base | ~145MB | ~1GB | Everyday use (default) |
| small | ~488MB | ~2GB | Balance of accuracy and speed |
| medium | ~1.5GB | ~5GB | Higher accuracy |
| large-v2 | ~3.1GB | ~10GB | High accuracy |
| large-v3 | ~3.1GB | ~10GB | Highest accuracy |

### wav2vec2 Alignment Models

Used to generate word-level timestamps. Storage location: `~/.cache/torch/hub/`

```bash
# English alignment model (built into torchaudio, downloaded automatically)
# Chinese alignment model
hf download jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
```

To download wav2vec2_fairseq_base_ls960_asr_ls960.pth manually:

```bash
# Download from torch hub.
python -c "import torchaudio; torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()"
```

The model file will be saved to `~/.cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth`.

### Speaker Diarization Models (Optional)

Speaker diarization requires authorized access to the pyannote models on Hugging Face.

**Step 1: Accept the Model Terms of Use**

Visit the following links and accept the user agreements:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

**Step 2: Get a Hugging Face Token**

Create an access token at https://huggingface.co/settings/tokens.

**Step 3: Configure the Environment Variable**

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Add this line to `~/.bashrc` or `~/.zshrc` to persist the setting.

**Step 4: Download the Models**

```bash
hf download pyannote/speaker-diarization-3.1 --token "$HF_TOKEN"
hf download pyannote/segmentation-3.0 --token "$HF_TOKEN"
```

## Verify the Installation

```bash
# Check that whisperx imports successfully.
python -c "import whisperx; print('WhisperX OK')"

# Check whether a GPU is available.
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Environment Variables

| Variable | Default | Description |
|------|--------|------|
| `WHISPERX_MODEL_SIZE` | large-v3 | Default model size |
| `WHISPERX_COMPUTE_TYPE` | float16 | Compute precision (float16/int8) |
| `HF_TOKEN` | - | Hugging Face token (required for speaker diarization) |

Model size and compute precision use this precedence: **explicit arguments for the current tool call > environment variables at server startup > defaults in the table above**.
`model_size` and `compute_type` can each be omitted independently; omitted values use the server configuration.
CPU execution still changes the compute precision to `int8`.

When launching through MCP, update the existing fields in `.mcp.profiles.json` under
`mcpServers.audio-transcription.env`, for example:

```json
{
  "WHISPERX_MODEL_SIZE": "base",
  "WHISPERX_COMPUTE_TYPE": "int8"
}
```

These defaults are read at server startup, so restart the MCP service after changing them. Explicit `env` values
in the launch configuration are not automatically changed by environment variables with the same names in the parent shell.
The existing `auto_fallback` mechanism may still try a larger model from the local cache if the transcription is too sparse.
This configuration determines the initial model and does not change the automatic fallback behavior.

## Available Tools

The segment timestamps, word-level timestamps, and formatted times returned by `transcribe_audio` are all relative to the **original input file**,
not the start of the extracted clip. For example, when extracting seconds `60–70`, text at second `1` of the clip is returned
at second `61` of the original file. The offset is `0` for a full file or when only an end time is specified.

If automatic fallback narrows the transcription interval, timestamps use the actual start of that extraction. The top-level
`start_time` / `end_time` fields describe the final selected interval. If a word's start or end time cannot be determined,
the corresponding endpoint is returned as `null`. If word alignment fails entirely, the transcribed segments are still returned with their timestamps adjusted,
along with an `alignment_warning`.

| Tool | Description |
|--------|------|
| `transcribe_audio` | Transcribes audio and video files with word-level timestamps |
| `transcribe_with_diarization` | Transcribes speech and identifies different speakers |
| `clear_whisperx_cache` | Clears the model cache to free GPU memory |
