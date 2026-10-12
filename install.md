# Installation Guide

## Prerequisites

- Python 3.11 (required by the profile installer)
- FFmpeg
- Git (needed to install the pinned SAM2 revision)
- A CUDA-capable GPU is optional

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## 1. Install dependency profiles

Install from a source checkout and run all commands below from the repository
root. `scripts/install_profiles.sh` is the installation entry point for project,
MCP, development, and evaluation Python dependencies. MCP dependencies are installed
once during setup. At inference time, every MCP server starts its already-installed
Python interpreter; no online installation is performed.

For a CPU installation:

```bash
bash scripts/install_profiles.sh all --torch-backend cpu
```

For the currently supported CUDA 12.6 stack and GPU PaddlePaddle:

```bash
bash scripts/install_profiles.sh all --torch-backend cu126 --paddle-backend gpu
```

The command creates these isolated environments:

| Profile | Environment | Servers |
|---|---|---|
| `base` | `.venvs/base` | Agent and lightweight/OpenCV/API-based MCPs |
| `vision` | `.venvs/vision` | Torch vision, detection, SAM/SAM2, depth, tracking |
| `audio` | `.venvs/audio` | WhisperX, diarization, audio events, emotion analysis |
| `ocr` | `.venvs/ocr` | PaddleOCR on NumPy 1.26 |
| `music` | `.venvs/music` | Shazamio on NumPy 2.2 |

Each profile has its own interpreter under `.venvs/<profile>/bin/python`.

Install only selected profiles when the complete tool suite is unnecessary:

```bash
bash scripts/install_profiles.sh base vision --torch-backend cpu
bash scripts/install_profiles.sh audio --torch-backend cu126
bash scripts/install_profiles.sh ocr --paddle-backend cpu
```

The `base` profile installs the project in editable mode, including its Agent
runtime dependencies, together with `requirements/profiles/base.txt` in one pip
dependency resolution. There is no separate root requirements installation step.

Development and evaluation dependencies are optional additions to `base`:

```bash
# build, pytest, black, and ruff
bash scripts/install_profiles.sh base --dev

# pandas, PyArrow, and Matplotlib for evaluation data
bash scripts/install_profiles.sh base --evaluation

# Both extras, with the audio environment as well
bash scripts/install_profiles.sh base audio --dev --evaluation

# Both extras, with all five environments
bash scripts/install_profiles.sh all --dev --evaluation
```

These options only affect `base` and require `base` or `all` in the selected
profiles. Preview an installation without changing environments:

```bash
bash scripts/install_profiles.sh base --dev --evaluation --dry-run
```

See `bash scripts/install_profiles.sh --help` for interpreter, environment
directory, and CPU/CUDA options.

The installer checks each environment with `pip check` and an import smoke test;
base also checks the Agent CLI help and imports the web demo.
These checks do not verify model inference, API credentials, or GPU drivers.
It never edits `/etc/apt`, installs operating-system packages, or silently ignores
dependency metadata with `--no-deps`.

## 2. Use the generated MCP configuration

This checkout includes `.mcp.profiles.json`. The installer updates each server's
profile interpreter without overwriting a separate `.mcp.json`. It selects the
source configuration in this order: an explicit `--config-source` file, the
existing output file, then the repository's `.mcp.profiles.json`. The standalone
configurator uses the same order with its `--source` option. Review the selected
file's service paths, endpoints, and environment values.

When no source exists, provide an existing file with
`--config-source /path/to/your/mcp.json`, or use `--skip-config` to install
dependencies only. An invalid selected configuration reports an error; it is not
silently replaced by another source.
Known service scripts are launched through `scripts/mcp_runtime.py`. The
configurator inserts this wrapper once, preserving Python flags, service arguments,
proxies, and existing settings. The wrapper prepares offline/model-cache policy
before importing the service and runs it from the repository root.

The installer also forwards the selected backends to runtime configuration:
`--torch-backend cpu/cu126` sets CPU/CUDA for the vision/audio profiles being
installed, and `--paddle-backend cpu/gpu` sets the OCR device when installing OCR.
Unselected vision/audio backends keep their device settings. Composite consumers
such as the base-profile `segment-deep-query` also receive the settings for their
selected vision or OCR backends.

To update runtime settings without reinstalling dependencies, use:

```bash
.venvs/base/bin/python scripts/configure_mcp_profiles.py \
  --device cpu --ocr-device cpu \
  --disable-server music-identification
```

`--device` controls local Torch backends, including models used by composite
services. `--ocr-device` independently sets PaddleOCR/PPStructure's `use_gpu`
option. CUDA requires compatible packages, drivers, and hardware. InsightFace
uses CPU providers when CPU is selected; CUDA selection retains the configured
ONNX providers and only uses providers available in that environment.
After a CPU configuration, add `CUDAExecutionProvider` back to
`INSIGHTFACE_PROVIDERS` to enable ONNX GPU execution if the installed runtime
supports it. Restart running agents after changing the configuration.

Add `--device-profile vision` or `--device-profile audio` to limit `--device` to
one dependency profile; repeat it to select both. Omitting device options preserves
existing preferences. `MCP_DEVICE` overrides saved global backend device values;
explicit `TOOL_*` overrides remain authoritative where supported.

Repeat `--disable-server NAME` to omit multiple services. Unknown names are
rejected; disabling an already-absent known service is safe. Disabled services are
removed from the output, so restoring them requires a source configuration that
contains them. The installer accepts the same repeated `--disable-server` option.
If you installed only selected profiles, omit services whose dependencies are not
installed before launching the agent.

For a new checkout, either copy it into place:

```bash
cp .mcp.profiles.json .mcp.json
```

or set `mcp_json_file: .mcp.profiles.json` in an experiment YAML. To convert an
existing configuration while preserving its arguments and environment values:

```bash
python3.11 scripts/configure_mcp_profiles.py \
  --source .mcp.json \
  --output .mcp.profiles.json
```

## 3. Dependency maintenance

Project runtime dependencies and optional extras live in `pyproject.toml`;
profile manifests live in `requirements/profiles/`. The installer resolves both
inputs together for `base`. Keep shared version pins compatible with project
dependency ranges. Tool-local `requirements.txt` files are compatibility
references to profiles, not extra installation steps; they must not contain
their own Torch, Transformers, OpenCV or NumPy pins.

Do not combine the OCR and music profiles in one Python environment: PaddleOCR is
kept on NumPy 1.26, while Shazamio declares NumPy 2.2 or newer.

## 4. Model Downloads

All model weights come from public upstream repositories. Preview download
commands without downloading:

```bash
bash scripts/download_models.sh --venv-root .venvs --dry-run all
```

Select the profile environment root so each component uses its own interpreter:

```bash
bash scripts/download_models.sh --venv-root .venvs all
```

`--venv-root` also accepts an absolute custom directory; relative paths are resolved
from the repository root. Without this option, downloads retain the existing
`PATH`-based behavior.

See [MODEL_SETUP.md](MODEL_SETUP.md) for per-model sources and selective downloads.

### Audio Transcription (WhisperX)

Models are typically downloaded automatically on first use. To pre-download for offline environments:

```bash
# Large-v3 model (~3.1 GB, highest accuracy)
.venvs/audio/bin/hf download Systran/faster-whisper-large-v3

# Word-level alignment model (auto-downloaded via torchaudio)
.venvs/audio/bin/python -c "import torchaudio; torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()"
```

### Speaker Diarization (Optional)

Requires HuggingFace access to pyannote models:

1. Accept the license agreements at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

2. Set your HuggingFace token:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

3. Download the models:

```bash
.venvs/audio/bin/hf download pyannote/speaker-diarization-3.1 --token "$HF_TOKEN"
.venvs/audio/bin/hf download pyannote/segmentation-3.0 --token "$HF_TOKEN"
```

## 5. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPERX_MODEL_SIZE` | `large-v3` | WhisperX model size |
| `WHISPERX_COMPUTE_TYPE` | `float16` | Compute precision (`float16` / `int8`) |
| `HF_TOKEN` | — | HuggingFace token (required for speaker diarization) |
| `MCP_ALLOW_MODEL_DOWNLOAD` | `0` | Allow runtime model downloads (`0` = disabled) |

## 6. Verify Installation

```bash
# Check dependency metadata and imports for each installed profile
.venvs/base/bin/python -m pip check
.venvs/base/bin/python scripts/verify_profile.py base
.venvs/vision/bin/python scripts/verify_profile.py vision
.venvs/audio/bin/python scripts/verify_profile.py audio
.venvs/ocr/bin/python scripts/verify_profile.py ocr
.venvs/music/bin/python scripts/verify_profile.py music

# Check the generated server-to-profile mapping
python3.11 scripts/configure_mcp_profiles.py --check

# Check FFmpeg
ffmpeg -version | head -1
```
