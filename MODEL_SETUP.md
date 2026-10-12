# MCP Model Downloads and Setup

This project does not bundle model weights or depend on internal shared directories.
MCP services obtain weights from the model authors' public Hugging Face repositories,
GitHub releases, or download links published by the official projects. After preparing
the required weights, set `MCP_ALLOW_MODEL_DOWNLOAD=0` to disable runtime downloads
where supported by the service.

## Quick Start

Run all commands from the repository root. First, use the unified installer to
prepare the profiles you need. This example installs the vision, audio, and OCR
environments. The vision and audio profiles include `hf` through `huggingface-hub`;
install `curl` and `unzip` separately as system tools:

```bash
bash scripts/install_profiles.sh vision audio ocr --torch-backend cpu --paddle-backend cpu
```

Preview the download commands without downloading any files:

```bash
bash scripts/download_models.sh --venv-root .venvs --dry-run all
```

Select the profile environment root to use the correct `hf` and `python` for each
component. Audio preloading requires `torchaudio`, while OCR preloading requires
`paddleocr`; mixed requests and `all` select these environments separately:

```bash
bash scripts/download_models.sh --venv-root .venvs all
```

You can also prepare individual components:

```bash
bash scripts/download_models.sh --venv-root .venvs grounding-dino sam sam2
bash scripts/download_models.sh --venv-root /absolute/path/to/profiles emotion
WHISPER_MODEL_SIZE=base bash scripts/download_models.sh --venv-root .venvs audio
```

Relative environment roots are resolved from the repository root. Missing selected
executables cause an error; the downloader does not fall back to another installed
Python. Omitting `--venv-root` preserves the existing `PATH`-based behavior for
callers that activate their own environment.

By default, Hugging Face models use the cache under `HF_HOME`, and checkpoints
that require explicit file paths are placed in the repository's `models/` directory.
Use `HF_HOME`, `TORCH_HOME`, and `MODEL_DIR` to customize these locations. Run
`bash scripts/download_models.sh --help` to list all components.

## Model Sources and Usage

| Model | Public upstream source | Tools | Default location or configuration |
|---|---|---|---|
| all-MiniLM-L6-v2 | [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | temporal-ground | `SBERT_MODEL=sentence-transformers/all-MiniLM-L6-v2` |
| Depth Anything V2 Small | [Hugging Face](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | depth-estimation | `DEPTH_MODEL=depth-anything/Depth-Anything-V2-Small-hf` |
| VideoMAE Kinetics | [Hugging Face](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) | action-classify | `VIDEOMAE_MODEL=MCG-NJU/videomae-base-finetuned-kinetics` |
| CLIP ViT-B/32 | [Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32) | moment-retrieval | `CLIP_MODEL=openai/clip-vit-base-patch32` |
| BERT base uncased | [Hugging Face](https://huggingface.co/google-bert/bert-base-uncased) | grounding-dino | `models/bert-base-uncased/` |
| GroundingDINO-B | [Official GitHub release](https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha) | grounding-dino | `models/GroundingDINO/groundingdino_swinb_cogcoor.pth` |
| SAM ViT-H | [Official GitHub repository](https://github.com/facebookresearch/segment-anything#model-checkpoints) | auto-segment, prompt-segment | `models/sam_vit_h_4b8939.pth` |
| SAM 2.1 Hiera-L | [Official GitHub repository](https://github.com/facebookresearch/sam2#download-checkpoints) | object-track, obj-count-sequence, motion-analyze, spatial-analyze | `models/sam2.1_hiera_large.pt` |
| faster-whisper | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3) | audio-transcription, speaker-diarization | `WHISPERX_MODEL_SIZE=large-v3` |
| pyannote 3.1 | [Hugging Face](https://huggingface.co/pyannote/speaker-diarization-3.1) | speaker-diarization | Hugging Face cache |
| ViT Face Expression | [Hugging Face](https://huggingface.co/trpakov/vit-face-expression) | emotion-analysis | `FER_MODEL=trpakov/vit-face-expression` |
| SpeechBrain IEMOCAP | [Hugging Face](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) | emotion-analysis | `SER_MODEL=speechbrain/emotion-recognition-wav2vec2-IEMOCAP` |
| InsightFace buffalo_l | [Official GitHub release](https://github.com/deepinsight/insightface/releases/tag/v0.7) | face-cluster, active-speaker-detection, gaze-estimation, emotion-analysis | `~/.insightface/models/buffalo_l/` |
| PANNs Cnn14 | [Official GitHub repository](https://github.com/qiuqiangkong/audioset_tagging_cnn#pretrained-models) | audio-event-detection | `models/Cnn14_mAP=0.431.pth` |
| PaddleOCR | [Official GitHub repository](https://github.com/PaddlePaddle/PaddleOCR) | ocr | PaddleX/PaddleOCR user cache |
| RAFT Large | [torchvision documentation](https://pytorch.org/vision/stable/models/raft.html) | optical-flow | `${TORCH_HOME:-~/.cache/torch}/hub/checkpoints/` |
| L2CS-Net Gaze360 | [Official GitHub model zoo](https://github.com/Ahmednull/L2CS-Net#models) | gaze-estimation | `models/L2CSNet_gaze360.pkl` |

The official PANNs project hosts its large checkpoints on Zenodo; the downloader
uses the original weights linked from its README. SAM and SAM2 weights are hosted
on Meta's `dl.fbaipublicfiles.com`, with links published in their official READMEs.

## Manual Downloads

Run the following `hf` commands in an installed vision or audio environment.
For example:

```bash
source .venvs/vision/bin/activate
```

### Hugging Face Models

Without `--local-dir`, `hf download` uses the standard Hugging Face cache. Services
can continue to load models by repository ID. Avoid placing user-specific
`snapshots/...` paths containing commit hashes in `.mcp.json`.

```bash
hf download sentence-transformers/all-MiniLM-L6-v2
hf download depth-anything/Depth-Anything-V2-Small-hf
hf download MCG-NJU/videomae-base-finetuned-kinetics
hf download openai/clip-vit-base-patch32
hf download Systran/faster-whisper-large-v3
hf download trpakov/vit-face-expression
hf download facebook/wav2vec2-base
```

GroundingDINO uses the following repository-local BERT directory by default:

```bash
hf download google-bert/bert-base-uncased --local-dir models/bert-base-uncased
```

The SpeechBrain service checks its save directory before loading the model.
Place the speech emotion recognition (SER) model there:

```bash
hf download speechbrain/emotion-recognition-wav2vec2-IEMOCAP \
  --local-dir "$HOME/.cache/speechbrain/speechbrain--emotion-recognition-wav2vec2-IEMOCAP"
```

### Checkpoints with Explicit File Paths

```bash
mkdir -p models/GroundingDINO

curl -fL --retry 3 -o models/GroundingDINO/groundingdino_swinb_cogcoor.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swinb_cogcoor.pth

curl -fL --retry 3 -o models/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

curl -fL --retry 3 -o models/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

curl -fL --retry 3 -o models/Cnn14_mAP=0.431.pth \
  'https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
```

torchvision downloads RAFT weights on first use. To download them in advance:

```bash
mkdir -p "${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints"
curl -fL --retry 3 \
  -o "${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth" \
  https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth
```

### InsightFace and PaddleOCR

InsightFace's `FaceAnalysis(name="buffalo_l")` downloads its models on first use
when online. For offline deployment, run the downloader's `insightface` component
in advance, or download `buffalo_l.zip` from the official release and extract it
into `~/.insightface/models/buffalo_l/`.

PaddleOCR/PaddleX also downloads weights from official sources when an OCR pipeline
is first created. Preload the basic Chinese OCR models with:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True .venvs/ocr/bin/python -c \
  "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)"
```

### Speaker Diarization (Gated Models)

`speaker-diarization-mcp` currently loads `pyannote/speaker-diarization-3.1`.
First, accept the model access conditions on these pages:

- <https://huggingface.co/pyannote/speaker-diarization-3.1>
- <https://huggingface.co/pyannote/segmentation-3.0>

Then set a token with read access:

```bash
export HF_TOKEN="hf_xxx"
PATH="$PWD/.venvs/audio/bin:$PATH" bash scripts/download_models.sh diarization
```

Without a token, the service uses its MFCC clustering fallback, and the `all`
download command skips gated models without failing.

### L2CS-Net and Moment-DETR

L2CS-Net publishes Gaze360 weights through the model zoo linked in its GitHub
README. The upstream download link may require browser interaction. Copy the
current download URL from the official page, then run:

```bash
L2CS_WEIGHTS_URL='Gaze360 download URL from the upstream model zoo' \
  bash scripts/download_models.sh gaze
```

If L2CS weights are unavailable, `gaze-estimation-mcp` falls back to head pose
estimation using facial landmarks.

`moment-retrieval-mcp` can run with the public `openai/clip-vit-base-patch32` model
alone. Without a Moment-DETR checkpoint, it uses the CLIP similarity backend.
`MOMENT_DETR_CHECKPOINT` accepts a local checkpoint compatible with this repository's
simplified inference architecture. Verify a checkpoint's source and compatibility
before distributing it as official Moment-DETR weights.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache root |
| `TORCH_HOME` | `~/.cache/torch` | PyTorch/torchvision cache root |
| `MCP_ALLOW_MODEL_DOWNLOAD` | `0` in the example configuration | Allow MCP runtime model downloads where supported |
| `HF_TOKEN` | Empty | Access token for gated Hugging Face models |
| `SBERT_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Temporal semantic model repository ID or local path |
| `DEPTH_MODEL` | `depth-anything/Depth-Anything-V2-Small-hf` | Depth model repository ID or local path |
| `VIDEOMAE_MODEL` | `MCG-NJU/videomae-base-finetuned-kinetics` | Action classification model repository ID or local path |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | CLIP model for moment retrieval |
| `GDINO_BERT_PATH` | `models/bert-base-uncased` | GroundingDINO text encoder directory |
| `GDINO_CHECKPOINT` | `models/GroundingDINO/groundingdino_swinb_cogcoor.pth` | GroundingDINO checkpoint file |
| `SAM_CHECKPOINT` | `models/sam_vit_h_4b8939.pth` | SAM checkpoint file |
| `SAM2_CHECKPOINT` | `models/sam2.1_hiera_large.pt` | SAM2 checkpoint file |
| `PANNS_CHECKPOINT` | `models/Cnn14_mAP=0.431.pth` | PANNs checkpoint file |
| `L2CS_WEIGHTS` | `models/L2CSNet_gaze360.pkl` | L2CS-Net checkpoint file |
| `WHISPERX_MODEL_SIZE` | `large-v3` | faster-whisper model size |
| `FER_MODEL` | `trpakov/vit-face-expression` | Facial expression model repository ID or local path |
| `SER_MODEL` | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | Speech emotion model repository ID or local path |

If you change `MODEL_DIR`, update the corresponding checkpoint environment variables
in your MCP configuration. Prefer repository IDs for Hugging Face models so their
caches can be resolved under each user's `HF_HOME`.

## Offline Checks

The following checks do not access the network or load large models:

```bash
bash -n scripts/download_models.sh
bash scripts/download_models.sh --dry-run all

test -f models/GroundingDINO/groundingdino_swinb_cogcoor.pth
test -f models/sam_vit_h_4b8939.pth
test -f models/sam2.1_hiera_large.pt
test -f models/Cnn14_mAP=0.431.pth
```

Each model is subject to its upstream license and terms of use. Review its model
card or repository license before redistribution or commercial use. This project's
license does not apply to third-party model weights.
