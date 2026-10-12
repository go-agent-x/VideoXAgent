# grounding-dino-mcp

Local open-set object detection MCP server based on GroundingDINO-B (SwinB backbone). Replaces the remote DINO-X API with fully local inference.

## Tools

| Tool | Description |
|------|-------------|
| `detect` | Detect specific object categories in an image |
| `count` | Count objects of a category with NMS dedup |
| `locate` | Locate objects by natural language phrase |
| `scene_scan` | Scan common or supplied object categories in a scene |
| `instance_segment` | Detect objects and compare sizes |
| `spatial_relation` | Detect two objects and compute spatial relationship |

`scene_scan` performs text-prompted category detection. Without `categories`, it
scans 48 common categories in batches of eight; this is not exhaustive object
discovery. Supply category phrases such as `["person", "traffic light"]` to focus
the scan. Multiword names remain intact. The result reports `categories_used`,
`detection_mode`, batch counts, and any errors or partial-scan warnings. A successful
empty result means no matches within that vocabulary; it does not mean an empty scene.

## Installation

Run from the repository root using Python 3.11:

```bash
bash scripts/install_profiles.sh vision --torch-backend cpu
source .venvs/vision/bin/activate
```

For the supported CUDA 12.6 stack, use `--torch-backend cu126` instead.

## Model Weights Setup

### 1. BERT (text encoder)

```bash
hf download google-bert/bert-base-uncased --local-dir models/bert-base-uncased
```

The model comes from its public [Hugging Face repository](https://huggingface.co/google-bert/bert-base-uncased).

### 2. GroundingDINO checkpoint

```bash
mkdir -p models/GroundingDINO
curl -fL --retry 3 \
  -o models/GroundingDINO/groundingdino_swinb_cogcoor.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swinb_cogcoor.pth
```

The checkpoint is published in the [official GroundingDINO GitHub Release](https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha). Both steps can also be run from the repository root with `bash scripts/download_models.sh grounding-dino`.

After setup, the directory structure should be:

```
models/
└── GroundingDINO/
    └── groundingdino_swinb_cogcoor.pth
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GDINO_CONFIG` | (auto-detected from package) | Path to GroundingDINO_SwinB_cfg.py |
| `GDINO_CHECKPOINT` | `models/GroundingDINO/groundingdino_swinb_cogcoor.pth` | Path to model weights |
| `GDINO_DEVICE` | `cuda` | Inference device |
| `GDINO_BERT_PATH` | `models/bert-base-uncased` | Path to BERT tokenizer |

## Run

```bash
.venvs/vision/bin/python tools/vision/grounding-dino-mcp/server.py
```
