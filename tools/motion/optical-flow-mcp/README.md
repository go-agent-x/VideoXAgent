# Optical Flow MCP Server

Dense optical flow analysis for video understanding, exposed as an MCP (Model Context Protocol) server.

## Features

- **compute_dense_flow** - Compute dense optical flow between two video frames at specified timestamps. Returns flow magnitude/direction statistics.
- **analyze_flow_segment** - Analyze optical flow patterns across a video segment. Computes flow between consecutive sampled frames and aggregates motion statistics.

## Backends

| Backend | Library | Accuracy | Speed | GPU |
|---------|---------|----------|-------|-----|
| RAFT (default) | torchvision | High | Medium | Yes |
| Farneback (fallback) | OpenCV | Medium | Fast | No |

The server automatically selects RAFT when available, falling back to Farneback if the model fails to load.

## Model Setup

RAFT requires the pretrained weights file `raft_large_C_T_SKHT_V2-ff5fadd5.pth`.

### From PyTorch

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L -o ~/.cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth \
  https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth
```

Alternatively, run `bash scripts/download_models.sh optical-flow` from the repository root.
The model is also downloaded automatically on first use when the network is reachable.

## Dependencies

```bash
bash scripts/install_profiles.sh vision --torch-backend cpu
```

Run the command from the repository root. Dependency versions are maintained in
`requirements/profiles/vision.txt`.

## Usage

```bash
.venvs/vision/bin/python tools/motion/optical-flow-mcp/server.py
```
