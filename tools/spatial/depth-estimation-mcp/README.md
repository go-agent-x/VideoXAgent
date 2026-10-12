# Depth Estimation MCP

A monocular depth estimation tool based on Depth-Anything-V2 that estimates per-pixel relative depth in an image.

## Installation

```bash
bash scripts/install_profiles.sh vision --torch-backend cpu
source .venvs/vision/bin/activate
```

Run these commands from the repository root. For CUDA 12.6, use `--torch-backend cu126` instead.

### Model Downloads

Download the model from the public [Hugging Face model repository](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) to the default cache:

```bash
hf download depth-anything/Depth-Anything-V2-Small-hf
# Alternatively, run from the repository root:
bash scripts/download_models.sh depth
```


## Environment Variables

| Variable | Description |
|------|------|
| `DEPTH_MODEL` | Depth-Anything-V2 model name or path; defaults to `depth-anything/Depth-Anything-V2-Small-hf` |
| `DEPTH_DEVICE` | Inference device; defaults to `cuda` |

## Available Tools

| Tool Name | Function |
|--------|------|
| `estimate_depth` | Estimate a per-pixel depth map for an image and return a visualization of depth regions |
| `compare_object_depth` | Compare the relative depth of two bounding-box regions to determine which is closer or farther away |
| `get_depth_at_points` | Get depth values at specified pixel coordinates in an image |
