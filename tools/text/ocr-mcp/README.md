
A PaddleOCR-based text recognition MCP service that supports full-image text recognition, region-based text recognition, table extraction, layout analysis, and video text timelines.

## Installation

Use Python 3.11 and run one of the following commands from the repository root:

```bash
# CPU
bash scripts/install_profiles.sh ocr --paddle-backend cpu

# GPU
bash scripts/install_profiles.sh ocr --paddle-backend gpu
```

### System Dependencies

No additional system dependencies are required.

### Model Downloads

PaddleOCR/PaddleX downloads model weights from official model sources on first initialization. To download them in advance, run the following command from the repository root:

```bash
PATH="$PWD/.venvs/ocr/bin:$PATH" bash scripts/download_models.sh ocr
```

For model versions and public download sources, see the [official PaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR).

## Environment Variables

| Variable | Description |
|------|------|
| `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` | Set to `True` to disable model source checks |

## Available Tools

| Tool Name | Function |
|--------|------|
| `ocr_frame_text` | Detect and recognize all text in an image |
| `ocr_region_read` | Recognize text within a specified bounding box (bbox) |
| `ocr_table_extract` | Extract table structures and data from an image |
| `ocr_layout_analysis` | Analyze the layout structure of a document image |
| `ocr_video_text_timeline` | Extract a timeline of on-screen text from video frames |
Filter files

