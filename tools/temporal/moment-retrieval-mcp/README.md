# Moment Retrieval MCP

Locate video segments that match a natural-language query using visual features.
The service exposes one MCP tool, `moment_retrieve`, and runs inference locally.
It does not require subtitles, precomputed captions, or an LLM/VLM API key.

The default backend uses CLIP frame-text similarity. A compatible Moment-DETR
checkpoint can enable an optional learned retrieval backend. Both require CLIP.

## Installation

Run all commands from the repository root with Python 3.11 available:

```bash
bash scripts/install_profiles.sh vision --torch-backend cpu
```

For the supported CUDA build, use `--torch-backend cu126` instead. The service
automatically selects CUDA when PyTorch reports it available; otherwise it uses
CPU. Installation checks dependencies, not retrieval accuracy or GPU drivers.

The local `requirements.txt` delegates to the shared `vision` profile. See the
[installation guide](../../../install.md) for environment setup.

## Prepare the CLIP Model

The default model is `openai/clip-vit-base-patch32`. Download it before starting
the MCP service:

```bash
source .venvs/vision/bin/activate
bash scripts/download_models.sh moment
```

The `moment` download component prepares CLIP only; it does not download a
Moment-DETR checkpoint. Set `HF_HOME` consistently during preparation and inference
if you use a custom Hugging Face cache. Alternatively, set `CLIP_MODEL` to an
absolute directory containing a complete CLIP model and processor.

See [Model Downloads and Setup](../../../MODEL_SETUP.md) for model sources and cache
configuration.

## MCP Configuration

Add this service to the MCP configuration selected by your Agent or MCP client:

```json
{
  "mcpServers": {
    "moment-retrieval": {
      "command": "./.venvs/vision/bin/python",
      "args": [
        "./scripts/mcp_runtime.py",
        "./tools/temporal/moment-retrieval-mcp/server.py"
      ],
      "env": {
        "CLIP_MODEL": "openai/clip-vit-base-patch32",
        "MCP_ALLOW_MODEL_DOWNLOAD": "0",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

Relative command paths assume the client starts from the repository root. Use
absolute paths when configuring a client that starts elsewhere. The runtime
wrapper applies cache paths and offline settings before importing model libraries.

To start the stdio server directly:

```bash
.venvs/vision/bin/python scripts/mcp_runtime.py tools/temporal/moment-retrieval-mcp/server.py
```

This command waits for MCP protocol messages; it does not run a query by itself.
Models load lazily on the first retrieval request. If you bypass the wrapper,
missing CLIP files may trigger an online download unless offline settings are
already configured.

## Tool: `moment_retrieve`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `video_path` | string | Yes | — | Absolute path to a local video readable by OpenCV. |
| `query` | string | Yes | — | A concise description of the visual event to find. |
| `top_k` | integer | No | `5` | Maximum number of returned segments; use a positive value. |
| `threshold` | number | No | `0.3` | Minimum segment score. Higher values retain fewer candidates. |
| `start_time` | number | No | `0` | Start of the search range, in seconds from the video start. |
| `end_time` | number | No | Video duration | End of the search range, in seconds from the video start. |
| `sample_fps` | number | No | `2.0` | Frame sampling rate; must be greater than zero. |

Use a valid range within the video, with `start_time < end_time`. CLIP truncates
query text to 77 tokens, so keep the description focused on visible content.

Example arguments to pass when calling the tool from an MCP client:

```json
{
  "video_path": "/absolute/path/to/video.mp4",
  "query": "A person opens a refrigerator and takes out a bottle.",
  "top_k": 3,
  "threshold": 0.3,
  "start_time": 0,
  "end_time": 120,
  "sample_fps": 2.0
}
```

Example decoded response, with illustrative values rather than measured results:

```json
{
  "success": true,
  "video_duration": 120.0,
  "moments": [
    {
      "start_time": 12.25,
      "end_time": 15.5,
      "confidence": 0.3542
    }
  ]
}
```

Segments are ranked by descending score, with overlapping candidates suppressed.
Timestamps refer to the original video, including when a search range is provided.
`confidence` is a backend-specific ranking score, not a calibrated probability.
In CLIP mode it is based on smoothed cosine similarity; Moment-DETR uses a sigmoid
prediction score. An empty `moments` list can be a successful result when no
candidate meets the threshold. The response does not include a backend name or
per-frame saliency scores.

## Environment Variables

Set these in the MCP service's `env` configuration, or in the environment when
starting the server directly. Restart the service after changing model settings.

| Variable | Default | Purpose |
|---|---|---|
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | Model repository ID or local model directory. |
| `HF_HOME` | `~/.cache/huggingface` | The service looks for model snapshots under this directory's `hub/` subdirectory. |
| `MOMENT_RETRIEVAL_CACHE` | `~/.cache/moment_retrieval` | Directory for cached video features (`.npz` files). |
| `MOMENT_DETR_CHECKPOINT` | Unset | Optional path to a compatible local Moment-DETR checkpoint; prefer an absolute path. |
| `MOMENT_DETR_CONFIG` | Built-in architecture defaults | Optional JSON object overriding checkpoint architecture settings. |
| `MCP_ALLOW_MODEL_DOWNLOAD` | `0` through the runtime wrapper | The wrapper sets Hugging Face and Transformers offline flags before loading the server. |

## Optional Moment-DETR Backend

Leave `MOMENT_DETR_CHECKPOINT` unset to use CLIP similarity retrieval. When a
checkpoint is configured and loads successfully, the service selects its
Moment-DETR backend. Missing or failed checkpoint loads fall back to CLIP; a
failure to load CLIP makes retrieval unavailable in either mode.

The implementation uses a simplified Moment-DETR architecture. Check tensor
names and shapes against `server.py` and validate predictions before enabling a
checkpoint. Loading uses `strict=False`, so a success message alone does not
establish compatibility. Set `MOMENT_DETR_CONFIG` only to match a verified
checkpoint; it is not needed for the default CLIP workflow.

## Caching and Long Videos

The first query samples and extracts features for the **entire video**, then
filters them to the requested search range. Narrowing `start_time` and `end_time`
does not reduce the first extraction pass. Subsequent queries can reuse the
features. Lower `sample_fps` reduces processing and memory costs but can miss
short events.

Cache keys include the absolute video path, sampling rate, and `CLIP_MODEL`
string. They do not include file contents or model revisions. If you replace a
video or model at the same path, remove its cached features or choose a new
`MOMENT_RETRIEVAL_CACHE` directory.

For Moment-DETR, search ranges with more than 5,000 sampled frames are downsampled
to 5,000 for inference. Ranges with more than 10,000 frames use the CLIP similarity
backend instead. These limits do not bound the initial full-video extraction.

## Troubleshooting

- **CLIP model unavailable:** prepare the model in the same `HF_HOME` used by the
  service, or point `CLIP_MODEL` to a complete local model directory.
- **Video not found or no frames extracted:** check the absolute path, permissions,
  and whether OpenCV can decode the file.
- **No frames in range:** check timestamps against the video duration and increase
  the range or sampling rate if necessary.
- **No matching moments:** try a more concrete visual query or lower `threshold`.
  The CLIP backend requires at least two sampled frames in the search range to
  form a candidate window.
- **Unexpected Moment-DETR results:** inspect stderr for missing checkpoint keys
  and fallback messages, then compare with the default CLIP backend.

Tool schemas are defined in [`_tool_defs/temporal.yaml`](../../../_tool_defs/temporal.yaml).
The implementation is in [`server.py`](server.py).
