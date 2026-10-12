# Temporal Ground MCP

An MCP service for temporal event localization using semantic matching with sentence-transformers. It locates the time segments in a video where events occur based on text queries.

## Installation

```bash
bash scripts/install_profiles.sh vision --torch-backend cpu
source .venvs/vision/bin/activate
```

Run these commands from the repository root.

### System Dependencies

No additional system dependencies are required.

### Model Download

Download the model from the public [Hugging Face model repository](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) to the default cache:

```bash
hf download sentence-transformers/all-MiniLM-L6-v2
# Or run from the repository root:
bash scripts/download_models.sh temporal-ground
```


## Environment Variables

| Variable | Description |
|------|------|
| `SBERT_MODEL` | Sentence-BERT model name. Default: `sentence-transformers/all-MiniLM-L6-v2` |

## Available Tools

| Tool Name | Description |
|--------|------|
| `locate_event_by_text` | Locates event time segments by semantically matching SRT subtitles |
| `locate_event_by_scene_caption` | Locates event time segments by semantically matching scene captions |
