# Tool Definitions

This directory contains YAML files defining all MCP tools in a decoupled manner.
Each YAML file represents a category of tools.

Service implementations are grouped under [`tools/`](../tools/README.md). The
[service catalog](../tools/catalog.json) maps registered tools to their entrypoints
and runtime profiles; this directory continues to hold the shared tool schemas.

## YAML Format

```yaml
tool_name:
  description: |
    Multi-line description of what the tool does.
    Can include usage hints and best practices.
  backend: grounding_dino  # Backend type: grounding_dino, vlm, script, whisperx, audio, etc.
  media_type: image  # Input media type: image, video, audio, none
  input_schema:
    type: object
    properties:
      param1:
        type: string
        description: Description of param1
      param2:
        type: number
        description: Description of param2
        default: 0.5
    required:
      - param1
  output_format:
    type: object
    properties:
      success:
        type: boolean
      result:
        type: string
```

## Fields

- `description`: Tool description shown to LLM. Should include "Best for:" hints.
- `backend`: Backend implementation type
  - `grounding_dino`: Grounding DINO local object detection
  - `vlm`: Vision Language Model (GPT-4V, etc.)
  - `script`: Local Python script
  - `whisperx`: WhisperX audio processing
  - `audio`: GPT-4o-audio API
- `media_type`: Type of media the tool processes
  - `image`: Single image
  - `video`: Video file
  - `audio`: Audio file
  - `none`: No media processing
- `input_schema`: JSON Schema for input parameters
- `output_format`: Expected output structure (documentation only)

## Files

- `grounding_dino.yaml`: Grounding DINO object detection and localization tools
- `ocr.yaml`: OCR and text extraction tools
- `temporal.yaml`: Temporal analysis and moment retrieval
- `spatial.yaml`: Spatial analysis and geometry
- `vlm_query.yaml`: VLM-based query tools
- `basic.yaml`: Basic video information tools
- `recognition.yaml`: Recognition and classification tools
