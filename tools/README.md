# MCP Tools

MCP implementations are organized by capability under
`tools/<category>/<service>-mcp/`, using the same categories as LVideoXAgent.
The 43 services register 71 tools. Their entrypoints, dependency profiles, and
registered tool names are listed in [catalog.json](catalog.json).

| Category | Services | Purpose |
| --- | ---: | --- |
| [media](media/) | 8 | Video input, metadata, extraction, cropping, and image composition |
| [vision](vision/) | 11 | Visual inspection, detection, segmentation, tracking, and recognition |
| [audio](audio/) | 4 | Transcription, diarization, audio events, and music identification |
| [text](text/) | 2 | OCR and subtitles |
| [motion](motion/) | 5 | Optical flow, frame differences, camera/object motion, and speed |
| [spatial](spatial/) | 3 | Geometry, spatial relations, and depth |
| [temporal](temporal/) | 3 | Scene changes, temporal grounding, and moment retrieval |
| [composite](composite/) | 7 | Scene understanding and tools combining multiple capabilities |

Categories describe what a service does. Dependency profiles (`base`, `vision`,
`audio`, `ocr`, and `music`) determine which Python environment it uses; the two
classifications are independent. For example, emotion analysis is a composite
service running in the audio environment.

The [segment deep-query tool](composite/segment-deep-query-mcp/README.md) can invoke
Grounding DINO in the separate vision environment, while keeping its VLM process
in base. Its detections are scoped to common or explicitly requested categories.

## Service Directory

| Category | Service directories |
| --- | --- |
| media | `youtube-cache-mcp`, `video-duration-mcp`, `frame-extract-mcp`, `audio-extract-mcp`, `video-segment-extract-mcp`, `spatial-crop-mcp`, `concat-images-mcp`, `thumbnail-navigator-mcp` |
| vision | `inspect-image-mcp`, `grounding-dino-mcp`, `auto-segment-mcp`, `prompt-segment-mcp`, `object-track-mcp`, `obj-count-sequence-mcp`, `attribute-recognition-mcp`, `action-classify-mcp`, `face-cluster-mcp`, `person-identification-mcp`, `gaze-estimation-mcp` |
| audio | `audio-transcription-mcp`, `speaker-diarization-mcp`, `audio-event-detection-mcp`, `music-identification-mcp` |
| text | `ocr-mcp`, `subtitle-mcp` |
| motion | `optical-flow-mcp`, `frame-diff-mcp`, `camera-motion-mcp`, `motion-analyze-mcp`, `speed-estimation-mcp` |
| spatial | `spatial-calc-mcp`, `spatial-analyze-mcp`, `depth-estimation-mcp` |
| temporal | `scene-change-mcp`, `temporal-ground-mcp`, `moment-retrieval-mcp` |
| composite | `emotion-analysis-mcp`, `active-speaker-detection-mcp`, `scene-query-mcp`, `scene-caption-mcp`, `segment-deep-query-mcp`, `interaction-recognition-mcp`, `vlm-query-sequence-mcp` |

## Configuration and Shared Paths

Run commands from the repository root. The profile configurator updates known
legacy entrypoints such as `./ocr-mcp/server.py` to their categorized locations,
while preserving server settings, interpreter flags, wrappers, and custom arguments:

```bash
.venvs/base/bin/python scripts/configure_mcp_profiles.py
```

It uses the existing `.mcp.profiles.json` by default. To migrate another local MCP
configuration, supply `--source /path/to/config.json` and an `--output` path.
Known script entrypoints automatically receive `scripts/mcp_runtime.py` before
the service path, once. Module-based and custom-script invocations are not
automatically wrapped. Use `--device cpu|cuda`, `--ocr-device cpu|gpu`, and repeated
`--disable-server NAME` options to configure runtime devices and service selection.
Unavailable historical services are not added to the catalog or removed from
your configuration unless explicitly selected with `--disable-server`.

Tool schemas remain in [`_tool_defs/`](../_tool_defs/), shared utilities in
[`_base/`](../_base/), and dependency manifests in
[`requirements/profiles/`](../requirements/profiles/). `models/`, `data/`, and
`output/` also remain at the repository root.

Cross-service imports use the shared helpers rather than assuming sibling paths:

```python
from _base.paths import PROJECT_ROOT, service_dir, service_entrypoint

detector_dir = service_dir("grounding-dino")
subtitle_server = service_entrypoint("subtitle")
tool_schemas = PROJECT_ROOT / "_tool_defs"
```

Add new services to the appropriate category and to `catalog.json`, and point
their local `requirements.txt` at `../../../requirements/profiles/<profile>.txt`.
Keep service names and registered tool names stable when moving implementations.

## Verification

```bash
.venvs/base/bin/python -m unittest discover -s _base/tests -p 'test_tool_layout*.py' -v
```

The layout tests check registration, configuration rewriting, dependency paths,
and lightweight MCP stdio calls from another working directory. They do not
download weights or validate model inference.
