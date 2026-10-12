# Segment Deep Query

This tool combines sampled video frames, optional object detections, optional OCR,
and a VLM answer. Grounding DINO requires text prompts: detections use a bounded
category scan, not native prompt-free object discovery.

## Focus the Detection Vocabulary

The optional `detection_categories` argument accepts up to 64 phrases, each at most
80 characters and eight words. For example:

```json
{
  "video_path": "/absolute/path/to/video.mp4",
  "start_time": 60,
  "end_time": 70,
  "query": "Is the person standing near a traffic light?",
  "num_frames": 4,
  "detection_categories": ["person", "traffic light"]
}
```

Omitting the argument scans 48 common categories in batches of eight, requiring
six detector passes per frame. Supplying relevant categories reduces this work.
The categories describe what to look for; the scan does not prove that other
objects are absent.

## Detector Environment

The service keeps its `base` profile so visual-only VLM queries remain available.
If Grounding DINO is importable in that process, it uses the local detector.
Otherwise it starts the existing Grounding DINO MCP service in the `vision`
environment, once per segment request, and processes all frames sequentially in
that session. Concurrent batches are serialized within this composite process.

Prepare the vision dependencies and Grounding DINO weights using the repository
[installation guide](../../../install.md) and [model setup guide](../../../MODEL_SETUP.md).
The profile configurator supplies `GDINO_PYTHON` for the selected environment root:

```bash
.venvs/base/bin/python scripts/configure_mcp_profiles.py --venv-root .venvs
```

An explicit custom `GDINO_PYTHON` is preserved. Relative interpreter paths are
resolved from the repository root; the default is `.venvs/vision/bin/python`.
The child uses `scripts/mcp_runtime.py` and inherits this service's device,
checkpoint, cache, and offline settings. It does not read another service's
separate MCP `env` block. `GDINO_TIMEOUT_SECONDS` controls vision MCP startup and
individual tool-call timeouts (default: 240 seconds).

OCR remains optional in the composite's own environment. The detector subprocess
does not merge the isolated vision and OCR dependencies.

## Evidence Status

`evidence_summary` includes per-frame detection status and the actual scanned
vocabulary. It distinguishes detections, successful scans with no matches, partial
scans, failures, and unavailable backends. These statuses are also sent to the VLM.
If the vision environment or weights are unavailable, the VLM can still answer
from frames and available OCR, while the result reports the missing detection
evidence instead of claiming that no objects were present.

The implementation is covered by mock-backend tests and a real MCP transport test
with a synthetic detector. Real model accuracy and Linux/GPU performance still
require the prepared model environment.
