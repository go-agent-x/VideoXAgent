#!/usr/bin/env python3
"""Frame Diff MCP Server — Inter-frame visual difference analysis."""
import asyncio
import sys
from pathlib import Path

# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")
sys.path.insert(0, str(Path(__file__).parent))

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response

# ── Business logic (from frame_diff_server.py) ──

import json
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from frame_diff_utils import diff_two_frames, diff_video_timestamps


# =========================================================================
# Tool handlers
# =========================================================================
async def _handle_diff_frames(arguments: dict) -> dict:
    image_a = arguments.get("image_path_a", "")
    image_b = arguments.get("image_path_b", "")
    threshold = float(arguments.get("ssim_threshold", 0.6))
    min_area = int(arguments.get("min_region_area", 100))

    if not image_a or not image_b:
        return {"success": False, "error": "Both image_path_a and image_path_b are required"}

    for p, name in [(image_a, "image_path_a"), (image_b, "image_path_b")]:
        if not os.path.isfile(p):
            return {"success": False, "error": f"{name} not found: {p}"}

    try:
        return diff_two_frames(image_a, image_b, threshold, min_area)
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


async def _handle_diff_timestamps(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    time_a = arguments.get("time_a")
    time_b = arguments.get("time_b")
    threshold = float(arguments.get("ssim_threshold", 0.6))
    min_area = int(arguments.get("min_region_area", 100))

    if not video_path:
        return {"success": False, "error": "video_path is required"}
    if time_a is None or time_b is None:
        return {"success": False, "error": "Both time_a and time_b are required"}

    try:
        return diff_video_timestamps(
            video_path, float(time_a), float(time_b), threshold, min_area
        )
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# MCP Tool Registration
# =========================================================================


async def handle_diff_two_frames(arguments: dict) -> dict:
    return await _handle_diff_frames(arguments)


async def handle_diff_video_timestamps(arguments: dict) -> dict:
    return await _handle_diff_timestamps(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="frame-diff",
        tool_defs_source="_tool_defs/misc.yaml",
        handlers={
            "diff_two_frames": handle_diff_two_frames,
            "diff_video_timestamps": handle_diff_video_timestamps,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
