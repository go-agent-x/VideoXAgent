#!/usr/bin/env python3
"""Action Classify MCP Server — Video segment action classification."""
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

# ── Business logic (from action_classify_server.py) ──

import json
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from action_utils import classify_action, classify_multi_segments


# =========================================================================
# Tool handlers
# =========================================================================
async def _handle_classify_action(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    start_time = float(arguments.get("start_time", 0))
    end_time_raw = arguments.get("end_time")
    end_time = float(end_time_raw) if end_time_raw is not None else None
    top_k = int(arguments.get("top_k", 5))

    if not video_path:
        return {"success": False, "error": "video_path is required"}

    try:
        result = classify_action(video_path, start_time, end_time, top_k)
        result["video_path"] = video_path
        result["analyzed_range"] = {
            "start_time": start_time,
            "end_time": end_time if end_time is not None else "video_end",
        }
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


async def _handle_classify_multi(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    segments_raw = arguments.get("segments", "[]")
    top_k = int(arguments.get("top_k", 5))

    if not video_path:
        return {"success": False, "error": "video_path is required"}

    # Parse segments
    if isinstance(segments_raw, str):
        try:
            segments = json.loads(segments_raw)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid segments JSON: {e}"}
    elif isinstance(segments_raw, list):
        segments = segments_raw
    else:
        return {"success": False, "error": "segments must be a JSON array"}

    if not segments:
        return {"success": False, "error": "segments list is empty"}

    try:
        return classify_multi_segments(video_path, segments, top_k)
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# MCP Tool Registration
# =========================================================================


async def handle_classify_action(arguments: dict) -> dict:
    return await _handle_classify_action(arguments)


async def handle_classify_action_multi_segments(arguments: dict) -> dict:
    return await _handle_classify_multi(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="action-classify",
        tool_defs_source="_tool_defs/action.yaml",
        handlers={
            "classify_action": handle_classify_action,
            "classify_action_multi_segments": handle_classify_action_multi_segments,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
