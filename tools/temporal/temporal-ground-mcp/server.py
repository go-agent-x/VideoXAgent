#!/usr/bin/env python3
"""Temporal Ground MCP Server — Text query temporal localization."""
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
sys.path.insert(0, str(Path(__file__).parent))  # to import from original temporal_ground_server.py

from _base.server_framework import create_mcp_server, run_server

# ── Business logic (from temporal_ground_server.py) ──

import json
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from temporal_utils import (
    locate_by_subtitle,
    locate_by_captions,
    find_srt_for_video,
)


# =========================================================================
# Tool handlers
# =========================================================================
async def _handle_locate_by_text(arguments: dict) -> dict:
    query = arguments.get("query", "")
    video_path = arguments.get("video_path", "")
    srt_path = arguments.get("srt_path", "")
    subtitle_dir = arguments.get("subtitle_dir", "")
    top_k = int(arguments.get("top_k", 5))
    min_score = float(arguments.get("min_score", 0.1))
    merge_gap = float(arguments.get("merge_gap", 2.0))
    start_time = arguments.get("start_time")
    end_time = arguments.get("end_time")

    if not query:
        return {"success": False, "error": "query is required"}

    # Resolve SRT path
    if not srt_path:
        if not video_path:
            return {"success": False, "error": "Either srt_path or video_path is required"}
        srt_path = find_srt_for_video(video_path, subtitle_dir)
        if not srt_path:
            return {
                "success": False,
                "error": f"No SRT subtitle file found for video: {video_path}. "
                         "Please provide srt_path explicitly or use locate_event_by_caption instead.",
            }

    if not os.path.isfile(srt_path):
        return {"success": False, "error": f"SRT file not found: {srt_path}"}

    time_range = None
    if start_time is not None and end_time is not None:
        time_range = (float(start_time), float(end_time))

    try:
        result = locate_by_subtitle(query, srt_path, top_k, min_score, merge_gap, time_range)
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


async def _handle_locate_by_caption(arguments: dict) -> dict:
    query = arguments.get("query", "")
    captions_raw = arguments.get("captions", "[]")
    top_k = int(arguments.get("top_k", 5))
    min_score = float(arguments.get("min_score", 0.1))

    if not query:
        return {"success": False, "error": "query is required"}

    # Parse captions
    if isinstance(captions_raw, str):
        try:
            captions = json.loads(captions_raw)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid captions JSON: {e}"}
    elif isinstance(captions_raw, list):
        captions = captions_raw
    else:
        return {"success": False, "error": "captions must be a JSON array"}

    if not captions:
        return {"success": False, "error": "captions list is empty"}

    try:
        return locate_by_captions(query, captions, top_k, min_score)
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# MCP Tool Registration
# =========================================================================


async def handle_locate_event_by_text(arguments: dict) -> dict:
    return await _handle_locate_by_text(arguments)


async def handle_locate_event_by_scene_caption(arguments: dict) -> dict:
    return await _handle_locate_by_caption(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="temporal-ground",
        tool_defs_source="_tool_defs/temporal.yaml",
        handlers={
            "locate_event_by_text": handle_locate_event_by_text,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
