#!/usr/bin/env python3
"""Camera Motion MCP Server — Analyze camera movement patterns in video."""
import asyncio
import sys
import traceback
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

from camera_motion_utils import (
    get_video_info,
    analyze_video_camera_motion,
    detect_motion_between_times,
)


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

async def handle_analyze_camera_motion(arguments: dict) -> dict:
    """Handle analyze_camera_motion tool call.
    
    Analyzes camera motion across a video segment, detecting and classifying
    motion events (pan, tilt, zoom, static, tracking).
    """
    video_path = arguments.get("video_path", "")
    start_time = float(arguments.get("start_time", 0))
    end_time_raw = arguments.get("end_time")
    sample_fps = float(arguments.get("sample_fps", 2.0))
    motion_threshold = float(arguments.get("motion_threshold", 5.0))
    
    if not video_path:
        return make_error_response("video_path is required")
    
    try:
        # Parse end_time
        end_time = float(end_time_raw) if end_time_raw is not None else None
        
        # Run analysis in thread pool to avoid blocking
        result = await asyncio.to_thread(
            analyze_video_camera_motion,
            video_path,
            start_time,
            end_time,
            sample_fps,
            motion_threshold,
        )
        
        return result
        
    except FileNotFoundError as e:
        return make_error_response(str(e))
    except Exception as e:
        return make_error_response(str(e), {"traceback": traceback.format_exc()})


async def handle_detect_camera_motion_type(arguments: dict) -> dict:
    """Handle detect_camera_motion_type tool call.
    
    Detects the primary camera motion type between two timestamps.
    """
    video_path = arguments.get("video_path", "")
    time_a = arguments.get("time_a")
    time_b = arguments.get("time_b")
    motion_threshold = float(arguments.get("motion_threshold", 5.0))
    
    if not video_path:
        return make_error_response("video_path is required")
    if time_a is None:
        return make_error_response("time_a is required")
    if time_b is None:
        return make_error_response("time_b is required")
    
    try:
        time_a = float(time_a)
        time_b = float(time_b)
        
        if time_a >= time_b:
            return make_error_response(f"time_a ({time_a}) must be less than time_b ({time_b})")
        
        # Run detection in thread pool
        result = await asyncio.to_thread(
            detect_motion_between_times,
            video_path,
            time_a,
            time_b,
            motion_threshold,
        )
        
        return result
        
    except FileNotFoundError as e:
        return make_error_response(str(e))
    except Exception as e:
        return make_error_response(str(e), {"traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_mcp_server(
        server_name="camera-motion-mcp",
        tool_defs_source="_tool_defs/misc.yaml",
        handlers={
            "analyze_camera_motion": handle_analyze_camera_motion,
            "detect_camera_motion_type": handle_detect_camera_motion_type,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
