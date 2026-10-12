#!/usr/bin/env python3
"""Motion Analyze MCP Server — Object motion trajectory and kinematics analysis."""
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

# ── Business logic (from motion_analyze_server.py) ──

import json
import os
import traceback

import numpy as np

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from motion_analyze_utils import (
    get_video_predictor,
    get_video_info,
    extract_frames_to_jpeg_dir,
    mask_to_bbox,
    mask_center,
    mask_area,
    analyze_motion,
    cleanup_gpu,
    cleanup_temp_dir,
)

# ---------------------------------------------------------------------------
# MCP Server init
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: parse JSON string or pass through
# ---------------------------------------------------------------------------
def _parse_json(val):
    if val is None:
        return None
    if isinstance(val, str):
        return json.loads(val)
    return val


# =========================================================================
# Tool: motion_analyze — Motion trajectory analysis
# =========================================================================
async def _handle_motion_analyze(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    init_frame_time = float(arguments.get("init_frame_time", 0))
    init_points = _parse_json(arguments.get("init_points", None))
    init_box = _parse_json(arguments.get("init_box", None))
    start_time = float(arguments.get("start_time", 0))
    end_time_raw = arguments.get("end_time", None)
    sample_fps = float(arguments.get("sample_fps", 3))

    if not video_path:
        return {"success": False, "error": "video_path is required"}
    if init_points is None and init_box is None:
        return {"success": False, "error": "At least one of init_points or init_box is required"}

    jpeg_dir = None
    try:
        info = get_video_info(video_path)
        end_time = float(end_time_raw) if end_time_raw is not None else info["duration"]

        jpeg_dir, frame_times, _ = extract_frames_to_jpeg_dir(
            video_path, sample_fps=sample_fps, start_time=start_time, end_time=end_time,
        )

        init_idx = min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - init_frame_time))

        predictor = get_video_predictor()
        inference_state = predictor.init_state(video_path=jpeg_dir, offload_video_to_cpu=True)

        obj_id = 1
        points_np = np.array(init_points, dtype=np.float32) if init_points else None
        labels_np = np.ones(len(init_points), dtype=np.int32) if init_points else None
        box_np = np.array(init_box, dtype=np.float32) if init_box else None

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=init_idx,
            obj_id=obj_id,
            points=points_np,
            labels=labels_np,
            box=box_np,
        )

        # Propagate forward + backward
        video_segments = {}
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
            video_segments[frame_idx] = (masks > 0.0).cpu().numpy()

        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            inference_state, start_frame_idx=init_idx, reverse=True
        ):
            if frame_idx not in video_segments:
                video_segments[frame_idx] = (masks > 0.0).cpu().numpy()

        # Build trajectory (only visible frames)
        trajectory = []
        for fi in sorted(video_segments.keys()):
            t = frame_times[fi] if fi < len(frame_times) else fi / sample_fps
            m = video_segments[fi][0][0]  # first object, first channel
            area = mask_area(m)
            if area > 50:
                bbox = mask_to_bbox(m)
                center = mask_center(m)
                trajectory.append({
                    "timestamp": round(t, 2),
                    "center": center,
                    "bbox": bbox,
                    "area": area,
                })

        # Analyze motion
        motion_summary = analyze_motion(trajectory)

        predictor.reset_state(inference_state)
        cleanup_gpu()
        cleanup_temp_dir(jpeg_dir)

        # Generate text summary
        ms = motion_summary
        if ms["is_stationary"]:
            summary_text = "Object is stationary (minimal movement detected)"
        else:
            summary_text = (
                f"Object moves primarily {ms['primary_direction']}, "
                f"avg speed {ms['avg_speed_px_per_sec']} px/s, "
                f"{ms['direction_changes']} direction change(s)"
            )

        return {
            "success": True,
            "trajectory": trajectory,
            "motion_summary": motion_summary,
            "summary": summary_text,
        }

    except Exception as e:
        cleanup_gpu()
        if jpeg_dir:
            cleanup_temp_dir(jpeg_dir)
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# Tool Registration
# =========================================================================
# =========================================================================
# Tool Call Router
# =========================================================================
TOOL_HANDLERS = {
    "motion_analyze": _handle_motion_analyze,
}


# =========================================================================
# Entry point
# =========================================================================


async def handle_motion_analyze(arguments: dict) -> dict:
    return await _handle_motion_analyze(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="motion-analyze",
        tool_defs_source="_tool_defs/misc.yaml",
        handlers={
            "motion_analyze": handle_motion_analyze,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
