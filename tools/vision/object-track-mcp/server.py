#!/usr/bin/env python3
"""Object-Track MCP Server — Single-object tracking across video frames."""
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
sys.path.insert(0, str(Path(__file__).parent))  # to import from original obj_track_server.py

from _base.server_framework import create_mcp_server, run_server

# ── Business logic (from obj_track_server.py) ──

import json
import os
import traceback

import numpy as np

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from obj_track_utils import (
    get_video_predictor,
    get_video_info,
    extract_frames_to_jpeg_dir,
    mask_to_bbox,
    mask_center,
    mask_area,
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
# Tool: object_track — Single-object tracking
# =========================================================================
async def _handle_object_track(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    init_frame_time = float(arguments.get("init_frame_time", 0))
    init_points = _parse_json(arguments.get("init_points", None))
    init_box = _parse_json(arguments.get("init_box", None))
    start_time = float(arguments.get("start_time", 0))
    end_time_raw = arguments.get("end_time", None)
    sample_fps = float(arguments.get("sample_fps", 2))

    if not video_path:
        return {"success": False, "error": "video_path is required"}
    if init_points is None and init_box is None:
        return {"success": False, "error": "At least one of init_points or init_box is required"}

    jpeg_dir = None
    try:
        # Get video info
        info = get_video_info(video_path)
        end_time = float(end_time_raw) if end_time_raw is not None else info["duration"]

        # Extract frames
        jpeg_dir, frame_times, _ = extract_frames_to_jpeg_dir(
            video_path, sample_fps=sample_fps, start_time=start_time, end_time=end_time,
        )

        # Find the frame index closest to init_frame_time
        init_idx = min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - init_frame_time))

        # Initialize SAM2 video predictor
        predictor = get_video_predictor()
        inference_state = predictor.init_state(video_path=jpeg_dir, offload_video_to_cpu=True)

        # Add prompt on the init frame
        obj_id = 1
        points_np = np.array(init_points, dtype=np.float32) if init_points else None
        labels_np = np.ones(len(init_points), dtype=np.int32) if init_points else None
        box_np = np.array(init_box, dtype=np.float32) if init_box else None

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=init_idx,
            obj_id=obj_id,
            points=points_np,
            labels=labels_np,
            box=box_np,
        )

        # Propagate forward
        video_segments = {}
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
            video_segments[frame_idx] = {
                "obj_ids": obj_ids,
                "masks": (masks > 0.0).cpu().numpy(),
            }

        # Also propagate backward from init frame
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            inference_state, start_frame_idx=init_idx, reverse=True
        ):
            if frame_idx not in video_segments:
                video_segments[frame_idx] = {
                    "obj_ids": obj_ids,
                    "masks": (masks > 0.0).cpu().numpy(),
                }

        # Build trajectory
        trajectory = []
        visible_count = 0
        first_seen = None
        last_seen = None

        for fi in sorted(video_segments.keys()):
            seg = video_segments[fi]
            t = frame_times[fi] if fi < len(frame_times) else fi / sample_fps

            # Find our object's mask (obj_id=1)
            obj_idx_in_list = None
            for oi, oid in enumerate(seg["obj_ids"]):
                if oid == obj_id:
                    obj_idx_in_list = oi
                    break

            if obj_idx_in_list is not None:
                m = seg["masks"][obj_idx_in_list][0]  # [H, W] binary mask
                area = mask_area(m)
                visible = area > 50  # Minimum visibility threshold
            else:
                visible = False
                area = 0

            if visible:
                bbox = mask_to_bbox(m)
                center = mask_center(m)
                trajectory.append({
                    "timestamp": round(t, 2),
                    "bbox": bbox,
                    "center": center,
                    "area": area,
                    "visible": True,
                })
                visible_count += 1
                if first_seen is None:
                    first_seen = round(t, 2)
                last_seen = round(t, 2)
            else:
                trajectory.append({
                    "timestamp": round(t, 2),
                    "bbox": [0, 0, 0, 0],
                    "center": [0, 0],
                    "area": 0,
                    "visible": False,
                })

        predictor.reset_state(inference_state)
        cleanup_gpu()
        cleanup_temp_dir(jpeg_dir)

        total_frames = len(trajectory)
        if visible_count == 0:
            summary = f"Object not found in any of the {total_frames} analyzed frames"
        else:
            summary = (
                f"Object tracked across {visible_count}/{total_frames} frames "
                f"from {first_seen}s to {last_seen}s"
            )

        return {
            "success": True,
            "object_id": obj_id,
            "total_frames": total_frames,
            "visible_frames": visible_count,
            "trajectory": trajectory,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "summary": summary,
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
    "object_track": _handle_object_track,
}


# =========================================================================
# Entry point
# =========================================================================


async def handle_object_track(arguments: dict) -> dict:
    return await _handle_object_track(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="object-track",
        tool_defs_source="_tool_defs/segment.yaml",
        handlers={"object_track": handle_object_track},
        is_file=True,
    )
    asyncio.run(run_server(app))
