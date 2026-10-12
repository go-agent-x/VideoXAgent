#!/usr/bin/env python3
"""Frame Extract MCP Server — Extract a frame from video at a given timestamp."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")

from _base.paths import PROJECT_ROOT

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response

# ── Business logic (from frame_extract_server.py) ──

import json
import os
import cv2

OUTPUT_DIR = PROJECT_ROOT / "output"


def extract_frame(video_path: str, time_sec: float, output_name: str = "") -> dict:
    """Extract a single frame at a specified time in the video."""
    if not os.path.exists(video_path):
        return {"success": False, "error": f"Video not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return {"success": False, "error": f"Invalid FPS ({fps}) — video may be corrupted"}

    if time_sec < 0:
        cap.release()
        return {"success": False, "error": f"time_sec must be non-negative, got {time_sec}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        duration = total_frames / fps
        if time_sec > duration:
            cap.release()
            return {"success": False, "error": f"time_sec ({time_sec}s) exceeds video duration ({duration:.2f}s)"}

    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"success": False, "error": f"Cannot read frame at {time_sec}s"}

    # Generate the output path.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not output_name:
        video_name = Path(video_path).stem
        output_name = f"{video_name}_frame_{time_sec}s.jpg"
    # Ensure the filename has an extension.
    if not any(output_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
        output_name = output_name + '.jpg'
    output_path = (OUTPUT_DIR / output_name).resolve()

    # Save using an absolute path.
    success = cv2.imwrite(str(output_path), frame)
    if not success:
        return {"success": False, "error": f"Failed to save image to {output_path}"}

    return {
        "success": True,
        "output_path": str(output_path),
        "timestamp": time_sec
    }


def extract_multiple_frames(video_path: str, time_points: list, output_prefix: str = "") -> dict:
    """Extract frames at multiple timestamps in the video."""
    if not os.path.exists(video_path):
        return {"success": False, "error": f"Video not found: {video_path}"}

    if not time_points or len(time_points) == 0:
        return {"success": False, "error": "time_points array is empty"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return {"success": False, "error": f"Invalid FPS ({fps}) — video may be corrupted"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else float('inf')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem

    extracted_frames = []
    errors = []

    for time_sec in time_points:
        if time_sec < 0:
            errors.append(f"Skipped negative time: {time_sec}s")
            continue
        if total_frames > 0 and time_sec > duration:
            errors.append(f"Skipped {time_sec}s (exceeds duration {duration:.2f}s)")
            continue

        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)

        ret, frame = cap.read()
        if not ret:
            errors.append(f"Cannot read frame at {time_sec}s")
            continue

        # Generate the output filename.
        if output_prefix:
            output_name = f"{output_prefix}_{time_sec}s.jpg"
        else:
            output_name = f"{video_name}_frame_{time_sec}s.jpg"

        output_path = (OUTPUT_DIR / output_name).resolve()

        success = cv2.imwrite(str(output_path), frame)
        if success:
            extracted_frames.append({
                "output_path": str(output_path),
                "timestamp": time_sec
            })
        else:
            errors.append(f"Failed to save frame at {time_sec}s")

    cap.release()

    if len(extracted_frames) == 0:
        return {
            "success": False,
            "error": "No frames were extracted",
            "errors": errors
        }

    return {
        "success": True,
        "extracted_count": len(extracted_frames),
        "requested_count": len(time_points),
        "frames": extracted_frames,
        "output_paths": [f["output_path"] for f in extracted_frames],
        "timestamps": [f["timestamp"] for f in extracted_frames],
        "errors": errors if errors else None
    }


async def handle_extract_frame(arguments: dict) -> dict:
    time_points = arguments.get("time_points")

    # If time_points is provided, use multi-frame extraction
    if time_points and isinstance(time_points, list) and len(time_points) > 0:
        return extract_multiple_frames(
            video_path=arguments.get("video_path", ""),
            time_points=time_points,
            output_prefix=arguments.get("output_name", ""),
        )

    # Otherwise, use single-frame extraction
    time_sec = arguments.get("time_sec")
    if time_sec is None:
        return {"success": False, "error": "Either time_sec or time_points must be provided"}

    return extract_frame(
        video_path=arguments.get("video_path", ""),
        time_sec=time_sec,
        output_name=arguments.get("output_name", ""),
    )


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="frame-extract-mcp",
        tool_defs_source="_tool_defs/basic.yaml",
        handlers={"extract_frame": handle_extract_frame},
        is_file=True,
    )
    asyncio.run(run_server(app))
