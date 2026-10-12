#!/usr/bin/env python3
"""Video Segment Extract MCP Server — Extract a time-range sub-clip from video."""
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

# ── Business logic (from video_segment_extract_server.py) ──

import json
import os
import subprocess
import shutil

OUTPUT_DIR = PROJECT_ROOT / "output"

SUPPORTED_FORMATS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".ts", ".mpg", ".mpeg", ".3gp"
}


def _parse_time(t) -> float:
    """Convert time parameters to seconds (float). Accept numbers and HH:MM:SS.mmm strings."""
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, str):
        t = t.strip()
        # HH:MM:SS.mmm format
        parts = t.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
        return float(t)
    return float(t)


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _get_duration_ffprobe(video_path: str) -> float | None:
    """Get the video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def extract_video_segment(
    video_path: str,
    start_time,
    end_time,
    output_name: str = "",
    output_format: str = "",
    preserve_audio: bool = True,
) -> dict:
    """Extract a continuous video segment, preserving both video and audio tracks."""

    # Validate the input.
    if not os.path.exists(video_path):
        return {"success": False, "error": f"Video not found: {video_path}"}

    suffix = Path(video_path).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        return {"success": False, "error": f"Unsupported format: {suffix}"}

    if not shutil.which("ffmpeg"):
        return {"success": False, "error": "ffmpeg not found in PATH"}

    start_sec = _parse_time(start_time)
    end_sec = _parse_time(end_time)

    if start_sec < 0:
        return {"success": False, "error": f"start_time must be >= 0, got {start_sec}"}
    if end_sec <= start_sec:
        return {"success": False, "error": f"end_time ({end_sec}) must be > start_time ({start_sec})"}

    # Optionally validate the duration.
    total_duration = _get_duration_ffprobe(video_path)
    clamped = False
    if total_duration is not None:
        if start_sec >= total_duration:
            return {"success": False, "error": f"start_time ({start_sec}s) exceeds video duration ({total_duration:.2f}s)"}
        if end_sec > total_duration:
            end_sec = total_duration
            clamped = True
            if end_sec <= start_sec:
                return {"success": False, "error": f"end_time clamped to duration ({total_duration:.2f}s) but that is <= start_time ({start_sec}s)"}

    # Determine the output format and path.
    if not output_format:
        output_format = suffix.lstrip(".")
    output_format = output_format.lower().lstrip(".")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not output_name:
        video_name = Path(video_path).stem
        output_name = f"{video_name}_segment_{start_sec:.1f}s-{end_sec:.1f}s.{output_format}"
    elif not Path(output_name).suffix:
        output_name = f"{output_name}.{output_format}"

    output_path = str((OUTPUT_DIR / output_name).resolve())

    # Build the ffmpeg command.
    # -ss before -i enables input seeking (fast).
    # -to specifies an end time relative to -ss, so use duration.
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",            # Overwrite the output.
        "-ss", str(start_sec),     # Seek to the start.
        "-i", video_path,          # Input file
        "-t", str(duration),       # Segment duration
        "-c", "copy",              # Stream copy (very fast, no re-encoding)
    ]

    if not preserve_audio:
        cmd.extend(["-an"])  # Remove the audio track.

    cmd.append(output_path)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "ffmpeg timed out (>300s)"}

    if result.returncode != 0:
        # Try re-encoding if stream copy fails (some formats or timestamps are incompatible).
        cmd_reencode = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", video_path,
            "-t", str(duration),
        ]
        if not preserve_audio:
            cmd_reencode.append("-an")
        cmd_reencode.append(output_path)

        try:
            result2 = subprocess.run(
                cmd_reencode, capture_output=True, text=True, timeout=600
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "ffmpeg re-encode timed out (>600s)"}

        if result2.returncode != 0:
            stderr = (result2.stderr or "")[-500:]
            return {"success": False, "error": f"ffmpeg failed: {stderr}"}

    # Validate the output.
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return {"success": False, "error": "Output file is empty or missing"}

    # Get the output duration.
    out_duration = _get_duration_ffprobe(output_path)

    resp = {
        "success": True,
        "output_path": output_path,
        "start_time": start_sec,
        "end_time": end_sec,
        "requested_duration": round(duration, 3),
        "actual_duration": round(out_duration, 3) if out_duration else None,
        "file_size_bytes": os.path.getsize(output_path),
        "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2),
        "preserve_audio": preserve_audio,
    }
    if clamped:
        resp["note"] = f"end_time was clamped from requested value to video duration ({total_duration:.2f}s)"
    return resp


async def handle_extract_video_segment(arguments: dict) -> dict:
    return extract_video_segment(
        video_path=arguments.get("video_path", ""),
        start_time=arguments.get("start_time", 0),
        end_time=arguments.get("end_time", 0),
        output_name=arguments.get("output_name", ""),
        output_format=arguments.get("output_format", ""),
        preserve_audio=arguments.get("preserve_audio", True),
    )


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="video-segment-extract-mcp",
        tool_defs_source="_tool_defs/basic.yaml",
        handlers={"extract_video_segment": handle_extract_video_segment},
        is_file=True,
    )
    asyncio.run(run_server(app))
