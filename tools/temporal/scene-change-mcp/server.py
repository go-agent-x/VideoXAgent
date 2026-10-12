#!/usr/bin/env python3
"""Scene Change MCP Server — Detect shot boundaries and scene transitions in video."""
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
sys.path.insert(0, str(Path(__file__).parent))  # to import from original scene_change_server.py

from _base.server_framework import create_mcp_server, run_server

# ── Business logic (from scene_change_server.py) ──

import json
import os
import traceback

import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector


# ---------------------------------------------------------------------------
# MCP Server init
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: format seconds to HH:MM:SS.mmm
# ---------------------------------------------------------------------------
def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Helper: get video duration via OpenCV
# ---------------------------------------------------------------------------
def _get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS for video: {video_path}")
    return frame_count / fps


# ---------------------------------------------------------------------------
# Helper: merge adjacent scenes to limit output size
# ---------------------------------------------------------------------------
def _merge_scenes(scenes: list, max_scenes: int) -> list:
    """Greedily merge shortest adjacent scene pairs until count <= max_scenes."""
    if len(scenes) <= max_scenes:
        return scenes
    merged = list(scenes)
    while len(merged) > max_scenes:
        best_i, best_dur = 0, float("inf")
        for i in range(len(merged) - 1):
            d = merged[i]["duration"] + merged[i + 1]["duration"]
            if d < best_dur:
                best_dur, best_i = d, i
        a, b = merged[best_i], merged[best_i + 1]
        merged = merged[:best_i] + [{
            "scene_id": a["scene_id"],
            "start_time": a["start_time"],
            "end_time": b["end_time"],
            "duration": round(b["end_time"] - a["start_time"], 3),
        }] + merged[best_i + 2:]
    for i, s in enumerate(merged):
        s["scene_id"] = i + 1
    return merged


# ---------------------------------------------------------------------------
# Core: detect scene changes
# ---------------------------------------------------------------------------
def detect_scenes(
    video_path: str,
    start_time: float = 0.0,
    end_time: float = None,
    threshold: float = None,
    detector_type: str = "content",
    min_scene_len: float = 0.5,
    max_scenes: int = 100,
) -> dict:
    """Detect scene changes in a video file.

    Args:
        video_path: Absolute path to video file.
        start_time: Start time in seconds (default 0).
        end_time: End time in seconds (default: video end).
        threshold: Detection sensitivity threshold (None = auto-select per detector).
            For content detector: higher = less sensitive (default 27.0).
            For adaptive detector: higher = less sensitive (default 3.0).
        detector_type: "content" or "adaptive".
        min_scene_len: Minimum scene length in seconds (default 0.5).

    Returns:
        dict with total_scenes, duration, scenes list.
    """
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"File not found: {video_path}"}

    try:
        video = open_video(video_path)
        fps = video.frame_rate

        # Clamp min_scene_len to frames
        min_scene_frames = max(1, int(min_scene_len * fps))

        # Create detector
        if detector_type == "adaptive":
            if threshold is None:
                threshold = 3.0  # adaptive default
            detector = AdaptiveDetector(
                adaptive_threshold=threshold,
                min_scene_len=min_scene_frames,
            )
        else:
            if threshold is None:
                threshold = 27.0  # content default
            detector = ContentDetector(
                threshold=threshold,
                min_scene_len=min_scene_frames,
            )

        scene_manager = SceneManager()
        scene_manager.add_detector(detector)

        # Set start/end
        if start_time > 0:
            video.seek(start_time)

        duration = _get_video_duration(video_path)
        actual_end = end_time if end_time is not None else duration
        duration_to_process = actual_end - start_time

        # Process
        scene_manager.detect_scenes(
            video=video,
            duration=duration_to_process if end_time is not None else None,
        )

        scene_list = scene_manager.get_scene_list()

        # Build result
        scenes = []
        for i, (scene_start, scene_end) in enumerate(scene_list):
            s_sec = scene_start.get_seconds()
            e_sec = scene_end.get_seconds()
            dur = e_sec - s_sec
            scenes.append({
                "scene_id": i + 1,
                "start_time": round(s_sec, 3),
                "end_time": round(e_sec, 3),
                "duration": round(dur, 3),
            })

        total_scenes_raw = len(scenes)

        # Merge if exceeds max_scenes (max_scenes <= 0 means no limit)
        merged = False
        if max_scenes > 0 and len(scenes) > max_scenes:
            scenes = _merge_scenes(scenes, max_scenes)
            merged = True

        result = {
            "success": True,
            "video_path": video_path,
            "analyzed_range": {
                "start_time": round(start_time, 3),
                "end_time": round(actual_end, 3),
                "duration": round(actual_end - start_time, 3),
            },
            "detector": detector_type,
            "threshold": threshold,
            "total_scenes": len(scenes),
            "scenes": scenes,
        }

        if merged:
            result["total_scenes_raw"] = total_scenes_raw
            result["merged"] = True
            # Add stats for context
            durations = [s["duration"] for s in scenes]
            avg_dur = sum(durations) / len(durations) if durations else 0
            if avg_dur < 3.0:
                pace = "very_fast_cutting"
            elif avg_dur < 8.0:
                pace = "fast_cutting"
            elif avg_dur < 20.0:
                pace = "moderate"
            elif avg_dur < 60.0:
                pace = "slow"
            else:
                pace = "very_slow"
            result["stats"] = {
                "avg_scene_duration": round(avg_dur, 2),
                "cutting_pace": pace,
            }
            result["note"] = (
                f"Merged from {total_scenes_raw} raw scenes to {len(scenes)}. "
                f"Use start_time/end_time for finer analysis of specific segments."
            )

        return result

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Core: scene summary statistics
# ---------------------------------------------------------------------------
def get_summary(
    video_path: str,
    start_time: float = 0.0,
    end_time: float = None,
    threshold: float = None,
    detector_type: str = "content",
) -> dict:
    """Get high-level summary of scene structure."""
    raw = detect_scenes(
        video_path, start_time, end_time, threshold, detector_type,
        max_scenes=0,  # no merging for summary — need raw stats
    )
    if not raw.get("success"):
        return raw

    scenes = raw["scenes"]
    if not scenes:
        return {
            "success": True,
            "video_path": video_path,
            "total_scenes": 0,
            "note": "No scene changes detected (single continuous shot).",
        }

    durations = [s["duration"] for s in scenes]
    avg_dur = sum(durations) / len(durations)
    min_dur = min(durations)
    max_dur = max(durations)

    # Classify pace
    if avg_dur < 3.0:
        pace = "very_fast_cutting"
    elif avg_dur < 8.0:
        pace = "fast_cutting"
    elif avg_dur < 20.0:
        pace = "moderate"
    elif avg_dur < 60.0:
        pace = "slow"
    else:
        pace = "very_slow"

    total_scene_changes = max(len(scenes) - 1, 0)

    return {
        "success": True,
        "video_path": video_path,
        "analyzed_range": raw["analyzed_range"],
        "total_scenes": raw["total_scenes"],
        "total_scene_changes": total_scene_changes,
        "avg_scene_duration": round(avg_dur, 2),
        "min_scene_duration": round(min_dur, 2),
        "max_scene_duration": round(max_dur, 2),
        "cutting_pace": pace,
        "scene_change_rate_per_min": round(
            total_scene_changes / max(raw["analyzed_range"]["duration"] / 60, 0.01), 2
        ),
        "representative_frames": [
            round((s["start_time"] + s["end_time"]) / 2, 3)
            for s in scenes[::max(1, len(scenes) // 50)]  # sample up to ~50 frames
        ],
    }


# =========================================================================
# MCP Tool Registration
# =========================================================================


async def handle_scene_change_detect(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    start_time = float(arguments.get("start_time", 0))
    end_time = arguments.get("end_time")
    if end_time is not None:
        end_time = float(end_time)
    raw_threshold = arguments.get("threshold")
    threshold = float(raw_threshold) if raw_threshold is not None else None
    detector_type = arguments.get("detector_type", "content")
    min_scene_len = float(arguments.get("min_scene_len", 0.5))
    max_scenes = int(arguments.get("max_scenes", 100))
    if max_scenes < 0:
        max_scenes = 100
    mode = arguments.get("mode", "detect")

    if mode == "summary":
        return get_summary(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            threshold=threshold,
            detector_type=detector_type,
        )
    else:
        return detect_scenes(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            threshold=threshold,
            detector_type=detector_type,
            min_scene_len=min_scene_len,
            max_scenes=max_scenes,
        )


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="scene-change",
        tool_defs_source="_tool_defs/temporal.yaml",
        handlers={"scene_change_detect": handle_scene_change_detect},
        is_file=True,
    )
    asyncio.run(run_server(app))
