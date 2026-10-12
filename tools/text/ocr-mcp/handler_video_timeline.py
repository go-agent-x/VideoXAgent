#!/usr/bin/env python3
"""Handler for ocr_video_text_timeline tool — Extract a timeline of on-screen text from video frames."""

import asyncio
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from difflib import SequenceMatcher

from ocr_common import run_ocr

_MAX_FRAMES = 25  # hard cap; auto-reduce sample_fps when exceeded


def _text_similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings. Returns float in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def _extract_frames(video_path, output_dir, start_time=0, end_time=None, sample_fps=1.0):
    """
    Extract frames from a video using ffmpeg.
    Returns: (success, list_of_frame_paths, error_message)
    """
    if not os.path.isfile(video_path):
        return False, [], f"Video file not found: {video_path}"

    os.makedirs(output_dir, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", video_path]
    if end_time is not None and end_time > start_time:
        cmd.extend(["-t", str(end_time - start_time)])
    cmd.extend(["-vf", f"fps={sample_fps}"])
    cmd.extend(["-q:v", "2", os.path.join(output_dir, "frame_%06d.jpg")])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return False, [], f"ffmpeg error: {result.stderr[:500]}"
    except subprocess.TimeoutExpired:
        return False, [], "ffmpeg timed out (300s limit)"
    except Exception as e:
        return False, [], f"ffmpeg execution error: {str(e)}"

    frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    ])

    return True, frames, ""


def _get_duration(video_path):
    """Get video duration in seconds using ffprobe. Returns None on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def _process_frames_sync(frame_paths, start_time, sample_fps, lang, min_confidence, sim_threshold):
    """Synchronous OCR processing — meant to run in a thread pool."""
    text_tracks = []

    for frame_idx, frame_path in enumerate(frame_paths):
        frame_time = round(start_time + frame_idx / sample_fps, 2)

        success, texts, _, _, _ = run_ocr(frame_path, lang=lang, det_thresh=min_confidence)
        if not success or not texts:
            continue

        for t in texts:
            matched = False
            for track in text_tracks:
                sim = _text_similarity(t["text"], track["text"])
                if sim >= sim_threshold:
                    track["last_seen"] = frame_time
                    track["appear_count"] += 1
                    track["confidence_sum"] += t["confidence"]
                    if t.get("position"):
                        track["positions"].append(t["position"])
                    matched = True
                    break

            if not matched:
                text_tracks.append({
                    "text": t["text"],
                    "first_seen": frame_time,
                    "last_seen": frame_time,
                    "appear_count": 1,
                    "confidence_sum": t["confidence"],
                    "positions": [t.get("position", "")],
                })

    return text_tracks


async def handle_ocr_video_text_timeline(arguments: dict) -> dict:
    """Extract a timeline of on-screen text from video frames."""
    video_path = arguments.get("video_path", "")
    start_time = arguments.get("start_time", 0)
    end_time = arguments.get("end_time")
    sample_fps = arguments.get("sample_fps", 1.0)
    lang = arguments.get("lang", "ch")
    min_confidence = arguments.get("min_confidence", 0.6)
    sim_threshold = arguments.get("similarity_threshold", 0.8)

    if not video_path:
        return {"success": False, "error": "video_path is required"}
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"Video file not found: {video_path}"}

    if end_time is None:
        duration = _get_duration(video_path)
        if duration is None:
            return {"success": False, "error": "Cannot determine video duration. Please specify end_time."}
        end_time = duration

    # Auto-reduce sample_fps if estimated frame count exceeds limit
    original_fps = sample_fps
    span = max(end_time - start_time, 0)
    estimated_frames = span * sample_fps
    if estimated_frames > _MAX_FRAMES and span > 0:
        sample_fps = round(_MAX_FRAMES / span, 3)
        sample_fps = max(sample_fps, 0.05)  # floor

    tmpdir = tempfile.mkdtemp(prefix="ocr_timeline_")
    try:
        ok, frame_paths, ffmpeg_err = _extract_frames(
            video_path, tmpdir,
            start_time=start_time,
            end_time=end_time,
            sample_fps=sample_fps,
        )
        if not ok:
            return {"success": False, "error": ffmpeg_err}

        if not frame_paths:
            return {
                "success": True, "video_path": video_path,
                "time_range": {"start_time": start_time, "end_time": end_time},
                "frames_analyzed": 0, "unique_texts": 0, "timeline": [],
                "summary": "No frames extracted from the specified range",
            }

        # Run synchronous OCR in thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        text_tracks = await loop.run_in_executor(
            None,
            _process_frames_sync,
            frame_paths, start_time, sample_fps, lang, min_confidence, sim_threshold,
        )

        timeline = []
        for track in text_tracks:
            pos_counts = defaultdict(int)
            for p in track["positions"]:
                if p:
                    pos_counts[p] += 1
            avg_position = max(pos_counts, key=pos_counts.get) if pos_counts else ""

            timeline.append({
                "text": track["text"],
                "first_seen": track["first_seen"],
                "last_seen": track["last_seen"],
                "appear_count": track["appear_count"],
                "avg_confidence": round(track["confidence_sum"] / track["appear_count"], 4),
                # "avg_position": avg_position,
            })

        timeline.sort(key=lambda x: x["first_seen"])

        result = {
            "success": True,
            "video_path": video_path,
            "time_range": {"start_time": start_time, "end_time": end_time},
            "sample_fps_used": sample_fps,
            "frames_analyzed": len(frame_paths),
            "unique_texts": len(timeline),
            "timeline": timeline,
            "summary": f"Found {len(timeline)} unique texts across {len(frame_paths)} frames",
        }
        if sample_fps != original_fps:
            result["fps_auto_reduced"] = (
                f"Requested {original_fps} fps would produce ~{int(span * original_fps)} frames "
                f"(exceeds limit {_MAX_FRAMES}), auto-reduced to {sample_fps} fps"
            )
        return result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
