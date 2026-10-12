#!/usr/bin/env python3
"""
Object-Count MCP Server — Cross-frame deduplicated object counting in video.

NEW IMPLEMENTATION: Uses DINO-X for initial detection + SAM2 VideoPredictor for tracking.
This is much lighter than the old SAM2 AutomaticMaskGenerator approach.
"""
import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np

# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")

from _base.paths import service_dir
sys.path.insert(0, str(Path(__file__).parent))

from _base.server_framework import create_mcp_server, run_server
from _base.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_sam2_cfg = get_config("sam2")
_raw_cfg = _sam2_cfg["model_cfg"]
if os.path.isabs(_raw_cfg):
    _marker = "/configs/"
    idx = _raw_cfg.find(_marker)
    SAM2_MODEL_CFG = _raw_cfg[idx + 1:] if idx >= 0 else _raw_cfg
else:
    SAM2_MODEL_CFG = _raw_cfg
SAM2_CHECKPOINT = _sam2_cfg["checkpoint"]
SAM2_DEVICE = _sam2_cfg["device"]

_MAX_SEGMENT_DURATION = 60  # Hard limit: max 60 seconds per call
_MAX_OBJECTS = 20  # Max objects to track simultaneously


# ---------------------------------------------------------------------------
# Lazy-loaded SAM2 VideoPredictor (same as object_track)
# ---------------------------------------------------------------------------
_video_predictor = None


def _get_video_predictor():
    """Load SAM2 video predictor once and cache it."""
    global _video_predictor
    if _video_predictor is None:
        from sam2.build_sam import build_sam2_video_predictor
        if not SAM2_CHECKPOINT:
            raise RuntimeError("SAM2_CHECKPOINT not set")
        logger.info("Loading SAM2 VideoPredictor (first call) ...")
        _video_predictor = build_sam2_video_predictor(
            config_file=SAM2_MODEL_CFG,
            ckpt_path=SAM2_CHECKPOINT,
            device=SAM2_DEVICE,
        )
        logger.info("SAM2 VideoPredictor loaded and cached.")
    return _video_predictor


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def _load_gdino():
    """Lazy-import Grounding DINO API utilities."""
    try:
        from gdino_api import call_gdino_api, nms_filter
        return call_gdino_api, nms_filter
    except ImportError:
        gdino_dir = str(service_dir("grounding-dino"))
        sys.path.insert(0, gdino_dir)
        from gdino_api import call_gdino_api, nms_filter
        return call_gdino_api, nms_filter


def _detect_objects_with_gdino(frame_path: str, target_category: str, box_threshold: float = 0.25):
    """
    Detect objects of target_category in the given frame.
    Returns list of {'bbox': [x1,y1,x2,y2], 'score': float, 'category': str}
    """
    try:
        call_gdino_api, nms_filter = _load_gdino()
    except ImportError as e:
        logger.error(f"Grounding DINO not available: {e}")
        return []

    result = call_gdino_api(
        image_source=frame_path,
        prompt=target_category,
        targets=["bbox"],
        box_threshold=box_threshold,
    )
    if not result.get("success") or not result.get("objects"):
        return []

    # Filter by category and apply NMS
    target_lower = target_category.lower()
    matched = [
        o for o in result["objects"]
        if target_lower in o.get("category", "").lower()
    ]
    deduped = nms_filter(matched, iou_threshold=0.5)
    return deduped[:_MAX_OBJECTS]  # Limit number of objects


# ---------------------------------------------------------------------------
# Frame extraction utilities (simplified from obj_track_utils)
# ---------------------------------------------------------------------------
import cv2
import tempfile
import shutil


def _get_video_info(video_path: str):
    """Get video metadata."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total_frames / fps if fps > 0 else 0
    return {"fps": fps, "total_frames": total_frames, "duration": duration, "width": width, "height": height}


def _extract_frames_to_jpeg_dir(video_path: str, sample_fps: float, start_time: float, end_time: float):
    """Extract frames to a temp JPEG directory (SAM2 format)."""
    info = _get_video_info(video_path)
    native_fps = info["fps"]

    frame_times = []
    t = start_time
    while t <= end_time + 1e-6:
        frame_times.append(round(t, 4))
        t += 1.0 / sample_fps

    jpeg_dir = tempfile.mkdtemp(prefix="sam2_count_")
    cap = cv2.VideoCapture(video_path)
    for i, t in enumerate(frame_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(jpeg_dir, f"{i:05d}.jpg"), frame)
        else:
            black = np.zeros((info["height"], info["width"], 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(jpeg_dir, f"{i:05d}.jpg"), black)
    cap.release()
    return jpeg_dir, frame_times, info


def _cleanup_temp_dir(path: str):
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _cleanup_gpu():
    import torch
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def _mask_to_bbox(mask: np.ndarray):
    """Convert binary mask to [x1,y1,x2,y2]."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2) + 1, int(y2) + 1]


def _mask_area(mask: np.ndarray) -> int:
    return int(np.sum(mask))


# =========================================================================
# Tool: tracked_object_count — DINO-X detection + SAM2 VideoPredictor tracking
# =========================================================================
async def _handle_object_count(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    start_time = float(arguments.get("start_time", 0))
    end_time_raw = arguments.get("end_time", None)
    sample_fps = float(arguments.get("sample_fps", 0.5))
    min_track_frames = int(arguments.get("min_track_frames", 2))
    target_category = arguments.get("target_category", "")

    if not video_path:
        return {"success": False, "error": "video_path is required"}

    if end_time_raw is None:
        return {
            "success": False,
            "error": "end_time is required. Use scene captions first to identify the relevant time range."
        }

    end_time = float(end_time_raw)
    segment_duration = end_time - start_time
    if segment_duration > _MAX_SEGMENT_DURATION:
        return {
            "success": False,
            "error": f"Segment too long: {segment_duration:.1f}s exceeds max {_MAX_SEGMENT_DURATION}s."
        }
    if segment_duration <= 0:
        return {"success": False, "error": f"Invalid time range: start_time >= end_time"}

    if not target_category:
        return {
            "success": False,
            "error": "target_category is required. Specify what to count (e.g., 'person', 'car', 'laptop')."
        }

    jpeg_dir = None
    try:
        # Step 1: Get video info and extract frames
        logger.info(f"[1/5] Getting video info: {video_path}")
        info = _get_video_info(video_path)
        logger.info(f"[2/5] Video duration={info['duration']:.1f}s, analyzing {start_time:.1f}-{end_time:.1f}s")

        logger.info(f"[3/5] Extracting frames at {sample_fps} fps...")
        jpeg_dir, frame_times, _ = _extract_frames_to_jpeg_dir(
            video_path, sample_fps=sample_fps, start_time=start_time, end_time=end_time
        )
        num_frames = len(frame_times)
        logger.info(f"[3/5] Extracted {num_frames} frames to {jpeg_dir}")

        if num_frames == 0:
            _cleanup_temp_dir(jpeg_dir)
            return {"success": False, "error": "No frames extracted"}

        # Step 2: Use DINO-X to detect target objects in the FIRST frame
        first_frame_path = os.path.join(jpeg_dir, "00000.jpg")
        logger.info(f"[4/5] Detecting '{target_category}' in first frame with DINO-X...")
        initial_detections = _detect_objects_with_gdino(first_frame_path, target_category)
        logger.info(f"[4/5] DINO-X found {len(initial_detections)} objects in first frame")

        if not initial_detections:
            _cleanup_temp_dir(jpeg_dir)
            return {
                "success": True,
                "unique_object_count": 0,
                "objects": [],
                "appearance_timeline": "",
                "summary": f"No '{target_category}' detected in the first frame ({start_time:.1f}s). "
                           "Try a different time range or check if objects appear later."
            }

        # Step 3: Initialize SAM2 VideoPredictor and add all detected objects as prompts
        logger.info(f"[5/5] Tracking {len(initial_detections)} objects with SAM2 VideoPredictor...")
        predictor = _get_video_predictor()
        inference_state = predictor.init_state(video_path=jpeg_dir, offload_video_to_cpu=True)

        # Add each detected object as a separate tracking target
        for obj_idx, det in enumerate(initial_detections):
            obj_id = obj_idx + 1
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            box_np = np.array(bbox, dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,  # First frame
                obj_id=obj_id,
                box=box_np,
            )

        # Step 4: Propagate tracking forward through all frames
        video_segments = {}
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
            video_segments[frame_idx] = {
                "obj_ids": list(obj_ids),
                "masks": (masks > 0.0).cpu().numpy(),
            }

        # Step 5: Analyze tracking results
        object_stats = {i + 1: {"visible_frames": [], "areas": []} for i in range(len(initial_detections))}

        for fi in sorted(video_segments.keys()):
            seg = video_segments[fi]
            t = frame_times[fi] if fi < len(frame_times) else fi / sample_fps

            for oi, obj_id in enumerate(seg["obj_ids"]):
                if obj_id in object_stats:
                    mask = seg["masks"][oi][0]  # [H, W]
                    area = _mask_area(mask)
                    if area > 50:  # Visibility threshold
                        object_stats[obj_id]["visible_frames"].append(fi)
                        object_stats[obj_id]["areas"].append(area)

        # Reset predictor state
        predictor.reset_state(inference_state)
        _cleanup_gpu()

        # Build output: filter by min_track_frames
        objects = []
        timeline_parts = []
        for obj_id, stats in object_stats.items():
            if len(stats["visible_frames"]) >= min_track_frames:
                first_fi = min(stats["visible_frames"])
                last_fi = max(stats["visible_frames"])
                first_t = frame_times[first_fi] if first_fi < len(frame_times) else 0
                last_t = frame_times[last_fi] if last_fi < len(frame_times) else 0
                avg_area = int(np.mean(stats["areas"])) if stats["areas"] else 0

                objects.append({
                    "object_id": obj_id,
                    "first_seen": round(first_t, 2),
                    "last_seen": round(last_t, 2),
                    "appeared_frames": len(stats["visible_frames"]),
                    "avg_area": avg_area,
                })
                timeline_parts.append(f"obj{obj_id}: [{first_t:.1f}-{last_t:.1f}]")

        _cleanup_temp_dir(jpeg_dir)

        summary = (
            f"{len(objects)} unique '{target_category}' objects tracked across {num_frames} frames "
            f"({start_time:.1f}s to {end_time:.1f}s). "
            f"Initially detected {len(initial_detections)} objects in first frame."
        )

        return {
            "success": True,
            "unique_object_count": len(objects),
            "initial_detections": len(initial_detections),
            "objects": objects,
            "appearance_timeline": ", ".join(timeline_parts),
            "summary": summary,
        }

    except Exception as e:
        _cleanup_gpu()
        if jpeg_dir:
            _cleanup_temp_dir(jpeg_dir)
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# Entry point
# =========================================================================
async def handle_object_count(arguments: dict) -> dict:
    return await _handle_object_count(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="object-count",
        tool_defs_source="_tool_defs/segment.yaml",
        handlers={"tracked_object_count": handle_object_count},
        is_file=True,
    )
    asyncio.run(run_server(app))
