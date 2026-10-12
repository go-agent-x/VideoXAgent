#!/usr/bin/env python3
"""
Utilities for object-count MCP tool.

Provides:
- Video frame extraction (OpenCV → JPEG directory)
- Bbox geometry: center, area
- Cross-frame object association (IoU + distance based greedy matching)
- GPU memory and temp directory management
"""

import gc
import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------
def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video metadata: fps, total_frames, duration, width, height."""
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
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "width": width,
        "height": height,
    }


def extract_frames_to_jpeg_dir(
    video_path: str,
    sample_fps: float = 2.0,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> Tuple[str, List[float], Dict[str, Any]]:
    """
    Extract video frames at sample_fps to a temporary JPEG directory.
    Files are named as "00000.jpg", "00001.jpg", etc.

    Returns:
        (jpeg_dir, frame_times, video_info)
        where frame_times[i] is the timestamp (seconds) of frame i.
    """
    info = get_video_info(video_path)
    native_fps = info["fps"]
    duration = info["duration"]

    if end_time is None or end_time > duration:
        end_time = duration

    # Calculate which frames to extract
    frame_times = []
    t = start_time
    while t <= end_time + 1e-6:
        frame_times.append(round(t, 4))
        t += 1.0 / sample_fps

    # Create temp directory
    jpeg_dir = tempfile.mkdtemp(prefix="sam2_frames_")

    cap = cv2.VideoCapture(video_path)
    for i, t in enumerate(frame_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(jpeg_dir, f"{i:05d}.jpg")
            cv2.imwrite(out_path, frame)
        else:
            # Write a black frame as placeholder
            h, w = info["height"], info["width"]
            black = np.zeros((h, w, 3), dtype=np.uint8)
            out_path = os.path.join(jpeg_dir, f"{i:05d}.jpg")
            cv2.imwrite(out_path, black)
    cap.release()

    return jpeg_dir, frame_times, info


# ---------------------------------------------------------------------------
# Bbox geometry utilities
# ---------------------------------------------------------------------------
def bbox_center(bbox: List[int]) -> List[int]:
    """Center from [x1,y1,x2,y2]."""
    return [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]


def bbox_area(bbox: List[int]) -> int:
    """Area from [x1,y1,x2,y2]."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


# ---------------------------------------------------------------------------
# Cross-frame object association
# ---------------------------------------------------------------------------
def compute_iou(bbox_a: List[int], bbox_b: List[int]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] bboxes."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = bbox_area(bbox_a)
    area_b = bbox_area(bbox_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_distance(c1: List[int], c2: List[int]) -> float:
    """Euclidean distance between two centers."""
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def associate_detections_across_frames(
    tracks: List[Dict],
    new_detections: List[Dict],
    iou_threshold: float = 0.3,
    max_distance: float = 150.0,
) -> Tuple[List[Dict], List[int]]:
    """
    Associate new frame detections with existing tracks via greedy IoU + distance matching.

    Each track: {"track_id": int, "bboxes": [...], "areas": [...], "frame_indices": [...]}
    Each detection: {"bbox": [...], "center": [...], "area": int}

    Returns:
        (updated_tracks, matched_track_ids_for_each_detection)
    """
    matched = [None] * len(new_detections)
    used_tracks = set()

    # Build cost matrix: (track_idx, det_idx) -> score
    candidates = []
    for ti, track in enumerate(tracks):
        last_bbox = track["bboxes"][-1]
        last_center = bbox_center(last_bbox)
        for di, det in enumerate(new_detections):
            iou = compute_iou(last_bbox, det["bbox"])
            dist = center_distance(last_center, det["center"])
            if iou >= iou_threshold or dist <= max_distance:
                score = iou - dist / 1000.0  # prefer high IoU
                candidates.append((score, ti, di))

    # Greedy match (highest score first)
    candidates.sort(key=lambda x: -x[0])
    for score, ti, di in candidates:
        if ti in used_tracks or matched[di] is not None:
            continue
        matched[di] = ti
        used_tracks.add(ti)

    # Update tracks and create new ones
    next_id = max((t["track_id"] for t in tracks), default=0) + 1
    for di, det in enumerate(new_detections):
        ti = matched[di]
        if ti is not None:
            tracks[ti]["bboxes"].append(det["bbox"])
            tracks[ti]["areas"].append(det["area"])
            tracks[ti]["frame_indices"].append(det["frame_index"])
        else:
            tracks.append({
                "track_id": next_id,
                "bboxes": [det["bbox"]],
                "areas": [det["area"]],
                "frame_indices": [det["frame_index"]],
            })
            matched[di] = len(tracks) - 1
            next_id += 1

    return tracks, matched


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------
def cleanup_gpu():
    """Free unused GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def cleanup_temp_dir(path: str):
    """Remove a temporary directory."""
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)