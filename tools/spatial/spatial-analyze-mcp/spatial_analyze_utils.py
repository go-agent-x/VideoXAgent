#!/usr/bin/env python3
"""
Utilities for spatial-analyze MCP tool.

Provides:
- SAM2 image predictor lazy loading (configured via environment variables)
- Video frame reading
- Mask geometry: bbox, center, area
- Spatial relation computation between objects
- GPU memory management
"""

import gc
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")
from _base.config import get_config

# ---------------------------------------------------------------------------
# Configuration from _base/config.py
# ---------------------------------------------------------------------------
_sam2_cfg = get_config("sam2")
SAM2_MODEL_CFG = _sam2_cfg["model_cfg"]
SAM2_CHECKPOINT = _sam2_cfg["checkpoint"]
SAM_DEVICE = _sam2_cfg["device"]

# ---------------------------------------------------------------------------
# Lazy-loaded model singleton
# ---------------------------------------------------------------------------
_image_predictor = None


def get_image_predictor():
    """Load SAM2 image predictor once and cache it."""
    global _image_predictor
    if _image_predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        if not SAM2_CHECKPOINT:
            raise RuntimeError(
                "SAM2_CHECKPOINT environment variable is not set. "
                "Please set it to the path of the SAM2 model weights."
            )
        model = build_sam2(
            config_file=SAM2_MODEL_CFG,
            ckpt_path=SAM2_CHECKPOINT,
            device=SAM_DEVICE,
        )
        _image_predictor = SAM2ImagePredictor(model)
    return _image_predictor


# ---------------------------------------------------------------------------
# Video frame reading
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


def read_frame_at_time(video_path: str, time_sec: float) -> np.ndarray:
    """Read a single frame from video at the given time (seconds). Returns RGB array."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Failed to read frame at time {time_sec}s")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Mask geometry utilities
# ---------------------------------------------------------------------------
def mask_to_bbox(mask: np.ndarray) -> List[int]:
    """Convert a binary mask to [x1, y1, x2, y2] bounding box."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2) + 1, int(y2) + 1]


def mask_center(mask: np.ndarray) -> List[int]:
    """Compute the centroid of a binary mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0]
    return [int(np.mean(xs)), int(np.mean(ys))]


def mask_area(mask: np.ndarray) -> int:
    """Count True pixels."""
    return int(np.sum(mask))


def bbox_center(bbox: List[int]) -> List[int]:
    """Center from [x1,y1,x2,y2]."""
    return [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]


def bbox_area(bbox: List[int]) -> int:
    """Area from [x1,y1,x2,y2]."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


# ---------------------------------------------------------------------------
# Spatial relation computation
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


def compute_spatial_relation(
    bbox_a: List[int], bbox_b: List[int],
    label_a: str = "A", label_b: str = "B",
) -> Dict[str, Any]:
    """
    Compute spatial relation between two objects from their bboxes.

    Returns dict with: horizontal, vertical, h_distance, v_distance,
    euclidean_distance, overlap_iou, description.
    """
    ca = bbox_center(bbox_a)
    cb = bbox_center(bbox_b)

    # Horizontal relation (A relative to B)
    if ca[0] < cb[0]:
        horizontal = "LEFT"
    elif ca[0] > cb[0]:
        horizontal = "RIGHT"
    else:
        horizontal = "ALIGNED"

    # Vertical relation (A relative to B)
    if ca[1] < cb[1]:
        vertical = "ABOVE"
    elif ca[1] > cb[1]:
        vertical = "BELOW"
    else:
        vertical = "ALIGNED"

    h_dist = abs(ca[0] - cb[0])
    v_dist = abs(ca[1] - cb[1])
    euc_dist = center_distance(ca, cb)
    iou = compute_iou(bbox_a, bbox_b)

    parts = []
    if horizontal != "ALIGNED":
        parts.append(f"to the {horizontal} of")
    if vertical != "ALIGNED":
        parts.append(f"{vertical}")
    if not parts:
        parts.append("overlapping with")

    desc = f"{label_a} is {' and '.join(parts)} {label_b}"

    return {
        "pair": [label_a, label_b],
        "horizontal": horizontal,
        "vertical": vertical,
        "h_distance": round(h_dist, 1),
        "v_distance": round(v_dist, 1),
        "euclidean_distance": round(euc_dist, 1),
        "overlap_iou": round(iou, 4),
        "description": desc,
    }


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------
def cleanup_gpu():
    """Free unused GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()