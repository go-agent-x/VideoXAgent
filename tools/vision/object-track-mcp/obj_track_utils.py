#!/usr/bin/env python3
"""
Utilities for video-track MCP tool.

Provides:
- SAM2 video predictor lazy loading (configured via environment variables)
- Video frame extraction (OpenCV → JPEG directory for SAM2 compatibility)
- Mask geometry: bbox, center, area
- GPU memory and temp directory management
"""

import gc
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
_raw_cfg = _sam2_cfg["model_cfg"]
# Hydra compose() needs a relative config name (e.g. "configs/sam2.1/…"),
# not an absolute filesystem path.  Strip the package prefix if present.
if os.path.isabs(_raw_cfg):
    _marker = "/configs/"
    idx = _raw_cfg.find(_marker)
    SAM2_MODEL_CFG = _raw_cfg[idx + 1:] if idx >= 0 else _raw_cfg
else:
    SAM2_MODEL_CFG = _raw_cfg
SAM2_CHECKPOINT = _sam2_cfg["checkpoint"]
SAM_DEVICE = _sam2_cfg["device"]

# ---------------------------------------------------------------------------
# Lazy-loaded model singleton
# ---------------------------------------------------------------------------
_video_predictor = None


def get_video_predictor():
    """Load SAM2 video predictor once and cache it."""
    global _video_predictor
    if _video_predictor is None:
        from sam2.build_sam import build_sam2_video_predictor
        if not SAM2_CHECKPOINT:
            raise RuntimeError(
                "SAM2_CHECKPOINT environment variable is not set. "
                "Please set it to the path of the SAM2 model weights."
            )
        _video_predictor = build_sam2_video_predictor(
            config_file=SAM2_MODEL_CFG,
            ckpt_path=SAM2_CHECKPOINT,
            device=SAM_DEVICE,
        )
    return _video_predictor


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
    Files are named as "00000.jpg", "00001.jpg", etc. (SAM2 format).

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