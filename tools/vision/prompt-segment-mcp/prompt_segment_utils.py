#!/usr/bin/env python3
"""
Utilities for prompt-segment MCP tool.

Provides:
- SAM model lazy loading (configured via environment variables)
- Image reading and validation
- Mask area / bbox / center calculations
- Color analysis (mean RGB, dominant color name)
- GPU memory management
"""

import gc
import os
import sys
from pathlib import Path
from typing import List, Tuple

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
_sam_cfg = get_config("sam")
SAM_MODEL_TYPE = _sam_cfg["model_type"]
SAM_CHECKPOINT = _sam_cfg["checkpoint"]
SAM_DEVICE = _sam_cfg["device"]

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------
_sam_model = None
_predictor = None


def _get_sam_model():
    """Load SAM model once and cache it."""
    global _sam_model
    if _sam_model is None:
        from segment_anything import sam_model_registry
        if not SAM_CHECKPOINT:
            raise RuntimeError(
                "SAM_CHECKPOINT environment variable is not set. "
                "Please set it to the path of the SAM model weights."
            )
        _sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        _sam_model.to(device=SAM_DEVICE)
    return _sam_model


def get_predictor():
    """Get a SamPredictor (lazy init)."""
    global _predictor
    if _predictor is None:
        from segment_anything import SamPredictor
        sam = _get_sam_model()
        _predictor = SamPredictor(sam)
    return _predictor


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------
def read_image(image_path: str) -> np.ndarray:
    """Read an image file and return it as RGB numpy array."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image (unsupported format?): {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
    """Count the number of True pixels in a mask."""
    return int(np.sum(mask))


# ---------------------------------------------------------------------------
# Color analysis
# ---------------------------------------------------------------------------
# Common color name mapping (HSV-based)
COLOR_RANGES = [
    # (name, h_low, h_high, s_low, s_high, v_low, v_high)
    ("red",     0,   10,  70, 255, 50, 255),
    ("red",   170,  180,  70, 255, 50, 255),
    ("orange",  10,   25,  70, 255, 50, 255),
    ("yellow",  25,   35,  70, 255, 50, 255),
    ("green",   35,   85,  70, 255, 50, 255),
    ("blue",    85,  130,  70, 255, 50, 255),
    ("purple", 130,  170,  70, 255, 50, 255),
    ("white",    0,  180,   0,  30, 200, 255),
    ("gray",     0,  180,   0,  30,  50, 200),
    ("black",    0,  180,   0, 255,   0,  50),
    ("brown",   10,   25,  70, 255,  20, 150),
]


def analyze_color(image_rgb: np.ndarray, mask: np.ndarray) -> Tuple[List[int], str]:
    """
    Analyze the color of a masked region.

    Returns:
        (mean_rgb, dominant_color_name)
    """
    # Extract pixels within the mask
    pixels = image_rgb[mask]
    if len(pixels) == 0:
        return [0, 0, 0], "unknown"

    # Mean RGB
    mean_rgb = [int(np.mean(pixels[:, 0])), int(np.mean(pixels[:, 1])), int(np.mean(pixels[:, 2]))]

    # Convert to HSV for dominant color detection
    # Use the mean color for simplicity
    pixel_bgr = np.uint8([[[mean_rgb[2], mean_rgb[1], mean_rgb[0]]]])
    pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(pixel_hsv[0]), int(pixel_hsv[1]), int(pixel_hsv[2])

    # Match against color ranges (priority order)
    # Check achromatic first
    if v < 50:
        return mean_rgb, "black"
    if s < 30 and v >= 200:
        return mean_rgb, "white"
    if s < 30:
        return mean_rgb, "gray"

    # Check chromatic colors
    for name, h_low, h_high, s_low, s_high, v_low, v_high in COLOR_RANGES:
        if name in ("white", "gray", "black"):
            continue
        if h_low <= h <= h_high and s_low <= s <= s_high and v_low <= v <= v_high:
            return mean_rgb, name

    return mean_rgb, "unknown"


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------
def cleanup_gpu():
    """Free unused GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()