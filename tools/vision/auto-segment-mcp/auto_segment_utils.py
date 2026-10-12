#!/usr/bin/env python3
"""
Utilities for auto-segment MCP tool.

Provides:
- SAM model lazy loading (configured via environment variables)
- Image reading and validation
- Mask bbox / center calculations
- GPU memory management
"""

import gc
import os
import sys
from pathlib import Path
from typing import List

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
_mask_generator = None


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


def get_mask_generator(**kwargs):
    """Get a SamAutomaticMaskGenerator (lazy init)."""
    global _mask_generator
    from segment_anything import SamAutomaticMaskGenerator
    sam = _get_sam_model()
    # Recreate if custom kwargs provided, otherwise reuse default
    if kwargs or _mask_generator is None:
        _mask_generator = SamAutomaticMaskGenerator(sam, **kwargs)
    return _mask_generator


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
def bbox_from_sam(ann: dict) -> List[int]:
    """Extract [x1, y1, x2, y2] from SAM annotation's bbox [x, y, w, h]."""
    x, y, w, h = ann["bbox"]
    return [int(x), int(y), int(x + w), int(y + h)]


def center_from_bbox(bbox: List[int]) -> List[int]:
    """Compute center from [x1, y1, x2, y2]."""
    return [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------
def cleanup_gpu():
    """Free unused GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()