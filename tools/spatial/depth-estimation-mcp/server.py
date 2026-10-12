#!/usr/bin/env python3
"""Depth Estimation MCP Server — Monocular depth estimation for spatial perception."""
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

from _base.paths import PROJECT_ROOT
sys.path.insert(0, str(Path(__file__).parent))

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response
from _base.config import get_config

# ── Business logic (from depth_estimation_server.py) ──

import json
import os
import threading
import traceback

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_depth_cfg = get_config("depth")
DEPTH_MODEL = _depth_cfg["model"]
DEPTH_DEVICE = _depth_cfg["device"]
OUTPUT_DIR = os.path.join(str(PROJECT_ROOT), "output")

# ---------------------------------------------------------------------------
# Lazy model loading (thread-safe, load once)
# ---------------------------------------------------------------------------
_model = None
_processor = None
_load_lock = threading.Lock()


def _load_model():
    """Load depth estimation model (lazy, thread-safe)."""
    global _model, _processor
    if _model is not None:
        return

    with _load_lock:
        if _model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        _processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, local_files_only=True)
        _model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, local_files_only=True)
        device = DEPTH_DEVICE if torch.cuda.is_available() else "cpu"
        _model = _model.to(device).eval()


def _predict_depth(image_bgr: np.ndarray) -> np.ndarray:
    """Run depth inference on a BGR image. Returns float32 depth map (H, W)."""
    import torch
    from PIL import Image

    _load_model()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    inputs = _processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(image_bgr.shape[0], image_bgr.shape[1]),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Depth map helpers
# ---------------------------------------------------------------------------

def _depth_to_zones(depth: np.ndarray, num_zones: int = 3) -> list[dict]:
    """Divide the depth range into near/mid/far zones."""
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        return [{"zone": "uniform", "depth_range": [d_min, d_max], "pixel_ratio": 1.0}]

    edges = np.linspace(d_min, d_max, num_zones + 1)
    # Depth Anything outputs inverse depth: higher value = closer to camera.
    # linspace bins go from d_min (farthest) to d_max (nearest), so labels
    # must be ordered far → near.
    labels = ["far", "mid", "near"] if num_zones == 3 else [f"zone_{i}" for i in range(num_zones)]
    total = depth.size
    zones = []
    for i in range(num_zones):
        lo, hi = edges[i], edges[i + 1]
        if i < num_zones - 1:
            mask = (depth >= lo) & (depth < hi)
        else:
            mask = (depth >= lo) & (depth <= hi)
        ratio = float(mask.sum()) / total
        zones.append({
            "zone": labels[i],
            "depth_range": [round(float(lo), 4), round(float(hi), 4)],
            "pixel_ratio": round(ratio, 4),
        })
    return zones


def _save_depth_colormap(depth: np.ndarray, output_path: str) -> str:
    """Save a depth map as a color-mapped image (near=warm, far=cool)."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    # Depth Anything outputs inverse depth: higher value = closer = higher norm.
    # INFERNO maps high values to warm colors, so no inversion needed for near=warm, far=cool.
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, colored)
    return output_path


def _depth_in_bbox(depth: np.ndarray, bbox: list) -> dict:
    """Get depth statistics inside a bbox [x1, y1, x2, y2]."""
    h, w = depth.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return {"error": "Invalid bbox (zero area after clipping)"}
    region = depth[y1:y2, x1:x2]
    return {
        "mean_depth": round(float(region.mean()), 4),
        "median_depth": round(float(np.median(region)), 4),
        "min_depth": round(float(region.min()), 4),
        "max_depth": round(float(region.max()), 4),
        "std_depth": round(float(region.std()), 4),
    }


def _relative_label(depth_a: float, depth_b: float, depth_range: float = 0.0, tolerance: float = 0.02) -> str:
    """Return a human-readable relative depth label.

    Note: Depth Anything outputs *relative inverse depth* — higher values mean CLOSER.

    Args:
        depth_a: Depth value for object A.
        depth_b: Depth value for object B.
        depth_range: Full scene depth range (max - min). When provided and
            > 1e-6, the difference is normalized by scene range for consistency
            with normalized_difference in compare_object_depth.
        tolerance: Ratio threshold below which objects are considered same depth.
    """
    denom = depth_range if depth_range > 1e-6 else max(abs(depth_a), abs(depth_b), 1e-6)
    diff_ratio = (depth_a - depth_b) / denom
    if abs(diff_ratio) < tolerance:
        return "same_depth"
    # Higher depth value = closer to camera in Depth Anything
    return "closer" if depth_a > depth_b else "farther"


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _handle_estimate_depth(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    save_visualization = arguments.get("save_visualization", True)
    num_zones = arguments.get("num_zones", 3)

    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not os.path.isfile(image_path):
        return {"success": False, "error": f"File not found: {image_path}"}
    if num_zones < 1:
        num_zones = 1

    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": f"Cannot read image: {image_path}"}

        depth = _predict_depth(image)
        h, w = depth.shape

        result = {
            "success": True,
            "image_path": image_path,
            "image_size": {"width": w, "height": h},
            "depth_statistics": {
                "min": round(float(depth.min()), 4),
                "max": round(float(depth.max()), 4),
                "mean": round(float(depth.mean()), 4),
                "std": round(float(depth.std()), 4),
            },
            "depth_zones": _depth_to_zones(depth, num_zones),
            "note": "Higher depth value = closer to camera (inverse depth). Use compare_object_depth with bboxes for spatial reasoning.",
        }

        if save_visualization:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            vis_path = os.path.join(OUTPUT_DIR, f"{stem}_depth.png")
            _save_depth_colormap(depth, vis_path)
            result["depth_visualization"] = vis_path

        return result

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


async def _handle_compare_object_depth(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    bbox_a = arguments.get("bbox_a")
    bbox_b = arguments.get("bbox_b")
    label_a = arguments.get("label_a", "A")
    label_b = arguments.get("label_b", "B")

    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not os.path.isfile(image_path):
        return {"success": False, "error": f"File not found: {image_path}"}
    if not bbox_a or not bbox_b:
        return {"success": False, "error": "Both bbox_a and bbox_b are required ([x1,y1,x2,y2])"}
    if len(bbox_a) != 4 or len(bbox_b) != 4:
        return {"success": False, "error": "Each bbox must have exactly 4 values [x1,y1,x2,y2]"}

    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": f"Cannot read image: {image_path}"}

        depth = _predict_depth(image)

        stats_a = _depth_in_bbox(depth, bbox_a)
        stats_b = _depth_in_bbox(depth, bbox_b)

        if "error" in stats_a:
            return {"success": False, "error": f"bbox_a invalid: {stats_a['error']}"}
        if "error" in stats_b:
            return {"success": False, "error": f"bbox_b invalid: {stats_b['error']}"}

        depth_range = float(depth.max() - depth.min())
        rel = _relative_label(stats_a["median_depth"], stats_b["median_depth"], depth_range=depth_range)
        depth_diff = abs(stats_a["median_depth"] - stats_b["median_depth"])
        normalized_diff = round(depth_diff / depth_range, 4) if depth_range > 1e-6 else 0.0

        if rel == "closer":
            description = f"{label_a} is closer to the camera than {label_b}"
        elif rel == "farther":
            description = f"{label_a} is farther from the camera than {label_b}"
        else:
            description = f"{label_a} and {label_b} are at approximately the same depth"

        return {
            "success": True,
            label_a: {"bbox": bbox_a, "depth": stats_a},
            label_b: {"bbox": bbox_b, "depth": stats_b},
            "comparison": {
                "relative_position": rel,
                "depth_difference": round(depth_diff, 4),
                "normalized_difference": normalized_diff,
                "description": description,
            },
            "note": "Higher depth value = closer to camera (inverse depth).",
        }

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


async def _handle_get_depth_at_points(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    points = arguments.get("points")
    labels = arguments.get("labels")
    radius = arguments.get("radius", 5)

    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not os.path.isfile(image_path):
        return {"success": False, "error": f"File not found: {image_path}"}
    if not points or not isinstance(points, list):
        return {"success": False, "error": "points is required (list of [x, y])"}

    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": f"Cannot read image: {image_path}"}

        depth = _predict_depth(image)
        h, w = depth.shape

        if labels and len(labels) != len(points):
            return {"success": False, "error": "labels length must match points length"}

        results = []
        for i, pt in enumerate(points):
            if len(pt) != 2:
                results.append({"error": f"Point {i} must be [x, y]"})
                continue
            x, y = int(pt[0]), int(pt[1])
            if x < 0 or x >= w or y < 0 or y >= h:
                results.append({"error": f"Point ({x},{y}) out of bounds ({w}x{h})"})
                continue

            # Sample a small region around the point for robustness
            r = max(1, int(radius))
            y1, y2 = max(0, y - r), min(h, y + r + 1)
            x1, x2 = max(0, x - r), min(w, x + r + 1)
            region = depth[y1:y2, x1:x2]

            entry = {
                "point": [x, y],
                "depth_value": round(float(depth[y, x]), 4),
                "local_mean": round(float(region.mean()), 4),
            }
            if labels:
                entry["label"] = labels[i]
            results.append(entry)

        # Rank by depth (closest first, higher depth = closer)
        valid = [r for r in results if "error" not in r]
        valid.sort(key=lambda r: r["local_mean"], reverse=True)
        for rank, r in enumerate(valid):
            r["rank_near_to_far"] = rank + 1

        return {
            "success": True,
            "image_size": {"width": w, "height": h},
            "points": results,
            "note": "Higher depth value = closer to camera. rank 1 = nearest.",
        }

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def handle_estimate_depth(arguments: dict) -> dict:
    return await _handle_estimate_depth(arguments)


async def handle_compare_object_depth(arguments: dict) -> dict:
    return await _handle_compare_object_depth(arguments)


async def handle_get_depth_at_points(arguments: dict) -> dict:
    return await _handle_get_depth_at_points(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="depth-estimation",
        tool_defs_source="_tool_defs/misc.yaml",
        handlers={
            "estimate_depth": handle_estimate_depth,
            "compare_object_depth": handle_compare_object_depth,
            "get_depth_at_points": handle_get_depth_at_points,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
