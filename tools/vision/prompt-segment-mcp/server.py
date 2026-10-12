#!/usr/bin/env python3
"""Prompt-Segment MCP Server — Prompt-driven precise segmentation."""
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
sys.path.insert(0, str(Path(__file__).parent))  # to import from original prompt_segment_server.py

from _base.server_framework import create_mcp_server, run_server

# ── Business logic (from prompt_segment_server.py) ──

import json
import os
import traceback

import numpy as np

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from prompt_segment_utils import (
    get_predictor,
    read_image,
    mask_to_bbox,
    mask_center,
    mask_area,
    analyze_color,
    cleanup_gpu,
)

# ---------------------------------------------------------------------------
# MCP Server init
# ---------------------------------------------------------------------------


# =========================================================================
# Tool: prompt_segment — Prompt-driven precise segmentation
# =========================================================================
async def _handle_prompt_segment(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    point_coords_raw = arguments.get("point_coords", None)
    point_labels_raw = arguments.get("point_labels", None)
    box_raw = arguments.get("box", None)
    multimask_output = arguments.get("multimask_output", True)

    if not image_path:
        return {"success": False, "error": "image_path is required"}

    point_coords = None
    point_labels = None
    box = None

    if point_coords_raw is not None:
        if isinstance(point_coords_raw, str):
            point_coords_raw = json.loads(point_coords_raw)
        point_coords = np.array(point_coords_raw, dtype=np.float32)

    if point_labels_raw is not None:
        if isinstance(point_labels_raw, str):
            point_labels_raw = json.loads(point_labels_raw)
        point_labels = np.array(point_labels_raw, dtype=np.int32)
    elif point_coords is not None:
        # Default: all foreground
        point_labels = np.ones(len(point_coords), dtype=np.int32)

    if box_raw is not None:
        if isinstance(box_raw, str):
            box_raw = json.loads(box_raw)
        box = np.array(box_raw, dtype=np.float32)

    if isinstance(multimask_output, str):
        multimask_output = multimask_output.lower() in ("true", "1", "yes")

    # Validate: at least one prompt
    if point_coords is None and box is None:
        return {"success": False, "error": "At least one of point_coords or box must be provided"}

    # Validate point_coords / point_labels match
    if point_coords is not None and point_labels is not None:
        if len(point_coords) != len(point_labels):
            return {
                "success": False,
                "error": f"point_coords length ({len(point_coords)}) != point_labels length ({len(point_labels)})",
            }

    try:
        image = read_image(image_path)
        h, w = image.shape[:2]
        total_pixels = h * w

        predictor = get_predictor()
        predictor.set_image(image)

        masks_np, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
        )

        # Build results
        result_masks = []
        best_idx = 0
        best_score = -1.0

        for i in range(masks_np.shape[0]):
            m = masks_np[i]
            score = float(scores[i])

            area = mask_area(m)
            bbox = mask_to_bbox(m)
            center = mask_center(m)
            mean_rgb, dominant_color = analyze_color(image, m)

            result_masks.append({
                "mask_id": i,
                "area": area,
                "bbox": bbox,
                "center": center,
                "iou_prediction": round(score, 4),
                "area_ratio": round(area / total_pixels, 4),
                "mean_color_rgb": mean_rgb,
                "dominant_color": dominant_color,
            })

            if score > best_score:
                best_score = score
                best_idx = i

        # Determine prompt type
        prompt_type = "point+box" if (point_coords is not None and box is not None) else (
            "point" if point_coords is not None else "box"
        )

        cleanup_gpu()

        return {
            "success": True,
            "masks": result_masks,
            "best_mask_index": best_idx,
            "prompt_type": prompt_type,
            "image_size": [w, h],
        }

    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        cleanup_gpu()
        return {"success": False, "error": f"Segmentation failed: {str(e)}", "traceback": traceback.format_exc()}


# =========================================================================
# Tool Registration
# =========================================================================
# =========================================================================
# Tool Call Router
# =========================================================================
TOOL_HANDLERS = {
    "prompt_segment": _handle_prompt_segment,
}


# =========================================================================
# Entry point
# =========================================================================


async def handle_prompt_segment(arguments: dict) -> dict:
    return await _handle_prompt_segment(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="prompt-segment",
        tool_defs_source="_tool_defs/segment.yaml",
        handlers={"prompt_segment": handle_prompt_segment},
        is_file=True,
    )
    asyncio.run(run_server(app))
