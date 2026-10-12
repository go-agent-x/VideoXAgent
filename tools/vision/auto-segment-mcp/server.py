#!/usr/bin/env python3
"""Auto-Segment MCP Server — Automatic full-image segmentation."""
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
sys.path.insert(0, str(Path(__file__).parent))  # to import from original auto_segment_server.py

from _base.server_framework import create_mcp_server, run_server

# ── Business logic (from auto_segment_server.py) ──

import json
import os
import traceback

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from auto_segment_utils import (
    get_mask_generator,
    read_image,
    bbox_from_sam,
    center_from_bbox,
    cleanup_gpu,
)

# ---------------------------------------------------------------------------
# MCP Server init
# ---------------------------------------------------------------------------


# =========================================================================
# Tool: auto_segment — Automatic full-image segmentation
# =========================================================================
async def _handle_auto_segment(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    points_per_side = int(arguments.get("points_per_side", 32))
    min_mask_area = int(arguments.get("min_mask_area", 100))
    pred_iou_thresh = float(arguments.get("pred_iou_thresh", 0.86))

    if not image_path:
        return {"success": False, "error": "image_path is required"}

    try:
        image = read_image(image_path)
        h, w = image.shape[:2]
        total_pixels = h * w

        generator = get_mask_generator(
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            min_mask_region_area=min_mask_area,
        )

        masks = generator.generate(image)

        # Filter by minimum area
        masks = [m for m in masks if m["area"] >= min_mask_area]

        # Sort by area descending
        masks.sort(key=lambda m: m["area"], reverse=True)

        result_masks = []
        for i, m in enumerate(masks):
            bbox = bbox_from_sam(m)
            center = center_from_bbox(bbox)
            result_masks.append({
                "mask_id": i + 1,
                "area": int(m["area"]),
                "bbox": bbox,
                "center": center,
                "predicted_iou": round(float(m["predicted_iou"]), 4),
                "stability_score": round(float(m["stability_score"]), 4),
                "area_ratio": round(m["area"] / total_pixels, 4),
            })

        # Summary
        if result_masks:
            largest = result_masks[0]["area"]
            smallest = result_masks[-1]["area"]
            summary = (
                f"Found {len(result_masks)} distinct regions. "
                f"Largest: {largest}px ({round(largest / total_pixels * 100, 1)}% of image). "
                f"Smallest: {smallest}px."
            )
        else:
            summary = "No regions detected with current settings."

        cleanup_gpu()

        return {
            "success": True,
            "total_masks": len(result_masks),
            "masks": result_masks,
            "summary": summary,
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
    "auto_segment": _handle_auto_segment,
}


# =========================================================================
# Entry point
# =========================================================================


async def handle_auto_segment(arguments: dict) -> dict:
    return await _handle_auto_segment(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="auto-segment",
        tool_defs_source="_tool_defs/segment.yaml",
        handlers={"auto_segment": handle_auto_segment},
        is_file=True,
    )
    asyncio.run(run_server(app))
