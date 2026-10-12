#!/usr/bin/env python3
"""Spatial-Analyze MCP Server — Multi-object spatial relation analysis."""
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

from _base.paths import service_dir
sys.path.insert(0, str(Path(__file__).parent))

from _base.server_framework import create_mcp_server, run_server

import json
import os
import traceback

sys.path.insert(0, str(service_dir("grounding-dino")))

from spatial_analyze_utils import (
    bbox_center,
    bbox_area,
    compute_spatial_relation,
)


def _detect_objects_gdino(image_path: str, categories: str, box_threshold: float) -> list:
    """Call Grounding DINO to detect objects of given categories."""
    try:
        from gdino_api import call_gdino_api, normalize_categories, nms_filter
    except ImportError:
        gdino_dir = str(service_dir("grounding-dino"))
        sys.path.insert(0, gdino_dir)
        from gdino_api import call_gdino_api, normalize_categories, nms_filter

    cats = normalize_categories(categories)
    result = call_gdino_api(
        image_source=image_path,
        prompt=cats,
        targets=["bbox"],
        box_threshold=box_threshold,
    )
    if not result or "objects" not in result:
        return []
    objects = result["objects"]
    objects = nms_filter(objects, iou_threshold=0.5)
    return objects


# =========================================================================
# Tool: spatial_analyze — Multi-object spatial relations via DINO-X detection
# =========================================================================
async def _handle_spatial_analyze(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    categories = arguments.get("categories", "")
    box_threshold = float(arguments.get("box_threshold", 0.25))

    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not categories:
        return {"success": False, "error": "categories is required"}
    if not os.path.isfile(image_path):
        return {"success": False, "error": f"Image not found: {image_path}"}

    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": f"Cannot read image: {image_path}"}
        img_h, img_w = img.shape[:2]

        # Detect objects with DINO-X
        objects = _detect_objects_gdino(image_path, categories, box_threshold)
        if not objects:
            return {
                "success": True,
                "objects": [],
                "relations": [],
                "image_size": [img_w, img_h],
                "summary": "No objects detected",
            }

        # Build per-object info
        objects_result = []
        for obj in objects:
            bbox = obj.get("bbox", [0, 0, 0, 0])
            cat = obj.get("category", "unknown")
            score = obj.get("score", 0.0)
            center = bbox_center(bbox)
            area = bbox_area(bbox)

            # Position label
            cx_ratio = center[0] / img_w if img_w > 0 else 0.5
            cy_ratio = center[1] / img_h if img_h > 0 else 0.5
            h_label = "left" if cx_ratio < 0.33 else ("right" if cx_ratio > 0.67 else "center")
            v_label = "top" if cy_ratio < 0.33 else ("bottom" if cy_ratio > 0.67 else "middle")
            position = f"{v_label}-{h_label}"

            objects_result.append({
                "category": cat,
                "bbox": [round(b, 1) for b in bbox],
                "center": center,
                "area": area,
                "area_ratio": round(area / (img_h * img_w), 4) if img_h * img_w > 0 else 0,
                "score": round(score, 3),
                "position": position,
            })

        # Compute pairwise spatial relations
        relations = []
        for i in range(len(objects_result)):
            for j in range(i + 1, len(objects_result)):
                label_a = f"{objects_result[i]['category']}_{i}"
                label_b = f"{objects_result[j]['category']}_{j}"
                rel = compute_spatial_relation(
                    objects_result[i]["bbox"],
                    objects_result[j]["bbox"],
                    label_a,
                    label_b,
                )
                relations.append(rel)

        # Build category counts
        cat_counts = {}
        for obj in objects_result:
            c = obj["category"]
            cat_counts[c] = cat_counts.get(c, 0) + 1

        summary = f"Detected {len(objects_result)} objects ({', '.join(f'{v} {k}' for k, v in cat_counts.items())}), {len(relations)} pairwise relations computed"

        return {
            "success": True,
            "objects": objects_result,
            "relations": relations,
            "category_counts": cat_counts,
            "image_size": [img_w, img_h],
            "summary": summary,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# Entry point
# =========================================================================
async def handle_spatial_analyze(arguments: dict) -> dict:
    return await _handle_spatial_analyze(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="spatial-analyze",
        tool_defs_source="_tool_defs/spatial.yaml",
        handlers={"spatial_analyze": handle_spatial_analyze},
        is_file=True,
    )
    asyncio.run(run_server(app))
