#!/usr/bin/env python3
"""Handler for count tool — Precise object counting with NMS deduplication."""

from gdino_api import call_gdino_api, nms_filter
async def handle_count(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    target_category = arguments.get("target_category", "")
    box_threshold = arguments.get("box_threshold", 0.25)
    nms_threshold = arguments.get("nms_threshold", 0.5)

    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not target_category:
        return {"success": False, "error": "target_category is required"}

    result = call_gdino_api(
        image_source=image_path,
        prompt=target_category,
        targets=["bbox"],
        box_threshold=box_threshold,
    )

    if not result.get("success"):
        return result

    # Filter by target category (case-insensitive partial match)
    target_lower = target_category.lower()
    matched = [
        o for o in result["objects"]
        if target_lower in o.get("category", "").lower()
    ]

    # Apply NMS dedup
    deduped = nms_filter(matched, iou_threshold=nms_threshold)

    scores = [o["score"] for o in deduped]
    avg_conf = round(sum(scores) / len(scores), 3) if scores else 0.0

    return {
        "success": True,
        "count": len(deduped),
        "objects": deduped,
        "confidence_summary": f"{len(deduped)} detections, avg confidence {avg_conf}",
    }
