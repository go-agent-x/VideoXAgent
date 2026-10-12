#!/usr/bin/env python3
"""Handler for spatial_relation tool — Spatial relation analysis between two objects."""

from gdino_api import call_gdino_api, compute_spatial_relation, bbox_center
async def handle_spatial_relation(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    object_a = arguments.get("object_a", "")
    object_b = arguments.get("object_b", "")
    box_threshold = arguments.get("box_threshold", 0.25)

    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not object_a or not object_b:
        return {"success": False, "error": "both object_a and object_b are required"}

    prompt = f"{object_a} . {object_b}"

    result = call_gdino_api(
        image_source=image_path,
        prompt=prompt,
        targets=["bbox"],
        box_threshold=box_threshold,
    )

    if not result.get("success"):
        return result

    a_lower = object_a.lower()
    b_lower = object_b.lower()

    best_a = None
    best_b = None
    for obj in result["objects"]:
        cat = obj.get("category", "").lower()
        score = obj.get("score", 0)
        if a_lower in cat:
            if best_a is None or score > best_a["score"]:
                best_a = obj
        if b_lower in cat:
            if best_b is None or score > best_b["score"]:
                best_b = obj

    if not best_a:
        return {"success": False, "error": f"'{object_a}' not detected in image", "image_path": image_path}
    if not best_b:
        return {"success": False, "error": f"'{object_b}' not detected in image", "image_path": image_path}

    bbox_a = best_a.get("bbox", [])
    bbox_b = best_b.get("bbox", [])
    spatial = compute_spatial_relation(bbox_a, bbox_b)

    ca = bbox_center(bbox_a)
    cb = bbox_center(bbox_b)

    return {
        "success": True,
        "object_a": {"category": best_a["category"], "bbox": bbox_a, "center": [round(ca[0], 1), round(ca[1], 1)], "score": best_a["score"]},
        "object_b": {"category": best_b["category"], "bbox": bbox_b, "center": [round(cb[0], 1), round(cb[1], 1)], "score": best_b["score"]},
        **spatial,
        "image_path": image_path,
    }
