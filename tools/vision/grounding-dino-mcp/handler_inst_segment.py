#!/usr/bin/env python3
"""Handler for instance_segment tool — Instance segmentation with area analysis."""

from gdino_api import call_gdino_api, normalize_categories, compute_bbox_area, relative_size_label
async def handle_instance_segment(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    categories = arguments.get("categories", "")
    box_threshold = arguments.get("box_threshold", 0.25)

    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not categories:
        return {"success": False, "error": "categories is required"}

    prompt = normalize_categories(categories)

    result = call_gdino_api(
        image_source=image_path,
        prompt=prompt,
        targets=["bbox", "mask"],
        box_threshold=box_threshold,
    )

    if not result.get("success"):
        return result

    all_areas = [compute_bbox_area(o.get("bbox", [])) for o in result["objects"]]

    enriched = []
    for obj in result["objects"]:
        bbox = obj.get("bbox", [])
        area = compute_bbox_area(bbox)
        width = max(0, bbox[2] - bbox[0]) if len(bbox) >= 4 else 0
        height = max(0, bbox[3] - bbox[1]) if len(bbox) >= 4 else 0

        entry = {
            "category": obj.get("category", "object"),
            "score": obj.get("score", 0),
            "bbox": bbox,
            "bbox_area": round(area, 1),
            "bbox_width": round(width, 1),
            "bbox_height": round(height, 1),
            "relative_size": relative_size_label(area, all_areas),
        }
        entry["has_mask"] = "mask" in obj
        enriched.append(entry)

    ranking = sorted(enriched, key=lambda x: x["bbox_area"], reverse=True)
    size_ranking = [f"{e['category']}({int(e['bbox_area'])}px)" for e in ranking]

    return {
        "success": True,
        "objects": enriched,
        "size_ranking": size_ranking,
        "image_path": image_path,
    }
