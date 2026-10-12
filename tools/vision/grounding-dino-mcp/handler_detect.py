#!/usr/bin/env python3
"""Handler for detect tool — Text-prompted object detection (existence check)."""

from gdino_api import call_gdino_api, normalize_categories
async def handle_detect(arguments: dict) -> dict:
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
        targets=["bbox"],
        box_threshold=box_threshold,
    )

    if not result.get("success"):
        return result

    requested_cats = [c.strip().lower() for c in categories.replace(".", ",").replace(" ", ",").split(",") if c.strip()]
    detected_map = {cat: False for cat in requested_cats}
    cat_objects = {cat: [] for cat in requested_cats}

    for obj in result["objects"]:
        obj_cat = obj.get("category", "").lower()
        for cat in requested_cats:
            if cat in obj_cat:
                detected_map[cat] = True
                cat_objects[cat].append(obj)

    found = [c for c, v in detected_map.items() if v]
    not_found = [c for c, v in detected_map.items() if not v]

    summary_parts = []
    if found:
        found_strs = [f"{len(cat_objects[c])} {c}" for c in found]
        summary_parts.append("Found: " + ", ".join(found_strs))
    if not_found:
        summary_parts.append("Not found: " + ", ".join(not_found))

    return {
        "success": True,
        "detected_categories": detected_map,
        "object_count": len(result["objects"]),
        "objects": result["objects"],
        "summary": ". ".join(summary_parts),
        "image_path": image_path,
    }
