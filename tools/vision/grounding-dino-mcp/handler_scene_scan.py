#!/usr/bin/env python3
"""Handler for scene_scan: text-prompt scanning over a bounded vocabulary."""

from gdino_api import call_gdino_api


async def handle_scene_scan(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    box_threshold = arguments.get("box_threshold", 0.25)
    max_objects = arguments.get("max_objects", 50)

    if not image_path:
        return {"success": False, "error": "image_path is required"}

    result = call_gdino_api(
        image_source=image_path,
        prompt="<prompt_free>",
        targets=["bbox"],
        box_threshold=box_threshold,
        categories=arguments.get("categories"),
        max_objects=max_objects,
    )

    if not result.get("success"):
        return result

    objects = result["objects"]

    cat_counts: dict = {}
    for obj in objects:
        cat = obj.get("category", "object")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
    scene_parts = [f"{cnt} {cat}" for cat, cnt in sorted_cats]

    return {
        **result,
        "total_objects": len(objects),
        "category_counts": cat_counts,
        "objects": objects,
        "scene_summary": "Detected: " + ", ".join(scene_parts) if scene_parts else "No objects detected in the scanned category vocabulary",
        "image_path": image_path,
    }
