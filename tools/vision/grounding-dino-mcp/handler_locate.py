#!/usr/bin/env python3
"""Handler for locate tool — Natural language object locating."""

from gdino_api import call_gdino_api
async def handle_locate(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    phrase = arguments.get("phrase", "")
    box_threshold = arguments.get("box_threshold", 0.2)

    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not phrase:
        return {"success": False, "error": "phrase is required"}

    result = call_gdino_api(
        image_source=image_path,
        prompt=phrase,
        targets=["bbox"],
        box_threshold=box_threshold,
    )

    if not result.get("success"):
        return result

    objects = result["objects"]
    matched = len(objects) > 0
    best = max(objects, key=lambda o: o.get("score", 0)) if objects else None

    return {
        "success": True,
        "phrase": phrase,
        "matched": matched,
        "best_match": best,
        "all_matches": objects,
        "match_count": len(objects),
        "image_path": image_path,
    }
