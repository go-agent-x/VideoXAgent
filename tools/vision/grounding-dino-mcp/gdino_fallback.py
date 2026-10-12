#!/usr/bin/env python3
"""Bounded text-prompt category scans for the local Grounding DINO backend."""

import math
from typing import Any, Callable, Dict, List, Optional

# Common object categories for fallback sweep
_SWEEP_CATEGORIES = [
    "person", "car", "chair", "table", "dog", "cat", "bottle", "cup",
    "book", "phone", "laptop", "bag", "bicycle", "motorcycle", "bus",
    "truck", "bird", "horse", "cow", "sheep", "plant", "flower", "tree",
    "sign", "light", "clock", "tv", "monitor", "keyboard", "mouse",
    "door", "window", "bed", "couch", "refrigerator", "oven", "sink",
    "toilet", "umbrella", "hat", "shoe", "ball", "food", "fruit",
    "vegetable", "pizza", "cake", "sandwich",
]

_MAX_CATEGORIES = 64
_BATCH_SIZE = 8
_MAX_OBJECTS = 50
_VOCABULARY_NOTE = (
    "Detections are limited to the requested category vocabulary; "
    "an empty result does not establish that the image contains no objects."
)


def prompt_free_results_are_degraded(objects: List[Dict[str, Any]]) -> bool:
    """
    Check whether prompt-free detection results are degraded.

    Degraded means: too few objects, all generic "object" categories,
    or very low confidence scores — indicating the prompt-free mode
    didn't work well for this image.
    """
    if not objects:
        return True

    if len(objects) <= 1:
        return True

    categories = [obj.get("category", "object") for obj in objects]
    generic_count = sum(1 for c in categories if c.lower() in ("object", "unknown", ""))
    if generic_count / len(categories) > 0.5:
        return True

    avg_score = sum(obj.get("score", 0) for obj in objects) / len(objects)
    if avg_score < 0.15:
        return True

    return False


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_category_sweep(
    call_gdino_api: Callable,
    image_source: str,
    box_threshold: float = 0.25,
    max_objects: int = 50,
    iou_merge_threshold: float = 0.5,
    *,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Scan a bounded vocabulary in batches, then merge boxes by score and IoU.

    Local Grounding DINO requires text prompts. Each category is a complete
    phrase (for example, ``traffic light``), not a whitespace-delimited token.
    Backend failures are reported separately from successful empty detections.
    """
    metadata: Dict[str, Any] = {
        "detection_mode": "category_sweep",
        "partial": False,
        "vocabulary_limited": True,
        "vocabulary_note": _VOCABULARY_NOTE,
        "categories_used": [],
        "batches_attempted": 0,
        "batches_succeeded": 0,
        "batches_failed": 0,
        "warnings": [],
        "errors": [],
    }
    try:
        for name, value in (("box_threshold", box_threshold), ("iou_merge_threshold", iou_merge_threshold)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        if isinstance(max_objects, bool) or not isinstance(max_objects, int) or not 1 <= max_objects <= _MAX_OBJECTS:
            raise ValueError(f"max_objects must be an integer between 1 and {_MAX_OBJECTS}")
        requested = _SWEEP_CATEGORIES if categories is None else categories
        if not isinstance(requested, (list, tuple)) or not 1 <= len(requested) <= _MAX_CATEGORIES:
            raise ValueError(f"categories must contain between 1 and {_MAX_CATEGORIES} phrases")
        vocabulary = []
        seen = set()
        for category in requested:
            if not isinstance(category, str):
                raise ValueError("Each category must be a non-empty string")
            phrase = " ".join(category.split())
            if not phrase or len(phrase) > 80 or len(phrase.split()) > 8 or phrase == "<prompt_free>":
                raise ValueError("Each category must be a non-empty phrase of at most 80 characters and 8 words")
            if phrase.casefold() not in seen:
                vocabulary.append(phrase)
                seen.add(phrase.casefold())
    except ValueError as exc:
        return {**metadata, "success": False, "object_count": 0, "objects": [], "error": str(exc)}

    all_objects: List[Dict[str, Any]] = []

    # At most eight calls, without skipping later vocabulary after early hits.
    for i in range(0, len(vocabulary), _BATCH_SIZE):
        batch = vocabulary[i : i + _BATCH_SIZE]
        prompt = " . ".join(batch)
        metadata["categories_used"].extend(batch)
        metadata["batches_attempted"] += 1
        try:
            result = call_gdino_api(
                image_source=image_source,
                prompt=prompt,
                targets=["bbox"],
                box_threshold=box_threshold,
            )
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        if result.get("success"):
            metadata["batches_succeeded"] += 1
            all_objects.extend(result.get("objects", []))
        else:
            metadata["batches_failed"] += 1
            metadata["errors"].append(
                f"Batch {metadata['batches_attempted']}: {result.get('error') or 'Detection failed'}"
            )

    # Deduplicate by IoU (NMS-style, keep highest score)
    all_objects.sort(key=lambda o: o.get("score", 0), reverse=True)
    keep: List[Dict[str, Any]] = []
    for obj in all_objects:
        bbox = obj.get("bbox", [])
        if len(bbox) < 4:
            continue
        is_dup = False
        for kept in keep:
            if _iou(bbox, kept.get("bbox", [])) > iou_merge_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(obj)
        if len(keep) >= max_objects:
            break

    succeeded = metadata["batches_succeeded"] > 0
    metadata["partial"] = succeeded and metadata["batches_failed"] > 0
    if metadata["partial"]:
        metadata["warnings"] = list(metadata["errors"])
    result = {**metadata, "success": succeeded, "object_count": len(keep), "objects": keep}
    if not succeeded:
        result["error"] = "All category-sweep batches failed: " + "; ".join(metadata["errors"])
    return result


def relative_size_label_from_areas(area: float, all_areas: List[float]) -> str:
    """
    Compute a relative size label for an object given all object areas.

    Returns one of: "tiny", "small", "medium", "large", "dominant", "unknown-size".
    """
    valid_areas = [a for a in all_areas if a > 0]
    if not valid_areas or area <= 0:
        return "unknown-size"

    max_area = max(valid_areas)
    if max_area <= 0:
        return "unknown-size"

    ratio = area / max_area

    if ratio < 0.05:
        return "tiny"
    elif ratio < 0.2:
        return "small"
    elif ratio < 0.5:
        return "medium"
    elif ratio < 0.8:
        return "large"
    else:
        return "dominant"
