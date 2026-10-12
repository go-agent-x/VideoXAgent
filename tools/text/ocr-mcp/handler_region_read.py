#!/usr/bin/env python3
"""Handler for ocr_region_read tool — Recognize text within a specified bounding box region."""

import os

from ocr_common import run_ocr_on_array


def _crop_image(image_path, bbox, padding=10):
    """
    Crop a region from an image with optional padding.

    Returns:
        (cropped_array, actual_bbox, error_message)
    """
    import cv2

    if not os.path.isfile(image_path):
        return None, [], f"Image file not found: {image_path}"

    img = cv2.imread(image_path)
    if img is None:
        return None, [], f"Failed to read image: {image_path}"

    h, w = img.shape[:2]
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)

    if x2 <= x1 or y2 <= y1:
        return None, [], f"Invalid bbox after clamping: [{x1},{y1},{x2},{y2}]"

    cropped = img[y1:y2, x1:x2]
    return cropped, [x1, y1, x2, y2], ""


async def handle_ocr_region_read(arguments: dict) -> dict:
    """Recognize text within a specified bounding box region of an image."""
    image_path = arguments.get("image_path", "")
    bbox = arguments.get("bbox")
    lang = arguments.get("lang", "ch")
    padding = arguments.get("padding", 10)

    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return {"success": False, "error": "bbox must be [x1, y1, x2, y2]"}

    cropped_img, actual_bbox, crop_error = _crop_image(image_path, bbox, padding=padding)
    if cropped_img is None:
        return {"success": False, "error": crop_error}

    success, texts, error = run_ocr_on_array(cropped_img, lang=lang, det_thresh=0.3)

    if not success:
        return {"success": False, "error": error}

    # Map bbox coordinates back to original image space
    offset_x = actual_bbox[0]
    offset_y = actual_bbox[1]
    for t in texts:
        t["bbox"] = [
            round(t["bbox"][0] + offset_x, 1),
            round(t["bbox"][1] + offset_y, 1),
            round(t["bbox"][2] + offset_x, 1),
            round(t["bbox"][3] + offset_y, 1),
        ]

    sorted_texts = sorted(texts, key=lambda t: (t["bbox"][1], t["bbox"][0]))
    full_text = " ".join(t["text"] for t in sorted_texts)

    return {
        "success": True,
        "image_path": image_path,
        "crop_bbox": actual_bbox,
        "text_count": len(texts),
        "texts": sorted_texts,
        "full_text": full_text,
        "summary": f"Read {len(texts)} text regions from specified bbox",
    }
