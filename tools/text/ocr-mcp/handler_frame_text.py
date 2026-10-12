#!/usr/bin/env python3
"""Handler for ocr_frame_text tool — Detect and recognize ALL text in a single image."""

from ocr_common import run_ocr


async def handle_ocr_frame_text(arguments: dict) -> dict:
    """Detect and recognize all text in an image."""
    image_path = arguments.get("image_path", "")
    lang = arguments.get("lang", "ch")
    det_thresh = arguments.get("det_thresh", 0.3)

    if not image_path:
        return {"success": False, "error": "image_path is required"}

    success, texts, error, _, _ = run_ocr(image_path, lang=lang, det_thresh=det_thresh)

    if not success:
        return {"success": False, "error": error}

    sorted_texts = sorted(texts, key=lambda t: (t["bbox"][1], t["bbox"][0]))
    full_text = "\n".join(t["text"] for t in sorted_texts)

    return {
        "success": True,
        "texts": sorted_texts,
        "full_text": full_text,
    }
