#!/usr/bin/env python3
"""Spatial-Crop MCP Server — Crop specified spatial regions from images or video frames."""
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

from _base.paths import PROJECT_ROOT
sys.path.insert(0, str(Path(__file__).parent))  # to import from original spatial_crop_server.py

from _base.server_framework import create_mcp_server, run_server

# ── Business logic (from spatial_crop_server.py) ──

import json
import os

import cv2


OUTPUT_DIR = PROJECT_ROOT / "output"


def _crop_image(img, bbox, padding=0):
    """
    Crop a region from a numpy image array.
    bbox: [x1, y1, x2, y2] in pixels.
    Returns: (cropped_array, actual_bbox, error_message)
    """
    h, w = img.shape[:2]

    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)

    if x2 <= x1 or y2 <= y1:
        return None, [], f"Invalid bbox after clamping: [{x1},{y1},{x2},{y2}] (image size {w}x{h})"

    cropped = img[y1:y2, x1:x2]
    return cropped, [x1, y1, x2, y2], ""


def crop_frame(image_path: str, bbox: list, padding: int = 0, output_name: str = "") -> dict:
    """Crop a bbox region from a single image file and save the result."""
    if not os.path.isfile(image_path):
        return {"success": False, "error": f"Image not found: {image_path}"}

    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": f"Failed to read image: {image_path}"}

    cropped, actual_bbox, err = _crop_image(img, bbox, padding)
    if err:
        return {"success": False, "error": err}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not output_name:
        stem = Path(image_path).stem
        output_name = f"{stem}_crop_{actual_bbox[0]}_{actual_bbox[1]}_{actual_bbox[2]}_{actual_bbox[3]}.jpg"
    if not any(output_name.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp"]):
        output_name += ".jpg"

    output_path = str((OUTPUT_DIR / output_name).resolve())
    cv2.imwrite(output_path, cropped)

    h, w = cropped.shape[:2]
    return {
        "success": True,
        "output_path": output_path,
        "crop_width": w,
        "crop_height": h,
        "actual_bbox": actual_bbox,
        "source_image": image_path,
    }


def crop_video_frames(
    video_path: str,
    times: list,
    bbox: list = None,
    bboxes: list = None,
    padding: int = 0,
    output_prefix: str = "",
) -> dict:
    """
    Extract frames from a video at specified timestamps and crop a region from each.
    - bbox: static [x1,y1,x2,y2] applied to all frames.
    - bboxes: per-frame list of [x1,y1,x2,y2], must match len(times).
    Exactly one of bbox or bboxes must be provided.
    """
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"Video not found: {video_path}"}

    if bbox is None and bboxes is None:
        return {"success": False, "error": "Must provide either 'bbox' (static) or 'bboxes' (per-frame)."}
    if bbox is not None and bboxes is not None:
        return {"success": False, "error": "Provide only one of 'bbox' or 'bboxes', not both."}
    if bboxes is not None and len(bboxes) != len(times):
        return {
            "success": False,
            "error": f"Length mismatch: {len(times)} times but {len(bboxes)} bboxes.",
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": f"Failed to open video: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not output_prefix:
        output_prefix = Path(video_path).stem

    results = []
    for idx, t in enumerate(times):
        t = float(t)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            results.append({
                "timestamp": t,
                "success": False,
                "error": f"Failed to read frame at {t}s (duration {duration:.1f}s)",
            })
            continue

        cur_bbox = bboxes[idx] if bboxes is not None else bbox
        cropped, actual_bbox, err = _crop_image(frame, cur_bbox, padding)
        if err:
            results.append({"timestamp": t, "success": False, "error": err})
            continue

        fname = f"{output_prefix}_crop_{t}s.jpg"
        output_path = str((OUTPUT_DIR / fname).resolve())
        cv2.imwrite(output_path, cropped)

        ch, cw = cropped.shape[:2]
        results.append({
            "timestamp": t,
            "success": True,
            "output_path": output_path,
            "crop_width": cw,
            "crop_height": ch,
            "actual_bbox": actual_bbox,
        })

    cap.release()

    successful = sum(1 for r in results if r.get("success"))
    return {
        "success": successful > 0,
        "total_requested": len(times),
        "total_cropped": successful,
        "video_path": video_path,
        "results": results,
    }


async def handle_spatial_crop(arguments: dict) -> dict:
    """Unified spatial crop handler that dispatches to crop_frame or crop_video_frames."""
    # If video_path is provided, use crop_video_frames; otherwise crop_frame
    if "video_path" in arguments and arguments["video_path"]:
        return crop_video_frames(
            video_path=arguments["video_path"],
            times=arguments.get("times", []),
            bbox=arguments.get("bbox"),
            bboxes=arguments.get("bboxes"),
            padding=arguments.get("padding", 0),
            output_prefix=arguments.get("output_prefix", ""),
        )
    else:
        return crop_frame(
            image_path=arguments.get("image_path", ""),
            bbox=arguments.get("bbox", []),
            padding=arguments.get("padding", 0),
            output_name=arguments.get("output_name", ""),
        )


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="spatial-crop",
        tool_defs_source="_tool_defs/spatial.yaml",
        handlers={"spatial_crop": handle_spatial_crop},
        is_file=True,
    )
    asyncio.run(run_server(app))
