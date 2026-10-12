#!/usr/bin/env python3
"""Concat Images MCP Server — Concatenate multiple images into a single composite image."""
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
sys.path.insert(0, str(Path(__file__).parent))

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response

# ── Business logic ──

import math
import os

import cv2
import numpy as np

OUTPUT_DIR = PROJECT_ROOT / "output"


def _format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS string."""
    if seconds < 0:
        return str(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def _add_timestamp_overlay(img: np.ndarray, timestamp: float) -> np.ndarray:
    """Overlay timestamp text on the bottom-left of the image."""
    text = _format_timestamp(timestamp)
    h, w = img.shape[:2]

    # Scale font size based on image width
    font_scale = max(0.4, min(1.2, w / 800))
    thickness = max(1, int(font_scale * 1.5))

    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    # Semi-transparent background bar
    bar_height = text_h + baseline + 6
    bar = img[h - bar_height : h, 0 : text_w + 10].copy()
    overlay = bar.copy()
    overlay[:] = (0, 0, 0)  # black
    cv2.addWeighted(overlay, 0.55, bar, 0.45, 0, bar)
    img[h - bar_height : h, 0 : text_w + 10] = bar

    # White text
    cv2.putText(
        img,
        text,
        (4, h - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return img


def _calc_grid_cols(n: int, cols_hint: int) -> int:
    """Calculate optimal column count for grid layout."""
    if cols_hint > 0:
        return min(cols_hint, n)
    # Auto: aim for roughly square grid
    return max(1, int(math.ceil(math.sqrt(n))))


def concat_images(
    image_paths: list[str],
    timestamps: list[float] | None = None,
    layout: str = "grid",
    cols: int = 0,
    output_name: str = "",
) -> dict:
    """
    Concatenate multiple images into a single composite image.

    Each sub-image retains its original resolution. Timestamps are overlaid
    as text if provided. Layout can be horizontal, vertical, or grid.
    """
    if not image_paths:
        return make_error_response("image_paths must be a non-empty list")

    n = len(image_paths)
    if n > 30:
        return make_error_response(f"Too many images ({n}), maximum is 30")

    # Validate file existence
    for i, p in enumerate(image_paths):
        if not os.path.isfile(p):
            return make_error_response(f"Image not found (index {i}): {p}")

    # Validate timestamps
    if timestamps is not None:
        if len(timestamps) != n:
            return make_error_response(
                f"timestamps length ({len(timestamps)}) must match image_paths length ({n})"
            )

    # Load all images
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            return make_error_response(f"Cannot read image: {p}")
        images.append(img)

    # Add timestamp overlays if provided
    if timestamps is not None:
        images = [_add_timestamp_overlay(img, ts) for img, ts in zip(images, timestamps)]

    # Compute layout
    layout = layout.lower()
    if layout not in ("horizontal", "vertical", "grid"):
        return make_error_response(f"Invalid layout '{layout}', use: horizontal, vertical, grid")

    composite = None

    if layout == "horizontal":
        # Pad heights to max height, preserve original widths
        max_h = max(img.shape[0] for img in images)
        padded = []
        for img in images:
            h, w = img.shape[:2]
            if h < max_h:
                pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
                padded.append(np.vstack([img, pad]))
            else:
                padded.append(img)
        composite = np.hstack(padded)

    elif layout == "vertical":
        # Pad widths to max width, preserve original heights
        max_w = max(img.shape[1] for img in images)
        padded = []
        for img in images:
            h, w = img.shape[:2]
            if w < max_w:
                pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
                padded.append(np.hstack([img, pad]))
            else:
                padded.append(img)
        composite = np.vstack(padded)

    else:  # grid
        grid_cols = _calc_grid_cols(n, cols)
        grid_rows = int(math.ceil(n / grid_cols))

        # Compute column widths and row heights
        # For each grid cell, determine max width per column and max height per row
        col_widths = [0] * grid_cols
        row_heights = [0] * grid_rows

        for idx, img in enumerate(images):
            r = idx // grid_cols
            c = idx % grid_cols
            col_widths[c] = max(col_widths[c], img.shape[1])
            row_heights[r] = max(row_heights[r], img.shape[0])

        # If last row is incomplete, fill remaining columns with width 0 (will be black padding)
        # Actually, for incomplete last row, we still need col_widths for empty cells = 0
        # But total_width should still use full col_widths to keep alignment consistent

        total_w = sum(col_widths)
        total_h = sum(row_heights)

        composite = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        for idx, img in enumerate(images):
            r = idx // grid_cols
            c = idx % grid_cols
            h, w = img.shape[:2]

            # Compute placement offset
            y_offset = sum(row_heights[:r])
            x_offset = sum(col_widths[:c])

            # Place image — center it within the cell
            cell_w = col_widths[c]
            cell_h = row_heights[r]
            x_pad = (cell_w - w) // 2
            y_pad = (cell_h - h) // 2

            composite[y_offset + y_pad : y_offset + y_pad + h,
                      x_offset + x_pad : x_offset + x_pad + w] = img

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not output_name:
        # Auto-generate name: first image stem + timestamps + concat_images
        first_stem = Path(image_paths[0]).stem
        if timestamps is not None:
            ts_parts = "_".join(_format_timestamp(ts).replace(":", "") for ts in timestamps)
            output_name = f"{first_stem}_{ts_parts}_concat_images.jpg"
        else:
            output_name = f"{first_stem}_concat_images.jpg"
    if not any(output_name.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp"]):
        output_name += ".jpg"

    output_path = str((OUTPUT_DIR / output_name).resolve())

    # Use high quality for JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    success = cv2.imwrite(output_path, composite, encode_params)
    if not success:
        # Fallback to PNG for lossless
        output_name_png = output_name.rsplit(".", 1)[0] + ".png"
        output_path = str((OUTPUT_DIR / output_name_png).resolve())
        success = cv2.imwrite(output_path, composite)
        if not success:
            return make_error_response(f"Failed to save composite image to {output_path}")

    ch, cw = composite.shape[:2]

    # Build per-image info with position in composite
    per_image_info = []
    for i, (p, img) in enumerate(zip(image_paths, images)):
        info = {
            "image_path": p,
            "original_width": img.shape[1],
            "original_height": img.shape[0],
        }
        if timestamps is not None:
            info["timestamp"] = timestamps[i]
            info["timestamp_formatted"] = _format_timestamp(timestamps[i])
        per_image_info.append(info)

    return make_success_response({
        "output_path": output_path,
        "image_context": "composite_grid",
        "composite_width": cw,
        "composite_height": ch,
        "num_images": n,
        "layout": layout,
        # "per_image_info": per_image_info,
    })


async def handle_concat_images(arguments: dict) -> dict:
    return concat_images(
        image_paths=arguments.get("image_paths", []),
        timestamps=arguments.get("timestamps"),
        layout=arguments.get("layout", "grid"),
        cols=arguments.get("cols", 0),
        output_name=arguments.get("output_name", ""),
    )


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="concat-images-mcp",
        tool_defs_source="_tool_defs/basic.yaml",
        handlers={"concat_images": handle_concat_images},
        is_file=True,
    )
    asyncio.run(run_server(app))