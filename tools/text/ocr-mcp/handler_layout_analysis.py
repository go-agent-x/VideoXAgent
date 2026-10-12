#!/usr/bin/env python3
"""Handler for ocr_layout_analysis tool — Analyze page/frame layout structure
using PPStructure (PaddleOCR's layout analysis pipeline).

For table regions, crops the detected area and applies OCR with row/column
grouping to extract structured table content (workaround for PaddlePaddle 2.6
incompatibility with SLANet v2 table model)."""

import os
import threading

from _base.device import ocr_device_options
from collections import defaultdict

from ocr_common import position_label, _suppress_stdout


# ---------------------------------------------------------------------------
# PPStructure engine — lazy singleton (keyed by language)
# Uses table=False to avoid PaddlePaddle Tensor dimension bug in table module.
# Table content is extracted separately via OCR on cropped regions.
# ---------------------------------------------------------------------------
_ppstructure_instances = {}
_ocr_instances = {}
_engine_lock = threading.Lock()


def _init_ppstructure(lang: str = "ch"):
    """Get or create a PPStructure instance for layout analysis."""
    if lang not in _ppstructure_instances:
        with _engine_lock:
            if lang not in _ppstructure_instances:
                from paddleocr import PPStructure
                with _suppress_stdout():
                    _ppstructure_instances[lang] = PPStructure(
                        table=False,
                        ocr=True,
                        show_log=False,
                        **ocr_device_options(),
                        lang=lang,
                    )
    return _ppstructure_instances[lang]


def _init_ocr(lang: str = "ch"):
    """Get or create a PaddleOCR instance for table region OCR."""
    if lang not in _ocr_instances:
        with _engine_lock:
            if lang not in _ocr_instances:
                from paddleocr import PaddleOCR
                with _suppress_stdout():
                    _ocr_instances[lang] = PaddleOCR(
                        use_angle_cls=True,
                        lang=lang,
                        show_log=False,
                        **ocr_device_options(),
                    )
    return _ocr_instances[lang]


def _ocr_table_region(img, bbox, lang: str = "ch"):
    """Extract structured table content by OCR on cropped table region.

    Returns a dict with 'rows' (list of lists) and 'text' (flat string).
    """
    x1, y1, x2, y2 = [int(x) for x in bbox]
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return {"rows": [], "text": ""}

    ocr_engine = _init_ocr(lang)
    with _suppress_stdout():
        ocr_results = ocr_engine.ocr(crop, cls=True)

    if not ocr_results or not ocr_results[0]:
        return {"rows": [], "text": ""}

    # Collect all text boxes with positions
    items = []
    for line in ocr_results[0]:
        text = line[1][0]
        pts = line[0]
        y_center = (pts[0][1] + pts[2][1]) / 2
        x_center = (pts[0][0] + pts[2][0]) / 2
        items.append((y_center, x_center, text))

    items.sort(key=lambda it: (it[0], it[1]))

    # Group into rows (y gap > 20px = new row)
    rows = []
    current_row = []
    last_y = -999
    for y, x, text in items:
        if y - last_y > 20:
            if current_row:
                rows.append(current_row)
            current_row = [(x, text)]
        else:
            current_row.append((x, text))
        last_y = y
    if current_row:
        rows.append(current_row)

    # Sort cells within each row by x position
    table_rows = []
    for row in rows:
        row.sort(key=lambda r: r[0])
        table_rows.append([t for _, t in row])

    flat_text = "\n".join(" | ".join(cells) for cells in table_rows)
    return {"rows": table_rows, "text": flat_text}


async def handle_ocr_layout_analysis(arguments: dict) -> dict:
    """Analyze the layout structure of an image using PPStructure layout detection."""
    image_path = arguments.get("image_path", "")
    lang = arguments.get("lang", "ch")

    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not os.path.isfile(image_path):
        return {"success": False, "error": f"Image file not found: {image_path}"}

    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": f"Failed to read image: {image_path}"}
        img_h, img_w = img.shape[:2]

        engine = _init_ppstructure(lang)
        with _suppress_stdout():
            results = engine(img)
    except Exception as e:
        return {"success": False, "error": f"Layout analysis error: {str(e)}"}

    if not results:
        return {
            "success": True,
            "region_count": 0,
            "regions": [],
            "layout_summary": "",
            "summary": "No layout regions detected",
        }

    regions = []
    type_counts = defaultdict(int)

    for item in results:
        region_type = item.get("type", "unknown")
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            bbox = [round(float(x), 1) for x in bbox]
        position = position_label(bbox, img_w, img_h) if len(bbox) == 4 else ""

        # Extract text content
        text_content = ""
        table_data = None

        if region_type == "table" and len(bbox) == 4:
            # For table regions: crop and OCR to get structured content
            table_result = _ocr_table_region(img, bbox, lang)
            text_content = table_result["text"]
            if table_result["rows"]:
                table_data = table_result["rows"]
        else:
            # For non-table regions: extract from PPStructure OCR results
            res = item.get("res", [])
            if isinstance(res, list):
                text_lines = []
                for line in res:
                    if isinstance(line, dict) and "text" in line:
                        text_lines.append(line["text"])
                    elif isinstance(line, (list, tuple)) and len(line) >= 2:
                        if isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
                            text_lines.append(str(line[1][0]))
                text_content = " ".join(text_lines)

        type_counts[region_type] += 1
        region_info = {
            "type": region_type,
            "bbox": bbox,
            "text": text_content[:500],
            "position": position,
        }
        if table_data is not None:
            region_info["table_rows"] = table_data
        regions.append(region_info)

    # Sort regions by reading order (top-to-bottom, left-to-right)
    regions.sort(key=lambda r: (r["bbox"][1] if r["bbox"] else 0,
                                r["bbox"][0] if r["bbox"] else 0))

    layout_parts = [f"{t}({c})" for t, c in sorted(type_counts.items())]
    layout_summary = ", ".join(layout_parts)

    summary_parts = [f"{c} {t}" for t, c in sorted(type_counts.items())]
    summary = (f"Detected {len(regions)} layout regions: {', '.join(summary_parts)}"
               if regions else "No layout regions detected")

    return {
        "success": True,
        "region_count": len(regions),
        "regions": regions,
        "layout_summary": layout_summary,
        "summary": summary,
    }
