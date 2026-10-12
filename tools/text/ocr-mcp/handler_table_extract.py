#!/usr/bin/env python3
"""Handler for ocr_table_extract tool — Extract table structures from images."""

import os
import re

from ocr_common import init_ocr, _suppress_stdout


def _ocr_to_table_rows(image_path: str, lang: str = "ch", row_tolerance: float = 15.0):
    """
    Run PaddleOCR 2.x on an image and cluster detected text lines into
    table rows based on their vertical (y) centre position.

    Returns:
        (cells_2d, table_bbox, error)
        cells_2d: list of rows, each row is a list of cell-text strings
                  sorted left-to-right.
        table_bbox: [x1, y1, x2, y2] encompassing all detected text.
    """
    ocr = init_ocr(lang)
    try:
        with _suppress_stdout():
            results = ocr.ocr(image_path, cls=True)
    except Exception as e:
        return [], [], f"OCR engine error: {str(e)}"

    if not results or not results[0]:
        return [], [], ""

    # Collect all text items with their bboxes
    items = []
    for line in results[0]:
        points, (text_str, confidence) = line
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        cy = (bbox[1] + bbox[3]) / 2.0
        items.append({"text": text_str, "bbox": bbox, "cy": cy, "confidence": confidence})

    if not items:
        return [], [], ""

    # Compute overall table bbox
    all_x1 = [it["bbox"][0] for it in items]
    all_y1 = [it["bbox"][1] for it in items]
    all_x2 = [it["bbox"][2] for it in items]
    all_y2 = [it["bbox"][3] for it in items]
    table_bbox = [
        round(min(all_x1), 1), round(min(all_y1), 1),
        round(max(all_x2), 1), round(max(all_y2), 1),
    ]

    # Sort items top-to-bottom, then cluster by vertical centre
    items.sort(key=lambda it: it["cy"])
    rows_clustered = []  # list of lists of items
    for item in items:
        placed = False
        for row in rows_clustered:
            row_cy = sum(it["cy"] for it in row) / len(row)
            if abs(item["cy"] - row_cy) <= row_tolerance:
                row.append(item)
                placed = True
                break
        if not placed:
            rows_clustered.append([item])

    # Sort each row left-to-right by x1, then extract text
    cells_2d = []
    for row in rows_clustered:
        row.sort(key=lambda it: it["bbox"][0])
        cells_2d.append([it["text"] for it in row])

    return cells_2d, table_bbox, ""


def _parse_table_html(html: str) -> list:
    """
    Parse simple HTML table into a list of rows, each row is a list of cell texts.
    Handles <td>, <th> tags from table recognition output.
    """
    rows = []
    tr_pattern = re.compile(r'<tr>(.*?)</tr>', re.DOTALL)
    td_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL)
    tag_strip = re.compile(r'<[^>]+>')

    for tr_match in tr_pattern.finditer(html):
        tr_content = tr_match.group(1)
        cells = []
        for td_match in td_pattern.finditer(tr_content):
            cell_text = tag_strip.sub('', td_match.group(1)).strip()
            cells.append(cell_text)
        if cells:
            rows.append(cells)

    return rows


async def handle_ocr_table_extract(arguments: dict) -> dict:
    """Extract table structures from an image using OCR + row clustering."""
    image_path = arguments.get("image_path", "")
    lang = arguments.get("lang", "ch")

    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not os.path.isfile(image_path):
        return {"success": False, "error": f"Image file not found: {image_path}"}

    try:
        cells, table_bbox, err = _ocr_to_table_rows(image_path, lang=lang)
    except Exception as e:
        return {"success": False, "error": f"Table extraction error: {str(e)}"}

    if err:
        return {"success": False, "error": err}

    tables = []
    if cells:
        rows = len(cells)
        cols = max((len(row) for row in cells), default=0)

        # Build a simple HTML table representation
        html_parts = ["<table>"]
        for row in cells:
            html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
        html_parts.append("</table>")
        html = "".join(html_parts)

        tables.append({
            "bbox": table_bbox,
            "rows": rows,
            "cols": cols,
            "cells": cells,
            "html": html,
        })

    return {
        "success": True,
        "image_path": image_path,
        "table_count": len(tables),
        "tables": tables,
        "summary": f"Extracted {len(tables)} table(s)"
                   + (f" with {tables[0]['rows']} rows x {tables[0]['cols']} cols" if tables else ""),
    }
