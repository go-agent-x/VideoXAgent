#!/usr/bin/env python3
"""Spatial Relation Calculator MCP Server — Pure Python geometry computation."""
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
sys.path.insert(0, str(Path(__file__).parent))  # to import from original spatial_calc_server.py

from _base.server_framework import create_mcp_server, run_server

# ── Business logic (from spatial_calc_server.py) ──

import json
import math
import os

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from geometry import (
    validate_bbox,
    bbox_center,
    bbox_area,
    bbox_dimensions,
    compute_iou,
    compute_overlap_ratios,
    compute_containment,
    direction_8,
    direction_4,
    center_distance,
    edge_distance,
    proximity_label,
    point_in_polygon,
    point_to_polygon_boundary_distance,
    polygon_area,
    bbox_containment_detail,
)

# ---------------------------------------------------------------------------
# MCP Server init
# ---------------------------------------------------------------------------


# ===========================================================================
# Tool 1: calc_bbox_relation
# ===========================================================================

async def _handle_calc_bbox_relation(arguments: dict) -> dict:
    """Full spatial relation analysis between two bounding boxes."""
    bbox_a = arguments.get("bbox_a")
    bbox_b = arguments.get("bbox_b")
    label_a = arguments.get("label_a", "A")
    label_b = arguments.get("label_b", "B")
    img_w = arguments.get("image_width", 0)
    img_h = arguments.get("image_height", 0)

    # Validate
    ok_a, err_a = validate_bbox(bbox_a, "bbox_a")
    if not ok_a:
        return {"success": False, "error": err_a}
    ok_b, err_b = validate_bbox(bbox_b, "bbox_b")
    if not ok_b:
        return {"success": False, "error": err_b}

    a = [float(x) for x in bbox_a[:4]]
    b = [float(x) for x in bbox_b[:4]]

    ca = bbox_center(a)
    cb = bbox_center(b)

    # Core computations
    iou = compute_iou(a, b)
    ra, rb = compute_overlap_ratios(a, b)
    containment = compute_containment(a, b)
    dir8 = direction_8(ca, cb)
    dir4 = direction_4(ca, cb)

    cdist = center_distance(ca, cb)
    edist = edge_distance(a, b)

    area_a = bbox_area(a)
    area_b = bbox_area(b)
    size_ratio = round(area_a / area_b, 3) if area_b > 0 else float("inf")

    if size_ratio < 0.8:
        size_desc = f"{label_a} is smaller than {label_b} (ratio {size_ratio})"
    elif size_ratio > 1.25:
        size_desc = f"{label_a} is larger than {label_b} (ratio {size_ratio})"
    else:
        size_desc = f"{label_a} and {label_b} are similar in size (ratio {size_ratio})"

    # Horizontal / vertical relation strings
    dx = ca[0] - cb[0]
    dy = ca[1] - cb[1]

    if abs(dx) < 15:
        h_str = f"{label_a} is horizontally aligned with {label_b}"
    elif dx < 0:
        h_str = f"{label_a} is LEFT of {label_b}"
    else:
        h_str = f"{label_a} is RIGHT of {label_b}"

    if abs(dy) < 15:
        v_str = f"{label_a} is vertically aligned with {label_b}"
    elif dy < 0:
        v_str = f"{label_a} is ABOVE {label_b}"
    else:
        v_str = f"{label_a} is BELOW {label_b}"

    # Proximity label
    prox = proximity_label(edist, img_w, img_h)

    # Summary
    overlap_note = ""
    if containment == "A_inside_B":
        overlap_note = f", {label_a} is inside {label_b}"
    elif containment == "B_inside_A":
        overlap_note = f", {label_b} is inside {label_a}"
    elif containment == "partial_overlap":
        overlap_note = f", partial overlap (IoU={round(iou, 3)})"

    summary = (
        f"{label_a} is to the {dir8} of {label_b}, "
        f"center distance {round(cdist, 1)}px"
        f"{overlap_note}"
    )

    result = {
        "success": True,
        "containment": containment,
        "overlap_iou": round(iou, 4),
        "overlap_ratio_a": round(ra, 4),
        "overlap_ratio_b": round(rb, 4),
        "direction_8": dir8,
        "direction_4": dir4,
        "composite_direction": dir8,
        "horizontal_relation": h_str,
        "vertical_relation": v_str,
        "center_distance_px": round(cdist, 1),
        "edge_distance_px": round(edist, 1),
        "proximity": prox,
        "a_center": [round(ca[0], 1), round(ca[1], 1)],
        "b_center": [round(cb[0], 1), round(cb[1], 1)],
        "a_area": round(area_a, 1),
        "b_area": round(area_b, 1),
        "relative_size": size_desc,
        "summary": summary,
    }

    # Optional normalized coordinates when image dimensions are known
    if img_w > 0 and img_h > 0:
        diag = math.sqrt(img_w ** 2 + img_h ** 2)
        result["a_center_norm"] = [round(ca[0] / img_w, 4), round(ca[1] / img_h, 4)]
        result["b_center_norm"] = [round(cb[0] / img_w, 4), round(cb[1] / img_h, 4)]
        result["center_distance_norm"] = round(cdist / diag, 4) if diag > 0 else 0
        result["edge_distance_norm"] = round(edist / diag, 4) if diag > 0 else 0

    return result


# ===========================================================================
# Tool 2: calc_point_in_region
# ===========================================================================

async def _handle_calc_point_in_region(arguments: dict) -> dict:
    """Test whether a point or bbox center is inside a polygon region."""
    point_input = arguments.get("point")
    region = arguments.get("region")
    label_point = arguments.get("label_point", "point")
    label_region = arguments.get("label_region", "region")

    # Validate point
    if not isinstance(point_input, (list, tuple)) or len(point_input) < 2:
        return {"success": False, "error": "point must be [x,y] or [x1,y1,x2,y2]"}

    # If bbox (4 values), take center
    is_bbox_input = False
    if len(point_input) >= 4:
        ok, err = validate_bbox(point_input, "point(as bbox)")
        if not ok:
            return {"success": False, "error": err}
        px, py = bbox_center([float(x) for x in point_input[:4]])
        is_bbox_input = True
    else:
        try:
            px, py = float(point_input[0]), float(point_input[1])
        except (TypeError, ValueError):
            return {"success": False, "error": "point contains non-numeric values"}

    # Validate region polygon
    if not isinstance(region, (list, tuple)) or len(region) < 3:
        return {"success": False, "error": "region must be a polygon with at least 3 vertices [[x1,y1],[x2,y2],...]"}

    polygon = []
    for i, v in enumerate(region):
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            return {"success": False, "error": f"region vertex {i} must be [x, y]"}
        try:
            polygon.append([float(v[0]), float(v[1])])
        except (TypeError, ValueError):
            return {"success": False, "error": f"region vertex {i} contains non-numeric values"}

    # Core computations
    inside = point_in_polygon(px, py, polygon)
    boundary_dist, nearest_edge = point_to_polygon_boundary_distance(px, py, polygon)
    region_area_val = polygon_area(polygon)

    status = "INSIDE" if inside else "OUTSIDE"
    summary = (
        f"{label_point}({round(px, 1)},{round(py, 1)}) is {status} {label_region}, "
        f"{round(boundary_dist, 1)}px from boundary"
    )

    result: dict = {
        "success": True,
        "point_xy": [round(px, 1), round(py, 1)],
        "is_inside": inside,
        "distance_to_boundary": round(boundary_dist, 1),
        "nearest_edge_index": nearest_edge,
        "region_area_px": round(region_area_val, 1),
        "region_vertex_count": len(polygon),
        "summary": summary,
    }

    # If input was bbox, also check whether all 4 corners are inside
    if is_bbox_input:
        corners = [
            (float(point_input[0]), float(point_input[1])),
            (float(point_input[2]), float(point_input[1])),
            (float(point_input[2]), float(point_input[3])),
            (float(point_input[0]), float(point_input[3])),
        ]
        corners_inside = [point_in_polygon(cx, cy, polygon) for cx, cy in corners]
        result["bbox_all_corners_inside"] = all(corners_inside)
        result["bbox_corners_inside_count"] = sum(corners_inside)

    return result


# ===========================================================================
# Tool 3: calc_multi_spatial
# ===========================================================================

async def _handle_calc_multi_spatial(arguments: dict) -> dict:
    """Batch spatial analysis of multiple objects relative to a reference."""
    objects = arguments.get("objects")
    reference = arguments.get("reference_object")
    img_w = arguments.get("image_width", 0)
    img_h = arguments.get("image_height", 0)

    if not isinstance(objects, (list, tuple)) or len(objects) == 0:
        return {"success": False, "error": "objects must be a non-empty list of {bbox, label}"}

    # Parse objects
    parsed_objs = []
    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            return {"success": False, "error": f"objects[{i}] must be a dict with 'bbox' and 'label'"}
        bbox = obj.get("bbox")
        ok, err = validate_bbox(bbox, f"objects[{i}].bbox")
        if not ok:
            return {"success": False, "error": err}
        parsed_objs.append({
            "label": obj.get("label", f"object_{i}"),
            "bbox": [float(x) for x in bbox[:4]],
        })

    # Parse reference
    if reference and isinstance(reference, dict):
        ref_bbox = reference.get("bbox")
        ok, err = validate_bbox(ref_bbox, "reference_object.bbox")
        if not ok:
            return {"success": False, "error": err}
        ref_label = reference.get("label", "reference")
        ref_bbox_f = [float(x) for x in ref_bbox[:4]]
    else:
        # Use first object as reference
        ref_label = parsed_objs[0]["label"]
        ref_bbox_f = parsed_objs[0]["bbox"]
        parsed_objs = parsed_objs[1:]

    ref_center = bbox_center(ref_bbox_f)

    relations = []
    directional_groups: dict = {}

    for obj in parsed_objs:
        obj_center = bbox_center(obj["bbox"])
        dist = center_distance(obj_center, ref_center)
        dir8 = direction_8(obj_center, ref_center)
        dir4 = direction_4(obj_center, ref_center)
        containment = compute_containment(obj["bbox"], ref_bbox_f)
        iou = compute_iou(obj["bbox"], ref_bbox_f)

        rel: dict = {
            "label": obj["label"],
            "direction_8": dir8,
            "direction_4": dir4,
            "distance_px": round(dist, 1),
            "containment": containment,
            "overlap_iou": round(iou, 4),
            "center": [round(obj_center[0], 1), round(obj_center[1], 1)],
        }
        relations.append(rel)

        # Build directional grouping
        if dir8 not in directional_groups:
            directional_groups[dir8] = []
        directional_groups[dir8].append(obj["label"])

    # Sort by distance
    relations_sorted = sorted(relations, key=lambda r: r["distance_px"])

    # Spatial map string
    map_parts = [f"{r['label']}({r['direction_8']},{r['distance_px']}px)" for r in relations_sorted]
    spatial_map = " | ".join(map_parts)

    nearest = relations_sorted[0] if relations_sorted else None
    farthest = relations_sorted[-1] if relations_sorted else None

    # Cluster summary
    cluster_parts = []
    for direction, labels in sorted(directional_groups.items()):
        cluster_parts.append(f"{direction}: [{', '.join(labels)}]")
    cluster_summary = "; ".join(cluster_parts) if cluster_parts else "no objects"

    return {
        "success": True,
        "object_count": len(relations),
        "reference": ref_label,
        "reference_center": [round(ref_center[0], 1), round(ref_center[1], 1)],
        "relations": relations_sorted,
        "spatial_map": spatial_map,
        "directional_grouping": directional_groups,
        "cluster_summary": cluster_summary,
        "nearest_to_reference": f"{nearest['label']} ({nearest['distance_px']}px)" if nearest else "N/A",
        "farthest_from_reference": f"{farthest['label']} ({farthest['distance_px']}px)" if farthest else "N/A",
    }


# ===========================================================================
# Tool 4: calc_bbox_in_bbox
# ===========================================================================

async def _handle_calc_bbox_in_bbox(arguments: dict) -> dict:
    """Quick test: is inner bbox contained within outer bbox?"""
    inner_bbox = arguments.get("inner_bbox")
    outer_bbox = arguments.get("outer_bbox")
    label_inner = arguments.get("label_inner", "object")
    label_outer = arguments.get("label_outer", "region")

    ok_i, err_i = validate_bbox(inner_bbox, "inner_bbox")
    if not ok_i:
        return {"success": False, "error": err_i}
    ok_o, err_o = validate_bbox(outer_bbox, "outer_bbox")
    if not ok_o:
        return {"success": False, "error": err_o}

    inner = [float(x) for x in inner_bbox[:4]]
    outer = [float(x) for x in outer_bbox[:4]]

    detail = bbox_containment_detail(inner, outer)

    status = detail["status"]
    if status == "inside":
        summary = (
            f"{label_inner} is INSIDE {label_outer}, "
            f"min margin {detail['min_margin']}px from boundary"
        )
    elif status == "outside":
        summary = (
            f"{label_inner} is OUTSIDE {label_outer}, "
            f"no overlap"
        )
    else:
        summary = (
            f"{label_inner} PARTIALLY overlaps {label_outer}, "
            f"overlap ratio {detail['overlap_ratio']}"
        )

    return {
        "success": True,
        "status": detail["status"],
        "is_inside": detail["is_inside"],
        "overlap_ratio": detail["overlap_ratio"],
        "margin_left": detail["margin_left"],
        "margin_right": detail["margin_right"],
        "margin_top": detail["margin_top"],
        "margin_bottom": detail["margin_bottom"],
        "min_margin": detail["min_margin"],
        "summary": summary,
    }


# ===========================================================================
# MCP Tool Registration
# ===========================================================================

# ===========================================================================
# Tool Call Router
# ===========================================================================

TOOL_HANDLERS = {
    "calc_bbox_relation": _handle_calc_bbox_relation,
    "calc_point_in_region": _handle_calc_point_in_region,
    "calc_multi_spatial": _handle_calc_multi_spatial,
    "calc_bbox_in_bbox": _handle_calc_bbox_in_bbox,
}


# ===========================================================================
# Entry point
# ===========================================================================


async def handle_calc_bbox_relation(arguments: dict) -> dict:
    return await _handle_calc_bbox_relation(arguments)


async def handle_calc_point_in_region(arguments: dict) -> dict:
    return await _handle_calc_point_in_region(arguments)


async def handle_calc_multi_spatial(arguments: dict) -> dict:
    return await _handle_calc_multi_spatial(arguments)


async def handle_calc_scene_zones(arguments: dict) -> dict:
    return await _handle_calc_bbox_in_bbox(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="spatial-calc",
        tool_defs_source="_tool_defs/spatial.yaml",
        handlers={
            "calc_bbox_relation": handle_calc_bbox_relation,
            "calc_point_in_region": handle_calc_point_in_region,
            "calc_multi_spatial": handle_calc_multi_spatial,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
