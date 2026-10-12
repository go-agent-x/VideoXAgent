#!/usr/bin/env python3
"""
Pure-Python geometry primitives for spatial relation calculation.

Zero external dependencies — all computations use only math stdlib.
Provides: bbox operations, IoU/overlap/containment, 8-direction compass,
distance metrics, polygon containment (ray-casting), and bbox-in-bbox shortcuts.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# Bbox basics
# ===========================================================================

def validate_bbox(bbox: Any, name: str = "bbox") -> Tuple[bool, str]:
    """
    Validate bbox format [x1, y1, x2, y2].

    Returns:
        (True, "") on success, (False, error_message) on failure.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False, f"{name} must be [x1, y1, x2, y2], got {bbox}"
    try:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except (TypeError, ValueError):
        return False, f"{name} contains non-numeric values"
    if x2 < x1 or y2 < y1:
        return False, f"{name} invalid: x2 must >= x1 and y2 must >= y1, got [{x1},{y1},{x2},{y2}]"
    return True, ""


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Return (cx, cy) of bbox [x1, y1, x2, y2]."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_area(bbox: List[float]) -> float:
    """Return pixel area of bbox [x1, y1, x2, y2]."""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_dimensions(bbox: List[float]) -> Tuple[float, float]:
    """Return (width, height) of bbox [x1, y1, x2, y2]."""
    return (max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1]))


# ===========================================================================
# Overlap / Containment
# ===========================================================================

def compute_iou(a: List[float], b: List[float]) -> float:
    """Compute Intersection over Union between two bboxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_overlap_ratios(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Return (overlap_ratio_a, overlap_ratio_b).

    overlap_ratio_a = intersection_area / area_a  (how much of A is covered by B).
    """
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    ra = inter / area_a if area_a > 0 else 0.0
    rb = inter / area_b if area_b > 0 else 0.0
    return (ra, rb)


def compute_containment(a: List[float], b: List[float]) -> str:
    """
    Determine containment relationship between bbox A and bbox B.

    Returns one of: "A_inside_B", "B_inside_A", "partial_overlap", "no_overlap".
    """
    ra, rb = compute_overlap_ratios(a, b)
    iou = compute_iou(a, b)

    if iou <= 0:
        return "no_overlap"
    if ra > 0.95:
        return "A_inside_B"
    if rb > 0.95:
        return "B_inside_A"
    return "partial_overlap"


# ===========================================================================
# Direction computation
# ===========================================================================

def direction_8(ca: Tuple[float, float], cb: Tuple[float, float]) -> str:
    """
    8-direction compass: direction of point A relative to point B.
    "A is to the {direction} of B".

    Uses atan2. Image coords (y increases downward) are negated
    so that "up in image" maps to "upper" in output.
    """
    dx = ca[0] - cb[0]
    dy = -(ca[1] - cb[1])  # negate: image y-down → math y-up

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "same_position"

    angle = math.degrees(math.atan2(dy, dx))  # [-180, 180]
    if angle < 0:
        angle += 360.0

    # Map to 8 sectors (each 45°, offset by 22.5°)
    # 0°=right, 90°=upper, 180°=left, 270°=lower
    if angle < 22.5 or angle >= 337.5:
        return "right"
    elif angle < 67.5:
        return "upper-right"
    elif angle < 112.5:
        return "upper"
    elif angle < 157.5:
        return "upper-left"
    elif angle < 202.5:
        return "left"
    elif angle < 247.5:
        return "lower-left"
    elif angle < 292.5:
        return "lower"
    else:
        return "lower-right"


def direction_4(ca: Tuple[float, float], cb: Tuple[float, float]) -> str:
    """
    4-direction: left / right / upper / lower of B.
    Picks the dominant axis.
    """
    dx = ca[0] - cb[0]
    dy = ca[1] - cb[1]  # image coords: positive = downward

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "same_position"

    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "lower" if dy > 0 else "upper"


# ===========================================================================
# Distance metrics
# ===========================================================================

def center_distance(ca: Tuple[float, float], cb: Tuple[float, float]) -> float:
    """Euclidean distance between two center points."""
    return math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)


def edge_distance(a: List[float], b: List[float]) -> float:
    """
    Minimum distance between edges of two bboxes.
    Returns 0.0 if they overlap.
    """
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 > ix1 and iy2 > iy1:
        return 0.0  # overlapping

    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return math.sqrt(dx * dx + dy * dy)


def proximity_label(distance: float, img_width: float = 0, img_height: float = 0) -> str:
    """
    Classify distance as touching / near / moderate / far.

    If image dimensions are provided, thresholds are based on diagonal percentage.
    Otherwise uses absolute pixel thresholds.
    """
    if img_width > 0 and img_height > 0:
        diagonal = math.sqrt(img_width ** 2 + img_height ** 2)
        ratio = distance / diagonal if diagonal > 0 else 0
        if ratio < 0.01:
            return "touching"
        elif ratio < 0.1:
            return "near"
        elif ratio < 0.3:
            return "moderate"
        else:
            return "far"
    else:
        # Absolute pixel thresholds (reasonable for ~640-1920px images)
        if distance < 5:
            return "touching"
        elif distance < 80:
            return "near"
        elif distance < 250:
            return "moderate"
        else:
            return "far"


# ===========================================================================
# Polygon geometry
# ===========================================================================

def point_in_polygon(px: float, py: float, polygon: List[List[float]]) -> bool:
    """
    Ray-casting algorithm: test if point (px, py) is inside a polygon.
    polygon is a list of [x, y] vertices (ordered, closed automatically).
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def point_to_segment_distance(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """Shortest distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def point_to_polygon_boundary_distance(
    px: float, py: float, polygon: List[List[float]]
) -> Tuple[float, int]:
    """
    Minimum distance from point to polygon boundary.
    Returns (min_distance, nearest_edge_index).
    """
    n = len(polygon)
    min_dist = float("inf")
    nearest_edge = 0
    for i in range(n):
        j = (i + 1) % n
        d = point_to_segment_distance(
            px, py,
            polygon[i][0], polygon[i][1],
            polygon[j][0], polygon[j][1],
        )
        if d < min_dist:
            min_dist = d
            nearest_edge = i
    return (min_dist, nearest_edge)


def polygon_area(polygon: List[List[float]]) -> float:
    """Shoelace formula for polygon area (absolute value)."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


# ===========================================================================
# Bbox-in-bbox shortcut
# ===========================================================================

def bbox_containment_detail(
    inner: List[float], outer: List[float]
) -> Dict[str, Any]:
    """
    Detailed containment analysis of inner bbox relative to outer bbox.

    Returns dict with:
        status: "inside" | "outside" | "partial"
        is_inside: bool
        overlap_ratio: float (how much of inner is within outer)
        margin_left/right/top/bottom: pixel margins (negative = exceeds boundary)
        min_margin: smallest margin (negative means outside on that side)
    """
    # Margins: positive means inner edge is within outer boundary
    margin_left = inner[0] - outer[0]
    margin_top = inner[1] - outer[1]
    margin_right = outer[2] - inner[2]
    margin_bottom = outer[3] - inner[3]

    min_margin = min(margin_left, margin_top, margin_right, margin_bottom)

    # Overlap ratio of inner
    ra, _ = compute_overlap_ratios(inner, outer)

    if min_margin >= 0:
        status = "inside"
    elif ra <= 0:
        status = "outside"
    else:
        status = "partial"

    return {
        "status": status,
        "is_inside": status == "inside",
        "overlap_ratio": round(ra, 4),
        "margin_left": round(margin_left, 1),
        "margin_right": round(margin_right, 1),
        "margin_top": round(margin_top, 1),
        "margin_bottom": round(margin_bottom, 1),
        "min_margin": round(min_margin, 1),
    }