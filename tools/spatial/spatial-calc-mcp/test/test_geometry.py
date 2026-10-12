#!/usr/bin/env python3
"""
Unit tests for geometry.py — pure Python spatial primitives.

Run: cd tools/spatial/spatial-calc-mcp && python3 -m pytest test/test_geometry.py -v
  or: cd tools/spatial/spatial-calc-mcp && python3 test/test_geometry.py
"""

import math
import os
import sys
import unittest

# Ensure geometry module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    point_to_segment_distance,
    point_to_polygon_boundary_distance,
    polygon_area,
    bbox_containment_detail,
)


# =========================================================================
# 4.1: Bbox basics
# =========================================================================
class TestBboxBasics(unittest.TestCase):

    def test_validate_bbox_ok(self):
        ok, err = validate_bbox([10, 20, 100, 200])
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_validate_bbox_float(self):
        ok, _ = validate_bbox([10.5, 20.3, 100.7, 200.1])
        self.assertTrue(ok)

    def test_validate_bbox_zero_size(self):
        ok, _ = validate_bbox([50, 50, 50, 50])
        self.assertTrue(ok)  # zero-area but valid

    def test_validate_bbox_bad_format(self):
        ok, err = validate_bbox([10, 20])
        self.assertFalse(ok)
        self.assertIn("must be", err)

    def test_validate_bbox_inverted(self):
        ok, err = validate_bbox([100, 200, 10, 20])
        self.assertFalse(ok)
        self.assertIn("invalid", err)

    def test_validate_bbox_non_numeric(self):
        ok, err = validate_bbox([10, "abc", 100, 200])
        self.assertFalse(ok)
        self.assertIn("non-numeric", err)

    def test_bbox_center(self):
        cx, cy = bbox_center([0, 0, 100, 200])
        self.assertAlmostEqual(cx, 50.0)
        self.assertAlmostEqual(cy, 100.0)

    def test_bbox_center_offset(self):
        cx, cy = bbox_center([10, 20, 50, 80])
        self.assertAlmostEqual(cx, 30.0)
        self.assertAlmostEqual(cy, 50.0)

    def test_bbox_area(self):
        self.assertAlmostEqual(bbox_area([0, 0, 100, 200]), 20000.0)

    def test_bbox_area_zero(self):
        self.assertAlmostEqual(bbox_area([50, 50, 50, 50]), 0.0)

    def test_bbox_dimensions(self):
        w, h = bbox_dimensions([10, 20, 110, 220])
        self.assertAlmostEqual(w, 100.0)
        self.assertAlmostEqual(h, 200.0)


# =========================================================================
# 4.2: IoU / overlap / containment
# =========================================================================
class TestOverlapContainment(unittest.TestCase):

    def test_iou_no_overlap(self):
        iou = compute_iou([0, 0, 50, 50], [100, 100, 200, 200])
        self.assertAlmostEqual(iou, 0.0)

    def test_iou_identical(self):
        iou = compute_iou([0, 0, 100, 100], [0, 0, 100, 100])
        self.assertAlmostEqual(iou, 1.0)

    def test_iou_partial(self):
        # Two 100x100 boxes overlapping by 50x100
        iou = compute_iou([0, 0, 100, 100], [50, 0, 150, 100])
        # inter = 50*100 = 5000, union = 10000+10000-5000 = 15000
        self.assertAlmostEqual(iou, 5000 / 15000, places=4)

    def test_overlap_ratios_full_containment(self):
        # Small box fully inside large box
        ra, rb = compute_overlap_ratios([40, 40, 60, 60], [0, 0, 100, 100])
        self.assertAlmostEqual(ra, 1.0)  # all of A is within B
        self.assertAlmostEqual(rb, 400 / 10000, places=4)  # small fraction of B

    def test_overlap_ratios_no_overlap(self):
        ra, rb = compute_overlap_ratios([0, 0, 10, 10], [100, 100, 200, 200])
        self.assertAlmostEqual(ra, 0.0)
        self.assertAlmostEqual(rb, 0.0)

    def test_containment_a_inside_b(self):
        result = compute_containment([40, 40, 60, 60], [0, 0, 100, 100])
        self.assertEqual(result, "A_inside_B")

    def test_containment_b_inside_a(self):
        result = compute_containment([0, 0, 100, 100], [40, 40, 60, 60])
        self.assertEqual(result, "B_inside_A")

    def test_containment_partial(self):
        result = compute_containment([0, 0, 100, 100], [50, 50, 150, 150])
        self.assertEqual(result, "partial_overlap")

    def test_containment_no_overlap(self):
        result = compute_containment([0, 0, 10, 10], [100, 100, 200, 200])
        self.assertEqual(result, "no_overlap")


# =========================================================================
# 4.3: Direction 8 and 4
# =========================================================================
class TestDirection(unittest.TestCase):
    """
    Test all 8 compass directions.
    Image coords: x-right, y-down.
    direction_8 computes "where A is relative to B".
    """

    def test_right(self):
        self.assertEqual(direction_8((200, 100), (100, 100)), "right")

    def test_left(self):
        self.assertEqual(direction_8((0, 100), (100, 100)), "left")

    def test_upper(self):
        # A is above B in image (smaller y)
        self.assertEqual(direction_8((100, 0), (100, 100)), "upper")

    def test_lower(self):
        # A is below B in image (larger y)
        self.assertEqual(direction_8((100, 200), (100, 100)), "lower")

    def test_upper_right(self):
        self.assertEqual(direction_8((200, 0), (100, 100)), "upper-right")

    def test_upper_left(self):
        self.assertEqual(direction_8((0, 0), (100, 100)), "upper-left")

    def test_lower_right(self):
        self.assertEqual(direction_8((200, 200), (100, 100)), "lower-right")

    def test_lower_left(self):
        self.assertEqual(direction_8((0, 200), (100, 100)), "lower-left")

    def test_same_position(self):
        self.assertEqual(direction_8((100, 100), (100, 100)), "same_position")

    # direction_4 tests
    def test_dir4_right(self):
        self.assertEqual(direction_4((200, 100), (100, 100)), "right")

    def test_dir4_left(self):
        self.assertEqual(direction_4((0, 100), (100, 100)), "left")

    def test_dir4_upper(self):
        self.assertEqual(direction_4((100, 0), (100, 100)), "upper")

    def test_dir4_lower(self):
        self.assertEqual(direction_4((100, 200), (100, 100)), "lower")

    def test_dir4_diagonal_picks_dominant(self):
        # dx=150, dy=50 → horizontal dominant → right
        self.assertEqual(direction_4((250, 150), (100, 100)), "right")
        # dx=50, dy=150 → vertical dominant → lower
        self.assertEqual(direction_4((150, 250), (100, 100)), "lower")


# =========================================================================
# 4.4: Distance, polygon, bbox_containment_detail
# =========================================================================
class TestDistanceAndPolygon(unittest.TestCase):

    def test_center_distance(self):
        d = center_distance((0, 0), (3, 4))
        self.assertAlmostEqual(d, 5.0)

    def test_edge_distance_no_overlap(self):
        d = edge_distance([0, 0, 10, 10], [20, 0, 30, 10])
        self.assertAlmostEqual(d, 10.0)

    def test_edge_distance_overlap(self):
        d = edge_distance([0, 0, 100, 100], [50, 50, 150, 150])
        self.assertAlmostEqual(d, 0.0)

    def test_edge_distance_diagonal_gap(self):
        # Two boxes with a diagonal gap
        d = edge_distance([0, 0, 10, 10], [20, 20, 30, 30])
        expected = math.sqrt(10**2 + 10**2)
        self.assertAlmostEqual(d, expected, places=3)

    def test_proximity_label_absolute(self):
        self.assertEqual(proximity_label(0), "touching")
        self.assertEqual(proximity_label(50), "near")
        self.assertEqual(proximity_label(150), "moderate")
        self.assertEqual(proximity_label(500), "far")

    def test_proximity_label_with_image(self):
        # Diagonal of 1000x1000 = ~1414
        self.assertEqual(proximity_label(1, 1000, 1000), "touching")
        self.assertEqual(proximity_label(100, 1000, 1000), "near")
        self.assertEqual(proximity_label(300, 1000, 1000), "moderate")
        self.assertEqual(proximity_label(600, 1000, 1000), "far")

    # Polygon tests
    def test_point_in_polygon_inside_square(self):
        square = [[0, 0], [100, 0], [100, 100], [0, 100]]
        self.assertTrue(point_in_polygon(50, 50, square))

    def test_point_in_polygon_outside_square(self):
        square = [[0, 0], [100, 0], [100, 100], [0, 100]]
        self.assertFalse(point_in_polygon(150, 50, square))

    def test_point_in_polygon_triangle(self):
        triangle = [[0, 0], [100, 0], [50, 100]]
        self.assertTrue(point_in_polygon(50, 30, triangle))
        self.assertFalse(point_in_polygon(0, 100, triangle))

    def test_polygon_area_square(self):
        square = [[0, 0], [100, 0], [100, 100], [0, 100]]
        self.assertAlmostEqual(polygon_area(square), 10000.0)

    def test_polygon_area_triangle(self):
        triangle = [[0, 0], [100, 0], [0, 100]]
        self.assertAlmostEqual(polygon_area(triangle), 5000.0)

    def test_point_to_polygon_boundary_distance(self):
        square = [[0, 0], [100, 0], [100, 100], [0, 100]]
        # Point at center of square, distance to nearest edge = 50
        dist, _ = point_to_polygon_boundary_distance(50, 50, square)
        self.assertAlmostEqual(dist, 50.0, places=1)

    def test_point_to_segment_distance(self):
        from geometry import point_to_segment_distance
        # Point directly above midpoint of horizontal segment
        d = point_to_segment_distance(50, 0, 0, 10, 100, 10)
        self.assertAlmostEqual(d, 10.0)


class TestBboxContainmentDetail(unittest.TestCase):

    def test_fully_inside(self):
        result = bbox_containment_detail([20, 20, 80, 80], [0, 0, 100, 100])
        self.assertEqual(result["status"], "inside")
        self.assertTrue(result["is_inside"])
        self.assertAlmostEqual(result["overlap_ratio"], 1.0)
        self.assertAlmostEqual(result["margin_left"], 20.0)
        self.assertAlmostEqual(result["margin_right"], 20.0)
        self.assertAlmostEqual(result["margin_top"], 20.0)
        self.assertAlmostEqual(result["margin_bottom"], 20.0)
        self.assertAlmostEqual(result["min_margin"], 20.0)

    def test_fully_outside(self):
        result = bbox_containment_detail([200, 200, 300, 300], [0, 0, 100, 100])
        self.assertEqual(result["status"], "outside")
        self.assertFalse(result["is_inside"])
        self.assertAlmostEqual(result["overlap_ratio"], 0.0)

    def test_partial_overlap(self):
        result = bbox_containment_detail([50, 50, 150, 150], [0, 0, 100, 100])
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["is_inside"])
        self.assertGreater(result["overlap_ratio"], 0.0)
        self.assertLess(result["overlap_ratio"], 1.0)
        # Right margin is negative (inner exceeds outer on right)
        self.assertLess(result["margin_right"], 0)
        # Bottom margin is negative
        self.assertLess(result["margin_bottom"], 0)

    def test_exact_match(self):
        result = bbox_containment_detail([0, 0, 100, 100], [0, 0, 100, 100])
        self.assertEqual(result["status"], "inside")
        self.assertTrue(result["is_inside"])
        self.assertAlmostEqual(result["overlap_ratio"], 1.0)
        self.assertAlmostEqual(result["min_margin"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)