"""
Frame difference analysis utilities.

Provides SSIM-based global similarity, pixel-level difference heatmap,
and connected-component change region extraction.
"""

import os
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def extract_frame_at_time(video_path: str, time_sec: float) -> np.ndarray:
    """Extract a single frame from video at specified time."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame at t={time_sec}s")
    return frame


def _load_image(path: str) -> np.ndarray:
    """Load image from file path."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def _resize_to_match(img_a: np.ndarray, img_b: np.ndarray) -> tuple:
    """Resize images to the same dimensions (use smaller)."""
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    if (h_a, w_a) == (h_b, w_b):
        return img_a, img_b
    # Resize to the smaller dimensions
    h = min(h_a, h_b)
    w = min(w_a, w_b)
    img_a = cv2.resize(img_a, (w, h))
    img_b = cv2.resize(img_b, (w, h))
    return img_a, img_b


def compute_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple:
    """Compute SSIM and return (score, diff_map).

    diff_map is a float array where lower values = more different.
    """
    score, diff_map = structural_similarity(
        gray_a, gray_b, full=True, data_range=255
    )
    return float(score), diff_map


def extract_change_regions(
    diff_map: np.ndarray,
    threshold: float = 0.6,
    min_area: int = 100,
) -> list[dict]:
    """Extract bounding boxes of changed regions from SSIM diff map.

    Args:
        diff_map: SSIM difference map (1.0 = identical, 0.0 = max diff).
        threshold: Pixels with SSIM below this are considered changed.
        min_area: Minimum region area in pixels.

    Returns:
        List of change region dicts with bbox, area, magnitude.
    """
    # Convert diff_map: low SSIM = high change
    change_mask = ((1.0 - diff_map) * 255).astype(np.uint8)
    thresh_val = int((1.0 - threshold) * 255)
    _, binary = cv2.threshold(change_mask, thresh_val, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = diff_map.shape[:2]
    regions = []
    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Average change magnitude in this region
        roi = change_mask[y:y+bh, x:x+bw]
        magnitude = float(np.mean(roi)) / 255.0

        regions.append({
            "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
            "area": area,
            "width": int(bw),
            "height": int(bh),
            "magnitude": round(magnitude, 4),
            "position": _get_position_label(x + bw // 2, y + bh // 2, w, h),
        })

    # Sort by area descending
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions


def _get_position_label(cx: int, cy: int, w: int, h: int) -> str:
    """Convert center coordinates to a 3x3 grid position label."""
    col = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
    row = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
    if row == "middle" and col == "center":
        return "center"
    return f"{row}-{col}"


def compute_pixel_diff_stats(gray_a: np.ndarray, gray_b: np.ndarray) -> dict:
    """Compute basic pixel-level difference statistics."""
    diff = cv2.absdiff(gray_a, gray_b)
    return {
        "mean_diff": round(float(np.mean(diff)), 2),
        "max_diff": int(np.max(diff)),
        "std_diff": round(float(np.std(diff)), 2),
        "changed_pixel_ratio": round(float(np.sum(diff > 25)) / diff.size, 4),
    }


def compute_color_histogram_diff(img_a: np.ndarray, img_b: np.ndarray) -> dict:
    """Compare color distributions between two frames."""
    diffs = {}
    for i, channel in enumerate(["blue", "green", "red"]):
        hist_a = cv2.calcHist([img_a], [i], None, [64], [0, 256])
        hist_b = cv2.calcHist([img_b], [i], None, [64], [0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
        diffs[f"{channel}_correlation"] = round(corr, 4)

    avg_corr = np.mean([diffs[f"{c}_correlation"] for c in ["blue", "green", "red"]])
    diffs["avg_color_correlation"] = round(float(avg_corr), 4)
    return diffs


# -------------------------------------------------------------------------
# Internal core — operates on numpy arrays (no file I/O)
# -------------------------------------------------------------------------
def _diff_arrays(
    img_a: np.ndarray,
    img_b: np.ndarray,
    ssim_threshold: float = 0.6,
    min_region_area: int = 100,
) -> dict:
    """Compare two BGR numpy arrays and return detailed diff analysis."""
    img_a, img_b = _resize_to_match(img_a, img_b)

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    ssim_score, diff_map = compute_ssim(gray_a, gray_b)
    change_regions = extract_change_regions(diff_map, ssim_threshold, min_region_area)
    pixel_stats = compute_pixel_diff_stats(gray_a, gray_b)
    color_stats = compute_color_histogram_diff(img_a, img_b)

    h, w = gray_a.shape
    total_changed_area = sum(r["area"] for r in change_regions)

    return {
        "success": True,
        "overall_similarity": round(ssim_score, 4),
        "image_size": {"width": w, "height": h},
        "change_summary": {
            "total_regions": len(change_regions),
            "total_changed_area": total_changed_area,
            "changed_area_ratio": round(total_changed_area / (w * h), 4) if w * h > 0 else 0,
            "interpretation": _interpret_similarity(ssim_score),
        },
        "change_regions": change_regions[:20],  # Cap at 20 regions
        "pixel_statistics": pixel_stats,
        "color_histogram": color_stats,
    }


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def diff_two_frames(
    image_path_a: str,
    image_path_b: str,
    ssim_threshold: float = 0.6,
    min_region_area: int = 100,
) -> dict:
    """Compare two frame images and return detailed diff analysis."""
    img_a = _load_image(image_path_a)
    img_b = _load_image(image_path_b)
    return _diff_arrays(img_a, img_b, ssim_threshold, min_region_area)


def diff_video_timestamps(
    video_path: str,
    time_a: float,
    time_b: float,
    ssim_threshold: float = 0.6,
    min_region_area: int = 100,
) -> dict:
    """Extract frames at two timestamps and compare them."""
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"Video not found: {video_path}"}

    try:
        frame_a = extract_frame_at_time(video_path, time_a)
        frame_b = extract_frame_at_time(video_path, time_b)
    except Exception as e:
        return {"success": False, "error": str(e)}

    result = _diff_arrays(frame_a, frame_b, ssim_threshold, min_region_area)
    result["video_path"] = video_path
    result["time_a"] = time_a
    result["time_b"] = time_b
    result["time_gap"] = round(abs(time_b - time_a), 2)
    return result


def _interpret_similarity(ssim: float) -> str:
    """Human-readable interpretation of SSIM score."""
    if ssim > 0.95:
        return "Nearly identical frames (minimal visual change)"
    elif ssim > 0.85:
        return "Very similar frames (subtle differences, e.g. small motion)"
    elif ssim > 0.70:
        return "Moderately similar (noticeable changes in some regions)"
    elif ssim > 0.50:
        return "Significant differences (major visual changes)"
    elif ssim > 0.30:
        return "Substantially different frames (likely different scenes)"
    else:
        return "Completely different frames (scene change or cut)"