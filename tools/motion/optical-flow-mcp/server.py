#!/usr/bin/env python3
"""Optical Flow MCP Server — Dense optical flow analysis for video understanding."""
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
sys.path.insert(0, str(Path(__file__).parent))

from _base.device import select_torch_device
from _base.server_framework import create_mcp_server, run_server

# ── Business logic (from optical_flow_server.py) ──

import gc
import json
import os
import threading
import traceback
from typing import Optional

import cv2
import numpy as np


# =========================================================================
# Backend state — dual backend with thread-safe lazy init
# =========================================================================
_model_lock = threading.Lock()
_raft_model = None
_raft_transforms = None
_backend = "uninitialized"  # "raft" | "farneback" | "uninitialized"


def _try_init_raft():
    """Attempt to load RAFT model from torchvision. Returns (model, transforms) or (None, None)."""
    try:
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        weights = Raft_Large_Weights.DEFAULT
        device = select_torch_device(torch)

        model = raft_large(weights=weights).to(device).eval()
        transforms = weights.transforms()

        print(f"[optical-flow] RAFT loaded on {device}", file=sys.stderr)
        return model, transforms
    except Exception as exc:
        print(f"[optical-flow] RAFT unavailable ({exc}), falling back to Farneback.",
              file=sys.stderr)
        return None, None


def _init_backend():
    """Initialise the optical flow backend (called once, thread-safe)."""
    global _raft_model, _raft_transforms, _backend
    if _backend != "uninitialized":
        return
    with _model_lock:
        if _backend != "uninitialized":
            return
        model, transforms = _try_init_raft()
        if model is not None:
            _raft_model = model
            _raft_transforms = transforms
            _backend = "raft"
        else:
            _backend = "farneback"


def _cleanup_gpu():
    """Free unused GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except ImportError:
        pass


# =========================================================================
# Frame extraction
# =========================================================================
def _extract_frame_at_time(video_path: str, time_sec: float) -> np.ndarray:
    """Extract a single frame from a video at the given timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata: fps={fps}, frames={total_frames}")

    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame at t={time_sec}s")
    return frame


def _sample_frame_pairs(video_path: str, start_time: float, end_time: float,
                        sample_fps: float) -> list[tuple]:
    """Sample consecutive frame pairs from a video segment.

    Returns list of (time_mid, frame_a, frame_b) tuples where time_mid is the
    midpoint between the two frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    actual_end = min(end_time, duration) if end_time is not None else duration
    if actual_end <= start_time:
        cap.release()
        return []

    interval = 1.0 / sample_fps
    timestamps = []
    t = start_time
    while t < actual_end:
        timestamps.append(t)
        t += interval
    # Ensure we have at least 2 timestamps for pairs
    if len(timestamps) < 2:
        timestamps = [start_time, actual_end]

    # Read all frames
    frames = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append((t, frame))

    cap.release()

    # Build consecutive pairs
    pairs = []
    for i in range(len(frames) - 1):
        t_a, f_a = frames[i]
        t_b, f_b = frames[i + 1]
        t_mid = round((t_a + t_b) / 2, 3)
        pairs.append((t_mid, f_a, f_b))

    return pairs


# =========================================================================
# Optical flow computation
# =========================================================================
def _resize_for_flow(frame: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Resize frame so the largest dimension <= max_dim. Returns (resized, scale)."""
    h, w = frame.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        # RAFT requires dimensions divisible by 8
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        # Recalculate scale after 8-divisible truncation
        scale = min(new_w / w, new_h / h)
        return cv2.resize(frame, (new_w, new_h)), scale
    # Also ensure divisible by 8 for RAFT
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    if new_w != w or new_h != h:
        return cv2.resize(frame, (new_w, new_h)), min(new_w / w, new_h / h)
    return frame, 1.0


def _compute_flow_raft(frame_a: np.ndarray, frame_b: np.ndarray,
                       max_dim: int = 520) -> np.ndarray:
    """Compute dense optical flow using RAFT. Returns flow (H, W, 2) in pixels."""
    import torch

    device = next(_raft_model.parameters()).device

    # Resize
    a_resized, _ = _resize_for_flow(frame_a, max_dim)
    b_resized, _ = _resize_for_flow(frame_b, max_dim)

    # Ensure same size
    h = min(a_resized.shape[0], b_resized.shape[0])
    w = min(a_resized.shape[1], b_resized.shape[1])
    h = (h // 8) * 8
    w = (w // 8) * 8
    a_resized = cv2.resize(a_resized, (w, h))
    b_resized = cv2.resize(b_resized, (w, h))

    # Compute actual scale from original frame_a to final processing size.
    # This accounts for both _resize_for_flow and the forced-same-size resize above.
    orig_h, orig_w = frame_a.shape[:2]
    actual_scale = min(w / orig_w, h / orig_h) if orig_w > 0 and orig_h > 0 else 1.0

    # BGR -> RGB -> tensor
    a_rgb = cv2.cvtColor(a_resized, cv2.COLOR_BGR2RGB)
    b_rgb = cv2.cvtColor(b_resized, cv2.COLOR_BGR2RGB)

    a_tensor = torch.from_numpy(a_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)
    b_tensor = torch.from_numpy(b_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # Apply RAFT transforms
    a_tensor, b_tensor = _raft_transforms(a_tensor, b_tensor)

    with torch.no_grad():
        flow_predictions = _raft_model(a_tensor, b_tensor)
        # RAFT returns a list of flow predictions (multi-scale); take the last (finest)
        flow = flow_predictions[-1]

    # flow shape: (1, 2, H, W) -> (H, W, 2)
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()

    # Scale flow back to original resolution
    if actual_scale < 1.0:
        flow_np = flow_np / actual_scale

    return flow_np


def _compute_flow_farneback(frame_a: np.ndarray, frame_b: np.ndarray,
                            max_dim: int = 640) -> np.ndarray:
    """Compute dense optical flow using Farneback. Returns flow (H, W, 2) in pixels."""
    a_resized, scale = _resize_for_flow(frame_a, max_dim)
    b_resized, _ = _resize_for_flow(frame_b, max_dim)

    # Ensure same size
    h = min(a_resized.shape[0], b_resized.shape[0])
    w = min(a_resized.shape[1], b_resized.shape[1])
    a_resized = cv2.resize(a_resized, (w, h))
    b_resized = cv2.resize(b_resized, (w, h))

    gray_a = cv2.cvtColor(a_resized, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b_resized, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )

    # Scale flow back to original resolution
    if scale < 1.0:
        flow = flow / scale

    return flow


def _compute_flow(frame_a: np.ndarray, frame_b: np.ndarray,
                  max_dim: int = 520) -> tuple[np.ndarray, str]:
    """Compute optical flow using the best available backend.

    Returns (flow_field, backend_name).
    """
    _init_backend()
    if _backend == "raft":
        try:
            flow = _compute_flow_raft(frame_a, frame_b, max_dim)
            return flow, "raft"
        except Exception as e:
            print(f"[optical-flow] RAFT failed ({e}), falling back to Farneback",
                  file=sys.stderr)
    flow = _compute_flow_farneback(frame_a, frame_b, min(max_dim, 640))
    return flow, "farneback"


# =========================================================================
# Flow field analysis
# =========================================================================
_DIRECTION_NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
_CAMERA_TYPES = {
    "E": "pan_right", "W": "pan_left",
    "N": "tilt_up", "S": "tilt_down",
    "NE": "pan_right_tilt_up", "NW": "pan_left_tilt_up",
    "SE": "pan_right_tilt_down", "SW": "pan_left_tilt_down",
}


def _angle_to_8dir(degrees: float) -> str:
    """Convert angle in degrees (0=right, CCW) to 8-direction compass label."""
    idx = int(((degrees + 22.5) % 360) / 45)
    return _DIRECTION_NAMES[idx]


def _angle_to_readable(degrees: float) -> str:
    """Convert angle to human-readable direction name."""
    mapping = {
        "E": "rightward", "W": "leftward", "N": "upward", "S": "downward",
        "NE": "upper-right", "NW": "upper-left",
        "SE": "lower-right", "SW": "lower-left",
    }
    return mapping.get(_angle_to_8dir(degrees), "unknown")


def _get_position_label(cx: float, cy: float, w: int, h: int) -> str:
    """Map center coordinates to a 3x3 grid position label."""
    col = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
    row = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "center")
    if row == "center" and col == "center":
        return "center"
    if row == "center":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def _analyze_flow_field(flow: np.ndarray, orig_width: int, orig_height: int) -> dict:
    """Analyze a dense flow field and return comprehensive statistics.

    Args:
        flow: (H, W, 2) flow field in pixels (dx, dy).
        orig_width, orig_height: original frame dimensions for position labels.

    Returns dict with: global_flow_stats, camera_motion, object_motion_regions,
        direction_histogram, magnitude_histogram.
    """
    h, w = flow.shape[:2]
    dx = flow[..., 0]
    dy = flow[..., 1]

    mag = np.sqrt(dx ** 2 + dy ** 2)
    ang = np.degrees(np.arctan2(-dy, dx)) % 360  # -dy because image y-axis is inverted

    # --- Global stats ---
    mean_mag = float(np.mean(mag))
    median_mag = float(np.median(mag))
    max_mag = float(np.max(mag))

    # Dominant direction (magnitude-weighted circular mean)
    if mean_mag > 0.3:
        weights = mag.flatten()
        angles_rad = np.radians(ang.flatten())
        sin_mean = float(np.average(np.sin(angles_rad), weights=weights))
        cos_mean = float(np.average(np.cos(angles_rad), weights=weights))
        dominant_deg = float(np.degrees(np.arctan2(sin_mean, cos_mean))) % 360
    else:
        dominant_deg = 0.0

    global_stats = {
        "mean_magnitude": round(mean_mag, 3),
        "median_magnitude": round(median_mag, 3),
        "max_magnitude": round(max_mag, 3),
        "dominant_direction": _angle_to_readable(dominant_deg),
        "dominant_direction_degrees": round(dominant_deg, 1),
    }

    # --- Camera motion estimation ---
    # Divide into 4x4 grid, compute median flow per cell
    grid_n = 4
    cell_h, cell_w = h // grid_n, w // grid_n
    cell_flows = []
    for gi in range(grid_n):
        for gj in range(grid_n):
            y0, y1 = gi * cell_h, (gi + 1) * cell_h
            x0, x1 = gj * cell_w, (gj + 1) * cell_w
            cell_dx = float(np.median(dx[y0:y1, x0:x1]))
            cell_dy = float(np.median(dy[y0:y1, x0:x1]))
            cell_flows.append((cell_dx, cell_dy))

    cell_mags = [np.sqrt(cdx ** 2 + cdy ** 2) for cdx, cdy in cell_flows]
    cell_angs = [float(np.degrees(np.arctan2(-cdy, cdx))) % 360 for cdx, cdy in cell_flows]

    # Background flow = median of all cell flows
    bg_dx = float(np.median([cf[0] for cf in cell_flows]))
    bg_dy = float(np.median([cf[1] for cf in cell_flows]))
    bg_mag = float(np.sqrt(bg_dx ** 2 + bg_dy ** 2))
    bg_ang = float(np.degrees(np.arctan2(-bg_dy, bg_dx))) % 360

    # Camera motion confidence: ratio of cells consistent with background
    if bg_mag > 0.5:
        consistent = 0
        for cm, ca in zip(cell_mags, cell_angs):
            if cm > 0.3:
                angle_diff = abs(ca - bg_ang)
                angle_diff = min(angle_diff, 360 - angle_diff)
                if angle_diff < 45:
                    consistent += 1
        confidence = consistent / len(cell_flows)
        is_camera_moving = confidence > 0.5
    else:
        confidence = 1.0
        is_camera_moving = False

    if is_camera_moving:
        camera_dir = _angle_to_8dir(bg_ang)
        camera_type = _CAMERA_TYPES.get(camera_dir, "moving")
    else:
        camera_type = "static"

    # Detect zoom: check if flow vectors point outward/inward from center
    if not is_camera_moving and bg_mag < 1.0:
        cx, cy = w / 2, h / 2
        radial_components = []
        for gi in range(grid_n):
            for gj in range(grid_n):
                cell_cx = (gj + 0.5) * cell_w
                cell_cy = (gi + 0.5) * cell_h
                # Vector from center to cell
                rx, ry = cell_cx - cx, cell_cy - cy
                r_len = max(np.sqrt(rx ** 2 + ry ** 2), 1e-6)
                # Project flow onto radial direction
                cdx, cdy = cell_flows[gi * grid_n + gj]
                radial = (cdx * rx + (-cdy) * ry) / r_len
                radial_components.append(radial)
        mean_radial = float(np.mean(radial_components))
        if abs(mean_radial) > 1.0 and all(r * mean_radial > 0 for r in radial_components if abs(r) > 0.3):
            is_camera_moving = True
            camera_type = "zoom_in" if mean_radial > 0 else "zoom_out"
            confidence = min(1.0, abs(mean_radial) / 3.0)
            bg_mag = abs(mean_radial)

    camera_motion = {
        "type": camera_type,
        "direction_degrees": round(bg_ang, 1) if is_camera_moving and "zoom" not in camera_type else None,
        "magnitude": round(bg_mag, 3),
        "confidence": round(confidence, 3),
        "is_camera_moving": is_camera_moving,
    }

    # --- Object motion regions (after subtracting camera flow) ---
    residual_dx = dx - bg_dx
    residual_dy = dy - bg_dy
    residual_mag = np.sqrt(residual_dx ** 2 + residual_dy ** 2)

    # Threshold: pixels with residual > 2px (or adaptive)
    threshold = max(2.0, mean_mag * 0.3)
    motion_mask = (residual_mag > threshold).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(100, h * w * 0.005)  # at least 0.5% of frame
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Scale bbox to original resolution
        scale_x = orig_width / w
        scale_y = orig_height / h
        bbox = [
            int(x * scale_x), int(y * scale_y),
            int((x + bw) * scale_x), int((y + bh) * scale_y),
        ]
        # Mean residual flow in region
        mask_region = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_region, [cnt], -1, 255, -1)
        region_pixels = mask_region > 0
        mean_rmag = float(np.mean(residual_mag[region_pixels]))
        region_ang = np.degrees(np.arctan2(
            -float(np.mean(residual_dy[region_pixels])),
            float(np.mean(residual_dx[region_pixels])),
        )) % 360

        cx_orig = (x + bw / 2) * scale_x
        cy_orig = (y + bh / 2) * scale_y

        regions.append({
            "bbox": bbox,
            "area": int(area * scale_x * scale_y),
            "mean_magnitude": round(mean_rmag, 3),
            "direction": _angle_to_readable(region_ang),
            "direction_degrees": round(region_ang, 1),
            "position": _get_position_label(cx_orig, cy_orig, orig_width, orig_height),
        })

    regions.sort(key=lambda r: r["area"], reverse=True)
    regions = regions[:10]

    # --- Direction histogram (8-bin, magnitude-weighted) ---
    dir_hist = {d: 0.0 for d in _DIRECTION_NAMES}
    total_weight = float(np.sum(mag))
    if total_weight > 0:
        for d in _DIRECTION_NAMES:
            idx = _DIRECTION_NAMES.index(d)
            low = (idx * 45 - 22.5) % 360
            high = (idx * 45 + 22.5) % 360
            if low < high:
                mask = (ang >= low) & (ang < high)
            else:
                mask = (ang >= low) | (ang < high)
            dir_hist[d] = round(float(np.sum(mag[mask])) / total_weight, 4)

    # --- Magnitude histogram (5-bin) ---
    mag_bins = {
        "static": float(np.sum(mag < 0.5)) / mag.size,
        "subtle": float(np.sum((mag >= 0.5) & (mag < 2))) / mag.size,
        "moderate": float(np.sum((mag >= 2) & (mag < 5))) / mag.size,
        "fast": float(np.sum((mag >= 5) & (mag < 15))) / mag.size,
        "very_fast": float(np.sum(mag >= 15)) / mag.size,
    }
    mag_bins = {k: round(v, 4) for k, v in mag_bins.items()}

    return {
        "global_flow_stats": global_stats,
        "camera_motion": camera_motion,
        "object_motion_regions": regions,
        "direction_histogram": dir_hist,
        "magnitude_histogram": mag_bins,
    }


def _build_summary(analysis: dict) -> str:
    """Generate a human-readable summary from flow analysis."""
    parts = []
    cam = analysis["camera_motion"]
    if cam["is_camera_moving"]:
        parts.append(f"Camera {cam['type'].replace('_', ' ')} (confidence {cam['confidence']})")
    else:
        parts.append("Camera static")

    regions = analysis["object_motion_regions"]
    if regions:
        largest = regions[0]
        parts.append(
            f"{len(regions)} moving region(s) detected, "
            f"largest at {largest['position']} moving {largest['direction']} "
            f"(magnitude {largest['mean_magnitude']})"
        )
    else:
        stats = analysis["global_flow_stats"]
        if stats["mean_magnitude"] < 0.5:
            parts.append("No significant motion detected")
        else:
            parts.append(f"Global motion {stats['dominant_direction']} "
                         f"(mean magnitude {stats['mean_magnitude']})")

    return ". ".join(parts) + "."


# =========================================================================
# Motion event detection (for segment analysis)
# =========================================================================
def _detect_motion_events(magnitudes: list[float], timestamps: list[float],
                          threshold_ratio: float = 0.3) -> list[dict]:
    """Detect motion onset/offset events from a magnitude timeline."""
    if not magnitudes or len(magnitudes) < 2:
        return []

    max_mag = max(magnitudes)
    if max_mag < 0.5:
        return []

    threshold = max(max_mag * threshold_ratio, 0.5)
    events = []
    in_motion = magnitudes[0] >= threshold

    for i in range(1, len(magnitudes)):
        was_in_motion = in_motion
        in_motion = magnitudes[i] >= threshold

        if in_motion and not was_in_motion:
            events.append({
                "type": "onset",
                "timestamp": round(timestamps[i], 3),
                "magnitude": round(magnitudes[i], 3),
            })
        elif not in_motion and was_in_motion:
            events.append({
                "type": "offset",
                "timestamp": round(timestamps[i], 3),
                "magnitude": round(magnitudes[i], 3),
            })

    return events


def _compute_periodicity(magnitudes: list[float], time_interval: float) -> dict:
    """Analyze periodicity in a magnitude series via autocorrelation."""
    if len(magnitudes) < 6:
        return {"is_periodic": False, "period_seconds": None, "confidence": 0.0}

    arr = np.array(magnitudes, dtype=np.float64)
    arr = arr - np.mean(arr)
    var = np.var(arr)
    if var < 1e-6:
        return {"is_periodic": False, "period_seconds": None, "confidence": 0.0}

    n = len(arr)
    autocorr = np.correlate(arr, arr, mode="full")
    autocorr = autocorr[n - 1:]  # keep only non-negative lags
    autocorr = autocorr / autocorr[0]  # normalize

    # Find first peak after lag 0 (skip lag 0 and 1)
    min_lag = 2
    max_lag = min(n // 2, len(autocorr))
    if max_lag <= min_lag:
        return {"is_periodic": False, "period_seconds": None, "confidence": 0.0}

    peaks = []
    for i in range(min_lag, max_lag):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1] if i + 1 < len(autocorr) else True:
            peaks.append((i, autocorr[i]))

    if not peaks:
        return {"is_periodic": False, "period_seconds": None, "confidence": 0.0}

    # Best peak
    best_lag, best_corr = max(peaks, key=lambda x: x[1])
    period_sec = round(best_lag * time_interval, 3)
    confidence = round(max(0.0, min(1.0, float(best_corr))), 3)

    return {
        "is_periodic": confidence > 0.3,
        "period_seconds": period_sec if confidence > 0.3 else None,
        "confidence": confidence,
    }


# =========================================================================
# Tool handlers
# =========================================================================
async def _handle_compute_dense_flow(arguments: dict) -> dict:
    """Handle compute_dense_flow tool call."""
    video_path = arguments.get("video_path", "")
    time_a = arguments.get("time_a")
    time_b = arguments.get("time_b")
    max_resolution = int(arguments.get("max_resolution", 520))

    if not video_path:
        return {"success": False, "error": "video_path is required"}
    if time_a is None or time_b is None:
        return {"success": False, "error": "Both time_a and time_b are required"}
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"File not found: {video_path}"}

    time_a = float(time_a)
    time_b = float(time_b)

    try:
        frame_a = _extract_frame_at_time(video_path, time_a)
        frame_b = _extract_frame_at_time(video_path, time_b)

        orig_h, orig_w = frame_a.shape[:2]

        flow, backend = _compute_flow(frame_a, frame_b, max_resolution)
        analysis = _analyze_flow_field(flow, orig_w, orig_h)
        summary = _build_summary(analysis)

        _cleanup_gpu()

        return {
            "success": True,
            "backend": backend,
            "video_path": video_path,
            "time_a": time_a,
            "time_b": time_b,
            "frame_size": {"width": orig_w, "height": orig_h},
            **analysis,
            "summary": summary,
        }

    except Exception as e:
        _cleanup_gpu()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


async def _handle_analyze_flow_segment(arguments: dict) -> dict:
    """Handle analyze_flow_segment tool call."""
    video_path = arguments.get("video_path", "")
    start_time = float(arguments.get("start_time", 0))
    end_time_raw = arguments.get("end_time")
    sample_fps = float(arguments.get("sample_fps", 2))
    roi_raw = arguments.get("roi")

    if not video_path:
        return {"success": False, "error": "video_path is required"}
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"File not found: {video_path}"}

    # Parse ROI
    roi = None
    if roi_raw:
        if isinstance(roi_raw, str):
            roi = json.loads(roi_raw)
        else:
            roi = roi_raw

    # Get video duration for end_time default
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    end_time = float(end_time_raw) if end_time_raw is not None else duration

    try:
        pairs = _sample_frame_pairs(video_path, start_time, end_time, sample_fps)
        if not pairs:
            return {"success": False, "error": "No frame pairs extracted"}

        timeline = []
        magnitudes = []
        timestamps = []
        camera_types = []
        all_directions = []
        backend_used = "farneback"

        for t_mid, f_a, f_b in pairs:
            # Apply ROI if specified
            if roi:
                x1, y1, x2, y2 = roi
                f_a_crop = f_a[y1:y2, x1:x2]
                f_b_crop = f_b[y1:y2, x1:x2]
                if f_a_crop.size == 0 or f_b_crop.size == 0:
                    continue
                flow, backend_used = _compute_flow(f_a_crop, f_b_crop, 520)
                crop_h, crop_w = f_a_crop.shape[:2]
                analysis = _analyze_flow_field(flow, crop_w, crop_h)
            else:
                flow, backend_used = _compute_flow(f_a, f_b, 520)
                orig_h, orig_w = f_a.shape[:2]
                analysis = _analyze_flow_field(flow, orig_w, orig_h)

            stats = analysis["global_flow_stats"]
            cam = analysis["camera_motion"]

            entry = {
                "timestamp": t_mid,
                "magnitude": stats["mean_magnitude"],
                "direction": stats["dominant_direction"],
                "camera_type": cam["type"],
            }
            timeline.append(entry)
            magnitudes.append(stats["mean_magnitude"])
            timestamps.append(t_mid)
            camera_types.append(cam["type"])
            all_directions.append(stats["dominant_direction"])

        _cleanup_gpu()

        if not timeline:
            return {"success": False, "error": "No valid flow computations"}

        # Motion events
        motion_events = _detect_motion_events(magnitudes, timestamps)

        # Camera motion summary
        from collections import Counter
        cam_counter = Counter(camera_types)
        total = len(camera_types)
        static_count = cam_counter.get("static", 0)
        camera_summary = {
            "dominant_type": cam_counter.most_common(1)[0][0],
            "static_ratio": round(static_count / total, 3),
            "moving_ratio": round(1 - static_count / total, 3),
            "types_detected": sorted(set(camera_types)),
        }

        # Segment stats
        dir_counter = Counter(all_directions)
        segment_stats = {
            "mean_magnitude": round(float(np.mean(magnitudes)), 3),
            "max_magnitude": round(float(np.max(magnitudes)), 3),
            "min_magnitude": round(float(np.min(magnitudes)), 3),
            "dominant_direction": dir_counter.most_common(1)[0][0],
            "motion_variability": round(float(np.std(magnitudes)), 3),
        }

        # Periodicity
        time_interval = 1.0 / sample_fps
        periodicity = _compute_periodicity(magnitudes, time_interval)

        # Summary text
        summary_parts = []
        if motion_events:
            onsets = [e for e in motion_events if e["type"] == "onset"]
            offsets = [e for e in motion_events if e["type"] == "offset"]
            if onsets and offsets:
                summary_parts.append(
                    f"Motion detected between {onsets[0]['time']}s-{offsets[0]['time']}s "
                    f"(onset at {onsets[0]['time']}s)"
                )
            elif onsets:
                summary_parts.append(f"Motion onset at {onsets[0]['time']}s")
        else:
            if segment_stats["mean_magnitude"] < 0.5:
                summary_parts.append("No significant motion in segment")
            else:
                summary_parts.append(
                    f"Continuous motion throughout segment "
                    f"(mean magnitude {segment_stats['mean_magnitude']})"
                )

        if camera_summary["moving_ratio"] > 0.3:
            summary_parts.append(
                f"Camera {camera_summary['dominant_type'].replace('_', ' ')} "
                f"for {camera_summary['moving_ratio'] * 100:.0f}% of segment"
            )

        if periodicity["is_periodic"]:
            summary_parts.append(
                f"Periodic motion detected (period ~{periodicity['period_seconds']}s)"
            )

        return {
            "success": True,
            "backend": backend_used,
            "video_path": video_path,
            "analyzed_range": {
                "start_time": start_time,
                "end_time": end_time,
                "duration": round(end_time - start_time, 3),
            },
            "sample_fps": sample_fps,
            "roi": roi,
            "timeline": timeline,
            "motion_events": motion_events,
            "camera_motion_summary": camera_summary,
            "segment_stats": segment_stats,
            "periodicity": periodicity,
            "summary": ". ".join(summary_parts) + "." if summary_parts else "No analysis.",
        }

    except Exception as e:
        _cleanup_gpu()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# MCP Server
# =========================================================================

TOOL_HANDLERS = {
    "compute_dense_flow": _handle_compute_dense_flow,
    "analyze_flow_segment": _handle_analyze_flow_segment,
}


# =========================================================================
# Entry point
# =========================================================================


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="optical-flow",
        tool_defs_source="_tool_defs/misc.yaml",
        handlers={
            "compute_dense_flow": _handle_compute_dense_flow,
            "analyze_flow_segment": _handle_analyze_flow_segment,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
