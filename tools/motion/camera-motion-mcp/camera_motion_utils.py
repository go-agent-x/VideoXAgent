#!/usr/bin/env python3
"""
Utilities for camera-motion MCP tool.

Provides:
- Video frame extraction at specified timestamps
- Dense optical flow computation (Farneback)
- Camera motion classification based on flow patterns
- Motion event sequence analysis and merging
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------
def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video metadata: fps, total_frames, duration, width, height.
    
    Args:
        video_path: Absolute path to video file.
        
    Returns:
        Dict with keys: fps, total_frames, duration, width, height.
        
    Raises:
        FileNotFoundError: If video file does not exist.
        ValueError: If video cannot be opened.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total_frames / fps if fps > 0 else 0
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "width": width,
        "height": height,
    }


def extract_frame_at_time(video_path: str, time_sec: float) -> Optional[np.ndarray]:
    """Extract a single frame at specified timestamp.
    
    Args:
        video_path: Absolute path to video file.
        time_sec: Timestamp in seconds.
        
    Returns:
        BGR frame as numpy array, or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_frames_at_times(
    video_path: str,
    timestamps: List[float],
) -> List[Tuple[float, np.ndarray]]:
    """Extract frames at multiple timestamps efficiently.
    
    Args:
        video_path: Absolute path to video file.
        timestamps: List of timestamps in seconds.
        
    Returns:
        List of (timestamp, frame) tuples for successfully extracted frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append((ts, frame))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Optical flow computation
# ---------------------------------------------------------------------------
def compute_optical_flow(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    max_resolution: int = 520,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dense optical flow between two frames using Farneback method.
    
    Args:
        frame_a: First frame (BGR).
        frame_b: Second frame (BGR).
        max_resolution: Max dimension for processing (downsample if larger).
        
    Returns:
        Tuple of (flow, magnitude) where:
        - flow: [H, W, 2] array of (dx, dy) displacement vectors
        - magnitude: [H, W] array of flow magnitudes
    """
    # Convert to grayscale
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    
    # Downsample if needed
    h, w = gray_a.shape[:2]
    scale = 1.0
    if max(h, w) > max_resolution:
        scale = max_resolution / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        gray_a = cv2.resize(gray_a, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray_b = cv2.resize(gray_b, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Compute Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    
    # Compute magnitude
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    
    # Scale flow and magnitude back to original resolution if downsampled
    if scale < 1.0:
        flow = flow / scale
        magnitude = magnitude / scale

    return flow, magnitude


def compute_flow_statistics(
    flow: np.ndarray,
    magnitude: np.ndarray,
) -> Dict[str, float]:
    """Compute global statistics from optical flow field.
    
    Args:
        flow: [H, W, 2] optical flow field.
        magnitude: [H, W] flow magnitude.
        
    Returns:
        Dict with flow statistics.
    """
    h, w = magnitude.shape
    
    # Global mean flow vector
    mean_dx = float(np.mean(flow[..., 0]))
    mean_dy = float(np.mean(flow[..., 1]))
    mean_magnitude = float(np.mean(magnitude))
    max_magnitude = float(np.max(magnitude))
    
    # Flow direction (angle in radians, 0 = right, pi/2 = down)
    mean_angle = math.atan2(mean_dy, mean_dx)
    
    # Horizontal vs vertical ratio
    abs_dx = np.abs(flow[..., 0])
    abs_dy = np.abs(flow[..., 1])
    total_motion = np.sum(abs_dx) + np.sum(abs_dy) + 1e-6
    horizontal_ratio = float(np.sum(abs_dx) / total_motion)
    vertical_ratio = float(np.sum(abs_dy) / total_motion)
    
    # Detect radial pattern (zoom)
    cy, cx = h // 2, w // 2
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    radial_x = x_coords - cx
    radial_y = y_coords - cy
    radial_norm = np.sqrt(radial_x ** 2 + radial_y ** 2) + 1e-6
    radial_x_unit = radial_x / radial_norm
    radial_y_unit = radial_y / radial_norm
    
    # Dot product of flow with radial direction (positive = zoom out, negative = zoom in)
    radial_component = flow[..., 0] * radial_x_unit + flow[..., 1] * radial_y_unit
    mean_radial = float(np.mean(radial_component))
    radial_consistency = float(np.std(radial_component))
    
    return {
        "mean_dx": round(mean_dx, 3),
        "mean_dy": round(mean_dy, 3),
        "mean_magnitude": round(mean_magnitude, 3),
        "max_magnitude": round(max_magnitude, 3),
        "mean_angle": round(mean_angle, 4),
        "horizontal_ratio": round(horizontal_ratio, 3),
        "vertical_ratio": round(vertical_ratio, 3),
        "mean_radial": round(mean_radial, 3),
        "radial_consistency": round(radial_consistency, 3),
    }


# ---------------------------------------------------------------------------
# Camera motion classification
# ---------------------------------------------------------------------------
MOTION_TYPES = [
    "static",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "zoom_in",
    "zoom_out",
    "tracking",
    "complex",
]


def classify_camera_motion(
    flow_stats: Dict[str, float],
    magnitude_threshold: float = 5.0,
    zoom_radial_threshold: float = 2.0,
    direction_ratio_threshold: float = 0.65,
) -> Tuple[str, float]:
    """Classify camera motion type based on flow statistics.
    
    Args:
        flow_stats: Statistics from compute_flow_statistics().
        magnitude_threshold: Below this = static.
        zoom_radial_threshold: Radial component threshold for zoom detection.
        direction_ratio_threshold: H/V ratio threshold for pan/tilt.
        
    Returns:
        Tuple of (motion_type, confidence).
    """
    mean_mag = flow_stats["mean_magnitude"]
    mean_dx = flow_stats["mean_dx"]
    mean_dy = flow_stats["mean_dy"]
    h_ratio = flow_stats["horizontal_ratio"]
    v_ratio = flow_stats["vertical_ratio"]
    mean_radial = flow_stats["mean_radial"]
    radial_consistency = flow_stats["radial_consistency"]
    
    # Static: very low motion
    if mean_mag < magnitude_threshold:
        confidence = min(1.0, (magnitude_threshold - mean_mag) / magnitude_threshold)
        return "static", round(confidence, 2)
    
    # Check for zoom (radial pattern)
    # Zoom out: positive radial (flow points outward from center)
    # Zoom in: negative radial (flow points inward to center)
    if abs(mean_radial) > zoom_radial_threshold and radial_consistency < mean_mag * 0.5:
        if mean_radial > 0:
            motion_type = "zoom_out"
        else:
            motion_type = "zoom_in"
        confidence = min(1.0, abs(mean_radial) / (zoom_radial_threshold * 2))
        return motion_type, round(confidence, 2)
    
    # Check for pan (horizontal) vs tilt (vertical)
    if h_ratio > direction_ratio_threshold:
        # Horizontal dominant -> pan
        # Note: positive dx = image moves right = camera pans left
        # negative dx = image moves left = camera pans right
        if mean_dx > 0:
            motion_type = "pan_left"
        else:
            motion_type = "pan_right"
        confidence = min(1.0, h_ratio)
        return motion_type, round(confidence, 2)
    
    if v_ratio > direction_ratio_threshold:
        # Vertical dominant -> tilt
        # positive dy = image moves down = camera tilts up
        # negative dy = image moves up = camera tilts down
        if mean_dy > 0:
            motion_type = "tilt_up"
        else:
            motion_type = "tilt_down"
        confidence = min(1.0, v_ratio)
        return motion_type, round(confidence, 2)
    
    # Mixed motion: could be tracking or complex
    # If motion is consistent (low variance) it's likely tracking
    if radial_consistency < mean_mag * 0.3:
        return "tracking", 0.6
    
    return "complex", 0.5


# ---------------------------------------------------------------------------
# Video segment analysis
# ---------------------------------------------------------------------------
def analyze_video_camera_motion(
    video_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    sample_fps: float = 2.0,
    motion_threshold: float = 5.0,
) -> Dict[str, Any]:
    """Analyze camera motion across a video segment.
    
    Args:
        video_path: Absolute path to video file.
        start_time: Start time in seconds.
        end_time: End time in seconds (None = video end).
        sample_fps: Frame sampling rate.
        motion_threshold: Magnitude threshold for static detection.
        
    Returns:
        Dict with camera_events list, global_summary, and statistics.
    """
    info = get_video_info(video_path)
    duration = info["duration"]
    
    if end_time is None or end_time > duration:
        end_time = duration
    if start_time >= end_time:
        return {
            "success": False,
            "error": f"Invalid time range: {start_time} >= {end_time}",
        }
    
    # Generate timestamps
    timestamps = []
    t = start_time
    while t < end_time:
        timestamps.append(t)
        t += 1.0 / sample_fps
    if not timestamps:
        timestamps = [start_time]
    
    # Extract frames
    frames = extract_frames_at_times(video_path, timestamps)
    if len(frames) < 2:
        return {
            "success": False,
            "error": "Could not extract enough frames for analysis",
        }
    
    # Analyze frame pairs
    events = []
    for i in range(len(frames) - 1):
        ts_a, frame_a = frames[i]
        ts_b, frame_b = frames[i + 1]
        
        flow, magnitude = compute_optical_flow(frame_a, frame_b)
        stats = compute_flow_statistics(flow, magnitude)
        motion_type, confidence = classify_camera_motion(stats, motion_threshold)
        
        events.append({
            "start_time": round(ts_a, 3),
            "end_time": round(ts_b, 3),
            "type": motion_type,
            "confidence": confidence,
            "magnitude": stats["mean_magnitude"],
            "flow_stats": stats,
        })
    
    # Merge consecutive events of same type
    merged_events = merge_consecutive_events(events)
    
    # Compute summary statistics
    total_duration = end_time - start_time
    motion_counts = {}
    motion_durations = {}
    for ev in merged_events:
        t = ev["type"]
        dur = ev["end_time"] - ev["start_time"]
        motion_counts[t] = motion_counts.get(t, 0) + 1
        motion_durations[t] = motion_durations.get(t, 0) + dur
    
    # Find dominant motion
    dominant_motion = "static"
    max_dur = 0
    for t, d in motion_durations.items():
        if t != "static" and d > max_dur:
            max_dur = d
            dominant_motion = t
    
    static_ratio = motion_durations.get("static", 0) / total_duration if total_duration > 0 else 1.0
    
    # Generate summary text
    summary_parts = []
    for ev in merged_events:
        if ev["type"] != "static":
            summary_parts.append(
                f"{ev['type']} ({ev['start_time']:.1f}s-{ev['end_time']:.1f}s)"
            )
    if summary_parts:
        global_summary = "Camera motion: " + ", ".join(summary_parts)
    else:
        global_summary = "Camera is mostly static throughout the segment"
    
    return {
        "success": True,
        "camera_events": merged_events,
        "global_summary": global_summary,
        "dominant_motion": dominant_motion,
        "static_ratio": round(static_ratio, 3),
        "motion_counts": motion_counts,
        "duration": round(total_duration, 3),
    }


def merge_consecutive_events(
    events: List[Dict[str, Any]],
    same_type_gap: float = 0.1,
) -> List[Dict[str, Any]]:
    """Merge consecutive events of the same type.
    
    Args:
        events: List of event dicts with type, start_time, end_time.
        same_type_gap: Max gap to consider events consecutive.
        
    Returns:
        Merged event list.
    """
    if not events:
        return []
    
    merged = []
    current = events[0].copy()
    
    for ev in events[1:]:
        # Check if same type and consecutive
        if (ev["type"] == current["type"] and 
            ev["start_time"] - current["end_time"] <= same_type_gap):
            # Extend current event
            current["end_time"] = ev["end_time"]
            current["confidence"] = (current["confidence"] + ev["confidence"]) / 2
            current["magnitude"] = (current["magnitude"] + ev["magnitude"]) / 2
        else:
            # Save current and start new
            merged.append(current)
            current = ev.copy()
    
    merged.append(current)
    
    # Remove flow_stats from merged events (too verbose)
    for ev in merged:
        ev.pop("flow_stats", None)
    
    return merged


def detect_motion_between_times(
    video_path: str,
    time_a: float,
    time_b: float,
    motion_threshold: float = 5.0,
) -> Dict[str, Any]:
    """Detect camera motion type between two specific timestamps.
    
    Args:
        video_path: Absolute path to video file.
        time_a: First timestamp in seconds.
        time_b: Second timestamp in seconds.
        motion_threshold: Magnitude threshold for static detection.
        
    Returns:
        Dict with motion_type, confidence, and flow_stats.
    """
    frame_a = extract_frame_at_time(video_path, time_a)
    frame_b = extract_frame_at_time(video_path, time_b)
    
    if frame_a is None:
        return {"success": False, "error": f"Failed to extract frame at {time_a}s"}
    if frame_b is None:
        return {"success": False, "error": f"Failed to extract frame at {time_b}s"}
    
    flow, magnitude = compute_optical_flow(frame_a, frame_b)
    stats = compute_flow_statistics(flow, magnitude)
    motion_type, confidence = classify_camera_motion(stats, motion_threshold)
    
    return {
        "success": True,
        "motion_type": motion_type,
        "confidence": confidence,
        "time_a": time_a,
        "time_b": time_b,
        "flow_stats": stats,
    }
