#!/usr/bin/env python3
"""Thumbnail Navigator Utilities — Core functions for generating thumbnail grids."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into human-readable timestamp string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string like "MM:SS" or "HH:MM:SS"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def add_timestamp_label(
    image: np.ndarray,
    timestamp: float,
    font_scale: float = 0.5,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Add timestamp label at the bottom of an image.
    
    Args:
        image: Input image (BGR format)
        timestamp: Time in seconds
        font_scale: Font size scale
        alpha: Background transparency (0-1)
        
    Returns:
        Image with timestamp label
    """
    result = image.copy()
    h, w = result.shape[:2]
    
    # Format timestamp
    time_str = format_timestamp(timestamp)
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(time_str, font, font_scale, thickness)
    
    # Draw semi-transparent background at bottom
    padding = 4
    bg_top = h - text_h - 2 * padding - baseline
    
    overlay = result.copy()
    cv2.rectangle(overlay, (0, bg_top), (w, h), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    
    # Draw text centered
    text_x = (w - text_w) // 2
    text_y = h - padding - baseline
    cv2.putText(result, time_str, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return result


def extract_thumbnails(
    video_path: str,
    timestamps: List[float],
    width: int = 320
) -> List[Tuple[float, np.ndarray]]:
    """
    Extract thumbnails at specified timestamps from a video.
    
    Args:
        video_path: Path to the video file
        timestamps: List of timestamps in seconds
        width: Desired thumbnail width (height calculated to maintain aspect ratio)
        
    Returns:
        List of (timestamp, image) tuples
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS ({fps}) — video may be corrupted")
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if original_width <= 0 or original_height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions")
    
    # Calculate scaled height (maintain aspect ratio)
    scale = width / original_width
    height = int(original_height * scale)
    
    thumbnails = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            # Resize to thumbnail size
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            thumbnails.append((ts, resized))
    
    cap.release()
    return thumbnails


def create_grid(
    thumbnails: List[Tuple[float, np.ndarray]],
    columns: int = 5,
    show_timestamp: bool = True,
    gap: int = 2
) -> np.ndarray:
    """
    Create a grid image from thumbnails.
    
    Args:
        thumbnails: List of (timestamp, image) tuples
        columns: Number of columns in the grid
        show_timestamp: Whether to add timestamp labels
        gap: Gap between thumbnails in pixels
        
    Returns:
        Grid image (BGR format)
    """
    if not thumbnails:
        raise ValueError("No thumbnails to create grid")
    
    # Add timestamp labels if requested
    labeled_thumbnails = []
    for ts, img in thumbnails:
        if show_timestamp:
            img = add_timestamp_label(img, ts)
        labeled_thumbnails.append(img)
    
    # Get thumbnail dimensions (assume all same size)
    thumb_h, thumb_w = labeled_thumbnails[0].shape[:2]
    
    # Calculate grid dimensions
    rows = (len(labeled_thumbnails) + columns - 1) // columns
    
    # Create canvas with gap
    grid_w = columns * thumb_w + (columns - 1) * gap
    grid_h = rows * thumb_h + (rows - 1) * gap
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Place thumbnails
    for idx, img in enumerate(labeled_thumbnails):
        row = idx // columns
        col = idx % columns
        y = row * (thumb_h + gap)
        x = col * (thumb_w + gap)
        grid[y:y+thumb_h, x:x+thumb_w] = img
    
    return grid


def calculate_timestamps(
    start_time: float,
    end_time: float,
    interval: Optional[float] = None,
    max_thumbnails: int = 20
) -> Tuple[List[float], float]:
    """
    Calculate sampling timestamps based on parameters.
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        interval: Sampling interval in seconds (auto-calculated if None)
        max_thumbnails: Maximum number of thumbnails (used when interval is None)
        
    Returns:
        Tuple of (list of timestamps, actual interval used)
    """
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError(f"Invalid time range: {start_time}s to {end_time}s")
    
    # Auto-calculate interval if not specified
    if interval is None:
        interval = max(1.0, duration / max_thumbnails)
    
    # Generate timestamps
    timestamps = []
    t = start_time
    while t < end_time:
        timestamps.append(t)
        t += interval
    
    # Always include a frame near the end if we have room
    if timestamps and (end_time - timestamps[-1]) > interval * 0.5:
        timestamps.append(end_time - 0.1)  # Slightly before end to avoid edge issues
    
    return timestamps, interval


def get_video_info(video_path: str) -> dict:
    """
    Get basic video information.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dict with fps, duration, width, height, frame_count
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    duration = frame_count / fps if fps > 0 else 0
    
    return {
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }
