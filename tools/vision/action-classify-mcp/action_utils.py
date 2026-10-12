"""
Shared utilities for video action classification.

Two-tier backend:
  1. VideoMAE (preferred) — Kinetics-400 label classification via HuggingFace
  2. OpenCV motion analysis (fallback) — optical flow + motion pattern descriptors
"""

import os
import sys
import threading
from typing import Optional

import cv2
import numpy as np

from _base.device import select_torch_device

# -------------------------------------------------------------------------
# Backend state
# -------------------------------------------------------------------------
_model_lock = threading.Lock()
_videomae_pipeline = None
_backend = "uninitialized"  # "videomae" | "motion" | "uninitialized"


def _try_init_videomae():
    """Attempt to load VideoMAE pipeline. Returns pipeline or None."""
    try:
        import torch
        from transformers import pipeline
        from pathlib import Path

        model_name = os.environ.get(
            "VIDEOMAE_MODEL",
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )

        # Check if model is locally cached; if not, skip network download
        # to avoid hanging in network-restricted environments
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        model_local = Path(model_name)
        is_local = model_local.is_dir() and (model_local / "config.json").exists()

        if not is_local:
            # Check HF cache for this model
            hub_cache = Path(cache_dir) / "hub"
            model_slug = "models--" + model_name.replace("/", "--")
            cached = hub_cache / model_slug
            if not cached.is_dir():
                print(f"[action_utils] VideoMAE model '{model_name}' not in local cache, "
                      "skipping download attempt.", file=sys.stderr)
                return None

        # Force offline mode to prevent hanging on network requests
        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            pipe = pipeline(
                "video-classification",
                model=model_name,
                device=0 if select_torch_device(torch) == "cuda" else -1,
            )
        finally:
            if prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_offline

        if pipe.model is None:
            return None
        return pipe
    except Exception as exc:
        print(f"[action_utils] VideoMAE unavailable ({exc}), falling back to motion analysis.",
              file=sys.stderr)
        return None


def _init_backend():
    """Initialise the classification backend (called once, thread-safe)."""
    global _videomae_pipeline, _backend
    if _backend != "uninitialized":
        return
    with _model_lock:
        if _backend != "uninitialized":
            return
        pipe = _try_init_videomae()
        if pipe is not None:
            _videomae_pipeline = pipe
            _backend = "videomae"
        else:
            _backend = "motion"


def get_backend() -> str:
    _init_backend()
    return _backend


# -------------------------------------------------------------------------
# Frame sampling from video
# -------------------------------------------------------------------------
def sample_video_frames(video_path: str, start_time: float = 0,
                        end_time: Optional[float] = None,
                        num_frames: int = 16) -> list[tuple]:
    """Sample evenly-spaced frames from a video segment.

    Returns list of (time_sec, bgr_numpy) tuples.
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

    segment_duration = actual_end - start_time
    interval = segment_duration / max(num_frames, 1)

    frames = []
    for i in range(num_frames):
        t = start_time + i * interval
        if t > actual_end:
            t = actual_end
        target_ms = max(t * 1000.0, 0.0)
        cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        ret, frame = cap.read()
        if ret:
            frames.append((round(t, 3), frame))

    cap.release()
    return frames


# -------------------------------------------------------------------------
# VideoMAE classification
# -------------------------------------------------------------------------
def _classify_videomae(video_path: str, start_time: float, end_time: Optional[float],
                       top_k: int = 5) -> dict:
    """Classify action using VideoMAE pipeline."""
    import torch
    import tempfile

    # If we need a specific segment, extract it first
    # Otherwise, use the full video path directly
    if start_time > 0 or end_time is not None:
        # Need to extract segment - use frames approach
        frames_data = sample_video_frames(video_path, start_time, end_time, num_frames=16)
        if not frames_data:
            return {"success": False, "error": "No frames extracted"}

        # Convert BGR→RGB numpy arrays
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in frames_data]

        # Save frames as temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_video_path = tmp_file.name

        try:
            # Write frames to temporary video
            height, width = rgb_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_video_path, fourcc, 16.0, (width, height))

            for frame in rgb_frames:
                # Convert RGB back to BGR for cv2.VideoWriter
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            out.release()

            # Use the temporary video file
            results = _videomae_pipeline(tmp_video_path, top_k=top_k)
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)
    else:
        # Use full video directly
        results = _videomae_pipeline(video_path, top_k=top_k)

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    actions = []
    for r in results:
        actions.append({
            "label": r["label"],
            "score": round(float(r["score"]), 4),
        })

    return {
        "success": True,
        "backend": "videomae",
        "actions": actions,
    }


# -------------------------------------------------------------------------
# Motion-based analysis (fallback)
# -------------------------------------------------------------------------
# Descriptive motion labels ordered by activity level
_MOTION_LABELS = {
    "stationary": "Scene is nearly static with minimal motion",
    "subtle_motion": "Subtle motion detected (talking head, small gestures)",
    "slow_movement": "Slow deliberate movement (walking, gentle actions)",
    "moderate_activity": "Moderate activity (working, interacting, sports warmup)",
    "fast_action": "Fast-paced action (running, sports play, dance)",
    "very_fast_action": "Very rapid motion (fighting, extreme sports, fast cuts)",
    "camera_pan": "Dominant camera panning motion (landscape sweep, tracking shot)",
    "camera_zoom": "Camera zoom in/out detected",
    "scene_transition": "Multiple scene transitions (montage, slideshow)",
}


def _compute_optical_flow_stats(frames: list[np.ndarray]) -> dict:
    """Compute optical flow statistics across consecutive frame pairs."""
    if len(frames) < 2:
        return {"avg_magnitude": 0, "max_magnitude": 0, "flow_direction": "none",
                "temporal_variance": 0, "spatial_uniformity": 0}

    magnitudes = []
    directions = []
    spatial_stds = []

    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        # Resize for speed
        h, w = gray1.shape
        scale = min(320.0 / max(h, w), 1.0)
        if scale < 1.0:
            gray1 = cv2.resize(gray1, None, fx=scale, fy=scale)
            gray2 = cv2.resize(gray2, None, fx=scale, fy=scale)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        avg_mag = float(np.mean(mag))
        magnitudes.append(avg_mag)

        # Dominant direction (weighted by magnitude)
        if avg_mag > 0.5:
            weights = mag.flatten()
            angles = ang.flatten()
            # Circular weighted mean to handle 0/360 boundary correctly
            angles_rad = np.deg2rad(angles)
            mean_sin = np.average(np.sin(angles_rad), weights=weights + 1e-8)
            mean_cos = np.average(np.cos(angles_rad), weights=weights + 1e-8)
            mean_angle = float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360)
            directions.append(mean_angle)

        # Spatial uniformity: std of flow magnitude across image
        spatial_stds.append(float(np.std(mag)))

    avg_mag = float(np.mean(magnitudes)) if magnitudes else 0
    max_mag = float(np.max(magnitudes)) if magnitudes else 0
    temporal_var = float(np.std(magnitudes)) if len(magnitudes) > 1 else 0
    spatial_uniformity = 1.0 - min(float(np.mean(spatial_stds)) / max(avg_mag + 1e-8, 1), 1.0)

    # Determine dominant direction
    if not directions or avg_mag < 0.5:
        dominant_dir = "stationary"
    else:
        # Circular mean to handle 0/360 boundary correctly
        dirs_rad = np.deg2rad(directions)
        mean_dir = float(np.rad2deg(np.arctan2(
            np.mean(np.sin(dirs_rad)),
            np.mean(np.cos(dirs_rad))
        )) % 360)
        if 45 <= mean_dir < 135:
            dominant_dir = "downward"
        elif 135 <= mean_dir < 225:
            dominant_dir = "leftward"
        elif 225 <= mean_dir < 315:
            dominant_dir = "upward"
        else:
            dominant_dir = "rightward"

    return {
        "avg_magnitude": round(avg_mag, 3),
        "max_magnitude": round(max_mag, 3),
        "flow_direction": dominant_dir,
        "temporal_variance": round(temporal_var, 3),
        "spatial_uniformity": round(spatial_uniformity, 3),
    }


def _compute_frame_differences(frames: list[np.ndarray]) -> dict:
    """Compute structural differences between consecutive frames."""
    if len(frames) < 2:
        return {"avg_diff": 0, "max_diff": 0, "scene_changes": 0}

    diffs = []
    scene_change_count = 0
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        h, w = gray1.shape
        scale = min(320.0 / max(h, w), 1.0)
        if scale < 1.0:
            gray1 = cv2.resize(gray1, None, fx=scale, fy=scale)
            gray2 = cv2.resize(gray2, None, fx=scale, fy=scale)

        diff = cv2.absdiff(gray1, gray2)
        mean_diff = float(np.mean(diff))
        diffs.append(mean_diff)

        if mean_diff > 40:  # Likely scene change
            scene_change_count += 1

    return {
        "avg_diff": round(float(np.mean(diffs)), 2) if diffs else 0,
        "max_diff": round(float(np.max(diffs)), 2) if diffs else 0,
        "scene_changes": scene_change_count,
    }


def _classify_motion_pattern(flow_stats: dict, diff_stats: dict) -> list[dict]:
    """Classify motion pattern from flow and difference statistics."""
    results = []
    avg_mag = flow_stats["avg_magnitude"]
    max_mag = flow_stats["max_magnitude"]
    temporal_var = flow_stats["temporal_variance"]
    spatial_uni = flow_stats["spatial_uniformity"]
    avg_diff = diff_stats["avg_diff"]
    scene_changes = diff_stats["scene_changes"]

    # Scene transition detection
    if scene_changes >= 3:
        results.append({"label": "scene_transition", "score": min(0.9, 0.5 + scene_changes * 0.1)})

    # Camera motion detection (high spatial uniformity = camera motion)
    if spatial_uni > 0.7 and avg_mag > 2.0:
        if flow_stats["flow_direction"] in ("leftward", "rightward"):
            results.append({"label": "camera_pan", "score": round(min(0.95, 0.6 + spatial_uni * 0.3), 4)})
        elif flow_stats["flow_direction"] in ("upward", "downward"):
            results.append({"label": "camera_zoom", "score": round(min(0.85, 0.5 + spatial_uni * 0.3), 4)})

    # Activity level classification
    if avg_mag < 0.3 and avg_diff < 3:
        results.append({"label": "stationary", "score": 0.95})
    elif avg_mag < 1.0:
        results.append({"label": "subtle_motion", "score": round(0.8 - avg_mag * 0.3, 4)})
    elif avg_mag < 2.5:
        results.append({"label": "slow_movement", "score": round(0.7, 4)})
    elif avg_mag < 5.0:
        results.append({"label": "moderate_activity", "score": round(0.7, 4)})
    elif avg_mag < 10.0:
        results.append({"label": "fast_action", "score": round(0.7, 4)})
    else:
        results.append({"label": "very_fast_action", "score": round(0.7, 4)})

    # High temporal variance → intermittent action
    if temporal_var > 2.0 and avg_mag > 1.0:
        results.append({"label": "intermittent_activity", "score": round(min(0.8, 0.4 + temporal_var * 0.1), 4)})

    # Sort by score descending, add descriptions
    results.sort(key=lambda x: x["score"], reverse=True)
    for r in results:
        r["description"] = _MOTION_LABELS.get(r["label"], r["label"])

    return results[:5]


def _classify_motion(video_path: str, start_time: float, end_time: Optional[float],
                     top_k: int = 5) -> dict:
    """Classify action using optical flow motion analysis."""
    frames_data = sample_video_frames(video_path, start_time, end_time, num_frames=16)
    if not frames_data:
        return {"success": False, "error": "No frames extracted"}

    bgr_frames = [f for _, f in frames_data]
    flow_stats = _compute_optical_flow_stats(bgr_frames)
    diff_stats = _compute_frame_differences(bgr_frames)
    actions = _classify_motion_pattern(flow_stats, diff_stats)

    return {
        "success": True,
        "backend": "motion_analysis",
        "actions": actions[:top_k],
        "motion_stats": {
            "optical_flow": flow_stats,
            "frame_difference": diff_stats,
        },
        "note": (
            "Using motion analysis fallback (VideoMAE model not available). "
            "Labels describe motion patterns rather than specific Kinetics-400 actions."
        ),
    }


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def classify_action(video_path: str, start_time: float = 0,
                    end_time: Optional[float] = None,
                    top_k: int = 5) -> dict:
    """Classify action in a video segment (auto-selects backend)."""
    _init_backend()

    if not os.path.isfile(video_path):
        return {"success": False, "error": f"File not found: {video_path}"}

    if _backend == "videomae":
        return _classify_videomae(video_path, start_time, end_time, top_k)
    else:
        return _classify_motion(video_path, start_time, end_time, top_k)


def classify_multi_segments(video_path: str, segments: list[dict],
                            top_k: int = 5) -> dict:
    """Classify actions in multiple video segments."""
    _init_backend()

    if not os.path.isfile(video_path):
        return {"success": False, "error": f"File not found: {video_path}"}

    results = []
    for seg in segments:
        start = float(seg.get("start_time", 0))
        end_raw = seg.get("end_time")
        end = float(end_raw) if end_raw is not None else None
        label = seg.get("label", f"{start}-{end}")

        if _backend == "videomae":
            r = _classify_videomae(video_path, start, end, top_k)
        else:
            r = _classify_motion(video_path, start, end, top_k)

        r["segment"] = {"start_time": start, "end_time": end, "label": label}
        results.append(r)

    return {
        "success": True,
        "video_path": video_path,
        "total_segments": len(results),
        "segment_results": results,
    }