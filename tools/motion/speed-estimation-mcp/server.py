#!/usr/bin/env python3
"""Speed Estimation MCP Server — Estimate speed and acceleration from trajectories."""
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

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response

# ── Business logic (from speed_estimation_server.py) ──

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# MCP Server init
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Speed unit parsing
# ---------------------------------------------------------------------------
_SPEED_PATTERN = re.compile(
    r"([0-9]*\.?[0-9]+)\s*(km/h|kmh|kph|m/s|mph|ft/s|knots?|km/s)",
    re.IGNORECASE,
)

_TO_M_PER_S = {
    "km/h": 1 / 3.6,
    "kmh": 1 / 3.6,
    "kph": 1 / 3.6,
    "mph": 0.44704,
    "m/s": 1.0,
    "ft/s": 0.3048,
    "knot": 0.514444,
    "knots": 0.514444,
    "km/s": 1000.0,
}


def parse_speed(speed_str: str) -> Optional[Tuple[float, str, float]]:
    """Parse '133km/h' → (133.0, 'km/h', 36.944 m/s). Returns None on failure."""
    m = _SPEED_PATTERN.search(speed_str)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    factor = _TO_M_PER_S.get(unit, 1.0)
    return value, m.group(2), value * factor


def m_per_s_to_multi(v: float) -> Dict[str, float]:
    """Convert m/s to common units."""
    return {
        "m/s": round(v, 3),
        "km/h": round(v * 3.6, 2),
        "mph": round(v / 0.44704, 2),
    }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def _dist(c1: List[float], c2: List[float]) -> float:
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def compute_kinematics(
    trajectory: List[Dict],
    fps: Optional[float] = None,
    scale: Optional[float] = None,  # meters per pixel
    frame_size: Optional[Tuple[int, int]] = None,  # (width, height) for normalized thresholds
) -> Dict[str, Any]:
    """
    Compute speed and acceleration profiles from trajectory.

    Args:
        trajectory: list of {time?, center:[x,y], bbox?, area?}
        fps: used to generate time if 'time' field absent
        scale: meters per pixel (for real-world conversion)
        frame_size: (width, height) in pixels; used to normalize motion state
                    thresholds so classification is resolution-independent.

    Returns:
        Full kinematics dict.
    """
    n = len(trajectory)
    if n < 2:
        return {
            "num_points": n,
            "speed_profile_px": [],
            "acceleration_profile_px": [],
            "avg_speed_px_per_sec": 0.0,
            "max_speed_px_per_sec": 0.0,
            "avg_acceleration_px_per_sec2": 0.0,
            "max_acceleration_px_per_sec2": 0.0,
            "motion_state": "stationary",
            "speed_classification": "stationary",
        }

    # --- Build time array ---
    first = trajectory[0]
    has_time = isinstance(first, dict) and "timestamp" in first and first["timestamp"] is not None
    if has_time:
        times = []
        for i, t in enumerate(trajectory):
            if isinstance(t, dict) and "timestamp" in t and t["timestamp"] is not None:
                times.append(float(t["timestamp"]))
            elif fps and fps > 0:
                times.append(i / fps)
            else:
                times.append(float(i))
    elif fps and fps > 0:
        times = [i / fps for i in range(n)]
    else:
        # Assume 1 second between points
        times = list(range(n))

    centers = []
    for t in trajectory:
        # Support both list/tuple format (from YAML schema) and dict format (legacy)
        if isinstance(t, (list, tuple)):
            if len(t) == 4:
                c = [(t[0] + t[2]) / 2, (t[1] + t[3]) / 2]
            elif len(t) >= 2:
                c = [float(t[0]), float(t[1])]
            else:
                c = [0, 0]
        elif isinstance(t, dict):
            c = t.get("center")
            if c is None:
                bbox = t.get("bbox")
                if bbox and len(bbox) == 4:
                    c = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                else:
                    c = [0, 0]
        else:
            c = [0, 0]
        centers.append([float(c[0]), float(c[1])])

    # --- Speed profile (px/s) ---
    speed_profile = []
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        d = _dist(centers[i - 1], centers[i])
        v = d / dt if dt > 0 else 0.0
        t_mid = (times[i - 1] + times[i]) / 2
        speed_profile.append({
            "timestamp": round(t_mid, 4),
            "speed_px_per_sec": round(v, 2),
        })

    # --- Acceleration profile (px/s²) ---
    accel_profile = []
    for i in range(1, len(speed_profile)):
        dt = speed_profile[i]["timestamp"] - speed_profile[i - 1]["timestamp"]
        dv = speed_profile[i]["speed_px_per_sec"] - speed_profile[i - 1]["speed_px_per_sec"]
        a = dv / dt if dt > 0 else 0.0
        t_mid = (speed_profile[i - 1]["timestamp"] + speed_profile[i]["timestamp"]) / 2
        accel_profile.append({
            "timestamp": round(t_mid, 4),
            "acceleration_px_per_sec2": round(a, 2),
        })

    speeds_px = [s["speed_px_per_sec"] for s in speed_profile]
    accels_px = [a["acceleration_px_per_sec2"] for a in accel_profile] if accel_profile else [0.0]

    avg_speed = sum(speeds_px) / len(speeds_px) if speeds_px else 0.0
    max_speed = max(speeds_px) if speeds_px else 0.0
    min_speed = min(speeds_px) if speeds_px else 0.0
    avg_accel = sum(accels_px) / len(accels_px) if accels_px else 0.0
    max_accel = max(accels_px) if accels_px else 0.0
    min_accel = min(accels_px) if accels_px else 0.0

    # --- Motion state ---
    # Normalize thresholds to be resolution-independent.
    # When frame_size is available, express thresholds as fraction of frame diagonal.
    # When scale (m/px) is available, use real-world m/s values directly.
    if scale and scale > 0:
        # Real-world path: use absolute m/s thresholds (resolution-independent)
        avg_speed_real = avg_speed * scale
        avg_accel_real = avg_accel * scale
        if avg_speed_real < 0.05:  # < 0.05 m/s ≈ 0.18 km/h
            motion_state = "stationary"
        elif avg_accel_real > 1.0:  # > 1.0 m/s²
            motion_state = "accelerating"
        elif avg_accel_real < -1.0:  # < -1.0 m/s²
            motion_state = "decelerating"
        else:
            motion_state = "constant_speed"
    elif frame_size:
        # Pixel path with resolution normalization: thresholds as % of frame diagonal
        diag = math.sqrt(frame_size[0] ** 2 + frame_size[1] ** 2)
        stationary_thresh = diag * 0.004   # ~0.4% of diagonal per second
        accel_thresh = diag * 0.008        # ~0.8% of diagonal per second²
        if avg_speed < stationary_thresh:
            motion_state = "stationary"
        elif avg_accel > accel_thresh:
            motion_state = "accelerating"
        elif avg_accel < -accel_thresh:
            motion_state = "decelerating"
        else:
            motion_state = "constant_speed"
    else:
        # Fallback: absolute pixel thresholds (legacy, not resolution-aware)
        if avg_speed < 5.0:
            motion_state = "stationary"
        elif avg_accel > 10.0:
            motion_state = "accelerating"
        elif avg_accel < -10.0:
            motion_state = "decelerating"
        else:
            motion_state = "constant_speed"

    # --- Real-world conversion if scale is available ---
    result = {
        "num_points": n,
        "duration": round(times[-1] - times[0], 3),
        "speed_profile_px": speed_profile,
        "acceleration_profile_px": accel_profile,
        "avg_speed_px_per_sec": round(avg_speed, 2),
        "max_speed_px_per_sec": round(max_speed, 2),
        "min_speed_px_per_sec": round(min_speed, 2),
        "avg_acceleration_px_per_sec2": round(avg_accel, 2),
        "max_acceleration_px_per_sec2": round(max_accel, 2),
        "min_acceleration_px_per_sec2": round(min_accel, 2),
        "motion_state": motion_state,
    }

    if scale and scale > 0:
        # Convert px → meters
        avg_speed_m = avg_speed * scale
        max_speed_m = max_speed * scale
        avg_accel_m = avg_accel * scale
        max_accel_m = max_accel * scale

        result["real_world"] = {
            "scale_m_per_px": round(scale, 6),
            "avg_speed": m_per_s_to_multi(avg_speed_m),
            "max_speed": m_per_s_to_multi(max_speed_m),
            "avg_acceleration_m_per_sec2": round(avg_accel_m, 3),
            "max_acceleration_m_per_sec2": round(max_accel_m, 3),
        }

        # Enrich profiles with real-world values
        for sp in result["speed_profile_px"]:
            sp["speed_m_per_sec"] = round(sp["speed_px_per_sec"] * scale, 3)
            sp["speed_km_per_h"] = round(sp["speed_px_per_sec"] * scale * 3.6, 2)
        for ap in result["acceleration_profile_px"]:
            ap["acceleration_m_per_sec2"] = round(ap["acceleration_px_per_sec2"] * scale, 3)

        result["speed_classification"] = _classify_speed(avg_speed_m)
    else:
        result["speed_classification"] = "unknown (no real-world calibration)"

    return result


def _classify_speed(speed_m_s: float) -> str:
    """Classify speed into human-readable categories."""
    kmh = speed_m_s * 3.6
    if kmh < 1:
        return "stationary"
    elif kmh < 5:
        return "very_slow (walking pace)"
    elif kmh < 15:
        return "slow (jogging/cycling)"
    elif kmh < 50:
        return "medium (vehicle in city)"
    elif kmh < 120:
        return "fast (highway speed)"
    elif kmh < 300:
        return "very_fast (high-speed rail / tennis serve)"
    else:
        return "extreme (aircraft / projectile)"


# =========================================================================
# Tool handler
# =========================================================================
async def _handle_estimate_speed(arguments: dict) -> dict:
    trajectory_raw = arguments.get("trajectory")
    fps = arguments.get("fps")
    known_speed = arguments.get("known_speed")
    reference_length_px = arguments.get("reference_length_px")
    reference_length_m = arguments.get("reference_length_m")
    frame_width = arguments.get("frame_width")
    frame_height = arguments.get("frame_height")

    # Build frame_size tuple if both dimensions provided
    frame_size = None
    if frame_width is not None and frame_height is not None:
        frame_size = (int(frame_width), int(frame_height))

    # --- Parse trajectory ---
    if trajectory_raw is None:
        return {"success": False, "error": "trajectory is required"}

    if isinstance(trajectory_raw, str):
        try:
            trajectory = json.loads(trajectory_raw)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid trajectory JSON: {e}"}
    else:
        trajectory = trajectory_raw

    if not isinstance(trajectory, list) or len(trajectory) < 2:
        return {"success": False, "error": "trajectory must be a list with at least 2 points"}

    if fps is not None:
        fps = float(fps)

    # --- Determine scale (meters per pixel) ---
    scale = None

    # Method 1: from known reference speed + trajectory pixel speed
    if known_speed and isinstance(known_speed, str):
        parsed = parse_speed(known_speed)
        if parsed:
            ref_value, ref_unit, ref_m_per_s = parsed
            # Quick pre-compute pixel speed to derive scale
            pre_result = compute_kinematics(trajectory, fps=fps, scale=None)
            avg_px_speed = pre_result["avg_speed_px_per_sec"]
            if avg_px_speed > 0:
                scale = ref_m_per_s / avg_px_speed
        else:
            return {"success": False, "error": f"Cannot parse known_speed: '{known_speed}'. Use format like '133km/h', '60mph', '36.9m/s'"}

    # Method 2: from reference object size
    if scale is None and reference_length_px and reference_length_m:
        ref_px = float(reference_length_px)
        ref_m = float(reference_length_m)
        if ref_px > 0:
            scale = ref_m / ref_px

    # --- Compute ---
    result = compute_kinematics(trajectory, fps=fps, scale=scale, frame_size=frame_size)

    # --- Generate summary ---
    summary_parts = []
    summary_parts.append(f"Trajectory: {result['num_points']} points over {result['duration']}s")
    summary_parts.append(f"Pixel speed: avg={result['avg_speed_px_per_sec']} px/s, max={result['max_speed_px_per_sec']} px/s")

    if "real_world" in result:
        rw = result["real_world"]
        summary_parts.append(
            f"Real-world speed: avg={rw['avg_speed']['km/h']} km/h, max={rw['max_speed']['km/h']} km/h"
        )
        summary_parts.append(
            f"Acceleration: avg={rw['avg_acceleration_m_per_sec2']} m/s², max={rw['max_acceleration_m_per_sec2']} m/s²"
        )

    summary_parts.append(f"Motion: {result['motion_state']}, Classification: {result['speed_classification']}")

    result["summary"] = " | ".join(summary_parts)
    result["success"] = True

    # Truncate large profiles for readability (keep first 20 + last 5)
    for key in ("speed_profile_px", "acceleration_profile_px"):
        profile = result.get(key, [])
        if len(profile) > 30:
            result[key] = profile[:20] + [{"...": f"({len(profile) - 25} entries omitted)"}] + profile[-5:]

    return result


# =========================================================================
# Tool Registration
# =========================================================================
# =========================================================================
# Tool Call Router
# =========================================================================
TOOL_HANDLERS = {
    "estimate_speed": _handle_estimate_speed,
}


# =========================================================================
# Entry point
# =========================================================================


async def handle_speed_estimate(arguments: dict) -> dict:
    return await _handle_estimate_speed(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="speed-estimation",
        tool_defs_source="_tool_defs/misc.yaml",
        handlers={
            "speed_estimate": handle_speed_estimate,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
