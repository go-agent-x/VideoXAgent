#!/usr/bin/env python3
"""VLM Query Interval MCP Server — Unified multi-frame VLM query on video intervals."""
import asyncio
import base64
import json
import os
import sys
import traceback
import subprocess
from pathlib import Path
from typing import Optional
 
sys.path.insert(0, str(Path(__file__).parent))
# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")
 
from _base.server_framework import create_mcp_server, run_server, make_error_response
from _base.config import get_config, pick_api_key, normalize_vlm_url
from _base.vlm_objectivity import build_objective_vlm_prompt
from _base.subtitle_utils import (
    parse_timestamp,
    format_seconds_to_timestamp,
    find_subtitle_file,
    parse_subtitle_file,
    filter_subtitles_by_time_range,
    format_subtitles_as_text,
)
 
import cv2
import httpx
import numpy as np
 
# ---------------------------------------------------------------------------
# Configuration from _base/config.py
# ---------------------------------------------------------------------------
_vlm_cfg = get_config("vlm")
VLM_API_URL = normalize_vlm_url(_vlm_cfg["api_url"])
VLM_API_KEY = pick_api_key("VLM_API_KEY")
VLM_MODEL_NAME = _vlm_cfg["model"]
VLM_REPORT_USAGE = os.environ.get("VLM_REPORT_USAGE", "").lower() == "true"
VLM_MODEL_FALLBACKS = tuple(f.strip() for f in _vlm_cfg["fallbacks"].split(",") if f.strip())
 
MAX_IMAGE_DIMENSION = 2048
MAX_SAMPLE_FRAMES = 32
VLM_TIMEOUT = 360.0
MAX_VLM_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
FALLBACK_STATUS_CODES = {400, 401, 403, 404, 415, 422}
FALLBACK_ERROR_MARKERS = (
    "content policy",
    "resource has been temporarily blocked",
    "does not support",
    "vision",
    "image",
    "multimodal",
    "invalid model",
    "model_not_found",
    "not found",
    "unsupported",
)
 
 
def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default
 
 
def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default
 
 
QUERY_SEGMENT_DETAIL_CONCURRENCY = _env_int("QSD_TOOL_CONCURRENCY", 2)
QUERY_SEGMENT_DETAIL_TIMEOUT = _env_float("QSD_TOOL_TIMEOUT", 240.0)
 
_QUERY_SEGMENT_DETAIL_SEMAPHORE = asyncio.Semaphore(QUERY_SEGMENT_DETAIL_CONCURRENCY)
 
 
# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------
def _get_duration(video_path: str) -> float:
    """Return video duration in seconds. Uses ffprobe for accuracy (handles VFR),
    falls back to OpenCV frame_count/fps if ffprobe is unavailable."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    # Fallback: OpenCV (less accurate for variable-frame-rate video)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS: {video_path}")
    return n / fps
 
 
def _sample_timestamps_for_mode(start: float, end: float, n: int) -> list[float]:
    """Sample timestamps with mixed-mode strategy: near boundaries + evenly distributed middle.
 
    Uses a small edge_pad to avoid landing exactly on cut frames (which may be black
    or transitional), while still capturing the interval's boundaries.
    """
    dur = end - start
    if n <= 1:
        return [start + dur / 2]
    edge_pad = min(0.25, dur * 0.05)
    first = start + edge_pad
    last = end - edge_pad
    if last <= first:
        first, last = start, end
    if n == 2:
        return [first, last]
    step = (last - first) / (n - 1)
    return [first + step * i for i in range(n)]
 
 
def _extract_frames_uniform(
    video_path: str, start: float, end: float, n: int
) -> list[tuple[float, np.ndarray]]:
    """Extract n frames from [start, end] using boundary-aware sampling.
 
    Uses CAP_PROP_POS_MSEC (PTS-based seek) rather than CAP_PROP_POS_FRAMES so that
    variable-frame-rate (VFR) video is handled correctly.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    if end <= start:
        cap.release()
        return []
    timestamps = _sample_timestamps_for_mode(start, end, n)
    frames = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append((ts, frame))
    cap.release()
    return frames
 
 
def _extract_frames_at_timestamps(
    video_path: str, timestamps: list[float]
) -> list[tuple[float, np.ndarray]]:
    """Extract frames at exact specified timestamps (seconds).
 
    Uses CAP_PROP_POS_MSEC for VFR safety.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames = []
    for ts in sorted(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append((ts, frame))
    cap.release()
    return frames
 
 
def _to_b64(frame: np.ndarray, quality: int = 85) -> str:
    h, w = frame.shape[:2]
    if max(h, w) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")
 
 
# ---------------------------------------------------------------------------
# Prompt configuration
# ---------------------------------------------------------------------------
_TASK_INSTRUCTION = (
    "Inspect this known video interval using the sampled frames. Answer the query "
    "with the level of temporal reasoning it requires: describe visible content "
    "and, when relevant, observed changes or progression across the sampled frames."
)
 
 
# ---------------------------------------------------------------------------
# VLM API with retry + model fallback
# ---------------------------------------------------------------------------
def _candidate_models() -> list[str]:
    candidates: list[str] = []
    seen = set()
    for model in (VLM_MODEL_NAME, *VLM_MODEL_FALLBACKS):
        normalized = model.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(normalized)
    return candidates
 
 
def _response_text(resp: httpx.Response) -> str:
    try:
        payload = resp.json()
    except Exception:
        return resp.text
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        message = payload.get("message")
        if isinstance(message, str) and message:
            return message
    return json.dumps(payload, ensure_ascii=False)
 
 
def _should_try_fallback(status_code: int, detail: str) -> bool:
    lowered = detail.lower()
    return status_code in FALLBACK_STATUS_CODES or any(
        marker in lowered for marker in FALLBACK_ERROR_MARKERS
    )
 
 
async def _call_vlm_with_retry(
    content: list[dict], max_tokens: int
) -> dict:
    """Call VLM API with retry logic and model fallback chain."""
    headers = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"
 
    result = {
        "response": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None,
        "model": None,
        "attempted_models": [],
    }
 
    async with httpx.AsyncClient(timeout=VLM_TIMEOUT) as client:
        for model_name in _candidate_models():
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": os.environ.get("VLM_ENABLE_THINKING", "false").lower() == "true"},
            }
            result["attempted_models"].append(model_name)
 
            for attempt in range(1, MAX_VLM_RETRIES + 1):
                try:
                    resp = await client.post(
                        VLM_API_URL, json=payload, headers=headers
                    )
                    if (
                        resp.status_code in RETRYABLE_STATUS_CODES
                        and attempt < MAX_VLM_RETRIES
                    ):
                        await asyncio.sleep(0.6 * attempt)
                        continue
 
                    resp.raise_for_status()
                    data = resp.json()
 
                    if "usage" in data:
                        usage = data["usage"]
                        result["input_tokens"] = usage.get("prompt_tokens", 0)
                        result["output_tokens"] = usage.get("completion_tokens", 0)
 
                    if "choices" in data and len(data["choices"]) > 0:
                        result["response"] = data["choices"][0]["message"]["content"]
                        result["model"] = model_name
                        result["error"] = None
                    else:
                        # Log full payload to stderr; return short error to agent.
                        print(
                            f"[vlm-query-sequence] invalid API response model={model_name} "
                            f"payload={json.dumps(data, ensure_ascii=False)[:500]}",
                            file=sys.stderr,
                            flush=True,
                        )
                        result["response"] = ""
                        result["model"] = model_name
                        result["error"] = "Invalid API response format"
                    break
 
                except (httpx.TimeoutException, httpx.CloseError) as e:
                    if attempt < MAX_VLM_RETRIES:
                        await asyncio.sleep(0.6 * attempt)
                        continue
                    error_type = "timeout" if isinstance(e, httpx.TimeoutException) else "connection closed"
                    print(
                        f"[vlm-query-sequence] VLM {error_type} after {VLM_TIMEOUT}s model={model_name} err={e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result["response"] = ""
                    result["error"] = f"API {error_type}"
                except httpx.HTTPStatusError as e:
                    detail = _response_text(e.response)[:500].replace("\n", " ")
                    print(
                        f"[vlm-query-sequence] VLM HTTP {e.response.status_code} "
                        f"model={model_name} detail={detail}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result["response"] = ""
                    result["error"] = f"HTTP {e.response.status_code}"
                    # Only skip retries and move on to the fallback model when
                    # the error signals a permanent incompatibility with this
                    # model (auth / unsupported / content-policy). For other
                    # non-retryable 4xx/5xx we let the retry loop finish on the
                    # same model before the outer loop advances.
                    if _should_try_fallback(e.response.status_code, detail):
                        break
                except Exception as e:
                    print(
                        f"[vlm-query-sequence] VLM call failed model={model_name} err={e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result["response"] = ""
                    result["error"] = str(e)
                    break
 
            if result["error"] is None:
                break
 
    return result
 
 
# ---------------------------------------------------------------------------
# Temporal-aware prompt builder
# ---------------------------------------------------------------------------
def _build_content(
    frames: list[tuple[float, np.ndarray]],
    query: str,
    interval_duration: float,
    total_duration: float,
    start_time: float,
    end_time: float,
    sampling_start_time: float,
    sampling_end_time: float,
    subtitle_text: str,
) -> list[dict]:
    """Build VLM content with temporal prompt and interleaved frames."""
    n = len(frames)
    s_str = format_seconds_to_timestamp(start_time)
    e_str = format_seconds_to_timestamp(end_time)
    t_str = format_seconds_to_timestamp(total_duration)
    context_parts = [
        "Strict chronological frame sequence.",
        f"Frames: {n}.",
        f"Segment duration: {interval_duration:.1f}s.",
        f"Segment time range: {s_str} - {e_str}.",
        f"Total video duration: {t_str}.",
    ]
    # Only surface the sampling window separately when it differs from the
    # requested segment (typically explicit_timestamps that was clamped).
    if (sampling_start_time, sampling_end_time) != (start_time, end_time):
        ss_str = format_seconds_to_timestamp(sampling_start_time)
        se_str = format_seconds_to_timestamp(sampling_end_time)
        context_parts.append(f"Frame sampling window: {ss_str} - {se_str}.")
    context_parts.append(
        "The sequence is sampled and may not include every important frame."
    )
    context = " ".join(context_parts)
    evidence_parts = []
    if subtitle_text:
        evidence_parts.append(f"Subtitles during this interval:\n{subtitle_text}")
    evidence = "\n\n".join(evidence_parts)
 
    temporal_prompt = build_objective_vlm_prompt(
        task_instruction=_TASK_INSTRUCTION,
        user_query=query,
        context=context,
        evidence=evidence,
    )
 
    content: list[dict] = [{"type": "text", "text": temporal_prompt}]
 
    for i, (ts, frame) in enumerate(frames):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_to_b64(frame)}"},
            }
        )
        label = f"Frame {i + 1}/{n} at {format_seconds_to_timestamp(ts)}"
        content.append({"type": "text", "text": label})
 
    return content
 
 
# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
async def _video_vlm_query(
    video_path: str,
    start_time: float,
    end_time: float,
    query: str,
    max_frames: Optional[int] = None,
    subtitle_dir: str = "",
    frame_timestamps: Optional[list] = None,
) -> dict:
    def _err(msg: str) -> dict:
        return {"success": False, "response": "", "error": msg}
 
    if not video_path:
        return _err("video_path is required")
    if not query:
        return _err("query is required")
    if not os.path.isfile(video_path):
        return _err(f"Video not found: {video_path}")
 
    start_time = parse_timestamp(start_time)
 
    try:
        total_duration = await asyncio.to_thread(_get_duration, video_path)
    except Exception as e:
        return _err(f"Cannot get duration: {e}")
 
    # Default end_time to video duration when not provided
    if end_time is None:
        end_time = total_duration
    else:
        end_time = parse_timestamp(end_time)
 
    # Clamp to video bounds BEFORE comparison to avoid post-clamp inversion
    # (e.g. start_time=90, end_time=100 on a 80s video would become start=90, end=80)
    start_time = max(0.0, min(start_time, total_duration))
    end_time = max(0.0, min(end_time, total_duration))
 
    if end_time <= start_time:
        return _err(
            f"end_time ({end_time:.2f}s) must be > start_time ({start_time:.2f}s) "
            f"after clamping to video duration ({total_duration:.2f}s)"
        )
 
    interval_duration = end_time - start_time
 
    # Frame extraction: frame_timestamps (explicit) > num_frames (required)
    sampling_start_time = start_time
    sampling_end_time = end_time
    warnings: list[str] = []
    if frame_timestamps is not None and len(frame_timestamps) > 0:
        if max_frames is not None:
            warnings.append(
                "Both frame_timestamps and num_frames were provided; "
                "frame_timestamps takes precedence and num_frames is ignored."
            )
        parsed_ts = sorted(set(parse_timestamp(t) for t in frame_timestamps))
        # Clamp to requested [start_time, end_time] interval (already within video bounds)
        parsed_ts = [t for t in parsed_ts if start_time <= t <= end_time]
        parsed_ts = parsed_ts[:MAX_SAMPLE_FRAMES]
        if parsed_ts:
            frames = await asyncio.to_thread(
                _extract_frames_at_timestamps, video_path, parsed_ts
            )
            sampling_start_time = min(parsed_ts)
            sampling_end_time = max(parsed_ts)
        else:
            return _err(
                f"All provided frame_timestamps fell outside the requested interval "
                f"[{start_time:.2f}s, {end_time:.2f}s]; no frames could be extracted. "
                "Ensure timestamps fall within start_time and end_time."
            )
    else:
        if max_frames is None:
            return _err("num_frames is required: decide based on interval duration and query complexity")
        frame_count = max(1, min(int(max_frames), MAX_SAMPLE_FRAMES))
        # Agent controls sampling; use exactly the requested [start_time, end_time] interval
        sampling_start_time = start_time
        sampling_end_time = end_time
        frames = await asyncio.to_thread(
            _extract_frames_uniform,
            video_path,
            sampling_start_time,
            sampling_end_time,
            frame_count,
        )
 
    if not frames:
        return _err("No frames extracted")
 
    # Load subtitles for interval
    srt_path = find_subtitle_file(video_path, subtitle_dir)
    subtitle_text = ""
    if srt_path:
        subs = parse_subtitle_file(srt_path)
        relevant = filter_subtitles_by_time_range(
            subs,
            sampling_start_time,
            sampling_end_time,
        )
        subtitle_text = format_subtitles_as_text(relevant, max_length=1000)
 
    # Build temporal-aware VLM content (cv2 resize/encode is CPU-heavy → thread).
    content = await asyncio.to_thread(
        _build_content,
        frames=frames,
        query=query,
        interval_duration=interval_duration,
        total_duration=total_duration,
        start_time=start_time,
        end_time=end_time,
        sampling_start_time=sampling_start_time,
        sampling_end_time=sampling_end_time,
        subtitle_text=subtitle_text,
    )
 
    # Call VLM (retries on timeout / 5xx; falls through to fallback model on other errors)
    vlm_result = await _call_vlm_with_retry(content, int(os.environ.get("VLM_MAX_TOKENS", "8192")))
 
    # Cost / model telemetry — written to stderr, not returned to the agent.
    print(json.dumps({
        "tool": "query_segment_detail",
        "model": vlm_result.get("model"),
        "attempted_models": vlm_result.get("attempted_models", []),
        "input_tokens": vlm_result.get("input_tokens", 0),
        "output_tokens": vlm_result.get("output_tokens", 0),
        "frames_sampled": len(frames),
        "error": vlm_result.get("error"),
    }, ensure_ascii=False), file=sys.stderr, flush=True)
 
    result = {
        "success": vlm_result["error"] is None,
        "response": vlm_result["response"],
        "frames_sampled": len(frames),
        "frame_timestamps": [format_seconds_to_timestamp(ts) for ts, _ in frames],
        **({
            "vlm_usage": {
                "input_tokens": vlm_result.get("input_tokens", 0),
                "output_tokens": vlm_result.get("output_tokens", 0),
                "total_tokens": vlm_result.get("input_tokens", 0) + vlm_result.get("output_tokens", 0),
                "model": vlm_result.get("model") or VLM_MODEL_NAME,
            }
        } if VLM_REPORT_USAGE else {}),
    }
    # Only include sampling_time_range when it differs from the requested
    # [start_time, end_time] — typically when explicit_timestamps was clamped.
    if (sampling_start_time, sampling_end_time) != (start_time, end_time):
        result["sampling_time_range"] = {
            "start_formatted": format_seconds_to_timestamp(sampling_start_time),
            "end_formatted": format_seconds_to_timestamp(sampling_end_time),
        }
    if warnings:
        result["warnings"] = warnings
    if vlm_result.get("error"):
        result["error"] = vlm_result["error"]
    return result
 
 
# ===========================================================================
# MCP Server
# ===========================================================================
 
 
async def handle_query_segment_detail(arguments: dict) -> dict:
    try:
        async with _QUERY_SEGMENT_DETAIL_SEMAPHORE:
            return await asyncio.wait_for(
                _video_vlm_query(
                    video_path=arguments.get("video_path", ""),
                    start_time=arguments.get("start_time", 0),
                    end_time=arguments.get("end_time"),
                    query=arguments.get("query", ""),
                    max_frames=arguments.get("num_frames"),
                    subtitle_dir=arguments.get("subtitle_dir", ""),
                    frame_timestamps=arguments.get("frame_timestamps"),
                ),
                timeout=QUERY_SEGMENT_DETAIL_TIMEOUT,
            )
    except asyncio.TimeoutError:
        msg = f"query_segment_detail timeout after {QUERY_SEGMENT_DETAIL_TIMEOUT:.1f}s"
        print(f"[vlm-query-sequence] {msg}", file=sys.stderr, flush=True)
        return make_error_response(msg)
    except Exception as e:
        print(
            f"[vlm-query-sequence] unhandled error: {e}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return make_error_response(str(e))
 
 
if __name__ == "__main__":
    app = create_mcp_server(
        server_name="vlm-query-sequence-mcp",
        tool_defs_source="_tool_defs/vlm_query.yaml",
        handlers={"query_segment_detail": handle_query_segment_detail},
        is_file=True,
    )
    asyncio.run(run_server(app))
