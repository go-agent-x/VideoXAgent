#!/usr/bin/env python3
"""Scene Query MCP Server — Dense-frame VLM deep dive for a specific video interval."""
import asyncio
import base64
import json
import os
import sys
import traceback
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

from _base.paths import PROJECT_ROOT

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response
from _base.config import get_config, pick_api_key, normalize_vlm_url
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
MAX_SAMPLE_FRAMES = 30
VLM_TIMEOUT = 240.0
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
_PROJECT_ROOT = PROJECT_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS: {video_path}")
    return n / fps


def _extract_frames_uniform(
    video_path: str, start: float, end: float, n: int
) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or end <= start:
        cap.release()
        return []
    dur = end - start
    if n == 1:
        timestamps = [start + dur / 2]
    else:
        step = dur / (n + 1)
        timestamps = [start + step * (i + 1) for i in range(n)]
    frames = []
    for ts in timestamps:
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
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


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


async def _call_vlm(content: list[dict], max_tokens: int) -> dict:
    """Call VLM API with retry logic and model fallback."""
    headers = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"

    result = {"response": "", "input_tokens": 0, "output_tokens": 0, "error": None}

    async with httpx.AsyncClient(timeout=VLM_TIMEOUT) as client:
        for model_name in _candidate_models():
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": os.environ.get("VLM_ENABLE_THINKING", "false").lower() == "true"},
            }

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
                        result["input_tokens"] = data["usage"].get("prompt_tokens", 0)
                        result["output_tokens"] = data["usage"].get("completion_tokens", 0)

                    if "choices" in data and data["choices"]:
                        result["response"] = data["choices"][0]["message"]["content"]
                        result["error"] = None
                    else:
                        result["response"] = "[Unexpected API response]"
                        result["error"] = "Invalid format"
                    break

                except httpx.TimeoutException:
                    if attempt < MAX_VLM_RETRIES:
                        await asyncio.sleep(0.6 * attempt)
                        continue
                    result["response"] = f"[VLM API timeout] model={model_name}"
                    result["error"] = "API timeout"
                except httpx.HTTPStatusError as e:
                    detail = _response_text(e.response)[:500].replace("\n", " ")
                    result["response"] = (
                        f"[VLM API HTTP {e.response.status_code}] "
                        f"model={model_name} {detail}"
                    )
                    result["error"] = f"HTTP {e.response.status_code}"
                    if _should_try_fallback(e.response.status_code, detail):
                        break
                except Exception as e:
                    result["response"] = f"[VLM API error: {e}] model={model_name}"
                    result["error"] = str(e)
                    break

            if result["error"] is None:
                break

    return result


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
async def _query_scene_detail(
    video_path: str, start_time: float, end_time: float,
    query: str, sample_frames: int = 8, subtitle_dir: str = "",
) -> dict:
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"Video not found: {video_path}"}

    start_time = parse_timestamp(start_time)
    end_time = parse_timestamp(end_time)
    if end_time <= start_time:
        return {"success": False, "error": "end_time must be > start_time"}
    sample_frames = max(1, min(sample_frames, MAX_SAMPLE_FRAMES))

    try:
        duration = _get_duration(video_path)
    except Exception as e:
        return {"success": False, "error": f"Cannot get duration: {e}"}

    frames = _extract_frames_uniform(video_path, start_time, end_time, sample_frames)
    if not frames:
        return {"success": False, "error": "No frames extracted"}

    # Load subtitle for this interval
    srt_path = find_subtitle_file(video_path, subtitle_dir)
    subtitle_text = ""
    if srt_path:
        subs = parse_subtitle_file(srt_path)
        relevant = filter_subtitles_by_time_range(subs, start_time, end_time)
        subtitle_text = format_subtitles_as_text(relevant, max_length=800)

    # Build VLM content
    t_str = format_seconds_to_timestamp(duration)
    s_str = format_seconds_to_timestamp(start_time)
    e_str = format_seconds_to_timestamp(end_time)

    prompt = (
        f"Analyze these {len(frames)} frames from a video segment "
        f"(time range: {s_str} - {e_str}, total duration: {t_str}).\n\n"
        f"Question: {query}\n\n"
        f"Answer in detail based on the visual information."
    )
    if subtitle_text:
        prompt += f"\n\nSubtitles during this interval: {subtitle_text}"

    content = [{"type": "text", "text": prompt}]
    for ts, f in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_to_b64(f)}"}})
        content.append({"type": "text", "text": f"[Frame at {format_seconds_to_timestamp(ts)}]"})

    vlm_result = await _call_vlm(content, int(os.environ.get("VLM_MAX_TOKENS", "8192")))

    return {
        "success": vlm_result["error"] is None,
        "video_path": video_path,
        "time_range": {
            "start_time": round(start_time, 3), "end_time": round(end_time, 3),
            "start_formatted": s_str, "end_formatted": e_str,
        },
        "query": query,
        "response": vlm_result["response"],
        "frames_sampled": len(frames),
        "frame_timestamps": [format_seconds_to_timestamp(ts) for ts, _ in frames],
        "has_subtitles": bool(subtitle_text),
        **({
            "vlm_usage": {
                "input_tokens": vlm_result.get("input_tokens", 0),
                "output_tokens": vlm_result.get("output_tokens", 0),
                "total_tokens": vlm_result.get("input_tokens", 0) + vlm_result.get("output_tokens", 0),
                "model": vlm_result.get("model") or VLM_MODEL_NAME,
            }
        } if VLM_REPORT_USAGE else {}),
        "error": vlm_result.get("error"),
    }


# ===========================================================================
# MCP Server
# ===========================================================================


async def handle_query_scene_detail(arguments: dict) -> dict:
    try:
        return await _query_scene_detail(
            video_path=arguments.get("video_path", ""),
            start_time=arguments.get("start_time", 0),
            end_time=arguments.get("end_time", 0),
            query=arguments.get("query", ""),
            sample_frames=int(arguments.get("sample_frames", 8)),
            subtitle_dir=arguments.get("subtitle_dir", ""),
        )
    except Exception as e:
        return make_error_response(str(e), {"traceback": traceback.format_exc()})


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="scene-query-mcp",
        tool_defs_source="_tool_defs/vlm_query.yaml",
        handlers={"query_scene_detail": handle_query_scene_detail},
        is_file=True,
    )
    asyncio.run(run_server(app))
