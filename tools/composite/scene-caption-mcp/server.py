#!/usr/bin/env python3
"""Scene Caption MCP Server — Scene-driven overview captioning with VLM."""
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

from _base.server_framework import create_mcp_server, run_server, make_error_response
from _base.config import get_config, pick_api_key, normalize_vlm_url
from _base.subtitle_utils import (
    parse_timestamp,
    format_seconds_to_timestamp,
    find_subtitle_file,
    parse_subtitle_file,
    group_subtitles_by_scenes,
)
from _base.vlm_objectivity import (
    build_objective_vlm_prompt,
    SCENE_CAPTION_OUTPUT_CONTRACT,
    SCENE_DIRECTED_QUERY_OUTPUT_CONTRACT,
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
MAX_CONCURRENT_VLM = 5
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
    if end <= start:
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
# Scene detection (self-contained, uses PySceneDetect)
# ---------------------------------------------------------------------------
def _detect_scenes(video_path: str, threshold: float = 27.0, min_len: float = 0.5) -> list[dict]:
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        fps = video.frame_rate
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=max(1, int(min_len * fps))))
        sm.detect_scenes(video)
        scenes = []
        for i, (s, e) in enumerate(sm.get_scene_list()):
            ss, ee = s.get_seconds(), e.get_seconds()
            scenes.append({
                "scene_id": i + 1,
                "start_time": round(ss, 3),
                "end_time": round(ee, 3),
                "duration": round(ee - ss, 3),
                "representative_frame": round((ss + ee) / 2, 3),
            })
        return scenes
    except Exception as exc:
        print(f"[scene-caption] Scene detection failed: {exc}", file=sys.stderr)
        return []


def _merge_scenes(scenes: list[dict], max_scenes: int) -> list[dict]:
    if len(scenes) <= max_scenes:
        return scenes
    merged = list(scenes)
    while len(merged) > max_scenes:
        best_i, best_dur = 0, float("inf")
        for i in range(len(merged) - 1):
            d = merged[i]["duration"] + merged[i + 1]["duration"]
            if d < best_dur:
                best_dur, best_i = d, i
        a, b = merged[best_i], merged[best_i + 1]
        merged = merged[:best_i] + [{
            "scene_id": a["scene_id"],
            "start_time": a["start_time"],
            "end_time": b["end_time"],
            "duration": round(b["end_time"] - a["start_time"], 3),
            "representative_frame": round((a["start_time"] + b["end_time"]) / 2, 3),
        }] + merged[best_i + 2:]
    for i, s in enumerate(merged):
        s["scene_id"] = i + 1
    return merged


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


_vlm_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_VLM)


async def _call_vlm(content: list[dict], max_tokens: int) -> dict:
    """Call VLM API with retry logic, model fallback, and concurrency limit."""
    headers = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"

    result = {"response": "", "model": None, "input_tokens": 0, "output_tokens": 0, "error": None, "attempted_models": []}

    async with httpx.AsyncClient(timeout=VLM_TIMEOUT) as client:
        for model_name in _candidate_models():
            result["attempted_models"].append(model_name)
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": os.environ.get("VLM_ENABLE_THINKING", "false").lower() == "true"},
            }

            for attempt in range(1, MAX_VLM_RETRIES + 1):
                # Semaphore scoped to the network call only so that retry
                # backoff sleeps and fallback-model switching do not hold slots.
                should_retry = False
                try:
                    async with _vlm_semaphore:
                        resp = await client.post(
                            VLM_API_URL, json=payload, headers=headers
                        )
                        if (
                            resp.status_code in RETRYABLE_STATUS_CODES
                            and attempt < MAX_VLM_RETRIES
                        ):
                            should_retry = True
                        else:
                            resp.raise_for_status()
                            data = resp.json()

                    if should_retry:
                        await asyncio.sleep(0.6 * attempt)
                        continue

                    if "usage" in data:
                        result["input_tokens"] = data["usage"].get("prompt_tokens", 0)
                        result["output_tokens"] = data["usage"].get("completion_tokens", 0)

                    if "choices" in data and data["choices"]:
                        result["response"] = data["choices"][0]["message"]["content"]
                        result["model"] = model_name
                        result["error"] = None
                    else:
                        print(
                            f"[scene-caption] invalid API response model={model_name} "
                            f"payload={json.dumps(data, ensure_ascii=False)[:500]}",
                            file=sys.stderr,
                            flush=True,
                        )
                        result["response"] = ""
                        result["model"] = model_name
                        result["error"] = "Invalid API response format"
                    break

                except httpx.TimeoutException:
                    if attempt < MAX_VLM_RETRIES:
                        await asyncio.sleep(0.6 * attempt)
                        continue
                    print(
                        f"[scene-caption] VLM timeout after {VLM_TIMEOUT}s model={model_name}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result["response"] = ""
                    result["error"] = "API timeout"
                except httpx.HTTPStatusError as e:
                    detail = _response_text(e.response)[:500].replace("\n", " ")
                    print(
                        f"[scene-caption] VLM HTTP {e.response.status_code} "
                        f"model={model_name} detail={detail}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result["response"] = ""
                    result["error"] = f"HTTP {e.response.status_code}"
                    # Only skip retries and move on to the fallback model when
                    # the error signals a permanent incompatibility with this
                    # model (auth / unsupported / content-policy).
                    if _should_try_fallback(e.response.status_code, detail):
                        break
                except Exception as e:
                    print(
                        f"[scene-caption] VLM call failed model={model_name} err={e}",
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
# Caption one scene
# ---------------------------------------------------------------------------
async def _caption_one(
    video_path: str, scene: dict, total_dur: float,
    sub_text: str = "", custom_query: Optional[str] = None,
    max_tokens_query: int = int(os.environ.get("VLM_MAX_TOKENS", "8192")), max_tokens_overview: int = int(os.environ.get("VLM_MAX_TOKENS", "8192")),
    num_frames: Optional[int] = None,
) -> dict:
    s_str, e_str, t_str = format_seconds_to_timestamp(scene["start_time"]), format_seconds_to_timestamp(scene["end_time"]), format_seconds_to_timestamp(total_dur)

    # Dynamic frame count: 1 frame per 5 seconds, min 1, max 5
    if num_frames is None:
        scene_duration = scene["end_time"] - scene["start_time"]
        num_frames = min(max(1, int(scene_duration / 5)), 5)

    frames = await asyncio.to_thread(
        _extract_frames_uniform, video_path, scene["start_time"], scene["end_time"], num_frames
    )
    if not frames:
        return {"caption": "[No frames extracted]", "error": "frame_extraction_failed"}

    # Build content with frames (cv2 resize/encode is CPU-heavy → thread).
    def _build_frame_blocks(frames):
        blocks = []
        for ts, f in frames:
            blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_to_b64(f)}"}})
            blocks.append({"type": "text", "text": f"[Frame at {format_seconds_to_timestamp(ts)}]"})
        return blocks

    content = await asyncio.to_thread(_build_frame_blocks, frames)

    context = (
        f"Scene time range: {s_str} - {e_str}. "
        f"Video total duration: {t_str}. "
        f"Frames sampled from this scene: {len(frames)}. "
        "These frames are a sampled view of the scene, not exhaustive coverage."
    )
    evidence = f"Subtitles during this scene:\n{sub_text[:500]}" if sub_text else ""

    custom_query_str = (custom_query or "").strip()
    if custom_query_str:
        tpl = build_objective_vlm_prompt(
            task_instruction=(
                "Answer the scene-level query for this scene using only the "
                "sampled frames and explicitly included subtitles. This mode is "
                "for candidate discovery or scene-level filtering, not a final "
                "answer for detailed interval reasoning."
            ),
            user_query=custom_query_str,
            context=context,
            evidence=evidence,
            output_contract=SCENE_DIRECTED_QUERY_OUTPUT_CONTRACT,
        )
        max_tokens = max_tokens_query
    else:
        tpl = build_objective_vlm_prompt(
            task_instruction=(
                "Produce a factual overview caption for this scene. This is an "
                "indexing/overview tool, not a final answer for detailed reasoning."
            ),
            context=context,
            evidence=evidence,
            output_contract=SCENE_CAPTION_OUTPUT_CONTRACT,
        )
        max_tokens = max_tokens_overview

    content.append({"type": "text", "text": tpl})
    r = await _call_vlm(content, max_tokens)

    print(json.dumps({
        "tool": "caption_video_scenes",
        "scene_id": scene.get("scene_id"),
        "model": r.get("model"),
        "attempted_models": r.get("attempted_models", []),
        "input_tokens": r.get("input_tokens", 0),
        "output_tokens": r.get("output_tokens", 0),
        "frames_sampled": len(frames),
        "error": r.get("error"),
    }, ensure_ascii=False), file=sys.stderr, flush=True)

    return {
        "caption": r["response"],
        "input_tokens": r.get("input_tokens", 0),
        "output_tokens": r.get("output_tokens", 0),
        "model": r.get("model"),
        "error": r.get("error"),
    }


# ---------------------------------------------------------------------------
# Main logic: caption all scenes
# ---------------------------------------------------------------------------
async def _caption_video_scenes(
    video_path: str, scene_threshold: float = 27.0, subtitle_dir: str = "",
    custom_query: Optional[str] = None, max_scenes: int = 30,
    max_tokens: int = int(os.environ.get("VLM_MAX_TOKENS", "8192")), num_frames: Optional[int] = None,
) -> dict:
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"Video not found: {video_path}"}
    try:
        duration = await asyncio.to_thread(_get_duration, video_path)
    except Exception as e:
        return {"success": False, "error": f"Cannot get duration: {e}"}

    scenes = await asyncio.to_thread(_detect_scenes, video_path, scene_threshold)
    if not scenes:
        scenes = [{"scene_id": 1, "start_time": 0, "end_time": round(duration, 3),
                    "duration": round(duration, 3),
                    "representative_frame": round(duration / 2, 3)}]
    scenes = _merge_scenes(scenes, max_scenes)

    srt_path = find_subtitle_file(video_path, subtitle_dir)
    subs = parse_subtitle_file(srt_path) if srt_path else []
    scene_subs = group_subtitles_by_scenes(subs, scenes)

    tasks = []
    for sc in scenes:
        sub_list = scene_subs.get(sc["scene_id"], [])
        sub_text = " ".join(s["text"] for s in sub_list)
        tasks.append(_caption_one(video_path, sc, duration, sub_text, custom_query,
                                   max_tokens_query=max_tokens, max_tokens_overview=max_tokens,
                                   num_frames=num_frames))
    results = await asyncio.gather(*tasks)
    total_vlm_input_tokens = sum(cr.get("input_tokens", 0) for cr in results)
    total_vlm_output_tokens = sum(cr.get("output_tokens", 0) for cr in results)
    vlm_call_count = sum(1 for cr in results if "input_tokens" in cr or "output_tokens" in cr)

    out_scenes = []
    for sc, cr in zip(scenes, results):
        sub_list = scene_subs.get(sc["scene_id"], [])
        sub_text = " ".join(s["text"] for s in sub_list).strip()
        scene_out = {
            "scene_id": sc["scene_id"],
            "start_time": sc["start_time"],
            "end_time": sc["end_time"],
            "caption": cr["caption"],
        }
        if cr.get("error"):
            scene_out["error"] = cr["error"]
        if sub_text:
            scene_out["subtitle_text"] = sub_text
        out_scenes.append(scene_out)

    return {
        "success": True,
        "scenes": out_scenes,
        **({
            "vlm_usage": {
                "input_tokens": total_vlm_input_tokens,
                "output_tokens": total_vlm_output_tokens,
                "total_tokens": total_vlm_input_tokens + total_vlm_output_tokens,
                "calls": vlm_call_count,
                "model": VLM_MODEL_NAME,
            }
        } if VLM_REPORT_USAGE else {}),
    }


# ===========================================================================
# MCP Server
# ===========================================================================


async def handle_caption_video_scenes(arguments: dict) -> dict:
    try:
        num_frames_raw = arguments.get("num_frames")
        num_frames = int(num_frames_raw) if num_frames_raw is not None else None
        return await _caption_video_scenes(
            video_path=arguments.get("video_path", ""),
            scene_threshold=float(arguments.get("scene_threshold", 27.0)),
            subtitle_dir=arguments.get("subtitle_dir", ""),
            custom_query=arguments.get("custom_query") or None,
            max_scenes=max(1, int(arguments.get("max_scenes", 30))),
            max_tokens=int(arguments.get("max_tokens", os.environ.get("VLM_MAX_TOKENS", "8192"))),
            num_frames=num_frames,
        )
    except Exception as e:
        print(
            f"[scene-caption] unhandled error: {e}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return make_error_response(str(e))


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="scene-caption-mcp",
        tool_defs_source="_tool_defs/vlm_query.yaml",
        handlers={"caption_video_scenes": handle_caption_video_scenes},
        is_file=True,
    )
    asyncio.run(run_server(app))
