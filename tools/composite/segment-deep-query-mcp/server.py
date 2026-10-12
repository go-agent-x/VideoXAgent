#!/usr/bin/env python3
"""Segment Deep Query MCP Server — Multi-modal deep analysis combining detection, OCR, and VLM."""
import asyncio
import base64
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")

from _base.paths import service_dir

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response
from _base.config import get_config, pick_api_key, normalize_vlm_url
from detection_backend import detect_frames
from utils.vlm_thinking_retry import (
    resolve_max_tokens,
    maybe_thinking_off_retry,
    build_chat_template_kwargs,
    get_max_concurrent_vlm,
)

import httpx

try:
    import cv2
    import numpy as np
except ImportError:
    raise ImportError("OpenCV is required. From the repository root, run: bash scripts/install_profiles.sh base")

# ---------------------------------------------------------------------------
# Optional detection / OCR backends (best-effort import from sibling servers)
# ---------------------------------------------------------------------------
_HAS_DETECTION = False
_HAS_OCR = False

try:
    _gdino_dir = str(service_dir("grounding-dino"))
    if _gdino_dir not in sys.path:
        sys.path.insert(0, _gdino_dir)
    from gdino_api import call_gdino_api  # noqa: E402
    _HAS_DETECTION = True
except Exception:
    pass

try:
    _ocr_dir = str(service_dir("ocr"))
    if _ocr_dir not in sys.path:
        sys.path.insert(0, _ocr_dir)
    from ocr_common import run_ocr_on_array  # noqa: E402
    _HAS_OCR = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# VLM configuration
# ---------------------------------------------------------------------------
_vlm_cfg = get_config("vlm")
VLM_API_URL = normalize_vlm_url(_vlm_cfg["api_url"])
VLM_API_KEY = pick_api_key("VLM_API_KEY")
VLM_MODEL_NAME = _vlm_cfg["model"]
VLM_REPORT_USAGE = os.environ.get("VLM_REPORT_USAGE", "").lower() == "true"

MAX_SAMPLE_FRAMES = 30

MAX_CONCURRENT_VLM = get_max_concurrent_vlm()
_vlm_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_VLM)


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frames(
    video_path: str, start_time: float, end_time: float, num_frames: int,
) -> list[tuple[float, np.ndarray]]:
    """Extract uniformly sampled frames from a video segment."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = end_time - start_time
    if duration <= 0:
        cap.release()
        return []

    num_frames = max(1, min(num_frames, MAX_SAMPLE_FRAMES))
    if num_frames == 1:
        timestamps = [start_time + duration / 2]
    else:
        step = duration / num_frames
        timestamps = [start_time + i * step for i in range(num_frames)]

    frames = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append((ts, frame))

    cap.release()
    return frames


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode a numpy frame to JPEG base64."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# Detection / OCR helpers (executed via asyncio.to_thread)
# ---------------------------------------------------------------------------
async def _detect_objects(frame: np.ndarray, categories=None) -> dict:
    """Detect one frame through the same bounded backend as a full segment."""
    results = await detect_frames(
        [frame], categories, local_detector=call_gdino_api if _HAS_DETECTION else None,
    )
    return results[0]


def _detection_info(result: dict, timestamp: float) -> dict:
    """Keep backend failures separate from successful scans with no matches."""
    success = bool(result.get("success"))
    if not success:
        unavailable = result.get("unavailable") or result.get("status") == "unavailable"
        status = "unavailable" if unavailable else "failed"
    elif result.get("partial"):
        status = "partial"
    else:
        status = "detected" if result.get("objects") else "no_detections"
    return {
        "timestamp": timestamp,
        "status": status,
        "success": success,
        "backend": result.get("backend", "in_process"),
        "detection_mode": result.get("detection_mode", "category_sweep"),
        "categories_used": result.get("categories_used", []),
        "vocabulary_limited": result.get("vocabulary_limited", True),
        "error": result.get("error"),
        "errors": result.get("errors", []),
        "warnings": result.get("warnings", []),
    }


async def _ocr_text(frame: np.ndarray) -> dict:
    """Run PaddleOCR on a single frame."""
    if not _HAS_OCR:
        return {"success": False, "texts": [], "error": "OCR unavailable"}

    try:
        success, texts, err = await asyncio.to_thread(run_ocr_on_array, frame)
        return {"success": success, "texts": texts if success else [], "error": err}
    except Exception as e:
        return {"success": False, "texts": [], "error": str(e)}


# ---------------------------------------------------------------------------
# VLM call with structured evidence
# ---------------------------------------------------------------------------
def _build_evidence_text(detections: list, ocr_texts: list, detection_info=None) -> str:
    """Format detection + OCR results as concise evidence text."""
    parts = []
    if detection_info:
        status = detection_info.get("status")
        if status in ("failed", "unavailable"):
            parts.append(
                f"Object detection {status}: {detection_info.get('error') or 'backend did not provide evidence'}. "
                "Do not infer object absence from this failure."
            )
        elif status == "partial":
            parts.append("Object detection partially succeeded; some category batches failed.")
        elif status == "no_detections":
            parts.append(
                "No matches within the scanned category vocabulary. "
                "This does not establish that the frame contains no objects."
            )
        else:
            parts.append("Object detection uses text-prompted category scanning.")
        if detection_info.get("categories_used"):
            parts.append("Category vocabulary: " + ", ".join(detection_info["categories_used"]))
        if detection_info.get("warnings"):
            parts.append("Detection warnings: " + "; ".join(detection_info["warnings"]))
    if detections:
        lines = [
            f"  {o.get('category', '?')} (conf={o.get('score', 0):.2f})"
            for o in detections[:15]
        ]
        parts.append("Detected objects:\n" + "\n".join(lines))
    if ocr_texts:
        lines = [
            f'  "{t.get("text", "")}" (conf={t.get("confidence", 0):.2f})'
            for t in ocr_texts[:10]
        ]
        parts.append("On-screen text (OCR):\n" + "\n".join(lines))
    return "\n".join(parts)


async def _call_vlm(
    frames_data: list[dict],
    start_time: float,
    end_time: float,
    query: str,
    max_tokens: int,
) -> dict:
    """Send frames + evidence to VLM and return the response."""
    max_tokens = resolve_max_tokens("segment_deep_query", max_tokens)
    content: list[dict[str, Any]] = []

    has_det = any(f.get("detections") for f in frames_data)
    has_ocr = any(f.get("ocr_texts") for f in frames_data)

    modalities = ["visual frames"]
    if has_det:
        modalities.append("object detection results")
    if has_ocr:
        modalities.append("OCR text")

    intro = (
        f"Analyze {len(frames_data)} keyframes from video segment "
        f"({format_timestamp(start_time)} - {format_timestamp(end_time)}).\n\n"
        f"Available evidence: {', '.join(modalities)}.\n\n"
        f"User question: {query}\n\n"
        "Answer based on ALL available evidence. Be precise and detailed."
    )
    content.append({"type": "text", "text": intro})

    frame_ts_list: list[str] = []
    for fd in frames_data:
        ts = fd["timestamp"]
        frame_ts_list.append(format_timestamp(ts))

        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{fd['base64']}"},
        })

        evidence = _build_evidence_text(
            fd.get("detections", []), fd.get("ocr_texts", []), fd.get("detection_info"),
        )
        label = f"[Frame at {format_timestamp(ts)}]"
        if evidence:
            label += "\n" + evidence
        content.append({"type": "text", "text": label})

    payload = {
        "model": VLM_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    _ctk = build_chat_template_kwargs(VLM_MODEL_NAME)
    if _ctk is not None:
        payload["chat_template_kwargs"] = _ctk
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"

    async with httpx.AsyncClient(timeout=240.0) as client:
        try:
            async with _vlm_semaphore:
                resp = await client.post(VLM_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()

            usage = body.get("usage", {})
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            text = ""
            thinking_off = False

            if "choices" in body and body["choices"]:
                retry_info = await maybe_thinking_off_retry(
                    data=body, payload=payload, client=client,
                    url=VLM_API_URL, headers=headers,
                    server_tag="segment-deep-query",
                    semaphore=None,
                )
                if retry_info["retried"]:
                    in_tok += retry_info["extra_input_tokens"]
                    out_tok += retry_info["extra_output_tokens"]
                final = retry_info["data"]
                if "choices" in final and final["choices"]:
                    text = final["choices"][0]["message"].get("content") or ""
                    if retry_info["retried"] and retry_info["retry_succeeded"]:
                        thinking_off = True

            out = {
                "response": text,
                "frame_timestamps": frame_ts_list,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
            if thinking_off:
                out["thinking_off_retry"] = True
            return out
        except httpx.TimeoutException:
            return {"response": "", "error": "VLM API timeout"}
        except httpx.HTTPStatusError as e:
            return {"response": "", "error": f"VLM HTTP {e.response.status_code}"}
        except Exception as e:
            return {"response": "", "error": str(e)}


# ---------------------------------------------------------------------------
# MCP handler
# ---------------------------------------------------------------------------
async def handle_segment_deep_query(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    query = arguments.get("query", "")
    start_time = arguments.get("start_time")
    end_time = arguments.get("end_time")
    num_frames = int(arguments.get("num_frames", 4))
    max_tokens = int(arguments.get("max_tokens", os.environ.get("VLM_MAX_TOKENS", "8192")))
    detection_categories = arguments.get("detection_categories")

    if detection_categories is not None:
        if not isinstance(detection_categories, list) or not 1 <= len(detection_categories) <= 64:
            return make_error_response("detection_categories must contain 1 to 64 category phrases")
        normalized_categories = []
        for category in detection_categories:
            if not isinstance(category, str):
                return make_error_response("detection_categories must contain non-empty strings")
            phrase = " ".join(category.split())
            if not phrase or len(phrase) > 80 or len(phrase.split()) > 8 or phrase == "<prompt_free>":
                return make_error_response("detection_categories must use phrases of at most 80 characters and 8 words")
            normalized_categories.append(phrase)
        detection_categories = normalized_categories

    if not video_path:
        return make_error_response("video_path is required")
    if start_time is None or end_time is None:
        return make_error_response("start_time and end_time are required")
    if not query:
        return make_error_response("query is required")
    if not os.path.exists(video_path):
        return make_error_response(f"Video not found: {video_path}")

    start_time = float(start_time)
    end_time = float(end_time)
    if start_time >= end_time:
        return make_error_response("start_time must be less than end_time")

    num_frames = max(1, min(num_frames, MAX_SAMPLE_FRAMES))

    try:
        # 1. Extract frames
        frames = extract_frames(video_path, start_time, end_time, num_frames)
        if not frames:
            return make_error_response("Failed to extract frames from the specified range")

        # 2. One detector batch shares a vision process/model across all frames.
        # OCR remains concurrent with detection and retains its optional backend.
        ocr_tasks = [_ocr_text(f) for _, f in frames]
        det_results, ocr_results = await asyncio.gather(
            detect_frames(
                [frame for _, frame in frames], detection_categories,
                local_detector=call_gdino_api if _HAS_DETECTION else None,
            ),
            asyncio.gather(*ocr_tasks),
        )

        # 3. Assemble per-frame evidence
        frames_data = []
        total_objects = 0
        total_texts = 0
        detection_frame_status = []
        for i, (ts, frame) in enumerate(frames):
            det = det_results[i]
            ocr = ocr_results[i]

            detections = det.get("objects", []) if det.get("success") else []
            ocr_texts = ocr.get("texts", []) if ocr.get("success") else []
            total_objects += len(detections)
            total_texts += len(ocr_texts)
            info = _detection_info(det, ts)
            detection_frame_status.append(info)

            frames_data.append({
                "timestamp": ts,
                "base64": frame_to_base64(frame),
                "detections": detections,
                "ocr_texts": ocr_texts,
                "detection_info": info,
            })

        # 4. VLM call with enriched prompt
        vlm = await _call_vlm(frames_data, start_time, end_time, query, max_tokens)

        if vlm.get("error"):
            return make_error_response(f"VLM call failed: {vlm['error']}")

        detection_succeeded = sum(info["success"] for info in detection_frame_status)
        detection_failed = len(frames) - detection_succeeded
        if all(info["status"] == "unavailable" for info in detection_frame_status):
            detection_status = "unavailable"
        elif detection_succeeded == 0:
            detection_status = "failed"
        elif detection_failed or any(info["status"] == "partial" for info in detection_frame_status):
            detection_status = "partial"
        else:
            detection_status = "ok"

        return make_success_response(
            query=query,
            start_time=start_time,
            end_time=end_time,
            num_frames=len(frames_data),
            response=vlm["response"],
            frame_timestamps=vlm.get("frame_timestamps", []),
            **({"vlm_usage": {
                "input_tokens": vlm.get("input_tokens", 0),
                "output_tokens": vlm.get("output_tokens", 0),
                "total_tokens": vlm.get("input_tokens", 0) + vlm.get("output_tokens", 0),
                "model": VLM_MODEL_NAME,
            }} if VLM_REPORT_USAGE else {}),
            evidence_summary={
                "detection_available": detection_succeeded > 0,
                "detection_status": detection_status,
                "detection_frames_succeeded": detection_succeeded,
                "detection_frames_failed": detection_failed,
                "detection_frame_status": detection_frame_status,
                "detection_categories": detection_categories,
                "detection_categories_used": list(dict.fromkeys(
                    category for info in detection_frame_status for category in info["categories_used"]
                )),
                "ocr_available": _HAS_OCR,
                "total_objects_detected": total_objects,
                "total_text_regions": total_texts,
            },
        )

    except Exception as e:
        return make_error_response(f"Processing failed: {str(e)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = create_mcp_server(
        server_name="segment-deep-query-mcp",
        tool_defs_source="_tool_defs/vlm_query.yaml",
        handlers={"segment_deep_query": handle_segment_deep_query},
        is_file=True,
    )
    asyncio.run(run_server(app))
