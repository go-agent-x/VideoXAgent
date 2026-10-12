#!/usr/bin/env python3
"""Inspect Image MCP Server — Visual question answering on a single image via VLM."""
import asyncio
import base64
import json
import os
import sys
import traceback
from pathlib import Path

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

import httpx
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    raise ImportError("OpenCV is required. From the repository root, run: bash scripts/install_profiles.sh base")


# ---------------------------------------------------------------------------
# Configuration from _base/config.py
# ---------------------------------------------------------------------------
_vlm_cfg = get_config("vlm")
VLM_API_URL = normalize_vlm_url(_vlm_cfg["api_url"])
VLM_API_KEY = pick_api_key("VLM_API_KEY")
VLM_MODEL_NAME = _vlm_cfg["model"]
VLM_REPORT_USAGE = os.environ.get("VLM_REPORT_USAGE", "").lower() == "true"
VLM_MODEL_FALLBACKS = tuple(f.strip() for f in _vlm_cfg["fallbacks"].split(",") if f.strip())

# Image constraints
MAX_IMAGE_DIMENSION = 2048  # Resize if any side exceeds this
MAX_COMPOSITE_IMAGE_DIMENSION = 4096  # Preserve more detail for concat_images outputs
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
COMPOSITE_PATH_MARKERS = ("concat_images", "contact_sheet", "montage")

_vlm_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_VLM)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_and_encode_image(
    image_path: str,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> tuple[str, int, int, int, int]:
    """
    Load image, optionally resize, encode to base64 JPEG.

    Returns:
        (base64_string, original_width, original_height, encoded_width, encoded_height)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    enc_h, enc_w = orig_h, orig_w

    # Resize if too large
    if max(orig_h, orig_w) > max_dimension:
        scale = max_dimension / max(orig_h, orig_w)
        enc_w = int(orig_w * scale)
        enc_h = int(orig_h * scale)
        img = cv2.resize(img, (enc_w, enc_h), interpolation=cv2.INTER_AREA)

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64, orig_w, orig_h, enc_w, enc_h


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


def _normalize_image_context(image_path: str, image_context: str) -> str:
    """Normalize optional caller-provided image context for prompt routing.

    "unknown" and empty inputs fall through to filename-based detection so that
    the yaml default still benefits from auto-identification of grid outputs.
    """
    value = (image_context or "").strip().lower().replace("-", "_")
    if value in {"single_image", "composite_grid", "thumbnail_grid"}:
        return value
    aliases = {
        "single": "single_image",
        "single_frame": "single_image",
        "image": "single_image",
        "frame": "single_image",
        "composite": "composite_grid",
        "grid": "composite_grid",
        "contact_sheet": "composite_grid",
        "concat": "composite_grid",
        "concat_images": "composite_grid",
        "thumbnail": "thumbnail_grid",
    }
    if value in aliases:
        return aliases[value]

    name = Path(image_path).name.lower()
    if "thumbnail_grid" in name:
        return "thumbnail_grid"
    if any(marker in name for marker in COMPOSITE_PATH_MARKERS):
        return "composite_grid"
    return "unknown"


def _inspect_task_instruction(image_context: str) -> str:
    if image_context == "single_image":
        return (
            "Answer the question about this single image or video frame. "
            "Focus on directly visible content. Do not infer motion, "
            "sequence, or events beyond what is shown."
        )
    if image_context == "thumbnail_grid":
        return (
            "Answer the question about this low-resolution thumbnail grid. Use it "
            "for broad navigation and candidate timestamp selection. Do not treat "
            "thumbnail-level evidence as reliable for fine visual details."
        )
    if image_context == "composite_grid":
        return (
            "Answer the question about this multi-image "
            "composite/contact sheet. Compare visible details or sampled progression "
            "across the provided tiles when supported by tile order/timestamps. Do "
            "not fill in unobserved content between tiles."
        )
    return (
        "Answer the question about this image. Use this tool for direct visual "
        "verification of a still image/frame, not temporal reasoning or dense OCR. "
        "If the image appears to be a grid/contact sheet/composite, treat each tile "
        "as a separate still image and do not infer continuous motion or events "
        "between tiles."
    )


async def _call_vlm(
    image_b64: str,
    query: str,
    max_tokens: int = int(os.environ.get("VLM_MAX_TOKENS", "8192")),
) -> dict:
    """
    Call VLM API with a single image and query.

    Returns dict with: response, input_tokens, output_tokens, error, model
    """
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        },
        {"type": "text", "text": query},
    ]

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

    async with httpx.AsyncClient(timeout=240.0) as client:
        for model_name in _candidate_models():
            result["attempted_models"].append(model_name)
            logger.info(f"[VLM Request] model={model_name}, url={VLM_API_URL}")
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
                        resp = await client.post(VLM_API_URL, json=payload, headers=headers)
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
                            f"[inspect-image] invalid API response model={model_name} "
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
                        f"[inspect-image] VLM timeout model={model_name}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result["response"] = ""
                    result["error"] = "API timeout"
                except httpx.HTTPStatusError as e:
                    detail = _response_text(e.response)[:500].replace("\n", " ")
                    print(
                        f"[inspect-image] VLM HTTP {e.response.status_code} "
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
                        f"[inspect-image] VLM call failed model={model_name} err={e}",
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
# MCP Server
# ---------------------------------------------------------------------------


async def handle_inspect_image(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    query = arguments.get("query", "")
    image_context = arguments.get("image_context", "")

    if not image_path:
        return make_error_response("image_path is required")
    if not query:
        return make_error_response("query is required")
    if not os.path.isfile(image_path):
        return make_error_response(f"File not found: {image_path}")

    try:
        normalized_context = _normalize_image_context(image_path, image_context)
        max_encoded_dimension = (
            MAX_COMPOSITE_IMAGE_DIMENSION
            if normalized_context == "composite_grid"
            else MAX_IMAGE_DIMENSION
        )

        img_b64, orig_w, orig_h, enc_w, enc_h = _load_and_encode_image(
            image_path,
            max_dimension=max_encoded_dimension,
        )
        was_resized = (orig_w != enc_w) or (orig_h != enc_h)
        # Only surface context when it adds info not already carried by the
        # task_instruction (which already describes single/thumbnail/composite).
        # Downscale notice tells the VLM fine detail may be lost.
        context = (
            f"Image was downscaled from {orig_w}x{orig_h} to {enc_w}x{enc_h} before encoding."
            if was_resized else ""
        )
        objective_query = build_objective_vlm_prompt(
            task_instruction=_inspect_task_instruction(normalized_context),
            user_query=query,
            context=context,
        )
        vlm_result = await _call_vlm(img_b64, objective_query)

        # Cost / model telemetry — written to stderr, not returned to the agent
        # (keeps agent context clean while enabling offline log analysis).
        print(json.dumps({
            "tool": "inspect_image",
            "model": vlm_result.get("model"),
            "attempted_models": vlm_result.get("attempted_models", []),
            "input_tokens": vlm_result.get("input_tokens", 0),
            "output_tokens": vlm_result.get("output_tokens", 0),
            "error": vlm_result.get("error"),
        }, ensure_ascii=False), file=sys.stderr, flush=True)

        vlm_usage = {
            "input_tokens": vlm_result.get("input_tokens", 0),
            "output_tokens": vlm_result.get("output_tokens", 0),
            "total_tokens": vlm_result.get("input_tokens", 0) + vlm_result.get("output_tokens", 0),
            "model": vlm_result.get("model") or VLM_MODEL_NAME,
        }

        if vlm_result["error"]:
            return make_error_response(vlm_result["error"], {"vlm_usage": vlm_usage} if VLM_REPORT_USAGE else {})
        return {
            "success": True,
            "response": vlm_result["response"],
            **({"vlm_usage": vlm_usage} if VLM_REPORT_USAGE else {}),
        }

    except Exception as e:
        # Log full traceback to stderr for dev debugging; keep the
        # agent-facing response minimal (no traceback dict to avoid
        # context pollution).
        print(
            f"[inspect-image] unhandled error: {e}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return make_error_response(str(e))


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="inspect-image-mcp",
        tool_defs_source="_tool_defs/vlm_query.yaml",
        handlers={"inspect_image": handle_inspect_image},
        is_file=True,
    )
    asyncio.run(run_server(app))
