#!/usr/bin/env python3
"""Attribute Recognition MCP Server — Classify visual attributes via VLM."""
import asyncio
import base64
import json
import os
import re
import sys
import traceback
from pathlib import Path

# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")
sys.path.insert(0, str(Path(__file__).parent))

from _base.server_framework import create_mcp_server, run_server, make_error_response
from _base.config import get_config, pick_api_key, normalize_vlm_url

import httpx

try:
    import cv2
    import numpy as np
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
VLM_MODEL_FALLBACKS = tuple(
    f.strip() for f in _vlm_cfg["fallbacks"].split(",") if f.strip()
)

MAX_IMAGE_DIMENSION = 4096
MAX_VLM_RETRIES = 3
VLM_TIMEOUT = 240.0
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_and_encode_image(image_path: str) -> tuple[str, int, int, int, int]:
    """Load image, optionally resize, encode to base64 JPEG.

    Returns:
        (base64_string, original_width, original_height, encoded_width, encoded_height)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    enc_h, enc_w = orig_h, orig_w

    if max(orig_h, orig_w) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(orig_h, orig_w)
        enc_w = int(orig_w * scale)
        enc_h = int(orig_h * scale)
        img = cv2.resize(img, (enc_w, enc_h), interpolation=cv2.INTER_AREA)

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64, orig_w, orig_h, enc_w, enc_h


def _crop_and_encode_region(
    img: np.ndarray, bbox: list[float]
) -> tuple[str, list[int]]:
    """Crop a region from the image and encode to base64 JPEG.

    Returns:
        (base64_string, clamped_bbox)
    """
    h, w = img.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clamping: [{x1},{y1},{x2},{y2}]")

    crop = img[y1:y2, x1:x2]
    if max(crop.shape[:2]) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(crop.shape[:2])
        crop = cv2.resize(
            crop,
            (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    _, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64, [x1, y1, x2, y2]


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


async def _call_vlm(image_b64: str, query: str, max_tokens: int = int(os.environ.get("VLM_MAX_TOKENS", "8192"))) -> dict:
    """Call VLM API with a single image and query, with model fallback and retry."""
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
                        result["response"] = (
                            f"[Unexpected API response format: "
                            f"{json.dumps(data, ensure_ascii=False)[:200]}]"
                        )
                        result["error"] = "Invalid API response format"
                    break

                except httpx.TimeoutException:
                    if attempt < MAX_VLM_RETRIES:
                        await asyncio.sleep(0.6 * attempt)
                        continue
                    result["response"] = (
                        f"[VLM API timeout after {VLM_TIMEOUT}s] model={model_name}"
                    )
                    result["error"] = "API timeout"
                except httpx.HTTPStatusError as e:
                    detail = _response_text(e.response)[:500].replace("\n", " ")
                    result["response"] = (
                        f"[VLM API error: HTTP {e.response.status_code}] "
                        f"model={model_name} {detail}"
                    )
                    result["error"] = f"HTTP {e.response.status_code}"
                    if _should_try_fallback(e.response.status_code, detail):
                        break
                except Exception as e:
                    result["response"] = (
                        f"[VLM API call failed] model={model_name} {e}"
                    )
                    result["error"] = str(e)
                    break

            if result["error"] is None:
                break

    return result


def _build_prompt_full_image(attributes: list[str]) -> str:
    """Build prompt for whole-image attribute recognition."""
    attr_str = ", ".join(attributes)
    return f"""Analyze the visual attributes of objects and regions in this image.
For each prominent object or region visible, classify the following attributes: {attr_str}.

Output ONLY a JSON array. Each element:
{{
  "object": "<brief object description>",
  "attributes": {{
    "<attribute_name>": "<attribute_value>",
    ...
  }}
}}

Rules:
- Only report attributes that are clearly visible — do not hallucinate.
- Cover all prominent objects/regions in the image.
- For each requested attribute, provide a concise value string.
- If an attribute is not applicable or not determinable for an object, omit it.

Output ONLY the JSON array, no other text."""


def _build_prompt_with_regions(
    attributes: list[str], regions: list[dict]
) -> str:
    """Build prompt for region-specific attribute recognition."""
    attr_str = ", ".join(attributes)
    region_lines = []
    for idx, region in enumerate(regions):
        bbox = region.get("bbox", [])
        label = region.get("label", f"region_{idx}")
        bbox_str = f"[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]"
        region_lines.append(f"  [{idx}] \"{label}\" at bbox {bbox_str}")

    return f"""Analyze the visual attributes of the following objects/regions in this image.
Regions to analyze:
{chr(10).join(region_lines)}

For each region, classify the following attributes: {attr_str}.

Output ONLY a JSON array. Each element:
{{
  "region_index": <integer>,
  "label": "<region label>",
  "bbox": [x1, y1, x2, y2],
  "attributes": {{
    "<attribute_name>": "<attribute_value>",
    ...
  }}
}}

Rules:
- Only report attributes that are clearly visible — do not hallucinate.
- For each requested attribute, provide a concise value string.
- If an attribute is not applicable or not determinable for a region, omit it.

Output ONLY the JSON array, no other text."""


def _parse_vlm_response(response_text: str) -> list:
    """Parse VLM response into a list of dicts."""
    text = response_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Strategy 1: Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON array between [ and ]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract individual JSON objects
    objects = []
    for m in re.finditer(r"\{[^{}]+\}", text):
        try:
            obj = json.loads(m.group())
            if "attributes" in obj:
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    return objects


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------
async def handle_attribute_recognition(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    attributes_str = arguments.get("attributes", "")
    regions_str = arguments.get("regions", "")
    max_tokens = arguments.get("max_tokens", os.environ.get("VLM_MAX_TOKENS", "8192"))

    if not image_path:
        return make_error_response("image_path is required")
    if not attributes_str:
        return make_error_response("attributes is required")
    if not os.path.isfile(image_path):
        return make_error_response(f"File not found: {image_path}")

    max_tokens = max(64, min(int(max_tokens), int(os.environ.get("VLM_MAX_TOKENS", "8192"))))

    # Parse attributes
    attributes = [a.strip() for a in attributes_str.split(",") if a.strip()]
    if not attributes:
        return make_error_response("No valid attributes provided")

    # Parse optional regions
    regions = []
    if regions_str:
        try:
            regions = json.loads(regions_str)
            if not isinstance(regions, list):
                return make_error_response("regions must be a JSON array")
            for r in regions:
                if "bbox" not in r or len(r["bbox"]) != 4:
                    return make_error_response(
                        "Each region must have a 'bbox' field with [x1, y1, x2, y2]"
                    )
        except json.JSONDecodeError as e:
            return make_error_response(f"Invalid regions JSON: {e}")

    try:
        # Load image
        img_b64, orig_w, orig_h, enc_w, enc_h = _load_and_encode_image(image_path)

        # Build prompt
        if regions:
            prompt = _build_prompt_with_regions(attributes, regions)
        else:
            prompt = _build_prompt_full_image(attributes)

        # Call VLM
        vlm_result = await _call_vlm(img_b64, prompt, max_tokens)

        vlm_usage = {
            "input_tokens": vlm_result.get("input_tokens", 0),
            "output_tokens": vlm_result.get("output_tokens", 0),
            "total_tokens": vlm_result.get("input_tokens", 0) + vlm_result.get("output_tokens", 0),
            "model": vlm_result.get("model") or VLM_MODEL_NAME,
        }

        if vlm_result["error"]:
            return {
                "success": False,
                "error": vlm_result["error"],
                "vlm_response": vlm_result["response"],
                **({"vlm_usage": vlm_usage} if VLM_REPORT_USAGE else {}),
            }

        # Parse response
        parsed = _parse_vlm_response(vlm_result["response"])

        return {
            "success": True,
            "image_path": image_path,
            "image_size": {"width": orig_w, "height": orig_h},
            "requested_attributes": attributes,
            "region_count": len(regions) if regions else None,
            "attributes": parsed,
            "attribute_count": len(parsed),
            **({"vlm_usage": vlm_usage} if VLM_REPORT_USAGE else {}),
        }

    except Exception as e:
        return make_error_response(str(e), {"traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = create_mcp_server(
        server_name="attribute-recognition-mcp",
        tool_defs_source="_tool_defs/recognition.yaml",
        handlers={"attribute_recognition": handle_attribute_recognition},
        is_file=True,
    )
    asyncio.run(run_server(app))
