#!/usr/bin/env python3
"""Person Identification MCP Server — Identify individuals across images via VLM."""
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
# Configuration
# ---------------------------------------------------------------------------
_vlm_cfg = get_config("vlm")
VLM_API_URL = normalize_vlm_url(_vlm_cfg["api_url"])
VLM_API_KEY = pick_api_key("VLM_API_KEY")
VLM_MODEL_NAME = _vlm_cfg["model"]
VLM_MODEL_FALLBACKS = tuple(
    f.strip() for f in _vlm_cfg["fallbacks"].split(",") if f.strip()
)

MAX_IMAGE_DIMENSION = 4096
MAX_IMAGES = 16
MAX_VLM_RETRIES = 3
VLM_TIMEOUT = 90.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
FALLBACK_STATUS_CODES = {400, 401, 403, 404, 415, 422}
VLM_REPORT_USAGE = os.environ.get("VLM_REPORT_USAGE", "").lower() == "true"
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
def _load_and_encode_image(image_path: str) -> tuple[str, int, int]:
    """Load image, optionally resize, encode to base64 JPEG.

    Returns:
        (base64_string, width, height)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64, w, h


def _draw_regions_on_image(image_path: str, regions: list[dict]) -> str:
    """Draw bounding boxes on an image for the given regions, return base64."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
    ]
    for idx, region in enumerate(regions):
        bbox = region["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = colors[idx % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"[{region.get('image_index', '?')}:{idx}]"
        cv2.putText(img, label, (x1 + 2, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


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


async def _call_vlm(image_b64_list: list[str], query: str, max_tokens: int = int(os.environ.get("VLM_MAX_TOKENS", "8192"))) -> dict:
    """Call VLM API with multiple images and a query, with model fallback and retry."""
    content = []
    for b64 in image_b64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": query})

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


def _build_prompt(
    image_count: int,
    regions: list[dict] | None,
    reference_identities: list[str] | None,
) -> str:
    """Build the VLM prompt for person identification."""
    image_desc = (
        f"You are given {image_count} image(s). They are labeled Image 0 through Image {image_count - 1} "
        f"in the order provided."
    )

    region_block = ""
    if regions:
        region_lines = []
        for idx, r in enumerate(regions):
            bbox = r["bbox"]
            img_idx = r.get("image_index", 0)
            bbox_str = f"[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]"
            region_lines.append(f"  Region {idx}: Image {img_idx}, bbox {bbox_str}")
        region_block = (
            "\n\nFocus on these specific person regions (marked with colored bounding boxes):\n"
            + "\n".join(region_lines)
        )

    identity_block = ""
    if reference_identities:
        names = ", ".join(reference_identities)
        identity_block = (
            f"\n\nKnown reference identities to match against: {names}\n"
            f"Try to match visible persons to these known identities based on facial features, "
            f"clothing, body build, hair, and other visual cues."
        )

    return f"""{image_desc}{region_block}{identity_block}

Identify and recognize all individuals visible in the images. For each person, analyze:
- Facial features (face shape, hair style/color, facial hair, glasses, etc.)
- Body features (build, height estimate, posture)
- Clothing and accessories
- Whether the same person appears across multiple images

Output ONLY a JSON object with this structure:
{{
  "persons": [
    {{
      "person_id": "<unique_id, e.g. person_0>",
      "identity": "<matched name from reference list, or 'unknown'>",
      "confidence": <float 0.0-1.0>,
      "appearances": [
        {{
          "image_index": <int>,
          "bbox_approx": [x1, y1, x2, y2],
          "description": "<brief visual description>"
        }}
      ],
      "distinguishing_features": "<key features that identify this person>"
    }}
  ],
  "cross_image_matches": [
    {{
      "person_id": "<person_id>",
      "image_indices": [<list of image indices where this person appears>],
      "match_confidence": <float 0.0-1.0>,
      "reasoning": "<why these are the same person>"
    }}
  ]
}}

Rules:
- Only report persons clearly visible — do not hallucinate.
- Assign consistent person_id across images for the same individual.
- If reference identities are provided, match only when confident; otherwise use "unknown".
- cross_image_matches should only include persons appearing in 2+ images.

Output ONLY the JSON object, no other text."""


def _parse_vlm_response(response_text: str) -> dict:
    """Parse VLM response into a dict."""
    text = response_text.strip()

    # Remove markdown code fences
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
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON object between { and }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text
    return {"raw_response": text}


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------
async def handle_person_identification(arguments: dict) -> dict:
    image_paths_str = arguments.get("image_paths", "")
    regions_str = arguments.get("regions", "")
    reference_str = arguments.get("reference_identities", "")
    max_tokens = arguments.get("max_tokens", os.environ.get("VLM_MAX_TOKENS", "8192"))

    if not image_paths_str:
        return make_error_response("image_paths is required")

    # Parse image paths
    image_paths = [p.strip() for p in image_paths_str.split(",") if p.strip()]
    if not image_paths:
        return make_error_response("No valid image paths provided")
    if len(image_paths) > MAX_IMAGES:
        return make_error_response(
            f"Too many images ({len(image_paths)}). Maximum is {MAX_IMAGES} per call."
        )

    # Validate files exist
    for p in image_paths:
        if not os.path.isfile(p):
            return make_error_response(f"File not found: {p}")

    max_tokens = max(64, min(int(max_tokens), int(os.environ.get("VLM_MAX_TOKENS", "8192"))))

    # Parse optional regions
    regions = None
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

    # Parse optional reference identities
    reference_identities = None
    if reference_str:
        reference_identities = [n.strip() for n in reference_str.split(",") if n.strip()]

    try:
        # Load and encode images
        image_b64_list = []
        image_sizes = []
        for idx, path in enumerate(image_paths):
            # If regions target this image, draw bounding boxes
            img_regions = [r for r in (regions or []) if r.get("image_index", 0) == idx]
            if img_regions:
                b64 = _draw_regions_on_image(path, img_regions)
                img = cv2.imread(path)
                h, w = img.shape[:2]
                image_sizes.append({"width": w, "height": h})
            else:
                b64, w, h = _load_and_encode_image(path)
                image_sizes.append({"width": w, "height": h})
            image_b64_list.append(b64)

        # Build prompt and call VLM
        prompt = _build_prompt(len(image_paths), regions, reference_identities)
        vlm_result = await _call_vlm(image_b64_list, prompt, max_tokens)

        vlm_usage = None
        if VLM_REPORT_USAGE:
            vlm_usage = {
                "input_tokens": vlm_result.get("input_tokens", 0),
                "output_tokens": vlm_result.get("output_tokens", 0),
                "total_tokens": vlm_result.get("input_tokens", 0) + vlm_result.get("output_tokens", 0),
                "model": vlm_result.get("model") or VLM_MODEL_NAME,
            }

        if vlm_result["error"]:
            result = {
                "success": False,
                "error": vlm_result["error"],
                "vlm_response": vlm_result["response"],
            }
            if VLM_REPORT_USAGE:
                result["vlm_usage"] = vlm_usage
            return result

        # Parse response
        parsed = _parse_vlm_response(vlm_result["response"])

        persons = parsed.get("persons", [])
        cross_matches = parsed.get("cross_image_matches", [])

        result = {
            "success": True,
            "image_paths": image_paths,
            "image_count": len(image_paths),
            "image_sizes": image_sizes,
            "persons": persons,
            "person_count": len(persons),
            "cross_image_matches": cross_matches,
            "reference_identities": reference_identities,
        }
        if VLM_REPORT_USAGE:
            result["vlm_usage"] = vlm_usage
        return result

    except Exception as e:
        return make_error_response(str(e), {"traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = create_mcp_server(
        server_name="person-identification-mcp",
        tool_defs_source="_tool_defs/recognition.yaml",
        handlers={"person_identification": handle_person_identification},
        is_file=True,
    )
    asyncio.run(run_server(app))
