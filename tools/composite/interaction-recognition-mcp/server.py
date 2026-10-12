#!/usr/bin/env python3
"""Interaction Recognition MCP Server — Structured entity interaction detection."""
import asyncio
import base64
import json
import math
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

from _base.paths import service_dir
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(service_dir("grounding-dino")))

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response
from _base.config import get_config, pick_api_key, normalize_vlm_url
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
    raise ImportError("OpenCV and NumPy are required. From the repository root, run: bash scripts/install_profiles.sh vision")


from gdino_api import call_gdino_api

# ---------------------------------------------------------------------------
# Configuration from _base/config.py
# ---------------------------------------------------------------------------
_vlm_cfg = get_config("vlm")
VLM_API_URL = normalize_vlm_url(_vlm_cfg["api_url"])
VLM_API_KEY = pick_api_key("VLM_API_KEY")
VLM_MODEL_NAME = _vlm_cfg["model"]
VLM_REPORT_USAGE = os.environ.get("VLM_REPORT_USAGE", "").lower() == "true"

MAX_IMAGE_DIMENSION = 2048
VLM_TIMEOUT = 240.0

MAX_CONCURRENT_VLM = get_max_concurrent_vlm()
_vlm_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_VLM)

# Distinct colors for bbox annotation (BGR for OpenCV)
_COLORS = [
    (0, 0, 255),    # red
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 128, 255),  # orange
    (128, 0, 255),  # purple
    (0, 255, 128),  # spring green
    (255, 128, 0),  # sky blue
]

# Human-like categories for prioritizing interaction pairs
_HUMAN_CATS = {"person", "man", "woman", "boy", "girl", "child", "human", "people",
               "player", "athlete", "rider", "pedestrian", "worker", "baby"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bbox_center(bbox: list) -> tuple:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _bbox_area(bbox: list) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(bbox_a: list, bbox_b: list) -> float:
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _overlap_ratio(inner_bbox: list, outer_bbox: list) -> float:
    inner_area = _bbox_area(inner_bbox)
    if inner_area <= 0:
        return 0.0
    return _intersection_area(inner_bbox, outer_bbox) / inner_area


def _is_human(category: str) -> bool:
    cat_lower = category.lower()
    return any(h in cat_lower for h in _HUMAN_CATS)


def _fallback_interactions(entities: list, candidate_pairs: list, max_interactions: int) -> list:
    """
    Conservative geometry/category fallback when the VLM is unavailable.
    """
    vehicle_terms = {"car", "truck", "bus", "motorcycle", "bicycle", "boat", "train", "airplane"}
    seat_terms = {"chair", "couch", "bench", "bed", "sofa"}
    support_terms = {"table", "desk"}

    interactions = []
    for idx_a, idx_b, norm_dist in candidate_pairs:
        ent_a = entities[idx_a]
        ent_b = entities[idx_b]
        cat_a = ent_a["category"].lower()
        cat_b = ent_b["category"].lower()
        bbox_a = ent_a["bbox"]
        bbox_b = ent_b["bbox"]

        a_human = _is_human(cat_a)
        b_human = _is_human(cat_b)
        overlap_a_in_b = _overlap_ratio(bbox_a, bbox_b)
        overlap_b_in_a = _overlap_ratio(bbox_b, bbox_a)

        predicate = None
        confidence = 0.0
        agent = ent_a
        patient = ent_b
        description = ""

        if a_human and any(term in cat_b for term in vehicle_terms) and overlap_a_in_b > 0.65:
            predicate = "inside"
            confidence = min(0.95, 0.55 + overlap_a_in_b * 0.35)
            description = f"{ent_a['category']} is inside {ent_b['category']}."
        elif b_human and any(term in cat_a for term in vehicle_terms) and overlap_b_in_a > 0.65:
            predicate = "inside"
            confidence = min(0.95, 0.55 + overlap_b_in_a * 0.35)
            agent = ent_b
            patient = ent_a
            description = f"{ent_b['category']} is inside {ent_a['category']}."
        elif a_human and any(term in cat_b for term in seat_terms) and overlap_a_in_b > 0.35:
            predicate = "sitting_on"
            confidence = min(0.9, 0.45 + overlap_a_in_b * 0.35)
            description = f"{ent_a['category']} appears to be sitting on {ent_b['category']}."
        elif b_human and any(term in cat_a for term in seat_terms) and overlap_b_in_a > 0.35:
            predicate = "sitting_on"
            confidence = min(0.9, 0.45 + overlap_b_in_a * 0.35)
            agent = ent_b
            patient = ent_a
            description = f"{ent_b['category']} appears to be sitting on {ent_a['category']}."
        elif a_human and any(term in cat_b for term in support_terms) and norm_dist < 0.12:
            predicate = "at"
            confidence = 0.45
            description = f"{ent_a['category']} is near {ent_b['category']}."
        elif b_human and any(term in cat_a for term in support_terms) and norm_dist < 0.12:
            predicate = "at"
            confidence = 0.45
            agent = ent_b
            patient = ent_a
            description = f"{ent_b['category']} is near {ent_a['category']}."
        elif norm_dist < 0.08:
            predicate = "near"
            confidence = 0.3
            description = f"{ent_a['category']} is near {ent_b['category']}."

        if not predicate:
            continue

        interactions.append({
            "predicate": predicate,
            "confidence": round(confidence, 3),
            "agent": {"entity_id": agent["id"], "category": agent["category"], "bbox": agent["bbox"]},
            "patient": {"entity_id": patient["id"], "category": patient["category"], "bbox": patient["bbox"]},
            "instrument": None,
            "triplet": f"{agent['category']} \u2192 {predicate} \u2192 {patient['category']}",
            "description": description,
        })

        if len(interactions) >= max_interactions:
            break

    return interactions


def _compute_candidate_pairs(
    entities: list,
    image_w: int,
    image_h: int,
    proximity_ratio: float,
    max_pairs: int = 50,
) -> list:
    """
    Generate candidate interacting pairs sorted by proximity.
    Returns list of (idx_a, idx_b, normalized_distance).
    """
    diag = math.sqrt(image_w ** 2 + image_h ** 2)
    if diag == 0:
        return []

    pairs = []
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            ca = _bbox_center(entities[i]["bbox"])
            cb = _bbox_center(entities[j]["bbox"])
            dist = math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
            norm_dist = dist / diag

            if norm_dist > proximity_ratio:
                continue

            # Boost priority for human-involved pairs
            has_human = _is_human(entities[i]["category"]) or _is_human(entities[j]["category"])
            sort_key = norm_dist * (0.5 if has_human else 1.0)
            pairs.append((i, j, round(norm_dist, 4), sort_key))

    pairs.sort(key=lambda x: x[3])
    return [(a, b, d) for a, b, d, _ in pairs[:max_pairs]]


def _annotate_image(img: np.ndarray, entities: list) -> np.ndarray:
    """Draw numbered bounding boxes with category labels on the image."""
    annotated = img.copy()
    for idx, ent in enumerate(entities):
        bbox = ent["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = _COLORS[idx % len(_COLORS)]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"[{idx}] {ent['category']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw label background
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw + 4, label_y + baseline), color, -1)
        cv2.putText(annotated, label, (x1 + 2, label_y - 2), font, font_scale, (255, 255, 255), thickness)

    return annotated


def _encode_image(img: np.ndarray) -> str:
    """Resize if needed and encode to base64 JPEG."""
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def _build_vlm_prompt(entities: list, candidate_pairs: list) -> str:
    """Build the structured VLM prompt for interaction recognition."""
    entity_lines = []
    for idx, ent in enumerate(entities):
        bbox_str = f"[{int(ent['bbox'][0])},{int(ent['bbox'][1])},{int(ent['bbox'][2])},{int(ent['bbox'][3])}]"
        entity_lines.append(f"  [{idx}] {ent['category']} (score: {ent['score']:.2f}) at bbox {bbox_str}")

    pair_strs = [f"[{a}]-[{b}]" for a, b, _ in candidate_pairs]

    prompt = f"""You are analyzing entity interactions in an image. Detected entities are marked with numbered colored bounding boxes:

{chr(10).join(entity_lines)}

Candidate pairs by proximity: {', '.join(pair_strs) if pair_strs else 'none'}

For each meaningful interaction clearly visible in the image, output a JSON array. Each element:
{{
  "predicate": "<verb or verb_phrase, e.g. holding, sitting_on, riding>",
  "agent_id": <integer, entity performing the action>,
  "patient_id": <integer, entity receiving/affected by the action>,
  "instrument_id": <integer or null, entity used as a tool/instrument>,
  "confidence": <float 0.0-1.0>,
  "description": "<one sentence describing the interaction>"
}}

Semantic role definitions:
- agent: the entity actively performing or initiating the action
- patient: the entity being acted upon, affected, or receiving the action
- instrument: an entity used as a tool or means to perform the action (null if none)

Common predicates: holding, sitting_on, standing_on, riding, eating, drinking_from, using, carrying, wearing, playing_with, looking_at, talking_to, leaning_against, pushing, pulling, throwing, catching, petting, feeding, hugging, kicking, hitting, reading, typing_on, pointing_at, reaching_for, climbing, driving, walking_with, chasing, leading

Rules:
- Only report interactions clearly visible in the image — do not hallucinate
- Each interaction must reference entities by their [id] numbers above
- Include interactions between any entity types (human-object, human-human, object-object)
- Prioritize human-involved interactions
- If no interactions are visible, return []

Output ONLY the JSON array, no other text."""

    return prompt


def _parse_vlm_interactions(response_text: str) -> list:
    """Parse VLM response into a list of interaction dicts, with fallback strategies."""
    text = response_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they are code fences
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
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract individual JSON objects
    objects = []
    for m in re.finditer(r'\{[^{}]+\}', text):
        try:
            obj = json.loads(m.group())
            if "predicate" in obj:
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    return objects


async def _call_vlm(image_b64: str, query: str, max_tokens: int | None = None) -> dict:
    """Call VLM API with annotated image and structured prompt."""
    max_tokens = resolve_max_tokens("recognize_interactions", max_tokens)
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        {"type": "text", "text": query},
    ]

    payload = {
        "model": VLM_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    _ctk = build_chat_template_kwargs(VLM_MODEL_NAME)
    if _ctk is not None:
        payload["chat_template_kwargs"] = _ctk

    headers = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"

    result = {
        "response": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None,
        "model": VLM_MODEL_NAME,
    }

    async with httpx.AsyncClient(timeout=VLM_TIMEOUT) as client:
        try:
            async with _vlm_semaphore:
                resp = await client.post(VLM_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            result["input_tokens"] = usage.get("prompt_tokens", 0)
            result["output_tokens"] = usage.get("completion_tokens", 0)

            if "choices" in data and len(data["choices"]) > 0:
                retry_info = await maybe_thinking_off_retry(
                    data=data, payload=payload, client=client,
                    url=VLM_API_URL, headers=headers,
                    server_tag="interaction-recognition",
                    semaphore=None,
                )
                if retry_info["retried"]:
                    result["input_tokens"] += retry_info["extra_input_tokens"]
                    result["output_tokens"] += retry_info["extra_output_tokens"]
                final = retry_info["data"]
                if "choices" in final and final["choices"]:
                    _content = final["choices"][0]["message"].get("content") or ""
                    if _content.strip():
                        result["response"] = _content
                        if retry_info["retried"] and retry_info["retry_succeeded"]:
                            result["thinking_off_retry"] = True
                    else:
                        result["error"] = "Empty content after thinking-off retry"
                else:
                    result["error"] = "Invalid VLM API response format"
            else:
                result["error"] = "Invalid VLM API response format"
        except httpx.TimeoutException:
            result["error"] = f"VLM API timeout after {VLM_TIMEOUT}s"
        except httpx.HTTPStatusError as e:
            result["error"] = f"VLM API HTTP {e.response.status_code}"
        except Exception as e:
            result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------
async def _handle_recognize_interactions(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    focus_entities = arguments.get("focus_entities", "")
    box_threshold = arguments.get("box_threshold", 0.2)
    max_interactions = arguments.get("max_interactions", 20)
    proximity_ratio = arguments.get("proximity_ratio", 0.5)

    if not image_path:
        return {"success": False, "error": "image_path is required"}

    # ---- Step 1: Entity Detection via DINO-X ----
    prompt = focus_entities.strip() if focus_entities else "<prompt_free>"
    detect_result = call_gdino_api(
        image_source=image_path,
        prompt=prompt,
        targets=["bbox"],
        box_threshold=box_threshold,
    )

    if not detect_result.get("success"):
        return detect_result

    raw_objects = detect_result.get("objects", [])
    if len(raw_objects) == 0:
        return {
            "success": True,
            "entities": [],
            "interactions": [],
            "interaction_count": 0,
            "entity_count": 0,
            "candidate_pairs_evaluated": 0,
            "image_path": image_path,
            "note": "No entities detected in the image",
        }

    # Build entity list (limit to top entities by score to keep VLM prompt manageable)
    raw_objects.sort(key=lambda o: o.get("score", 0), reverse=True)
    entities = []
    for idx, obj in enumerate(raw_objects[:30]):
        entities.append({
            "id": idx,
            "category": obj.get("category", "object"),
            "bbox": obj.get("bbox", []),
            "score": obj.get("score", 0.0),
        })

    if len(entities) < 2:
        return {
            "success": True,
            "entities": entities,
            "interactions": [],
            "interaction_count": 0,
            "entity_count": len(entities),
            "candidate_pairs_evaluated": 0,
            "image_path": image_path,
            "note": "Only one entity detected — interactions require at least two entities",
        }

    # ---- Step 2: Load image and compute dimensions ----
    if image_path.startswith(("http://", "https://")):
        # For URLs, use bboxes from detection to estimate image size
        all_x2 = [e["bbox"][2] for e in entities if len(e["bbox"]) >= 4]
        all_y2 = [e["bbox"][3] for e in entities if len(e["bbox"]) >= 4]
        image_w = int(max(all_x2)) if all_x2 else 1920
        image_h = int(max(all_y2)) if all_y2 else 1080
        img = None
    else:
        if not os.path.isfile(image_path):
            return {"success": False, "error": f"File not found: {image_path}"}
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": f"Cannot read image: {image_path}"}
        image_h, image_w = img.shape[:2]

    # ---- Step 3: Candidate pair generation ----
    candidate_pairs = _compute_candidate_pairs(
        entities, image_w, image_h, proximity_ratio, max_pairs=max_interactions * 3,
    )

    if not candidate_pairs:
        return {
            "success": True,
            "image_size": {"width": image_w, "height": image_h},
            "entities": entities,
            "interactions": [],
            "interaction_count": 0,
            "entity_count": len(entities),
            "candidate_pairs_evaluated": 0,
            "image_path": image_path,
            "note": "No entity pairs within proximity threshold",
        }

    # ---- Step 4: Annotate image and call VLM ----
    if img is not None:
        annotated = _annotate_image(img, entities)
        img_b64 = _encode_image(annotated)
    else:
        # URL image: cannot annotate locally, send raw detection info to VLM
        # Download image first
        try:
            import requests as req_lib
            r = req_lib.get(image_path, timeout=30)
            r.raise_for_status()
            arr = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                annotated = _annotate_image(img, entities)
                img_b64 = _encode_image(annotated)
                image_h, image_w = img.shape[:2]
            else:
                return {"success": False, "error": "Cannot decode image from URL"}
        except Exception as e:
            return {"success": False, "error": f"Cannot fetch image from URL: {e}"}

    vlm_prompt = _build_vlm_prompt(entities, candidate_pairs)
    vlm_result = await _call_vlm(img_b64, vlm_prompt, max_tokens=1536)
    vlm_usage = {
        "input_tokens": vlm_result.get("input_tokens", 0),
        "output_tokens": vlm_result.get("output_tokens", 0),
        "total_tokens": vlm_result.get("input_tokens", 0) + vlm_result.get("output_tokens", 0),
        "model": vlm_result.get("model") or VLM_MODEL_NAME,
    }

    if vlm_result["error"]:
        fallback_interactions = _fallback_interactions(entities, candidate_pairs, max_interactions=max_interactions)
        return {
            "success": True,
            "warning": f"VLM call failed: {vlm_result['error']}",
            "entities": entities,
            "interactions": fallback_interactions,
            "interaction_count": len(fallback_interactions),
            "entity_count": len(entities),
            "candidate_pairs_evaluated": len(candidate_pairs),
            "interaction_source": "heuristic_fallback",
            "image_path": image_path,
            **({"vlm_usage": vlm_usage} if VLM_REPORT_USAGE else {}),
        }

    # ---- Step 5: Parse VLM response into structured interactions ----
    raw_interactions = _parse_vlm_interactions(vlm_result["response"])

    entity_ids = {e["id"] for e in entities}
    interactions = []
    for raw in raw_interactions[:max_interactions]:
        predicate = raw.get("predicate", "")
        agent_id = raw.get("agent_id")
        patient_id = raw.get("patient_id")
        instrument_id = raw.get("instrument_id")
        confidence = raw.get("confidence", 0.5)
        description = raw.get("description", "")

        # Validate entity references
        if agent_id not in entity_ids or patient_id not in entity_ids:
            continue
        if instrument_id is not None and instrument_id not in entity_ids:
            instrument_id = None

        agent_ent = entities[agent_id]
        patient_ent = entities[patient_id]
        instrument_ent = entities[instrument_id] if instrument_id is not None else None

        triplet = f"{agent_ent['category']} \u2192 {predicate} \u2192 {patient_ent['category']}"
        if instrument_ent:
            triplet += f" (with {instrument_ent['category']})"

        interaction = {
            "predicate": predicate,
            "confidence": round(float(confidence), 3),
            "agent": {"entity_id": agent_id, "category": agent_ent["category"], "bbox": agent_ent["bbox"]},
            "patient": {"entity_id": patient_id, "category": patient_ent["category"], "bbox": patient_ent["bbox"]},
            "instrument": (
                {"entity_id": instrument_id, "category": instrument_ent["category"], "bbox": instrument_ent["bbox"]}
                if instrument_ent else None
            ),
            "triplet": triplet,
            "description": description,
        }
        interactions.append(interaction)

    return {
        "success": True,
        "image_size": {"width": image_w, "height": image_h},
        "entities": entities,
        "interactions": interactions,
        "interaction_count": len(interactions),
        "entity_count": len(entities),
        "candidate_pairs_evaluated": len(candidate_pairs),
        "interaction_source": "vlm",
        "image_path": image_path,
        **({"vlm_usage": vlm_usage} if VLM_REPORT_USAGE else {}),
    }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


async def handle_recognize_interactions(arguments: dict) -> dict:
    return await _handle_recognize_interactions(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="interaction-recognition",
        tool_defs_source="_tool_defs/action.yaml",
        handlers={
            "recognize_interactions": handle_recognize_interactions,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
