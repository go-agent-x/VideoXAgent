#!/usr/bin/env python3
"""
Grounding DINO local inference API.

Provides call_gdino_api() interface for all handlers.
"""

import base64
import contextlib
import io
import math
import os
import sys
import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
from PIL import Image

# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")
from _base.config import get_config

import groundingdino

_gdino_cfg = get_config("grounding_dino")
GDINO_CONFIG = os.environ.get("GDINO_CONFIG") or str(
    Path(groundingdino.__file__).parent / "config/GroundingDINO_SwinB_cfg.py"
)
GDINO_CHECKPOINT = _gdino_cfg["checkpoint"]
GDINO_DEVICE = _gdino_cfg["device"]

# ---------------------------------------------------------------------------
# Model singleton (thread-safe lazy load)
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")

                # Patch transformers BertModel for compatibility with groundingdino
                # get_head_mask was removed in transformers 5.x
                from transformers import BertModel
                if not hasattr(BertModel, 'get_head_mask'):
                    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                        if head_mask is not None:
                            import torch
                            if head_mask.dim() == 1:
                                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                            elif head_mask.dim() == 2:
                                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                            return head_mask
                        return [None] * num_hidden_layers
                    BertModel.get_head_mask = _get_head_mask

                # Patch get_extended_attention_mask to accept torch.device as 3rd arg.
                # groundingdino bertwarper passes `device` (torch.device) as the 3rd arg
                # to get_extended_attention_mask, but transformers 5.x expects `dtype`.
                _orig_attn_mask = BertModel.get_extended_attention_mask
                def _patched_attn_mask(self, attention_mask, input_shape, device_or_dtype=None):
                    if isinstance(device_or_dtype, torch.device):
                        return _orig_attn_mask(self, attention_mask, input_shape, None)
                    return _orig_attn_mask(self, attention_mask, input_shape, device_or_dtype)
                BertModel.get_extended_attention_mask = _patched_attn_mask

                from groundingdino.util.slconfig import SLConfig
                args = SLConfig.fromfile(GDINO_CONFIG)
                # Override text_encoder_type to local path
                if getattr(args, "text_encoder_type", "bert-base-uncased") == "bert-base-uncased":
                    _bert_local = _gdino_cfg.get("bert_path", "")
                    if _bert_local and os.path.isdir(_bert_local):
                        args.text_encoder_type = _bert_local
                from groundingdino.models import build_model as _build_gdino
                import torch
                # Suppress stdout during model build (tqdm, print statements)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    candidate = _build_gdino(args)
                    ckpt = torch.load(GDINO_CHECKPOINT, map_location="cpu")
                    candidate.load_state_dict(ckpt.get("model", ckpt), strict=False)
                candidate.eval()
                if GDINO_DEVICE == "cuda":
                    candidate.cuda()
                else:
                    candidate.to(GDINO_DEVICE)
                # Failed initialization must not leave an untrained or unusable
                # singleton for later category-sweep batches to reuse.
                _model = candidate
    return _model


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def encode_image_base64(image_path: str) -> str:
    """Encode local image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _load_image(image_source: str) -> Image.Image:
    """Load image from local path."""
    if image_source.startswith(("http://", "https://")):
        import requests
        from io import BytesIO
        resp = requests.get(image_source, timeout=30)
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_source).convert("RGB")


# ---------------------------------------------------------------------------
# Core API client (local inference)
# ---------------------------------------------------------------------------
def call_gdino_api(
    image_source: str,
    prompt: str,
    targets: List[str],
    box_threshold: float = 0.25,
    iou_threshold: float = 0.8,
    *,
    categories: Optional[List[str]] = None,
    max_objects: int = 50,
) -> Dict[str, Any]:
    """
    Local Grounding DINO inference — same interface as DINO-X API.

    Args:
        image_source: Local file path or URL.
        prompt: Text prompt. Legacy ``<prompt_free>`` requests a bounded category
            sweep because the local backend requires text prompts.
        targets: List of targets (only "bbox" supported; "mask"/"pose_keypoints" ignored).
        box_threshold: Confidence threshold for bounding boxes.
        iou_threshold: IoU threshold for NMS (used as text_threshold).
        categories: Optional category phrases for the legacy sentinel's sweep.
        max_objects: Maximum merged sweep results (1 to 50).

    Returns:
        dict with keys: success (bool), object_count (int), objects (list), [error].
    """
    if prompt == "<prompt_free>":
        from gdino_fallback import run_category_sweep
        return run_category_sweep(
            call_gdino_api=call_gdino_api,
            image_source=image_source,
            box_threshold=box_threshold,
            max_objects=max_objects,
            categories=categories,
        )

    if not image_source.startswith(("http://", "https://")) and not os.path.exists(image_source):
        return {"success": False, "error": f"Image not found: {image_source}", "objects": []}

    try:
        from groundingdino.util.inference import load_image, predict
        import numpy as np

        model = _get_model()

        # load_image returns (image_source_for_vis, image_tensor)
        image_pil = _load_image(image_source)
        w, h = image_pil.size

        # Use groundingdino's load_image which handles transform
        _, image_tensor = load_image(image_source)

        # Normalize prompt: ensure ends with "."
        text_prompt = prompt.strip()
        if not text_prompt.endswith("."):
            text_prompt += "."

        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=box_threshold,
            device=GDINO_DEVICE,
        )

        objects = []
        for box, score, phrase in zip(boxes, logits, phrases):
            # boxes are in cxcywh normalized format, convert to xyxy pixel coords
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            objects.append({
                "category": phrase,
                "score": round(float(score), 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })

        return {"success": True, "object_count": len(objects), "objects": objects}

    except Exception as e:
        return {"success": False, "error": str(e), "objects": []}


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------
def normalize_categories(categories: str) -> str:
    """Normalize user-supplied category string to dot-separated format."""
    cats = categories.replace(",", " ").replace(".", " ").split()
    return " . ".join(c.strip() for c in cats if c.strip())


# ---------------------------------------------------------------------------
# Bbox geometry
# ---------------------------------------------------------------------------
def bbox_center(bbox: List[float]):
    """Return center (cx, cy) of bbox [x1, y1, x2, y2]."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def compute_bbox_area(bbox: List[float]) -> float:
    """Compute area of bbox [x1, y1, x2, y2]."""
    if len(bbox) < 4:
        return 0.0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_spatial_relation(
    bbox_a: List[float], bbox_b: List[float]
) -> Dict[str, Any]:
    """Compute spatial relation between object A and object B."""
    ca = bbox_center(bbox_a)
    cb = bbox_center(bbox_b)
    h_dist = ca[0] - cb[0]
    v_dist = ca[1] - cb[1]

    if abs(h_dist) < 20:
        h_rel = "horizontally_aligned"
    elif h_dist < 0:
        h_rel = "LEFT"
    else:
        h_rel = "RIGHT"

    if abs(v_dist) < 20:
        v_rel = "vertically_aligned"
    elif v_dist < 0:
        v_rel = "ABOVE"
    else:
        v_rel = "BELOW"

    spatial_str = f"object_a is to the {h_rel} of and {v_rel} object_b"
    if h_rel == "horizontally_aligned" and v_rel == "vertically_aligned":
        spatial_str = "object_a and object_b overlap at center"

    return {
        "spatial_relation": spatial_str,
        "horizontal_relation": h_rel,
        "vertical_relation": v_rel,
        "horizontal_distance": round(abs(h_dist), 1),
        "vertical_distance": round(abs(v_dist), 1),
        "euclidean_distance": round(math.sqrt(h_dist ** 2 + v_dist ** 2), 1),
        "overlap_iou": round(compute_iou(bbox_a, bbox_b), 4),
    }


# ---------------------------------------------------------------------------
# Region labeling
# ---------------------------------------------------------------------------
def format_bbox_region(bbox: List[float], img_width: int = 0, img_height: int = 0) -> str:
    """Describe the region of a bbox within the image."""
    if len(bbox) < 4:
        return "unknown"
    cx, cy = bbox_center(bbox)
    if img_width > 0 and img_height > 0:
        rx = cx / img_width
        ry = cy / img_height
    else:
        rx = min(cx / 640, 1.0)
        ry = min(cy / 480, 1.0)

    if ry < 0.33:
        v = "top"
    elif ry < 0.66:
        v = "center"
    else:
        v = "bottom"

    if rx < 0.33:
        h = "left"
    elif rx < 0.66:
        h = "center"
    else:
        h = "right"

    if v == "center" and h == "center":
        return "center"
    if v == "center":
        return h
    if h == "center":
        return v
    return f"{v}-{h}"


def relative_size_label(area: float, all_areas: List[float]) -> str:
    """Classify an area as small / medium / large relative to peers."""
    if not all_areas:
        return "unknown"
    max_area = max(all_areas)
    if max_area == 0:
        return "unknown"
    ratio = area / max_area
    if ratio > 0.6:
        return "large"
    if ratio > 0.2:
        return "medium"
    return "small"


# ---------------------------------------------------------------------------
# NMS (Non-Maximum Suppression)
# ---------------------------------------------------------------------------
def nms_filter(objects: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Apply greedy NMS on a list of detection dicts."""
    if not objects:
        return []
    sorted_objs = sorted(objects, key=lambda o: o.get("score", 0), reverse=True)
    keep: List[Dict] = []
    for obj in sorted_objs:
        bbox = obj.get("bbox", [])
        if len(bbox) < 4:
            continue
        suppressed = False
        for kept in keep:
            if compute_iou(bbox, kept.get("bbox", [])) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(obj)
    return keep


# ---------------------------------------------------------------------------
# Pose / action inference (kept for interface compatibility, not used without pose_estimate)
# ---------------------------------------------------------------------------
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def parse_keypoints(raw_keypoints: Any) -> Dict[str, Dict[str, Any]]:
    """Parse raw keypoints into named dict."""
    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_keypoints, list):
        if len(raw_keypoints) >= 51:
            for i, name in enumerate(COCO_KEYPOINT_NAMES):
                idx = i * 3
                if idx + 2 < len(raw_keypoints):
                    result[name] = {
                        "x": round(raw_keypoints[idx], 1),
                        "y": round(raw_keypoints[idx + 1], 1),
                        "visible": raw_keypoints[idx + 2] > 0,
                    }
    return result


def infer_action_hints(keypoints: Dict[str, Dict[str, Any]]) -> List[str]:
    """Infer basic action hints from parsed keypoints."""
    hints: List[str] = []

    def _get(name: str) -> Optional[Tuple[float, float]]:
        kp = keypoints.get(name)
        if kp and kp.get("visible"):
            return (kp["x"], kp["y"])
        return None

    l_shoulder = _get("left_shoulder")
    r_shoulder = _get("right_shoulder")
    l_wrist = _get("left_wrist")
    r_wrist = _get("right_wrist")
    l_hip = _get("left_hip")
    r_hip = _get("right_hip")
    l_knee = _get("left_knee")
    r_knee = _get("right_knee")

    if l_wrist and l_shoulder and l_wrist[1] < l_shoulder[1] - 20:
        hints.append("left_hand_raised")
    if r_wrist and r_shoulder and r_wrist[1] < r_shoulder[1] - 20:
        hints.append("right_hand_raised")

    if l_hip and l_knee and r_hip and r_knee:
        avg_hip_y = (l_hip[1] + r_hip[1]) / 2
        avg_knee_y = (l_knee[1] + r_knee[1]) / 2
        hip_knee_dist = avg_knee_y - avg_hip_y
        if hip_knee_dist > 50:
            hints.append("standing")
        elif hip_knee_dist < 20:
            hints.append("sitting")

    if not hints:
        hints.append("neutral_pose")
    return hints
