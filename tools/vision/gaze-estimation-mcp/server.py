#!/usr/bin/env python3
"""Gaze Estimation MCP Server — Estimate gaze direction and head orientation."""
import asyncio
import gc
import json
import math
import os
import sys
import threading
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

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response
from _base.config import get_config
from _base.device import insightface_device_options

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration from _base/config.py
# ---------------------------------------------------------------------------
_gaze_cfg = get_config("gaze")
DEVICE = _gaze_cfg["device"]
L2CS_WEIGHTS = _gaze_cfg["weights"]
L2CS_NUM_BINS = 90
L2CS_INPUT_SIZE = 448

_insightface_cfg = get_config("insightface")
_INSIGHTFACE_MODEL = _insightface_cfg["model"]
_INSIGHTFACE_PROVIDERS = _insightface_cfg["providers"]

# ---------------------------------------------------------------------------
# Section 1: Face Detection Backend (pattern from face-detect-mcp)
# ---------------------------------------------------------------------------
_face_lock = threading.Lock()
_face_app = None
_face_backend = "uninitialized"  # "insightface" | "opencv"
_haar_cascade = None


def _try_init_insightface(det_size: tuple = (640, 640)):
    """Attempt to load InsightFace. Returns FaceAnalysis app or None."""
    try:
        os.environ.setdefault("ALBUMENTATIONS_CHECK_VERSION", "0")
        from insightface.app import FaceAnalysis

        provider_list, context = insightface_device_options(_INSIGHTFACE_PROVIDERS)

        app = FaceAnalysis(name=_INSIGHTFACE_MODEL, providers=provider_list)
        app.prepare(ctx_id=context, det_size=det_size)
        return app
    except Exception as exc:
        print(
            f"[gaze_estimation] InsightFace unavailable ({exc}), falling back to OpenCV.",
            file=sys.stderr,
        )
        return None


def _get_haar():
    global _haar_cascade
    if _haar_cascade is None:
        xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _haar_cascade = cv2.CascadeClassifier(xml)
    return _haar_cascade


def _init_face_backend(det_size: tuple = (640, 640)):
    global _face_app, _face_backend
    if _face_backend != "uninitialized":
        return
    with _face_lock:
        if _face_backend != "uninitialized":
            return
        app = _try_init_insightface(det_size)
        if app is not None:
            _face_app = app
            _face_backend = "insightface"
        else:
            _face_backend = "opencv"


def _detect_faces_insightface(img: np.ndarray) -> list[dict]:
    faces = _face_app.get(img)
    results = []
    for face in faces:
        bbox = face.bbox.astype(float).tolist()
        kps = face.kps.astype(float).tolist() if hasattr(face, "kps") and face.kps is not None else None
        results.append({
            "bbox": [round(b, 1) for b in bbox],
            "det_score": round(float(face.det_score), 4),
            "kps": kps,
        })
    results.sort(
        key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]),
        reverse=True,
    )
    return results


def _detect_faces_opencv(img: np.ndarray) -> list[dict]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = _get_haar()
    rects = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    h_img, w_img = img.shape[:2]
    results = []
    for (x, y, w, h) in rects:
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w_img, int(x + w))
        y2 = min(h_img, int(y + h))
        results.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "det_score": 0.90,
            "kps": None,
        })
    results.sort(
        key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]),
        reverse=True,
    )
    return results


def _detect_faces(img: np.ndarray, box_threshold: float = 0.5) -> list[dict]:
    _init_face_backend()
    if _face_backend == "insightface":
        faces = _detect_faces_insightface(img)
    else:
        faces = _detect_faces_opencv(img)
    return [f for f in faces if f["det_score"] >= box_threshold]


# ---------------------------------------------------------------------------
# Section 2: L2CS-Net Model Definition
# ---------------------------------------------------------------------------
_gaze_lock = threading.Lock()
_l2cs_model = None
_gaze_backend = "uninitialized"  # "l2cs" | "head_pose" | "none"
_torch_device = None


def _define_l2cs_model():
    """Define L2CS-Net model class. Returns the class."""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet50

    class L2CS(nn.Module):
        def __init__(self, num_bins=L2CS_NUM_BINS):
            super().__init__()
            base = resnet50(weights=None)
            # Remove the original FC layer, keep everything up to avgpool
            self.backbone = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
                base.avgpool,
            )
            self.fc_yaw = nn.Linear(2048, num_bins)
            self.fc_pitch = nn.Linear(2048, num_bins)

        def forward(self, x):
            features = self.backbone(x)
            features = features.flatten(1)
            yaw = self.fc_yaw(features)
            pitch = self.fc_pitch(features)
            return yaw, pitch

    return L2CS


def _try_init_l2cs():
    """Attempt to load L2CS-Net model from weights. Returns (model, device) or (None, None)."""
    if not L2CS_WEIGHTS or not os.path.isfile(L2CS_WEIGHTS):
        if L2CS_WEIGHTS:
            print(
                f"[gaze_estimation] L2CS weights not found: {L2CS_WEIGHTS}",
                file=sys.stderr,
            )
        else:
            print(
                "[gaze_estimation] L2CS_WEIGHTS not configured, will use head_pose fallback.",
                file=sys.stderr,
            )
        return None, None

    try:
        import torch

        device = torch.device(
            "cuda" if torch.cuda.is_available() and DEVICE == "cuda" else "cpu"
        )

        L2CS = _define_l2cs_model()
        model = L2CS(num_bins=L2CS_NUM_BINS)

        # Load weights — support both full state_dict and wrapped formats
        checkpoint = torch.load(L2CS_WEIGHTS, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Handle potential key prefix differences
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            cleaned[new_key] = v

        model.load_state_dict(cleaned, strict=False)
        model.to(device)
        model.eval()

        print(f"[gaze_estimation] L2CS-Net loaded on {device}.", file=sys.stderr)
        return model, device

    except Exception as exc:
        print(
            f"[gaze_estimation] Failed to load L2CS model ({exc}), using head_pose fallback.",
            file=sys.stderr,
        )
        return None, None


def _init_gaze_backend():
    global _l2cs_model, _gaze_backend, _torch_device
    if _gaze_backend != "uninitialized":
        return
    with _gaze_lock:
        if _gaze_backend != "uninitialized":
            return
        model, device = _try_init_l2cs()
        if model is not None:
            _l2cs_model = model
            _torch_device = device
            _gaze_backend = "l2cs"
        else:
            _gaze_backend = "head_pose"


# ---------------------------------------------------------------------------
# Section 3: L2CS-Net Inference
# ---------------------------------------------------------------------------
def _preprocess_face_l2cs(face_crop_bgr: np.ndarray):
    """Preprocess a face crop for L2CS-Net. Returns torch tensor (1, 3, 448, 448)."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((L2CS_INPUT_SIZE, L2CS_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(face_rgb)
    return tensor.unsqueeze(0)


def _predict_gaze_l2cs(face_crop_bgr: np.ndarray) -> tuple[float, float]:
    """Run L2CS-Net inference. Returns (yaw_degrees, pitch_degrees)."""
    import torch
    import torch.nn.functional as F

    inp = _preprocess_face_l2cs(face_crop_bgr).to(_torch_device)

    with torch.no_grad():
        yaw_logits, pitch_logits = _l2cs_model(inp)

    idx_tensor = torch.arange(L2CS_NUM_BINS, dtype=torch.float32, device=_torch_device)

    yaw_soft = F.softmax(yaw_logits, dim=1)
    pitch_soft = F.softmax(pitch_logits, dim=1)

    yaw_deg = float((yaw_soft * idx_tensor).sum(dim=1).item()) * 4.0 - 180.0
    pitch_deg = float((pitch_soft * idx_tensor).sum(dim=1).item()) * 4.0 - 180.0

    return yaw_deg, pitch_deg


# ---------------------------------------------------------------------------
# Section 4: solvePnP Head Pose Estimation (fallback)
# ---------------------------------------------------------------------------
# Canonical 3D face model points (nose-centered, mm)
# Order matches InsightFace kps: left_eye, right_eye, nose, left_mouth, right_mouth
# X: positive = subject's right.  Y: positive = down.  Z: positive = towards camera.
_MODEL_3D_POINTS = np.array([
    [-30.0, -32.7, -26.0],  # left eye center
    [30.0,  -32.7, -26.0],  # right eye center
    [0.0,    0.0,    0.0],  # nose tip (most protruding)
    [-28.9,  28.9, -24.1],  # left mouth corner
    [28.9,   28.9, -24.1],  # right mouth corner
], dtype=np.float64)


def _estimate_head_pose_pnp(
    landmarks_2d: np.ndarray,
    img_shape: tuple[int, int],
) -> tuple[float, float, float]:
    """Estimate head pose from 5 facial landmarks via solvePnP.

    Args:
        landmarks_2d: (5, 2) array of 2D keypoints in InsightFace order
                      [left_eye, right_eye, nose, left_mouth, right_mouth]
        img_shape: (height, width) of the image

    Returns:
        (yaw_degrees, pitch_degrees, roll_degrees)
    """
    h, w = img_shape[:2]
    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    pts_2d = landmarks_2d.astype(np.float64)
    success, rvec, tvec = cv2.solvePnP(
        _MODEL_3D_POINTS, pts_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)

    # Extract Euler angles using RQDecomp3x3 (standard for head pose)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    # angles[0] = pitch (around X), angles[1] = yaw (around Y), angles[2] = roll (around Z)
    pitch_deg = angles[0]
    yaw_deg = angles[1]
    roll_deg = angles[2]

    return (
        round(float(yaw_deg), 1),
        round(float(pitch_deg), 1),
        round(float(roll_deg), 1),
    )


# ---------------------------------------------------------------------------
# Section 5: Gaze Vector and Direction Label
# ---------------------------------------------------------------------------
def _yaw_pitch_to_vector(yaw_rad: float, pitch_rad: float) -> list[float]:
    """Convert yaw/pitch (radians) to a 3D unit gaze vector [x, y, z].

    Convention:
      x: positive = right
      y: positive = down
      z: positive = forward (into the screen)
    """
    x = -math.sin(yaw_rad) * math.cos(pitch_rad)
    y = -math.sin(pitch_rad)
    z = -math.cos(yaw_rad) * math.cos(pitch_rad)
    return [round(x, 4), round(y, 4), round(z, 4)]


def _direction_label(yaw_deg: float, pitch_deg: float) -> str:
    """Generate a human-readable gaze direction description."""
    parts = []

    abs_yaw = abs(yaw_deg)
    abs_pitch = abs(pitch_deg)

    if abs_yaw < 5 and abs_pitch < 5:
        return "looking straight ahead"

    # Horizontal component
    if abs_yaw >= 45:
        h = "far right" if yaw_deg > 0 else "far left"
    elif abs_yaw >= 20:
        h = "right" if yaw_deg > 0 else "left"
    elif abs_yaw >= 5:
        h = "slightly right" if yaw_deg > 0 else "slightly left"
    else:
        h = None

    # Vertical component
    if abs_pitch >= 30:
        v = "far up" if pitch_deg > 0 else "far down"
    elif abs_pitch >= 15:
        v = "up" if pitch_deg > 0 else "down"
    elif abs_pitch >= 5:
        v = "slightly up" if pitch_deg > 0 else "slightly down"
    else:
        v = None

    if h:
        parts.append(h)
    if v:
        parts.append(v)

    if not parts:
        return "looking forward"

    return "looking " + " and ".join(parts)


# ---------------------------------------------------------------------------
# Section 6: Face Crop Helper
# ---------------------------------------------------------------------------
def _crop_face(img_bgr: np.ndarray, bbox: list, expand_ratio: float = 0.3) -> np.ndarray:
    """Crop face region with optional expansion for gaze model context."""
    h_img, w_img = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    w_face = x2 - x1
    h_face = y2 - y1
    pad_x = int(w_face * expand_ratio)
    pad_y = int(h_face * expand_ratio)
    cx1 = max(0, int(x1) - pad_x)
    cy1 = max(0, int(y1) - pad_y)
    cx2 = min(w_img, int(x2) + pad_x)
    cy2 = min(h_img, int(y2) + pad_y)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return img_bgr[int(y1):int(y2), int(x1):int(x2)]
    return crop


# ---------------------------------------------------------------------------
# Section 7: GPU Cleanup
# ---------------------------------------------------------------------------
def _cleanup_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Section 8: Tool Handler
# ---------------------------------------------------------------------------
async def _handle_gaze_estimate(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not os.path.isfile(image_path):
        return {"success": False, "error": f"File not found: {image_path}"}

    box_threshold = float(arguments.get("box_threshold", 0.5))

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": f"Cannot read image: {image_path}"}

        _init_gaze_backend()
        faces = _detect_faces(img, box_threshold)

        face_results = []
        for i, face in enumerate(faces):
            entry: dict = {
                "face_id": i + 1,
                "bbox": face["bbox"],
                "det_score": face["det_score"],
            }

            face_crop = _crop_face(img, face["bbox"])

            # --- L2CS-Net gaze estimation ---
            if _gaze_backend == "l2cs":
                try:
                    yaw_deg, pitch_deg = _predict_gaze_l2cs(face_crop)
                    yaw_rad = math.radians(yaw_deg)
                    pitch_rad = math.radians(pitch_deg)

                    entry["gaze_yaw_degrees"] = round(yaw_deg, 1)
                    entry["gaze_pitch_degrees"] = round(pitch_deg, 1)
                    entry["gaze_vector_3d"] = _yaw_pitch_to_vector(yaw_rad, pitch_rad)
                    entry["looking_direction"] = _direction_label(yaw_deg, pitch_deg)
                except Exception as exc:
                    entry["gaze_error"] = str(exc)

            # --- Head pose via solvePnP (fallback or supplementary) ---
            kps = face.get("kps")
            if kps is not None and len(kps) >= 5:
                try:
                    kps_arr = np.array(kps[:5], dtype=np.float64)
                    yaw_hp, pitch_hp, roll_hp = _estimate_head_pose_pnp(
                        kps_arr, img.shape,
                    )
                    entry["head_pose"] = {
                        "yaw_degrees": yaw_hp,
                        "pitch_degrees": pitch_hp,
                        "roll_degrees": roll_hp,
                    }

                    # If L2CS not available, use head pose as approximate gaze
                    if _gaze_backend != "l2cs" and "gaze_yaw_degrees" not in entry:
                        yaw_rad = math.radians(yaw_hp)
                        pitch_rad = math.radians(pitch_hp)
                        entry["gaze_yaw_degrees"] = yaw_hp
                        entry["gaze_pitch_degrees"] = pitch_hp
                        entry["gaze_vector_3d"] = _yaw_pitch_to_vector(yaw_rad, pitch_rad)
                        entry["looking_direction"] = _direction_label(yaw_hp, pitch_hp)
                        entry["gaze_note"] = "Approximated from head pose (L2CS weights not available)"
                except Exception as exc:
                    entry["head_pose_error"] = str(exc)
            elif _gaze_backend != "l2cs":
                entry["gaze_note"] = "No landmarks available for head pose estimation"

            face_results.append(entry)

        _cleanup_gpu()

        return {
            "success": True,
            "image_path": image_path,
            "gaze_backend": _gaze_backend,
            "face_backend": _face_backend,
            "total_faces": len(face_results),
            "faces": face_results,
        }
    except Exception as e:
        _cleanup_gpu()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Section 9: MCP Server
# ---------------------------------------------------------------------------


async def handle_gaze_estimate(arguments: dict) -> dict:
    return await _handle_gaze_estimate(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="gaze-estimation",
        tool_defs_source="_tool_defs/face.yaml",
        handlers={
            "gaze_estimate": handle_gaze_estimate,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
