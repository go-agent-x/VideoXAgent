"""
Shared utilities for face detection, embedding extraction, and clustering.

Two-tier backend:
  1. InsightFace (preferred) — high-quality detection + recognition embeddings
  2. OpenCV Haar cascade (fallback) — always available, uses histogram features
     for rudimentary cross-frame matching
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")
from _base.config import get_config
from _base.device import insightface_device_options

# ---------------------------------------------------------------------------
# Configuration from _base/config.py
# ---------------------------------------------------------------------------
_insightface_cfg = get_config("insightface")
_INSIGHTFACE_MODEL = _insightface_cfg["model"]
_INSIGHTFACE_PROVIDERS = _insightface_cfg["providers"]

# -------------------------------------------------------------------------
# Backend state
# -------------------------------------------------------------------------
_model_lock = threading.Lock()
_face_app = None                 # InsightFace FaceAnalysis or None
_backend = "uninitialized"       # "insightface" | "opencv" | "uninitialized"
_haar_cascade = None


def _try_init_insightface(det_size: tuple = (640, 640)):
    """Attempt to load InsightFace. Returns FaceAnalysis app or None."""
    try:
        import os
        os.environ.setdefault("ALBUMENTATIONS_CHECK_VERSION", "0")
        import insightface
        from insightface.app import FaceAnalysis

        provider_list, context = insightface_device_options(_INSIGHTFACE_PROVIDERS)

        app = FaceAnalysis(name=_INSIGHTFACE_MODEL, providers=provider_list)
        app.prepare(ctx_id=context, det_size=det_size)
        return app
    except Exception as exc:
        print(f"[face_utils] InsightFace unavailable ({exc}), falling back to OpenCV.",
              file=sys.stderr)
        return None


def _get_haar():
    """Return the cached Haar cascade classifier."""
    global _haar_cascade
    if _haar_cascade is None:
        xml = os.path.join(cv2.data.haarcascades,
                           "haarcascade_frontalface_default.xml")
        _haar_cascade = cv2.CascadeClassifier(xml)
    return _haar_cascade


def _init_backend(det_size: tuple = (640, 640)):
    """Initialise the detection backend (called once, thread-safe)."""
    global _face_app, _backend
    if _backend != "uninitialized":
        return
    with _model_lock:
        if _backend != "uninitialized":
            return
        app = _try_init_insightface(det_size)
        if app is not None:
            _face_app = app
            _backend = "insightface"
        else:
            _backend = "opencv"


def get_backend() -> str:
    """Return current backend name after initialization."""
    _init_backend()
    return _backend


# -------------------------------------------------------------------------
# Histogram-based embedding (OpenCV fallback)
# -------------------------------------------------------------------------
_EMB_DIM = 192  # 64 (HSV-H) + 64 (HSV-S) + 64 (gradient orientation)


def _compute_hist_embedding(face_bgr: np.ndarray) -> np.ndarray:
    """Compute a lightweight histogram embedding for a face crop.

    Combines HSV colour histogram + gradient orientation histogram
    to form a 192-d vector, L2-normalised.
    """
    face_resized = cv2.resize(face_bgr, (64, 64))

    # HSV colour histogram (128 bins on H+S channels)
    hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()

    # Gradient orientation histogram (64 bins)
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    grad_hist = cv2.calcHist([angle.astype(np.float32)],
                             [0], None, [64], [0, 360]).flatten()

    emb = np.concatenate([h_hist, s_hist, grad_hist]).astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb /= norm
    return emb


# -------------------------------------------------------------------------
# OpenCV-based face detection
# -------------------------------------------------------------------------
def _detect_faces_opencv(img: np.ndarray) -> list[dict]:
    """Detect faces using OpenCV Haar cascade + histogram embedding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade = _get_haar()
    rects = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    h_img, w_img = img.shape[:2]
    results = []
    for (x, y, w, h) in rects:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        emb = _compute_hist_embedding(face_crop)
        results.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "det_score": 0.90,
            "age": None,
            "gender": None,
            "embedding": emb,
        })

    # Sort by area descending (largest face first)
    results.sort(key=lambda r: (r["bbox"][2] - r["bbox"][0]) *
                                (r["bbox"][3] - r["bbox"][1]), reverse=True)
    return results


# -------------------------------------------------------------------------
# InsightFace-based face detection
# -------------------------------------------------------------------------
def _detect_faces_insightface(img: np.ndarray) -> list[dict]:
    """Detect faces using InsightFace."""
    faces = _face_app.get(img)
    results = []
    for face in faces:
        bbox = face.bbox.astype(float).tolist()
        result = {
            "bbox": [round(b, 1) for b in bbox],
            "det_score": round(float(face.det_score), 4),
            "age": int(face.age) if hasattr(face, "age") and face.age is not None else None,
            "gender": ("M" if face.gender == 1 else "F") if hasattr(face, "gender") and face.gender is not None else None,
        }
        if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
            result["landmark_count"] = len(face.landmark_2d_106)
        elif hasattr(face, "kps") and face.kps is not None:
            result["landmark_count"] = len(face.kps)

        if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
            result["embedding"] = face.normed_embedding
        elif hasattr(face, "embedding") and face.embedding is not None:
            emb = face.embedding
            norm = np.linalg.norm(emb)
            result["embedding"] = emb / max(norm, 1e-8)

        results.append(result)
    return results


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def detect_faces_in_image(image_path: str, det_size: tuple = (640, 640)) -> list[dict]:
    """Detect faces in a single image (auto-selects backend).

    Returns list of dicts with keys:
        bbox, det_score, age, gender, embedding (numpy)
    """
    _init_backend(det_size)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if _backend == "insightface":
        return _detect_faces_insightface(img)
    else:
        return _detect_faces_opencv(img)


def extract_frames_at_fps(video_path: str, sample_fps: float = 1.0,
                          start_time: float = 0.0, end_time: Optional[float] = None) -> list[tuple]:
    """Extract frames from video at given FPS.

    Returns list of (frame_time_sec, frame_bgr_numpy).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    actual_end = end_time if end_time is not None else duration

    # Calculate which frames to grab
    interval_sec = 1.0 / sample_fps
    frames = []
    t = start_time
    while t < actual_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((round(t, 3), frame))
        t += interval_sec

    cap.release()
    return frames


def cluster_embeddings(embeddings: np.ndarray, distance_threshold: float = 0.6) -> np.ndarray:
    """Cluster face embeddings using Agglomerative Clustering with cosine distance.

    Args:
        embeddings: (N, D) normalized embedding matrix.
        distance_threshold: Cosine distance threshold for merging (default 0.6).

    Returns:
        labels: (N,) cluster labels.
    """
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) == 0:
        return np.array([], dtype=int)
    if len(embeddings) == 1:
        return np.array([0], dtype=int)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    return labels


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    a = emb_a.flatten()
    b = emb_b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
