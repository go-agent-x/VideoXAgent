#!/usr/bin/env python3
"""Emotion Analysis MCP Server — Facial expression + speech emotion + multimodal fusion."""
import asyncio
import gc
import json
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")

from _base.paths import PROJECT_ROOT
sys.path.insert(0, str(Path(__file__).parent))

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response
from _base.config import get_config
from _base.device import insightface_device_options

import cv2
import numpy as np

import os
os.environ["HF_HUB_OFFLINE"] = "1"

# ---------------------------------------------------------------------------
# Configuration from _base/config.py
# ---------------------------------------------------------------------------
_emotion_cfg = get_config("emotion")
DEVICE = _emotion_cfg["device"]
FER_MODEL_NAME = _emotion_cfg["fer_model"]
SER_MODEL_NAME = _emotion_cfg["ser_model"]
DEFAULT_FUSION_WEIGHT = _emotion_cfg["fusion_weight"]

_insightface_cfg = get_config("insightface")
_INSIGHTFACE_MODEL = _insightface_cfg["model"]
_INSIGHTFACE_PROVIDERS = _insightface_cfg["providers"]

OUTPUT_DIR = PROJECT_ROOT / "output"

SUPPORTED_AV_FORMATS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
}

# 7 Ekman emotions (FER output)
EKMAN_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Common categories for fusion (full 7 Ekman set — surprise and fear are distinct emotions)
COMMON_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Mapping from 7-class FER to common set (identity — all Ekman emotions preserved)
FER_TO_COMMON = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "sad": "sad",
    "happy": "happy",
    "surprise": "surprise",
    "neutral": "neutral",
}

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
            f"[emotion_analysis] InsightFace unavailable ({exc}), falling back to OpenCV.",
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
            "age": None,
            "gender": None,
        })
    results.sort(
        key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]),
        reverse=True,
    )
    return results


def _detect_faces_insightface(img: np.ndarray) -> list[dict]:
    faces = _face_app.get(img)
    results = []
    for face in faces:
        bbox = face.bbox.astype(float).tolist()
        results.append({
            "bbox": [round(b, 1) for b in bbox],
            "det_score": round(float(face.det_score), 4),
            "age": int(face.age) if hasattr(face, "age") and face.age is not None else None,
            "gender": ("M" if face.gender == 1 else "F") if hasattr(face, "gender") and face.gender is not None else None,
        })
    return results


def _detect_faces(img: np.ndarray, box_threshold: float = 0.5) -> list[dict]:
    _init_face_backend()
    if _face_backend == "insightface":
        faces = _detect_faces_insightface(img)
    else:
        faces = _detect_faces_opencv(img)
    return [f for f in faces if f["det_score"] >= box_threshold]


# ---------------------------------------------------------------------------
# Section 2: FER (Facial Expression Recognition) Backend
# ---------------------------------------------------------------------------
_fer_lock = threading.Lock()
_fer_pipeline = None
_fer_backend = "uninitialized"  # "transformers" | "simple"


def _try_init_fer():
    """Attempt to load HuggingFace FER pipeline. Returns pipeline or None."""
    try:
        import torch
        from transformers import pipeline as hf_pipeline

        model_name = FER_MODEL_NAME

        # Offline-first cache check (pattern from action-classify-single-mcp)
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        model_local = Path(model_name)
        is_local = model_local.is_dir() and (model_local / "config.json").exists()

        if not is_local:
            hub_cache = Path(cache_dir) / "hub"
            model_slug = "models--" + model_name.replace("/", "--")
            cached = hub_cache / model_slug
            if not cached.is_dir():
                print(
                    f"[emotion_analysis] FER model '{model_name}' not in local cache, "
                    "skipping download attempt.",
                    file=sys.stderr,
                )
                return None

        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            device_idx = 0 if torch.cuda.is_available() and DEVICE == "cuda" else -1
            pipe = hf_pipeline(
                "image-classification",
                model=model_name,
                device=device_idx,
            )
        finally:
            if prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_offline
        if pipe.model is None:
            return None
        return pipe
    except Exception as exc:
        print(
            f"[emotion_analysis] FER model unavailable ({exc}), using simple fallback.",
            file=sys.stderr,
        )
        return None


def _init_fer_backend():
    global _fer_pipeline, _fer_backend
    if _fer_backend != "uninitialized":
        return
    with _fer_lock:
        if _fer_backend != "uninitialized":
            return
        pipe = _try_init_fer()
        if pipe is not None:
            _fer_pipeline = pipe
            _fer_backend = "transformers"
        else:
            _fer_backend = "simple"


def _crop_face_with_padding(image_bgr: np.ndarray, bbox: list, pad_ratio: float = 0.2) -> np.ndarray:
    """Crop face region with padding for better FER context."""
    h_img, w_img = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    w_face = x2 - x1
    h_face = y2 - y1
    pad_x = int(w_face * pad_ratio)
    pad_y = int(h_face * pad_ratio)
    cx1 = max(0, int(x1) - pad_x)
    cy1 = max(0, int(y1) - pad_y)
    cx2 = min(w_img, int(x2) + pad_x)
    cy2 = min(h_img, int(y2) + pad_y)
    return image_bgr[cy1:cy2, cx1:cx2]


def _classify_face_emotion(face_crop_bgr: np.ndarray) -> list[dict]:
    """Classify emotion from a face crop image. Returns list of {label, score}."""
    _init_fer_backend()
    if _fer_backend != "transformers":
        return []

    from PIL import Image
    rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    results = _fer_pipeline(pil_img)
    emotions = []
    for r in results:
        label = r["label"].lower().strip()
        emotions.append({
            "label": label,
            "score": round(float(r["score"]), 4),
        })
    emotions.sort(key=lambda e: e["score"], reverse=True)
    return emotions


# ---------------------------------------------------------------------------
# Section 3: SER (Speech Emotion Recognition) Backend
# ---------------------------------------------------------------------------
_ser_lock = threading.Lock()
_ser_model = None
_ser_backend = "uninitialized"  # "speechbrain" | "unavailable"


def _patch_hf_hub_compat():
    """Fix SpeechBrain / huggingface_hub incompatibility.

    SpeechBrain 1.0.x passes deprecated ``use_auth_token`` to
    ``huggingface_hub.hf_hub_download``, which was removed in
    huggingface_hub >= 1.0.  Monkey-patch to convert it to ``token``.
    """
    try:
        import huggingface_hub as _hfh
        _orig = _hfh.hf_hub_download
        import inspect
        sig = inspect.signature(_orig)
        if "use_auth_token" not in sig.parameters:
            def _patched(*args, **kwargs):
                uat = kwargs.pop("use_auth_token", None)
                if uat is not None and "token" not in kwargs:
                    kwargs["token"] = uat
                return _orig(*args, **kwargs)
            _hfh.hf_hub_download = _patched
    except Exception:
        pass


def _patch_sb_wav2vec2_cache():
    """Patch SpeechBrain's HFTransformersInterface to use the default HF hub cache.

    SpeechBrain passes save_path as cache_dir to transformers' from_pretrained,
    and uses save_path to look for snapshot directories. In offline mode this
    breaks when models are cached in the standard HuggingFace hub directory
    (~/.cache/huggingface/hub) rather than SpeechBrain's own save_path.

    Fix: redirect save_path to the HF hub cache for the _from_pretrained call,
    and remove cache_dir override from AutoConfig calls in __init__.
    """
    try:
        from speechbrain.integrations.huggingface import huggingface as sb_hf
        from transformers import AutoConfig

        if getattr(sb_hf.HFTransformersInterface, "_cache_patched", False):
            return

        _hf_hub_cache = os.environ.get(
            "HF_HUB_CACHE",
            os.path.join(
                os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "hub",
            ),
        )

        _orig_init = sb_hf.HFTransformersInterface.__init__

        def _patched_init(self, source, save_path="", **kwargs):
            _real_config_from_pretrained = AutoConfig.from_pretrained

            def _config_no_cache(*a, **kw):
                kw.pop("cache_dir", None)
                return _real_config_from_pretrained(*a, **kw)

            AutoConfig.from_pretrained = staticmethod(_config_no_cache)
            try:
                _orig_init(self, source, save_path=save_path, **kwargs)
            finally:
                AutoConfig.from_pretrained = _real_config_from_pretrained

        _orig_from_pretrained = sb_hf.HFTransformersInterface._from_pretrained

        def _patched_from_pretrained(self, source, save_path, cache_dir, device=None, **kwargs):
            _orig_from_pretrained(
                self, source, save_path=_hf_hub_cache, cache_dir=cache_dir,
                device=device, **kwargs
            )

        _orig_load_feature_extractor = sb_hf.HFTransformersInterface.load_feature_extractor

        def _patched_load_feature_extractor(self, source, cache_dir=None, **kwargs):
            _orig_load_feature_extractor(self, source, cache_dir=None, **kwargs)

        sb_hf.HFTransformersInterface.__init__ = _patched_init
        sb_hf.HFTransformersInterface._from_pretrained = _patched_from_pretrained
        sb_hf.HFTransformersInterface.load_feature_extractor = _patched_load_feature_extractor
        sb_hf.HFTransformersInterface._cache_patched = True
    except Exception:
        pass


def _try_init_ser():
    """Attempt to load SpeechBrain emotion recognition model."""
    try:
        _patch_hf_hub_compat()
        _patch_sb_wav2vec2_cache()
        from speechbrain.inference.classifiers import EncoderClassifier
        import torch

        model_source = SER_MODEL_NAME
        device = "cuda" if torch.cuda.is_available() and DEVICE == "cuda" else "cpu"

        save_dir = os.path.expanduser(
            f"~/.cache/speechbrain/{model_source.replace('/', '--')}"
        )

        # Check if model is already downloaded locally
        hyperparams_file = os.path.join(save_dir, "hyperparams.yaml")
        if not os.path.isfile(hyperparams_file):
            # Check if model_source is a local path
            if os.path.isdir(model_source) and os.path.isfile(
                os.path.join(model_source, "hyperparams.yaml")
            ):
                pass  # Local path, proceed
            else:
                print(
                    f"[emotion_analysis] SER model '{model_source}' not in local cache "
                    f"({save_dir}), skipping (download may hang without network).",
                    file=sys.stderr,
                )
                return None

        model = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=save_dir,
            run_opts={"device": device},
        )
        return model
    except Exception as exc:
        print(
            f"[emotion_analysis] SER model unavailable ({exc})",
            file=sys.stderr,
        )
        return None


def _init_ser_backend():
    global _ser_model, _ser_backend
    if _ser_backend != "uninitialized":
        return
    with _ser_lock:
        if _ser_backend != "uninitialized":
            return
        model = _try_init_ser()
        if model is not None:
            _ser_model = model
            _ser_backend = "speechbrain"
        else:
            _ser_backend = "unavailable"


# ---------------------------------------------------------------------------
# Section 4: Audio extraction helper
# ---------------------------------------------------------------------------
def _extract_audio_segment(
    file_path: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> str:
    """Extract audio segment to 16kHz mono WAV temp file using ffmpeg."""
    s = start_sec or 0
    e = end_sec or 0
    tmp_path = f"/tmp/_emotion_audio_{os.getpid()}_{s:.1f}_{e:.1f}.wav"

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start_sec is not None and start_sec > 0:
        cmd.extend(["-ss", str(start_sec)])
    cmd.extend(["-i", file_path])
    if end_sec is not None:
        duration = end_sec - (start_sec or 0)
        if duration > 0:
            cmd.extend(["-t", str(duration)])
    cmd.extend(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp_path])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")
    return tmp_path


def _classify_speech_emotion(file_path: str, start_sec: Optional[float] = None,
                             end_sec: Optional[float] = None) -> dict:
    """Classify speech emotion from an audio/video file segment."""
    import torch
    import soundfile as sf

    _init_ser_backend()
    if _ser_backend != "speechbrain":
        return {"success": False, "error": "Speech emotion recognition model not available."}

    tmp_wav = None
    try:
        tmp_wav = _extract_audio_segment(file_path, start_sec, end_sec)

        data, sample_rate = sf.read(tmp_wav, dtype="float32")
        waveform = torch.from_numpy(data).unsqueeze(0).to(_ser_model.device)

        with torch.no_grad():
            feats = _ser_model.mods.wav2vec2(waveform)
            pooled = _ser_model.mods.avg_pool(feats)
            logits = _ser_model.mods.output_mlp(pooled)
            out_prob = _ser_model.hparams.softmax(logits)

        probs = out_prob.squeeze().cpu().tolist()

        # Get label names from the model's label encoder
        if hasattr(_ser_model, "hparams") and hasattr(_ser_model.hparams, "label_encoder"):
            le = _ser_model.hparams.label_encoder
            if hasattr(le, "ind2lab"):
                labels = [le.ind2lab[i] for i in range(len(probs))]
            else:
                labels = [f"class_{i}" for i in range(len(probs))]
        else:
            labels = [f"class_{i}" for i in range(len(probs))]

        emotions = []
        for label, prob in zip(labels, probs):
            emotions.append({
                "label": label.lower().strip(),
                "score": round(float(prob), 4),
            })
        emotions.sort(key=lambda e: e["score"], reverse=True)

        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "success": True,
            "backend": "speechbrain",
            "emotions": emotions,
            "dominant_emotion": emotions[0]["label"] if emotions else "unknown",
            "dominant_score": emotions[0]["score"] if emotions else 0.0,
        }
    except torch.cuda.OutOfMemoryError:
        _cleanup_gpu()
        return {
            "success": False,
            "error": "GPU out of memory during speech emotion analysis. "
                     "Try a shorter audio segment.",
        }
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Section 5: Frame extraction helper
# ---------------------------------------------------------------------------
def _extract_frame_at_time(video_path: str, time_sec: float) -> np.ndarray:
    """Extract a single frame from video at the given timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame at {time_sec}s")
        return frame
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Section 6: Emotion mapping and fusion
# ---------------------------------------------------------------------------
def _map_fer_to_common(fer_emotions: list[dict]) -> dict[str, float]:
    """Map 7-class FER emotions to common set for fusion."""
    common = {e: 0.0 for e in COMMON_EMOTIONS}
    for e in fer_emotions:
        target = FER_TO_COMMON.get(e["label"], "neutral")
        common[target] += e["score"]
    # Renormalize
    total = sum(common.values())
    if total > 0:
        common = {k: round(v / total, 4) for k, v in common.items()}
    return common


def _fuse_emotions(
    face_common: Optional[dict[str, float]],
    speech_emotions: Optional[list[dict]],
    w_face: float = 0.6,
) -> dict:
    """Weighted fusion of face and speech emotions."""
    modalities_used = []
    face_scores = None
    speech_scores = None

    if face_common:
        face_scores = face_common
        modalities_used.append("face")

    if speech_emotions:
        speech_scores = {e: 0.0 for e in COMMON_EMOTIONS}
        for e in speech_emotions:
            label = e["label"].lower().strip()
            if label in speech_scores:
                speech_scores[label] = e["score"]
        modalities_used.append("speech")

    if not modalities_used:
        return {
            "modalities_used": [],
            "emotions": [],
            "dominant_emotion": "unknown",
            "dominant_score": 0.0,
            "note": "No modality produced results.",
        }

    # Single modality fallback
    if len(modalities_used) == 1:
        scores = face_scores if face_scores else speech_scores
        actual_w_face = 1.0 if "face" in modalities_used else 0.0
    else:
        scores = {}
        for label in COMMON_EMOTIONS:
            scores[label] = round(
                w_face * face_scores[label] + (1 - w_face) * speech_scores[label],
                4,
            )
        actual_w_face = w_face

    emotions = [{"label": k, "score": v} for k, v in scores.items()]
    emotions.sort(key=lambda e: e["score"], reverse=True)

    return {
        "fusion_weights": {"face": round(actual_w_face, 2), "speech": round(1 - actual_w_face, 2)},
        "modalities_used": modalities_used,
        "emotions": emotions,
        "dominant_emotion": emotions[0]["label"] if emotions else "unknown",
        "dominant_score": emotions[0]["score"] if emotions else 0.0,
    }


# ---------------------------------------------------------------------------
# Section 7: GPU cleanup
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
# Section 8: Tool handlers
# ---------------------------------------------------------------------------
async def _handle_analyze_face_emotion(arguments: dict) -> dict:
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

        faces = _detect_faces(img, box_threshold)

        face_results = []
        for i, face in enumerate(faces):
            face_crop = _crop_face_with_padding(img, face["bbox"])
            if face_crop.size == 0:
                continue

            emotions = _classify_face_emotion(face_crop)
            dominant = emotions[0] if emotions else None

            entry = {
                "face_id": i + 1,
                "bbox": face["bbox"],
                "det_score": face["det_score"],
                "age": face.get("age"),
                "gender": face.get("gender"),
            }

            if emotions:
                entry["emotions"] = emotions
                entry["dominant_emotion"] = dominant["label"]
                entry["dominant_score"] = dominant["score"]
            else:
                entry["emotions"] = []
                entry["note"] = "FER model not available; only face detection provided."

            face_results.append(entry)

        result = {
            "success": True,
            "image_path": image_path,
            "face_backend": _face_backend,
            "fer_backend": _fer_backend,
            "total_faces": len(face_results),
            "faces": face_results,
        }
        return result

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


async def _handle_analyze_speech_emotion(arguments: dict) -> dict:
    file_path = arguments.get("file_path", "")
    if not file_path:
        return {"success": False, "error": "file_path is required"}
    if not os.path.isfile(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_AV_FORMATS:
        return {
            "success": False,
            "error": f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_AV_FORMATS)}",
        }

    start_time = arguments.get("start_time")
    end_time = arguments.get("end_time")
    start_sec = float(start_time) if start_time is not None else None
    end_sec = float(end_time) if end_time is not None else None

    try:
        result = _classify_speech_emotion(file_path, start_sec, end_sec)
        result["file_path"] = file_path
        result["analyzed_range"] = {
            "start_time": start_sec if start_sec is not None else 0,
            "end_time": end_sec if end_sec is not None else "file_end",
        }
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


async def _handle_analyze_multimodal_emotion(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    if not video_path:
        return {"success": False, "error": "video_path is required"}
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"File not found: {video_path}"}

    time_sec = arguments.get("time_sec")
    if time_sec is None:
        return {"success": False, "error": "time_sec is required"}
    time_sec = float(time_sec)

    audio_window = float(arguments.get("audio_window", 5.0))
    fusion_weight_raw = arguments.get("fusion_weight_face")
    w_face = float(fusion_weight_raw) if fusion_weight_raw is not None else DEFAULT_FUSION_WEIGHT

    try:
        # --- Face emotion analysis ---
        face_result = None
        face_common = None
        try:
            frame = _extract_frame_at_time(video_path, time_sec)
            faces = _detect_faces(frame)
            face_entries = []
            for i, face in enumerate(faces):
                face_crop = _crop_face_with_padding(frame, face["bbox"])
                if face_crop.size == 0:
                    continue
                emotions = _classify_face_emotion(face_crop)
                dominant = emotions[0] if emotions else None
                entry = {
                    "face_id": i + 1,
                    "bbox": face["bbox"],
                    "det_score": face["det_score"],
                    "age": face.get("age"),
                    "gender": face.get("gender"),
                }
                if emotions:
                    entry["emotions"] = emotions
                    entry["dominant_emotion"] = dominant["label"]
                    entry["dominant_score"] = dominant["score"]
                face_entries.append(entry)

            face_result = {
                "total_faces": len(face_entries),
                "faces": face_entries,
            }

            # Use primary (largest) face for fusion
            if face_entries and face_entries[0].get("emotions"):
                face_common = _map_fer_to_common(face_entries[0]["emotions"])
                face_result["primary_face_emotions_common"] = face_common

        except Exception as exc:
            face_result = {"error": str(exc)}

        # --- Speech emotion analysis ---
        speech_result = None
        speech_emotions = None
        try:
            audio_start = max(0, time_sec - audio_window / 2)
            audio_end = time_sec + audio_window / 2
            ser_out = _classify_speech_emotion(video_path, audio_start, audio_end)
            speech_result = {
                "analyzed_range": {"start_time": round(audio_start, 2), "end_time": round(audio_end, 2)},
            }
            if ser_out.get("success"):
                speech_result["emotions"] = ser_out["emotions"]
                speech_result["dominant_emotion"] = ser_out["dominant_emotion"]
                speech_result["dominant_score"] = ser_out["dominant_score"]
                speech_emotions = ser_out["emotions"]
            else:
                speech_result["error"] = ser_out.get("error", "SER failed")
        except Exception as exc:
            speech_result = {"error": str(exc)}

        # --- Multimodal fusion ---
        fused = _fuse_emotions(face_common, speech_emotions, w_face)

        return {
            "success": True,
            "video_path": video_path,
            "timestamp": time_sec,
            "face_emotion": face_result,
            "speech_emotion": speech_result,
            "fused_emotion": fused,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Section 9: MCP Server
# ---------------------------------------------------------------------------


async def handle_analyze_face_emotion(arguments: dict) -> dict:
    return await _handle_analyze_face_emotion(arguments)


async def handle_analyze_speech_emotion(arguments: dict) -> dict:
    return await _handle_analyze_speech_emotion(arguments)


async def handle_analyze_multimodal_emotion(arguments: dict) -> dict:
    return await _handle_analyze_multimodal_emotion(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="emotion-analysis",
        tool_defs_source="_tool_defs/face.yaml",
        handlers={
            "analyze_face_emotion": handle_analyze_face_emotion,
            "analyze_speech_emotion": handle_analyze_speech_emotion,
            "analyze_multimodal_emotion": handle_analyze_multimodal_emotion,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
