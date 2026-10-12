#!/usr/bin/env python3
"""Active Speaker Detection MCP Server — Determine who is speaking via lip-audio correlation."""
import asyncio
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import traceback
import wave
from collections import defaultdict
from pathlib import Path
from typing import Optional

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
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Configuration from _base/config.py
# ---------------------------------------------------------------------------
_insightface_cfg = get_config("insightface")
_INSIGHTFACE_MODEL = _insightface_cfg["model"]
_INSIGHTFACE_PROVIDERS = _insightface_cfg["providers"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# InsightFace 106-point landmark mouth indices (standard 2d106det scheme).
# Outer lip: indices 76-87, Inner lip: indices 88-95 — 20 points total.
MOUTH_INDICES_106 = list(range(76, 96))

# Supported video formats
SUPPORTED_VIDEO_FORMATS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".ts", ".mpg", ".mpeg", ".3gp",
}

# Audio extraction settings
AUDIO_SAMPLE_RATE = 16000  # 16kHz mono for analysis

# ---------------------------------------------------------------------------
# Backend state (InsightFace lazy init with double-checked locking)
# ---------------------------------------------------------------------------
_model_lock = threading.Lock()
_face_app = None
_backend = "uninitialized"  # "insightface" | "opencv" | "uninitialized"
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
            f"[active_speaker_detection] InsightFace unavailable ({exc}), "
            f"falling back to OpenCV.",
            file=sys.stderr,
        )
        return None


def _get_haar():
    global _haar_cascade
    if _haar_cascade is None:
        xml = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        _haar_cascade = cv2.CascadeClassifier(xml)
    return _haar_cascade


def _init_backend(det_size: tuple = (640, 640)):
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


# ---------------------------------------------------------------------------
# OpenCV histogram embedding fallback
# ---------------------------------------------------------------------------
def _compute_hist_embedding(face_bgr: np.ndarray) -> np.ndarray:
    face_resized = cv2.resize(face_bgr, (64, 64))
    hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    _, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    grad_hist = cv2.calcHist(
        [angle.astype(np.float32)], [0], None, [64], [0, 360]
    ).flatten()
    emb = np.concatenate([h_hist, s_hist, grad_hist]).astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb /= norm
    return emb


# ---------------------------------------------------------------------------
# Face detection with landmarks
# ---------------------------------------------------------------------------
def _detect_faces_insightface(img: np.ndarray) -> list[dict]:
    """Detect faces with InsightFace, extracting landmarks + embeddings."""
    faces = _face_app.get(img)
    results = []
    for face in faces:
        bbox = face.bbox.astype(float).tolist()
        result = {
            "bbox": [round(b, 1) for b in bbox],
            "det_score": round(float(face.det_score), 4),
            "age": (
                int(face.age)
                if hasattr(face, "age") and face.age is not None
                else None
            ),
            "gender": (
                ("M" if face.gender == 1 else "F")
                if hasattr(face, "gender") and face.gender is not None
                else None
            ),
            "landmarks_106": None,
            "kps": None,
        }
        # Extract embedding
        if (
            hasattr(face, "normed_embedding")
            and face.normed_embedding is not None
        ):
            result["embedding"] = face.normed_embedding
        elif hasattr(face, "embedding") and face.embedding is not None:
            emb = face.embedding
            norm = np.linalg.norm(emb)
            result["embedding"] = emb / max(norm, 1e-8)

        # Extract 106-point landmarks
        if (
            hasattr(face, "landmark_2d_106")
            and face.landmark_2d_106 is not None
        ):
            result["landmarks_106"] = face.landmark_2d_106.copy()
        # Extract 5-point keypoints
        if hasattr(face, "kps") and face.kps is not None:
            result["kps"] = face.kps.copy()

        results.append(result)
    return results


def _detect_faces_opencv(img: np.ndarray) -> list[dict]:
    """OpenCV Haar cascade fallback. No landmarks, histogram embedding."""
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
    for x, y, w, h in rects:
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w_img, int(x + w))
        y2 = min(h_img, int(y + h))
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        emb = _compute_hist_embedding(face_crop)
        results.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "det_score": 0.90,
                "age": None,
                "gender": None,
                "embedding": emb,
                "landmarks_106": None,
                "kps": None,
            }
        )
    results.sort(
        key=lambda r: (r["bbox"][2] - r["bbox"][0])
        * (r["bbox"][3] - r["bbox"][1]),
        reverse=True,
    )
    return results


def _detect_faces(img: np.ndarray, det_size: tuple = (640, 640)) -> list[dict]:
    _init_backend(det_size)
    if _backend == "insightface":
        return _detect_faces_insightface(img)
    return _detect_faces_opencv(img)


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------
def extract_frames_at_fps(
    video_path: str,
    sample_fps: float = 5.0,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> list[tuple]:
    """Extract frames at given FPS. Returns list of (time_sec, frame_bgr).

    Uses CAP_PROP_POS_MSEC for VFR-safe seeking; cap.read() failure
    serves as the natural boundary check (no need for frame-count math).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    # Use OpenCV duration as fallback only when end_time not provided.
    # For VFR videos total_frames/fps may be inaccurate, but cap.read()
    # returning False will stop the loop before we go past the real end.
    if end_time is not None:
        actual_end = end_time
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_end = total_frames / fps if fps > 0 and total_frames > 0 else float("inf")

    interval_sec = 1.0 / sample_fps
    frames = []
    t = start_time
    while t < actual_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break  # Reached real end of video — VFR-safe boundary
        frames.append((round(t, 3), frame))
        t += interval_sec

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Face embedding clustering
# ---------------------------------------------------------------------------
def cluster_embeddings(
    embeddings: np.ndarray, distance_threshold: float = 0.6
) -> np.ndarray:
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
    return clustering.fit_predict(embeddings)


# ---------------------------------------------------------------------------
# Mouth Aspect Ratio (MAR) computation
# ---------------------------------------------------------------------------
def _compute_mar_from_106(landmarks_106: np.ndarray) -> float:
    """Compute MAR from InsightFace 106-point landmarks.

    Uses indices 76-95 (outer+inner lip, 20 points).
    MAR = mouth_height / mouth_width.
    """
    mouth_pts = landmarks_106[MOUTH_INDICES_106]  # (20, 2)
    xs = mouth_pts[:, 0]
    ys = mouth_pts[:, 1]
    mouth_width = float(np.max(xs) - np.min(xs))
    mouth_height = float(np.max(ys) - np.min(ys))
    if mouth_width < 1e-6:
        return 0.0
    return mouth_height / mouth_width


def _compute_mar_from_kps(kps: np.ndarray, bbox: list[float]) -> float:
    """Approximate MAR from 5-point keypoints.

    kps indices: 0=left_eye, 1=right_eye, 2=nose, 3=left_mouth, 4=right_mouth.
    We can only estimate mouth width from corners. Use face bbox lower region
    as proxy for vertical opening — very approximate.
    """
    if kps is None or len(kps) < 5:
        return 0.0
    mouth_left = kps[3]
    mouth_right = kps[4]
    mouth_width = float(np.linalg.norm(mouth_right - mouth_left))
    if mouth_width < 1e-6:
        return 0.0
    # Estimate vertical using face bbox: mouth height is roughly proportional
    # to the distance from mouth corners to bottom of bbox
    mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2.0
    bbox_bottom = bbox[3]
    vertical_proxy = max(0.0, float(bbox_bottom - mouth_center_y))
    return vertical_proxy / mouth_width


def _compute_lower_face_diff(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    bbox: list[float],
) -> float:
    """Fallback: pixel difference in lower 40% of face bbox.

    Used when no landmarks available (OpenCV backend).
    Returns normalized mean absolute pixel difference.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h = y2 - y1
    lower_y1 = y1 + int(h * 0.6)  # lower 40%

    h_img, w_img = frame_curr.shape[:2]
    lower_y1 = max(0, min(lower_y1, h_img - 1))
    y2 = max(lower_y1 + 1, min(y2, h_img))
    x1 = max(0, min(x1, w_img - 1))
    x2 = max(x1 + 1, min(x2, w_img))

    crop_prev = cv2.cvtColor(frame_prev[lower_y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    crop_curr = cv2.cvtColor(frame_curr[lower_y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    if crop_prev.size == 0 or crop_curr.size == 0:
        return 0.0

    # Resize to same size for safety
    target_size = (32, 32)
    crop_prev = cv2.resize(crop_prev, target_size).astype(np.float32)
    crop_curr = cv2.resize(crop_curr, target_size).astype(np.float32)

    diff = np.mean(np.abs(crop_curr - crop_prev)) / 255.0
    return float(diff)


def _build_mar_timeseries(
    detections: list[dict],
) -> tuple[list[float], list[float]]:
    """Build MAR time series from a person's detections.

    Returns (timestamps, mar_values).
    """
    timestamps = []
    mar_values = []
    for det in detections:
        t = det["timestamp"]
        lm106 = det.get("landmarks_106")
        kps = det.get("kps")
        bbox = det["bbox"]

        if lm106 is not None:
            mar = _compute_mar_from_106(lm106)
        elif kps is not None:
            mar = _compute_mar_from_kps(kps, bbox)
        else:
            mar = 0.0

        timestamps.append(t)
        mar_values.append(mar)
    return timestamps, mar_values


def _compute_delta_mar(mar_values: list[float]) -> list[float]:
    """Compute frame-to-frame absolute change in MAR (lip movement signal)."""
    if not mar_values:
        return []
    deltas = [0.0]
    for i in range(1, len(mar_values)):
        deltas.append(abs(mar_values[i] - mar_values[i - 1]))
    return deltas


def _expand_signal_to_timeline(
    person_timestamps: list[float],
    person_values: list[float],
    all_timestamps: list[float],
) -> list[float]:
    """Expand a person's sparse signal to the full timeline.

    Frames where the person is not detected get value 0.0.
    """
    lookup = {}
    for t, v in zip(person_timestamps, person_values):
        lookup[t] = v
    return [lookup.get(t, 0.0) for t in all_timestamps]


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------
def _extract_audio_wav(
    video_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Extract audio from video as 16kHz mono WAV using ffmpeg.

    Returns (audio_float32_array, sample_rate).
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")

    tmp_wav = tempfile.mktemp(suffix=".wav", prefix="_asd_audio_")
    try:
        cmd = ["ffmpeg", "-y"]
        if start_time > 0:
            cmd += ["-ss", str(start_time)]
        cmd += ["-i", video_path]
        if end_time is not None:
            duration = end_time - start_time
            cmd += ["-t", str(duration)]
        cmd += [
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            tmp_wav,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(f"ffmpeg failed: {stderr_text}")

        # Read WAV file
        with wave.open(tmp_wav, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        if sampwidth == 2:
            samples = struct.unpack(f"<{n_frames * n_channels}h", raw_data)
            audio = np.array(samples, dtype=np.float32) / 32768.0
        elif sampwidth == 4:
            samples = struct.unpack(f"<{n_frames * n_channels}i", raw_data)
            audio = np.array(samples, dtype=np.float32) / 2147483648.0
        else:
            audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        return audio, sample_rate

    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def _compute_audio_energy(
    audio: np.ndarray,
    sample_rate: int,
    frame_timestamps: list[float],
    audio_start_time: float = 0.0,
    window_duration: float = 0.04,
) -> list[float]:
    """Compute RMS audio energy at each video frame timestamp.

    Returns list of energy values aligned with frame_timestamps.
    ``audio_start_time`` is the absolute video time corresponding to
    audio[0] (i.e. the start_time passed to _extract_audio_wav).
    """
    half_window_samples = int(window_duration / 2.0 * sample_rate)
    energy_values = []
    audio_len = len(audio)

    for t in frame_timestamps:
        center = int((t - audio_start_time) * sample_rate)
        start = max(0, center - half_window_samples)
        end = min(audio_len, center + half_window_samples)
        if end <= start:
            energy_values.append(0.0)
            continue
        window = audio[start:end]
        rms = float(np.sqrt(np.mean(window ** 2)))
        energy_values.append(rms)

    # Normalize to [0, 1]
    max_e = max(energy_values) if energy_values else 1.0
    if max_e > 1e-8:
        energy_values = [e / max_e for e in energy_values]

    return energy_values


def _detect_speech_segments_energy(
    energy_values: list[float],
    timestamps: list[float],
    threshold_ratio: float = 0.15,
    min_duration: float = 0.5,
    merge_gap: float = 0.3,
) -> list[dict]:
    """Detect speech segments from audio energy using threshold.

    Returns list of {start_sec, end_sec, duration_sec}.
    """
    if not energy_values or not timestamps:
        return []

    # Binary speech mask
    is_speech = [e > threshold_ratio for e in energy_values]

    segments = []
    in_segment = False
    seg_start = 0.0

    for i, (t, speech) in enumerate(zip(timestamps, is_speech)):
        if speech and not in_segment:
            seg_start = t
            in_segment = True
        elif not speech and in_segment:
            segments.append({"start_time": seg_start, "end_time": t})
            in_segment = False

    if in_segment:
        segments.append({"start_time": seg_start, "end_time": timestamps[-1]})

    # Merge close segments
    merged = []
    for seg in segments:
        if merged and (seg["start_time"] - merged[-1]["end_time"]) < merge_gap:
            merged[-1]["end_time"] = seg["end_time"]
        else:
            merged.append(dict(seg))

    # Filter short segments
    result = []
    for seg in merged:
        dur = seg["end_time"] - seg["start_time"]
        if dur >= min_duration:
            seg["duration"] = round(dur, 3)
            result.append(seg)

    return result


def _try_silero_vad(
    audio: np.ndarray,
    sample_rate: int,
    min_duration: float = 0.5,
    audio_start_time: float = 0.0,
) -> Optional[list[dict]]:
    """Attempt to use Silero VAD for more robust speech detection.

    Returns None if Silero/torch not available.
    """
    try:
        import torch

        model, utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        get_speech_timestamps = utils[0]

        audio_tensor = torch.from_numpy(audio).float()
        speech_ts = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=int(min_duration * 1000),
            return_seconds=True,
        )

        segments = []
        for ts in speech_ts:
            start = float(ts["start"]) + audio_start_time
            end = float(ts["end"]) + audio_start_time
            duration = float(ts["end"]) - float(ts["start"])
            segments.append(
                {
                    "start_time": round(start, 3),
                    "end_time": round(end, 3),
                    "duration": round(duration, 3),
                }
            )
        return segments

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Audio-Visual correlation
# ---------------------------------------------------------------------------
def _correlate_av(
    delta_mar_full: list[float],
    audio_energy: list[float],
    timestamps: list[float],
    start_sec: float,
    end_sec: float,
) -> float:
    """Compute Pearson correlation between delta_MAR and audio energy
    within a time window [start_sec, end_sec].

    Returns correlation coefficient (clamped to >= 0).
    """
    indices = [
        i
        for i, t in enumerate(timestamps)
        if start_sec <= t <= end_sec
    ]
    if len(indices) < 3:
        return 0.0

    x = [delta_mar_full[i] for i in indices]
    y = [audio_energy[i] for i in indices]

    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0

    r, _ = pearsonr(x, y)
    return max(0.0, float(r))


def _assign_active_speakers(
    persons_data: dict,
    speech_segments: list[dict],
    timestamps: list[float],
    audio_energy: list[float],
    min_confidence: float = 0.15,
) -> tuple[list[dict], list[dict]]:
    """For each speech segment, determine the active speaker.

    Returns (active_speaker_segments, speaking_timeline).
    """
    active_segments = []

    for seg in speech_segments:
        start = seg["start_time"]
        end = seg["end_time"]

        person_scores = []
        for pid, pdata in persons_data.items():
            delta_mar_full = pdata["delta_mar_full"]

            # Check person is visible during this segment
            visible_count = sum(
                1
                for i, t in enumerate(timestamps)
                if start <= t <= end and delta_mar_full[i] > 0
            )
            # Need at least 2 visible frames to correlate
            if visible_count < 2:
                person_scores.append({"person_id": pid, "score": 0.0})
                continue

            score = _correlate_av(
                delta_mar_full, audio_energy, timestamps, start, end
            )
            person_scores.append({"person_id": pid, "score": round(score, 4)})

        person_scores.sort(key=lambda x: x["score"], reverse=True)
        best = person_scores[0] if person_scores else None

        active_pid = None
        confidence = 0.0
        if best and best["score"] >= min_confidence:
            active_pid = best["person_id"]
            confidence = best["score"]

        active_segments.append(
            {
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "start_formatted": _fmt_time(start),
                "end_formatted": _fmt_time(end),
                "active_speaker_person_id": active_pid,
                "confidence": round(confidence, 4),
                "all_person_scores": person_scores,
            }
        )

    # Build per-frame speaking timeline
    timeline = []
    for i, t in enumerate(timestamps):
        # Find which segment this timestamp falls in
        active_pid = None
        conf = 0.0
        speech_active = False
        for aseg in active_segments:
            if aseg["start_time"] <= t <= aseg["end_time"]:
                speech_active = True
                active_pid = aseg["active_speaker_person_id"]
                conf = aseg["confidence"]
                break

        timeline.append(
            {
                "timestamp": round(t, 3),
                "speech_active": speech_active,
                "active_speaker_person_id": active_pid,
                "confidence": round(conf, 4),
            }
        )

    return active_segments, timeline


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
async def _handle_detect_active_speaker(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    start_time = float(arguments.get("start_time", 0))
    end_time_raw = arguments.get("end_time", None)
    end_time = float(end_time_raw) if end_time_raw is not None else None
    sample_fps = float(arguments.get("sample_fps", 5.0))
    distance_threshold = float(arguments.get("distance_threshold", 0.6))
    speech_min_duration = float(arguments.get("speech_min_duration", 0.5))

    # --- Validate ---
    if not video_path:
        return {"success": False, "error": "video_path is required"}
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"File not found: {video_path}"}
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_FORMATS:
        return {
            "success": False,
            "error": f"Unsupported format: {ext}. Supported: "
            f"{', '.join(sorted(SUPPORTED_VIDEO_FORMATS))}",
        }

    try:
        # --- Step 1: Extract frames ---
        frames = extract_frames_at_fps(
            video_path, sample_fps, start_time, end_time
        )
        if not frames:
            return {
                "success": True,
                "video_path": video_path,
                "note": "No frames extracted",
                "num_persons_detected": 0,
                "num_active_speakers": 0,
                "active_speaker_segments": [],
                "persons": [],
                "speaking_timeline": [],
            }

        timestamps = [t for t, _ in frames]
        frames_lookup = {t: f for t, f in frames}

        # --- Step 2: Detect faces + landmarks in every frame ---
        all_detections = []  # list of (time_sec, face_dict)
        for frame_time, frame_bgr in frames:
            faces = _detect_faces(frame_bgr)
            for face in faces:
                if "embedding" in face:
                    face_data = dict(face)
                    face_data["timestamp"] = frame_time
                    all_detections.append((frame_time, face_data))

        if not all_detections:
            return {
                "success": True,
                "video_path": video_path,
                "backend": _backend,
                "note": "No faces detected in any frame",
                "num_persons_detected": 0,
                "num_active_speakers": 0,
                "active_speaker_segments": [],
                "persons": [],
                "speaking_timeline": [],
            }

        # --- Step 3: Cluster embeddings -> person tracks ---
        embeddings = np.array([d[1]["embedding"] for d in all_detections])
        labels = cluster_embeddings(embeddings, distance_threshold)

        persons_data = defaultdict(lambda: {"detections": []})
        for idx, (frame_time, face_data) in enumerate(all_detections):
            label = int(labels[idx])
            persons_data[label]["detections"].append(face_data)

        # Determine landmark type
        has_106 = any(
            d.get("landmarks_106") is not None
            for _, d in all_detections
        )
        has_kps = any(
            d.get("kps") is not None
            for _, d in all_detections
        )
        landmark_type = (
            "landmark_2d_106"
            if has_106
            else "kps_5"
            if has_kps
            else "pixel_diff"
        )

        # --- Step 4: Build MAR time series per person ---
        for pid, pdata in persons_data.items():
            ts, mars = _build_mar_timeseries(pdata["detections"])
            delta_mars = _compute_delta_mar(mars)

            # If using pixel diff fallback (no landmarks), compute diff between
            # consecutive frames where this person appears
            if landmark_type == "pixel_diff" and len(pdata["detections"]) >= 2:
                dets = pdata["detections"]
                pixel_diffs = [0.0]
                for i in range(1, len(dets)):
                    t_prev = dets[i - 1]["timestamp"]
                    t_curr = dets[i]["timestamp"]
                    if (
                        t_prev in frames_lookup
                        and t_curr in frames_lookup
                    ):
                        diff = _compute_lower_face_diff(
                            frames_lookup[t_prev],
                            frames_lookup[t_curr],
                            dets[i]["bbox"],
                        )
                        pixel_diffs.append(diff)
                    else:
                        pixel_diffs.append(0.0)
                delta_mars = pixel_diffs

            pdata["timestamps"] = ts
            pdata["mar_values"] = mars
            pdata["delta_mars"] = delta_mars
            pdata["delta_mar_full"] = _expand_signal_to_timeline(
                ts, delta_mars, timestamps
            )

        # --- Step 5: Extract audio and compute energy ---
        audio_error = None
        audio_energy = [0.0] * len(timestamps)
        speech_segments = []
        vad_method = "none"

        try:
            audio, sr = _extract_audio_wav(
                video_path, start_time, end_time
            )
            audio_energy = _compute_audio_energy(
                audio, sr, timestamps, audio_start_time=start_time
            )

            # Try Silero VAD first
            silero_segs = _try_silero_vad(audio, sr, speech_min_duration, audio_start_time=start_time)
            if silero_segs is not None:
                speech_segments = silero_segs
                vad_method = "silero"
            else:
                speech_segments = _detect_speech_segments_energy(
                    audio_energy,
                    timestamps,
                    min_duration=speech_min_duration,
                )
                vad_method = "energy"

        except Exception as audio_exc:
            audio_error = str(audio_exc)
            vad_method = "failed"

        # --- Step 6: Assign active speakers ---
        active_segments = []
        speaking_timeline = []

        if speech_segments and len(persons_data) > 0:
            active_segments, speaking_timeline = _assign_active_speakers(
                persons_data,
                speech_segments,
                timestamps,
                audio_energy,
                min_confidence=0.15,
            )
        else:
            # No speech or no persons: build empty timeline
            speaking_timeline = [
                {
                    "timestamp": round(t, 3),
                    "speech_active": False,
                    "active_speaker_person_id": None,
                    "confidence": 0.0,
                }
                for t in timestamps
            ]

        # --- Step 7: Build persons summary ---
        # Reassign person IDs to 1-based sorted by first_seen
        sorted_pids = sorted(
            persons_data.keys(),
            key=lambda pid: persons_data[pid]["detections"][0]["timestamp"],
        )
        pid_map = {old: new + 1 for new, old in enumerate(sorted_pids)}

        # Remap IDs in active_segments and timeline
        for aseg in active_segments:
            if aseg["active_speaker_person_id"] is not None:
                aseg["active_speaker_person_id"] = pid_map.get(
                    aseg["active_speaker_person_id"],
                    aseg["active_speaker_person_id"],
                )
            for ps in aseg.get("all_person_scores", []):
                ps["person_id"] = pid_map.get(ps["person_id"], ps["person_id"])

        for tl in speaking_timeline:
            if tl["active_speaker_person_id"] is not None:
                tl["active_speaker_person_id"] = pid_map.get(
                    tl["active_speaker_person_id"],
                    tl["active_speaker_person_id"],
                )

        # Compute per-person speaking stats
        total_speech_time = sum(
            seg.get("duration", seg["end_time"] - seg["start_time"])
            for seg in speech_segments
        )

        persons_output = []
        active_speaker_pids = set()
        for old_pid in sorted_pids:
            new_pid = pid_map[old_pid]
            pdata = persons_data[old_pid]
            dets = pdata["detections"]
            times = [d["timestamp"] for d in dets]
            ages = [d["age"] for d in dets if d.get("age") is not None]
            genders = [d["gender"] for d in dets if d.get("gender") is not None]

            # Compute speaking time for this person
            speaking_time = 0.0
            speaking_confidence_sum = 0.0
            speaking_segment_count = 0
            for aseg in active_segments:
                if aseg["active_speaker_person_id"] == new_pid:
                    speaking_time += aseg["end_time"] - aseg["start_time"]
                    speaking_confidence_sum += aseg["confidence"]
                    speaking_segment_count += 1
                    active_speaker_pids.add(new_pid)

            persons_output.append(
                {
                    "person_id": new_pid,
                    "total_speaking_time": round(speaking_time, 3),
                    "speaking_ratio": (
                        round(speaking_time / total_speech_time, 3)
                        if total_speech_time > 0
                        else 0.0
                    ),
                    "avg_confidence": (
                        round(
                            speaking_confidence_sum / speaking_segment_count, 4
                        )
                        if speaking_segment_count > 0
                        else 0.0
                    ),
                    "speaking_segments": speaking_segment_count,
                    "first_seen": round(min(times), 3),
                    "first_seen_formatted": _fmt_time(min(times)),
                    "last_seen": round(max(times), 3),
                    "last_seen_formatted": _fmt_time(max(times)),
                    "total_detections": len(dets),
                    "avg_age": (
                        round(sum(ages) / len(ages))
                        if ages
                        else None
                    ),
                    "gender": (
                        max(set(genders), key=genders.count)
                        if genders
                        else None
                    ),
                }
            )

        result = {
            "success": True,
            "video_path": video_path,
            "analyzed_range": {
                "start_time": start_time,
                "end_time": end_time if end_time else "video_end",
            },
            "sample_fps": sample_fps,
            "backend": _backend,
            "landmark_type": landmark_type,
            "vad_method": vad_method,
            "frames_analyzed": len(frames),
            "total_face_detections": len(all_detections),
            "num_persons_detected": len(persons_output),
            "num_active_speakers": len(active_speaker_pids),
            "total_speech_segments": len(speech_segments),
            "total_speech_time": round(total_speech_time, 3),
            "active_speaker_segments": active_segments,
            "persons": persons_output,
            "speaking_timeline": speaking_timeline,
        }

        if audio_error:
            result["audio_warning"] = f"Audio extraction failed: {audio_error}"

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


async def handle_detect_active_speaker(arguments: dict) -> dict:
    return await _handle_detect_active_speaker(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="active-speaker-detection",
        tool_defs_source="_tool_defs/face.yaml",
        handlers={
            "detect_active_speaker": handle_detect_active_speaker,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
