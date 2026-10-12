#!/usr/bin/env python3
"""Moment Retrieval MCP Server — Visual temporal grounding."""
import asyncio
import sys
from pathlib import Path

# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")
sys.path.insert(0, str(Path(__file__).parent))  # to import from original moment_retrieval_server.py

from _base.device import select_torch_device
from _base.server_framework import create_mcp_server, run_server

# ── Business logic (from moment_retrieval_server.py) ──

import hashlib
import json
import logging
import math
import os
import threading
import traceback
from typing import Optional

log = logging.getLogger(__name__)

import cv2
import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy scalars/arrays when serializing to JSON."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
MOMENT_DETR_CHECKPOINT = os.environ.get("MOMENT_DETR_CHECKPOINT", "")
CACHE_DIR = os.environ.get(
    "MOMENT_RETRIEVAL_CACHE",
    os.path.expanduser("~/.cache/moment_retrieval"),
)

# Moment-DETR model hyperparameters (match standard QVHighlights checkpoint)
_DEFAULT_MODEL_CONFIG = {
    "d_model": 256,
    "nhead": 8,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "num_queries": 10,
    "input_video_dim": 512,
    "input_text_dim": 512,
    "dim_feedforward": 1024,
    "dropout": 0.1,
}

# Override via env JSON string, e.g. '{"d_model": 512, "num_queries": 20}'
_config_override = os.environ.get("MOMENT_DETR_CONFIG", "")
if _config_override:
    try:
        _DEFAULT_MODEL_CONFIG.update(json.loads(_config_override))
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Global backend state
# ---------------------------------------------------------------------------
_model_lock = threading.Lock()
_clip_model = None
_clip_processor = None
_clip_tokenizer = None
_moment_model = None
_backend = "uninitialized"  # "moment_detr" | "clip_similarity" | "unavailable"
_device = None

# ---------------------------------------------------------------------------
# CLIP model loading
# ---------------------------------------------------------------------------

def _try_init_clip():
    """Load CLIP model from local cache / directory. No download."""
    global _device
    import torch

    _device = select_torch_device(torch)

    model_name = CLIP_MODEL_NAME

    # Check if model is a local directory
    if os.path.isdir(model_name):
        model_dir = model_name
    else:
        # Resolve HuggingFace hub cache to snapshot path for reliable loading
        slug = model_name.replace("/", "--")
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        cache_model = os.path.join(hf_home, "hub", f"models--{slug}")
        snapshot_dir = None
        if os.path.isdir(cache_model):
            snapshots = os.path.join(cache_model, "snapshots")
            if os.path.isdir(snapshots):
                # Pick the snapshot that contains config.json (complete download)
                for rev in sorted(os.listdir(snapshots), reverse=True):
                    candidate = os.path.join(snapshots, rev)
                    if os.path.isfile(os.path.join(candidate, "config.json")):
                        snapshot_dir = candidate
                        break
        if snapshot_dir:
            model_dir = snapshot_dir
        else:
            print(
                f"[moment-retrieval] CLIP model '{model_name}' not found in "
                f"local cache ({cache_model}). Will attempt download.",
                file=sys.stderr,
            )
            model_dir = model_name

    orig_offline = os.environ.get("HF_HUB_OFFLINE")
    try:
        from transformers import CLIPModel, CLIPProcessor

        # Prefer safetensors to avoid torch.load CVE-2025-32434 issue
        # with torch < 2.6 + transformers >= 5.x
        load_kwargs = {}
        try:
            if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
                # Check if safetensors file exists in model_dir
                if os.path.isdir(model_dir) and os.path.isfile(
                    os.path.join(model_dir, "model.safetensors")
                ):
                    load_kwargs["use_safetensors"] = True
        except Exception:
            pass

        model = CLIPModel.from_pretrained(model_dir, **load_kwargs).to(_device).eval()
        processor = CLIPProcessor.from_pretrained(model_dir)
        print(f"[moment-retrieval] CLIP loaded: {model_name} on {_device}", file=sys.stderr)
        return model, processor
    except Exception as exc:
        print(f"[moment-retrieval] Failed to load CLIP: {exc}", file=sys.stderr)
        return None
    finally:
        if orig_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = orig_offline


# ---------------------------------------------------------------------------
# Moment-DETR model definition (simplified inference-only)
# ---------------------------------------------------------------------------

def _build_moment_detr_model():
    """Build a Moment-DETR model from config and load checkpoint."""
    import torch
    import torch.nn as nn

    cfg = _DEFAULT_MODEL_CONFIG

    class SinusoidalPE(nn.Module):
        """Sinusoidal positional encoding."""

        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * -(math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

        def forward(self, length: int):
            return self.pe[:, :length, :]

    class MomentDETR(nn.Module):
        """
        Simplified Moment-DETR for inference.

        Input:  video_features (B, T, D_v), text_features (B, L, D_t)
        Output: pred_spans (B, Q, 2)  -- (center, width) in [0,1]
                pred_logits (B, Q)    -- raw confidence logits
                saliency   (B, T)    -- per-clip saliency logits
        """

        def __init__(self):
            super().__init__()
            d = cfg["d_model"]
            nhead = cfg["nhead"]
            dff = cfg["dim_feedforward"]
            drop = cfg["dropout"]

            # Input projections
            self.video_proj = nn.Sequential(
                nn.Linear(cfg["input_video_dim"], d), nn.ReLU(), nn.Dropout(drop)
            )
            self.text_proj = nn.Sequential(
                nn.Linear(cfg["input_text_dim"], d), nn.ReLU(), nn.Dropout(drop)
            )

            # Positional encoding
            self.pos_enc = SinusoidalPE(d)

            # Transformer encoder
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=nhead, dim_feedforward=dff,
                dropout=drop, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, cfg["num_encoder_layers"])

            # Transformer decoder
            dec_layer = nn.TransformerDecoderLayer(
                d_model=d, nhead=nhead, dim_feedforward=dff,
                dropout=drop, batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(dec_layer, cfg["num_decoder_layers"])

            # Learnable moment queries
            self.moment_queries = nn.Embedding(cfg["num_queries"], d)

            # Prediction heads
            self.span_head = nn.Sequential(
                nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2), nn.Sigmoid()
            )
            self.confidence_head = nn.Sequential(
                nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1)
            )
            self.saliency_head = nn.Linear(d, 1)

        def forward(self, video_feat, text_feat):
            B, T, _ = video_feat.shape

            # Project
            v = self.video_proj(video_feat)   # (B, T, d)
            t = self.text_proj(text_feat)     # (B, L, d)
            if t.dim() == 2:
                t = t.unsqueeze(1)            # (B, 1, d)

            # Positional encoding for video tokens
            v = v + self.pos_enc(T)

            # Encode (video + text concatenated)
            src = torch.cat([v, t], dim=1)    # (B, T+L, d)
            memory = self.encoder(src)        # (B, T+L, d)

            # Decode with learnable moment queries
            queries = self.moment_queries.weight.unsqueeze(0).expand(B, -1, -1)
            decoded = self.decoder(queries, memory)  # (B, Q, d)

            # Predict
            spans = self.span_head(decoded)                        # (B, Q, 2)
            logits = self.confidence_head(decoded).squeeze(-1)     # (B, Q)

            video_memory = memory[:, :T, :]
            saliency = self.saliency_head(video_memory).squeeze(-1)  # (B, T)

            return {
                "pred_spans": spans,
                "pred_logits": logits,
                "saliency_scores": saliency,
            }

    # Build and load checkpoint
    model = MomentDETR()
    ckpt_path = MOMENT_DETR_CHECKPOINT

    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        # Attempt flexible loading (strict=False for architecture variants)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(
                f"[moment-retrieval] Moment-DETR: {len(missing)} missing keys "
                f"(first 5: {missing[:5]})",
                file=sys.stderr,
            )
        model = model.to(_device).eval()
        print(
            f"[moment-retrieval] Moment-DETR loaded from {ckpt_path} on {_device}",
            file=sys.stderr,
        )
        return model
    except Exception as exc:
        print(f"[moment-retrieval] Failed to load Moment-DETR: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Backend initialization (double-checked locking)
# ---------------------------------------------------------------------------

def _init_backend():
    global _clip_model, _clip_processor, _moment_model, _backend
    if _backend != "uninitialized":
        return
    with _model_lock:
        if _backend != "uninitialized":
            return

        # CLIP is required by both backends
        clip_result = _try_init_clip()
        if clip_result is None:
            _backend = "unavailable"
            return
        _clip_model, _clip_processor = clip_result

        # Try loading Moment-DETR
        moment = _build_moment_detr_model()
        if moment is not None:
            _moment_model = moment
            _backend = "moment_detr"
        else:
            _backend = "clip_similarity"
            print(
                "[moment-retrieval] Moment-DETR unavailable, using CLIP similarity fallback.",
                file=sys.stderr,
            )


def _get_backend() -> str:
    _init_backend()
    return _backend


# ---------------------------------------------------------------------------
# Video frame sampling
# ---------------------------------------------------------------------------

def _sample_video_frames(video_path: str, start_time: float = 0,
                         end_time: Optional[float] = None,
                         sample_fps: float = 2.0):
    """
    Sample video frames at a given FPS.
    Returns list of (timestamp_sec, frame_rgb_ndarray).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if end_time is None or end_time > duration:
        end_time = duration

    # Compute frame indices to sample
    step = 1.0 / sample_fps
    timestamps = []
    t = start_time + step / 2  # Start at half-step to center samples
    while t < end_time:
        timestamps.append(t)
        t += step

    if not timestamps:
        timestamps = [(start_time + end_time) / 2]

    frames = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append((ts, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames, duration


# ---------------------------------------------------------------------------
# CLIP feature extraction
# ---------------------------------------------------------------------------

def _extract_clip_features(frames_with_ts, batch_size: int = 32):
    """
    Extract CLIP visual features from a list of (timestamp, frame_rgb) tuples.
    Returns (features: np.ndarray(T,D), timestamps: np.ndarray(T,)).
    """
    import torch

    if not frames_with_ts:
        return np.zeros((0, 512), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    timestamps = np.array([t for t, _ in frames_with_ts], dtype=np.float32)
    frames = [f for _, f in frames_with_ts]

    all_features = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        inputs = _clip_processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _clip_model.get_image_features(**inputs)
            # transformers >= 5.x returns BaseModelOutputWithPooling
            if hasattr(outputs, "pooler_output"):
                outputs = outputs.pooler_output
            # L2 normalize
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        all_features.append(outputs.cpu().numpy())

    features = np.concatenate(all_features, axis=0)  # (T, D)
    return features, timestamps


def _encode_text_query(query: str):
    """
    Encode text query with CLIP text encoder.
    Returns np.ndarray of shape (D,).
    """
    import torch

    inputs = _clip_processor(text=[query], return_tensors="pt", padding=True,
                             truncation=True, max_length=77)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _clip_model.get_text_features(**inputs)
        # transformers >= 5.x returns BaseModelOutputWithPooling
        if hasattr(outputs, "pooler_output"):
            outputs = outputs.pooler_output
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
    return outputs.cpu().numpy().squeeze(0)  # (D,)


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------

def _get_cache_path(video_path: str, sample_fps: float) -> str:
    """Generate cache file path for extracted features."""
    key = f"{os.path.abspath(video_path)}|{sample_fps}|{CLIP_MODEL_NAME}"
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.npz")


def _load_cached_features(cache_path: str):
    """Load cached features. Returns (features, timestamps) or None."""
    if not os.path.isfile(cache_path):
        return None
    try:
        data = np.load(cache_path)
        return data["features"], data["timestamps"]
    except Exception:
        return None


def _save_features_cache(cache_path: str, features: np.ndarray, timestamps: np.ndarray):
    """Save extracted features to cache."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # np.savez auto-appends .npz, so use np.savez on the final path directly
        np.savez(cache_path, features=features, timestamps=timestamps)
    except Exception as exc:
        print(f"[moment-retrieval] Cache save failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Moment-DETR inference
# ---------------------------------------------------------------------------

def _moment_detr_retrieve(video_features: np.ndarray, text_features: np.ndarray,
                          timestamps: np.ndarray, video_duration: float,
                          top_k: int = 5, threshold: float = 0.3):
    """
    Run Moment-DETR inference.
    Returns list of (start_sec, end_sec, confidence) tuples + saliency scores.
    """
    import torch

    MAX_SEQ_LEN = 5000
    T = video_features.shape[0]

    if T > MAX_SEQ_LEN:
        log.warning(f"Frame count {T} exceeds Moment-DETR max positional encoding length "
                    f"{MAX_SEQ_LEN}, uniformly downsampling to {MAX_SEQ_LEN} frames")
        indices = np.linspace(0, T - 1, MAX_SEQ_LEN, dtype=int)
        video_features = video_features[indices]
        timestamps = timestamps[indices]
        T = MAX_SEQ_LEN

    # Prepare tensors
    v_tensor = torch.from_numpy(video_features).unsqueeze(0).to(_device)  # (1,T,D)
    t_tensor = torch.from_numpy(text_features).unsqueeze(0).to(_device)   # (1,D)

    with torch.no_grad():
        outputs = _moment_model(v_tensor, t_tensor)

    # Extract predictions
    spans = outputs["pred_spans"].cpu().numpy()[0]       # (Q, 2) center, width in [0,1]
    logits = outputs["pred_logits"].cpu().numpy()[0]     # (Q,)
    saliency = outputs["saliency_scores"].cpu().numpy()[0]  # (T,)

    # Convert spans to seconds
    search_start = float(timestamps[0]) if len(timestamps) > 0 else 0
    search_end = float(timestamps[-1]) if len(timestamps) > 0 else video_duration
    search_duration = search_end - search_start

    moments = []
    for i in range(spans.shape[0]):
        center, width = spans[i]
        score = float(1.0 / (1.0 + math.exp(-logits[i])))  # sigmoid

        start_sec = search_start + (center - width / 2) * search_duration
        end_sec = search_start + (center + width / 2) * search_duration

        # Clamp to valid range
        start_sec = max(search_start, min(start_sec, search_end))
        end_sec = max(start_sec + 0.1, min(end_sec, search_end))

        if score >= threshold:
            moments.append((start_sec, end_sec, score))

    # NMS
    moments = _temporal_nms(moments, iou_threshold=0.5)
    moments.sort(key=lambda x: x[2], reverse=True)

    # Saliency as list
    saliency_list = [float(s) for s in saliency]

    return moments[:top_k], saliency_list


# ---------------------------------------------------------------------------
# CLIP similarity fallback
# ---------------------------------------------------------------------------

def _clip_similarity_retrieve(video_features: np.ndarray, text_features: np.ndarray,
                              timestamps: np.ndarray, video_duration: float,
                              top_k: int = 5, threshold: float = 0.3):
    """
    Fallback: frame-level CLIP cosine similarity + multi-scale sliding window.
    Returns list of (start_sec, end_sec, confidence) tuples + similarity scores.
    """
    # Frame-level cosine similarity (features already L2-normalized)
    similarities = video_features @ text_features  # (T,)

    # Gaussian smoothing
    try:
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(similarities, sigma=1.5)
    except ImportError:
        # Manual simple moving average as fallback
        kernel = 3
        smoothed = np.convolve(similarities, np.ones(kernel) / kernel, mode="same")

    T = len(smoothed)
    if T == 0:
        return [], []

    # Multi-scale sliding window
    window_sizes = [2, 3, 5, 8, 12]
    moments = []
    for ws in window_sizes:
        if ws > T:
            continue
        for start_idx in range(T - ws + 1):
            end_idx = start_idx + ws
            window_score = float(np.mean(smoothed[start_idx:end_idx]))
            if window_score >= threshold:
                start_sec = float(timestamps[start_idx])
                end_sec = float(timestamps[min(end_idx - 1, T - 1)])
                # Use the timestamp of the last frame in window + half a step
                if end_idx < T:
                    end_sec = (float(timestamps[end_idx - 1]) + float(timestamps[end_idx])) / 2
                moments.append((start_sec, end_sec, window_score))

    # NMS
    moments = _temporal_nms(moments, iou_threshold=0.4)
    moments.sort(key=lambda x: x[2], reverse=True)

    similarity_list = [float(s) for s in similarities]

    return moments[:top_k], similarity_list


# ---------------------------------------------------------------------------
# Temporal NMS
# ---------------------------------------------------------------------------

def _temporal_nms(moments, iou_threshold: float = 0.5):
    """Non-maximum suppression on temporal moments (start, end, score)."""
    if not moments:
        return []

    # Sort by score descending
    moments = sorted(moments, key=lambda x: x[2], reverse=True)
    keep = []
    for m in moments:
        suppress = False
        for k in keep:
            iou = _temporal_iou(m, k)
            if iou > iou_threshold:
                suppress = True
                break
        if not suppress:
            keep.append(m)
    return keep


def _temporal_iou(a, b):
    """Compute IoU between two temporal segments (start, end, score)."""
    inter_start = max(a[0], b[0])
    inter_end = min(a[1], b[1])
    inter = max(0, inter_end - inter_start)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    if union <= 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

def moment_retrieve(video_path: str, query: str, top_k: int = 5,
                    threshold: float = 0.3, start_time: float = 0,
                    end_time: Optional[float] = None, sample_fps: float = 2.0):
    """
    Main moment retrieval function.

    Returns a result dict with moments and saliency/similarity scores.
    """
    import torch

    _init_backend()

    if _backend == "unavailable":
        return {
            "success": False,
            "error": "CLIP model not available. Cannot perform visual moment retrieval.",
        }

    if not os.path.isfile(video_path):
        return {"success": False, "error": f"Video file not found: {video_path}"}

    # 1. Extract or load cached CLIP features (for full video at given FPS)
    cache_path = _get_cache_path(video_path, sample_fps)
    cached = _load_cached_features(cache_path)

    if cached is not None:
        all_features, all_timestamps = cached
        # Sample frames for duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()
    else:
        frames_with_ts, video_duration = _sample_video_frames(
            video_path, start_time=0, end_time=None, sample_fps=sample_fps
        )
        if not frames_with_ts:
            return {"success": False, "error": "No frames could be extracted from video."}
        all_features, all_timestamps = _extract_clip_features(frames_with_ts)
        _save_features_cache(cache_path, all_features, all_timestamps)

    # 2. Restrict to search range
    if end_time is None:
        end_time = video_duration

    mask = (all_timestamps >= start_time) & (all_timestamps <= end_time)
    features = all_features[mask]
    timestamps = all_timestamps[mask]

    if len(features) == 0:
        return {
            "success": False,
            "error": f"No frames in range [{start_time}, {end_time}].",
        }

    # 3. Encode text query
    text_features = _encode_text_query(query)

    # 4. Run retrieval
    if _backend == "moment_detr":
        if len(features) > 10000:
            log.warning(f"Frame count {len(features)} is extremely large (>10000), "
                        "falling back to clip_similarity backend for stability")
            moments, score_list = _clip_similarity_retrieve(
                features, text_features, timestamps, video_duration,
                top_k=top_k, threshold=threshold,
            )
        else:
            moments, score_list = _moment_detr_retrieve(
                features, text_features, timestamps, video_duration,
                top_k=top_k, threshold=threshold,
            )
    else:
        moments, score_list = _clip_similarity_retrieve(
            features, text_features, timestamps, video_duration,
            top_k=top_k, threshold=threshold,
        )

    # 5. Format results (streamlined output - removed redundant fields)
    result_moments = []
    for s, e, c in moments:
        result_moments.append({
            "start_time": round(s, 2),
            "end_time": round(e, 2),
            "confidence": round(c, 4),
        })

    result = {
        "success": True,
        "video_duration": round(video_duration, 2),
        "moments": result_moments,
    }

    # Clean up CUDA cache
    if _device == "cuda":
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def handle_moment_retrieve(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    query = arguments.get("query", "")
    top_k = int(arguments.get("top_k", 5))
    threshold = float(arguments.get("threshold", 0.3))
    start_time = float(arguments.get("start_time", 0))
    end_time_raw = arguments.get("end_time")
    end_time = float(end_time_raw) if end_time_raw is not None else None
    sample_fps = float(arguments.get("sample_fps", 2.0))

    if not video_path:
        return {"success": False, "error": "video_path is required."}
    if not query:
        return {"success": False, "error": "query is required."}

    return moment_retrieve(
        video_path=video_path,
        query=query,
        top_k=top_k,
        threshold=threshold,
        start_time=start_time,
        end_time=end_time,
        sample_fps=sample_fps,
    )


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="moment-retrieval",
        tool_defs_source="_tool_defs/temporal.yaml",
        handlers={"moment_retrieve": handle_moment_retrieve},
        is_file=True,
    )
    asyncio.run(run_server(app))
