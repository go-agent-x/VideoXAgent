"""
Temporal grounding utilities — locate events in video by text query.

Backend: sentence-transformers — dense semantic embedding (required).

Subtitle parsing/discovery functions are imported from _base.subtitle_utils.
"""

import os
import sys
from pathlib import Path
import threading
from typing import Optional

import numpy as np

# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")

from _base.device import select_torch_device

from _base.subtitle_utils import (
    DATASET_SUBTITLE_CONFIG,
    detect_dataset_from_video_path,
    parse_timestamp,
    clean_subtitle_text,
    strip_html_tags,
    parse_srt_file,
    parse_json_subtitle_file,
    parse_jsonl_subtitle_file,
    parse_subtitle_file,
    find_subtitle_file,
)

# Alias for backward compatibility with this service's server.py
parse_srt_timestamp = parse_timestamp
find_srt_for_video = find_subtitle_file

# -------------------------------------------------------------------------
# Backend state
# -------------------------------------------------------------------------
_model_lock = threading.Lock()
_encoder = None
_backend = "uninitialized"  # "sbert" | "uninitialized"


def _init_sbert():
    """Load sentence-transformers encoder. Raises RuntimeError on failure."""
    from pathlib import Path
    model_name = os.environ.get(
        "SBERT_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    # Check local cache first
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_local = Path(model_name)
    is_local = model_local.is_dir()
    if not is_local:
        hub_cache = Path(cache_dir) / "hub"
        model_slug = "models--" + model_name.replace("/", "--")
        if not (hub_cache / model_slug).is_dir():
            raise RuntimeError(
                f"SBERT model '{model_name}' not found in local cache "
                f"(checked {hub_cache / model_slug}). "
                f"Please download the model first or set SBERT_MODEL env var to a valid path."
            )

    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from sentence_transformers import SentenceTransformer
        device_options = {}
        if os.environ.get("MCP_DEVICE") in ("cpu", "cuda"):
            import torch
            device_options["device"] = select_torch_device(torch)
        model = SentenceTransformer(model_name, **device_options)
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load SBERT model '{model_name}': {exc}"
        ) from exc
    finally:
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline


def _init_backend():
    global _encoder, _backend
    if _backend != "uninitialized":
        return
    with _model_lock:
        if _backend != "uninitialized":
            return
        _encoder = _init_sbert()
        _backend = "sbert"


def get_backend() -> str:
    _init_backend()
    return _backend


# -------------------------------------------------------------------------
# Text similarity
# -------------------------------------------------------------------------
def _sbert_similarity(query: str, texts: list[str]) -> np.ndarray:
    """Compute cosine similarities using sentence-transformers."""
    all_texts = [query] + texts
    embeddings = _encoder.encode(all_texts, normalize_embeddings=True)
    query_emb = embeddings[0:1]
    text_embs = embeddings[1:]
    sims = (text_embs @ query_emb.T).flatten()
    return sims


def compute_text_similarity(query: str, texts: list[str]) -> np.ndarray:
    """Compute text similarity using sentence-transformers."""
    _init_backend()
    return _sbert_similarity(query, texts)


# -------------------------------------------------------------------------
# Core grounding functions
# -------------------------------------------------------------------------
def _merge_adjacent_segments(segments: list[dict], gap_threshold: float = 2.0) -> list[dict]:
    """Merge subtitle segments that are temporally adjacent."""
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["start_time"] - prev["end_time"] <= gap_threshold:
            prev["end_time"] = max(prev["end_time"], seg["end_time"])
            prev["text"] = prev["text"] + " " + seg["text"]
            prev["score"] = max(prev["score"], seg["score"])
            prev["matched_subtitles"].extend(seg["matched_subtitles"])
        else:
            merged.append(seg.copy())

    return merged


def locate_by_subtitle(
    query: str,
    srt_path: str,
    top_k: int = 5,
    min_score: float = 0.1,
    merge_gap: float = 2.0,
    time_range: Optional[tuple] = None,
) -> dict:
    """Locate events by matching query against subtitle text."""
    subtitles = parse_subtitle_file(srt_path)
    if not subtitles:
        return {"success": False, "error": "No subtitles parsed from subtitle file"}

    if time_range:
        start_bound, end_bound = time_range
        subtitles = [s for s in subtitles
                     if s["end"] >= start_bound and s["start"] <= end_bound]
        if not subtitles:
            return {"success": False, "error": "No subtitles in specified time range"}

    texts = [s["text"] for s in subtitles]
    scores = compute_text_similarity(query, texts)

    scored = []
    for i, (sub, score) in enumerate(zip(subtitles, scores)):
        if float(score) >= min_score:
            scored.append({
                "start_time": sub["start"],
                "end_time": sub["end"],
                "text": sub["text"],
                "score": round(float(score), 4),
                "matched_subtitles": [{"index": sub["index"], "text": sub["text"]}],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = scored[:top_k * 3]
    top_candidates.sort(key=lambda x: x["start_time"])
    merged = _merge_adjacent_segments(top_candidates, merge_gap)
    merged.sort(key=lambda x: x["score"], reverse=True)
    results = merged[:top_k]

    return {
        "success": True,
        "backend": get_backend(),
        "query": query,
        "total_subtitles_searched": len(texts),
        "matches": results,
    }


def locate_by_captions(
    query: str,
    captions: list[dict],
    top_k: int = 5,
    min_score: float = 0.1,
) -> dict:
    """Locate events by matching query against pre-computed video captions."""
    if not captions:
        return {"success": False, "error": "No captions provided"}

    texts = [c.get("caption", c.get("text", "")) for c in captions]
    scores = compute_text_similarity(query, texts)

    results = []
    for i, (cap, score) in enumerate(zip(captions, scores)):
        if float(score) >= min_score:
            results.append({
                "start_time": cap.get("start_time", cap.get("start", 0)),
                "end_time": cap.get("end_time", cap.get("end", 0)),
                "caption": texts[i],
                "score": round(float(score), 4),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]

    return {
        "success": True,
        "backend": get_backend(),
        "query": query,
        "total_captions_searched": len(texts),
        "matches": results,
    }
