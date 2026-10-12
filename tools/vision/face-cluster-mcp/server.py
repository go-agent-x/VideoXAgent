#!/usr/bin/env python3
"""Face Cluster MCP Server — Face detection, cross-frame clustering, and matching."""
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
sys.path.insert(0, str(Path(__file__).parent))

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response

# ── Business logic (from face_cluster_server.py) ──

import json
import os
import traceback

import numpy as np

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from face_utils import (
    detect_faces_in_image,
    extract_frames_at_fps,
    cluster_embeddings,
    cosine_similarity,
)

# ---------------------------------------------------------------------------
# MCP Server init
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: format time
# ---------------------------------------------------------------------------
def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# =========================================================================
# Tool 1: face_detect — Single image face detection
# =========================================================================
async def _handle_face_detect(arguments: dict) -> dict:
    image_path = arguments.get("image_path", "")
    if not image_path:
        return {"success": False, "error": "image_path is required"}
    if not os.path.isfile(image_path):
        return {"success": False, "error": f"File not found: {image_path}"}

    try:
        faces = detect_faces_in_image(image_path)

        face_results = []
        for i, face in enumerate(faces):
            entry = {
                "face_id": i + 1,
                "bbox": face["bbox"],
                "det_score": face["det_score"],
                # "age": face.get("age"),
                "gender": face.get("gender"),
                "has_embedding": "embedding" in face,
            }
            face_results.append(entry)

        return {
            "success": True,
            "image_path": image_path,
            "total_faces": len(face_results),
            "faces": face_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# Tool 2: face_cluster_video — Cross-frame face clustering
# =========================================================================
async def _handle_face_cluster_video(arguments: dict) -> dict:
    video_path = arguments.get("video_path", "")
    start_time = float(arguments.get("start_time", 0))
    end_time_raw = arguments.get("end_time", None)
    end_time = float(end_time_raw) if end_time_raw is not None else None
    sample_fps = float(arguments.get("sample_fps", 1.0))
    distance_threshold = float(arguments.get("distance_threshold", 0.6))

    if not video_path:
        return {"success": False, "error": "video_path is required"}
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"File not found: {video_path}"}

    try:
        # Extract frames
        frames = extract_frames_at_fps(video_path, sample_fps, start_time, end_time)
        if not frames:
            return {"success": True, "unique_persons": 0, "total_face_detections": 0,
                    "note": "No frames extracted", "persons": []}

        # Detect faces in each frame
        all_detections = []  # (frame_time, face_data)
        for frame_time, frame_bgr in frames:
            import cv2
            # Save temp frame for InsightFace
            tmp_path = f"/tmp/_face_cluster_{os.getpid()}_{frame_time}.jpg"
            cv2.imwrite(tmp_path, frame_bgr)
            try:
                faces = detect_faces_in_image(tmp_path)
                for face in faces:
                    if "embedding" in face:
                        all_detections.append((frame_time, face))
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if not all_detections:
            return {"success": True, "unique_persons": 0, "total_face_detections": 0,
                    "note": "No faces detected in any frame", "persons": []}

        # Cluster embeddings
        embeddings = np.array([d[1]["embedding"] for d in all_detections])
        labels = cluster_embeddings(embeddings, distance_threshold)

        # Group by cluster
        from collections import defaultdict
        clusters = defaultdict(list)
        for idx, (frame_time, face_data) in enumerate(all_detections):
            label = int(labels[idx])
            clusters[label].append({
                "timestamp": frame_time,
                "timestamp_formatted": _fmt_time(frame_time),
                "bbox": face_data["bbox"],
                "age": face_data.get("age"),
                "gender": face_data.get("gender"),
                "det_score": face_data["det_score"],
            })

        # Build person summaries
        persons = []
        for person_id, (label, detections) in enumerate(sorted(clusters.items())):
            times = [d["timestamp"] for d in detections]
            ages = [d["age"] for d in detections if d["age"] is not None]
            genders = [d["gender"] for d in detections if d["gender"] is not None]

            # simplify the output for person appearances
            detections_simple_keys = []
            for d in detections:
                detections_simple_keys.append({
                    "timestamp": d["timestamp"],
                    "bbox": d["bbox"],
                })

            person = {
                "person_id": person_id + 1,
                "total_detections": len(detections),
                "first_seen": min(times),
                "first_seen_formatted": _fmt_time(min(times)),
                "last_seen": max(times),
                "last_seen_formatted": _fmt_time(max(times)),
                "avg_age": round(sum(ages) / len(ages)) if ages else None,
                "gender": max(set(genders), key=genders.count) if genders else None,
                "appearances": detections_simple_keys,
            }

            persons.append(person)

        # Sort by first appearance
        persons.sort(key=lambda p: p["first_seen"])
        for i, p in enumerate(persons):
            p["person_id"] = i + 1

        return {
            "success": True,
            "video_path": video_path,
            "analyzed_range": {
                "start_time": start_time,
                "end_time": end_time if end_time else "video_end",
            },
            "sample_fps": sample_fps,
            "frames_analyzed": len(frames),
            "total_face_detections": len(all_detections),
            "unique_persons": len(persons),
            "persons": persons,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# Tool 3: face_match — Compare two face images
# =========================================================================
async def _handle_face_match(arguments: dict) -> dict:
    image_a = arguments.get("image_a", "")
    image_b = arguments.get("image_b", "")

    if not image_a or not image_b:
        return {"success": False, "error": "Both image_a and image_b are required"}

    for path in [image_a, image_b]:
        if not os.path.isfile(path):
            return {"success": False, "error": f"File not found: {path}"}

    try:
        faces_a = detect_faces_in_image(image_a)
        faces_b = detect_faces_in_image(image_b)

        if not faces_a:
            return {"success": False, "error": f"No face detected in image_a: {image_a}"}
        if not faces_b:
            return {"success": False, "error": f"No face detected in image_b: {image_b}"}

        # Use the face with highest detection score from each image
        face_a = max(faces_a, key=lambda f: f["det_score"])
        face_b = max(faces_b, key=lambda f: f["det_score"])

        if "embedding" not in face_a or "embedding" not in face_b:
            return {"success": False, "error": "Failed to extract face embedding"}

        similarity = cosine_similarity(face_a["embedding"], face_b["embedding"])
        threshold = float(arguments.get("threshold", 0.4))
        is_same = similarity >= threshold

        return {
            "success": True,
            "image_a": image_a,
            "image_b": image_b,
            "similarity": round(similarity, 4),
            "threshold": threshold,
            "is_same_person": is_same,
            "face_a": {
                "bbox": face_a["bbox"],
                "age": face_a.get("age"),
                "gender": face_a.get("gender"),
            },
            "face_b": {
                "bbox": face_b["bbox"],
                "age": face_b.get("age"),
                "gender": face_b.get("gender"),
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# MCP Tool Registration
# =========================================================================


async def handle_face_detect(arguments: dict) -> dict:
    return await _handle_face_detect(arguments)


async def handle_face_cluster_video(arguments: dict) -> dict:
    return await _handle_face_cluster_video(arguments)


async def handle_face_match(arguments: dict) -> dict:
    return await _handle_face_match(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="face-cluster",
        tool_defs_source="_tool_defs/face.yaml",
        handlers={
            "face_detect": handle_face_detect,
            "face_cluster_video": handle_face_cluster_video,
            "face_match": handle_face_match,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
