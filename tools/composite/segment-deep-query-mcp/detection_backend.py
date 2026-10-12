"""Collect frame detections locally or through the isolated vision MCP process."""

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import tempfile
import threading


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_DETECTION_LOCK = threading.Lock()
_VISION_MCP_SEMAPHORE = asyncio.Semaphore(1)


def _failure(message, backend, *, status="failed"):
    result = {
        "success": False,
        "objects": [],
        "object_count": 0,
        "backend": backend,
        "status": status,
        "error": message,
    }
    if status == "unavailable":
        result["unavailable"] = True
    return result


def _normalize_result(result, backend, *, mode):
    if not isinstance(result, dict) or not isinstance(result.get("success"), bool):
        return _failure("Detection backend returned an invalid result object", backend)
    objects = result.get("objects", [])
    if not isinstance(objects, list) or any(not isinstance(obj, dict) for obj in objects):
        return _failure("Detection backend returned an invalid objects list", backend)
    normalized = dict(result)
    normalized["backend"] = backend
    normalized["objects"] = objects
    normalized.setdefault("object_count", len(objects))
    normalized.setdefault("detection_mode", mode)
    status = "failed"
    if normalized["success"]:
        status = "partial" if normalized.get("partial") else ("detected" if objects else "no_detections")
    normalized.setdefault("status", status)
    return normalized


def _write_frame(path, frame):
    # Keep the helper importable for tests and VLM-only installations.
    import cv2

    if not cv2.imwrite(str(path), frame):
        raise ValueError("OpenCV could not write the frame as JPEG")


def _detect_local_frame(frame, categories, detector):
    # The worker owns its files: cancelling the awaiting coroutine must not
    # unlink an image while an already-running inference thread still reads it.
    try:
        with tempfile.TemporaryDirectory(prefix="videox-gdino-") as directory:
            image_path = Path(directory) / "frame.jpg"
            _write_frame(image_path, frame)
            with _LOCAL_DETECTION_LOCK:
                result = detector(
                    image_source=str(image_path),
                    prompt="<prompt_free>",
                    targets=["bbox"],
                    box_threshold=0.25,
                    iou_threshold=0.8,
                    categories=categories,
                    max_objects=50,
                )
            return _normalize_result(
                result, "in_process", mode="category_sweep",
            )
    except Exception as exc:
        return _failure(f"Local detection failed: {exc}", "in_process")


def _parse_mcp_result(result):
    if getattr(result, "isError", False):
        details = [
            block.text for block in getattr(result, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        return _failure("Vision MCP tool error: " + "; ".join(details), "vision_mcp")
    for block in getattr(result, "content", []):
        if getattr(block, "type", None) != "text":
            continue
        try:
            decoded = json.loads(block.text)
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, dict):
            return _normalize_result(decoded, "vision_mcp", mode="category_sweep")
    return _failure("Vision MCP returned no JSON result object", "vision_mcp")


def _exception_message(exc):
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_exception_message(child) for child in exc.exceptions)
    return str(exc) or type(exc).__name__


def _timeout_seconds():
    try:
        seconds = float(os.environ.get("GDINO_TIMEOUT_SECONDS", "240"))
    except ValueError as exc:
        raise ValueError("GDINO_TIMEOUT_SECONDS must be a positive finite number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("GDINO_TIMEOUT_SECONDS must be a positive finite number")
    return seconds


async def _detect_via_vision(frames, categories, root):
    backend = "vision_mcp"
    raw_python = os.environ.get("GDINO_PYTHON") or str(root / ".venvs/vision/bin/python")
    python_path = Path(raw_python).expanduser()
    if not python_path.is_absolute():
        python_path = root / python_path
    # Do not resolve this symlink: Python needs the virtualenv executable path.
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        message = (
            f"Grounding DINO environment unavailable: {python_path}. "
            "Install the vision profile or set GDINO_PYTHON to its Python executable."
        )
        return [_failure(message, backend, status="unavailable") for _ in frames]
    try:
        timeout = _timeout_seconds()
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:
        return [_failure(str(exc), backend, status="unavailable") for _ in frames]

    # One child and one loaded model for the entire frame batch. Also serialize
    # concurrent segment queries so they cannot create competing GPU models.
    async with _VISION_MCP_SEMAPHORE:
        with tempfile.TemporaryDirectory(prefix="videox-gdino-") as directory:
            results = [None] * len(frames)
            image_paths = []
            for index, frame in enumerate(frames):
                image_path = Path(directory) / f"frame-{index}.jpg"
                try:
                    _write_frame(image_path, frame)
                    image_paths.append((index, image_path))
                except Exception as exc:
                    results[index] = _failure(f"Cannot write detection frame: {exc}", backend)
            if not image_paths:
                return results

            child_env = dict(os.environ)
            child_env["PATH"] = str(python_path.parent) + os.pathsep + child_env.get("PATH", "")
            parameters = StdioServerParameters(
                command=str(python_path),
                args=[
                    "-B", str(root / "scripts/mcp_runtime.py"),
                    str(root / "tools/vision/grounding-dino-mcp/server.py"),
                ],
                env=child_env,
                cwd=str(root),
            )
            initialized = False
            try:
                # Enter and leave MCP contexts in the same task. wait_for()
                # would move context entry into another task and break AnyIO's
                # cancel-scope ownership during cleanup.
                async with AsyncExitStack() as stack:
                    try:
                        async with asyncio.timeout(timeout):
                            read, write = await stack.enter_async_context(stdio_client(parameters))
                            client = await stack.enter_async_context(ClientSession(
                                read, write, read_timeout_seconds=timedelta(seconds=timeout),
                            ))
                            await client.initialize()
                    except TimeoutError as exc:
                        raise RuntimeError(f"Vision MCP startup timed out after {timeout:g} seconds") from exc
                    initialized = True
                    for index, image_path in image_paths:
                        arguments = {"image_path": str(image_path)}
                        if categories:
                            arguments["categories"] = list(categories)
                        try:
                            async with asyncio.timeout(timeout):
                                response = await client.call_tool("scene_scan", arguments)
                        except TimeoutError as exc:
                            raise RuntimeError(f"Vision MCP detection timed out after {timeout:g} seconds") from exc
                        results[index] = _parse_mcp_result(response)
            except Exception as exc:
                message = f"Vision MCP detection failed: {_exception_message(exc)}"
                for index, result in enumerate(results):
                    if result is None:
                        results[index] = _failure(
                            message, backend, status="failed" if initialized else "unavailable",
                        )
            return results


async def detect_frames(frames, categories=None, *, local_detector=None, root=PROJECT_ROOT):
    """Return one detection result per frame without requiring vision in base.

    Explicit categories are an ordered sequence of phrases (for example,
    ``["person", "traffic light"]``). Omit them to scan common categories.
    Errors are returned as evidence status; task cancellation still propagates.
    """
    frames = list(frames)
    if not frames:
        return []
    if categories is not None:
        if not isinstance(categories, (list, tuple)) or not 1 <= len(categories) <= 64 or any(
            not isinstance(category, str) or not category.strip()
            or len(" ".join(category.split())) > 80 or len(category.split()) > 8
            or category.strip() == "<prompt_free>" for category in categories
        ):
            backend = "in_process" if local_detector is not None else "vision_mcp"
            return [_failure("categories must contain 1 to 64 nonempty phrases (at most 80 characters and 8 words each)", backend) for _ in frames]
        categories = [" ".join(category.split()) for category in categories]
    if local_detector is not None:
        results = []
        for frame in frames:
            results.append(await asyncio.to_thread(_detect_local_frame, frame, categories, local_detector))
        return results
    return await _detect_via_vision(frames, categories, Path(root).absolute())
