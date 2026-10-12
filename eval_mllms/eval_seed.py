#!/usr/bin/env python3
"""VideoMME Benchmark Evaluation Script with VideoCut Tool (Function-Call Mode).

Quota reporting is optional and disabled by default. Set --quota-url, the YAML
quota_url field, or VLM_QUOTA_URL (in that priority order) to enable it. An explicit
empty value disables it. Queries use the evaluator API key as a Bearer token.
"""

import argparse
import atexit
import base64
import io
import json
import math
import os
import re
import subprocess
import sys
import termios
import threading
import time

import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import tempfile

import cv2
import httpx
import numpy as np


# ── Thread-safe Output ────────────────────────────────────────────────────────

class _ThreadLocalStdout:
    """Stdout wrapper that supports per-thread output buffering.

    When a thread calls ``start_buffering()``, all subsequent ``print()``
    output from *that* thread is captured in a thread-local StringIO buffer.
    Calling ``flush_buffer()`` atomically writes the entire buffer to the
    real stdout under a lock, preventing interleaved output from concurrent
    workers.  Threads that have not started buffering (e.g. the main thread)
    write directly to real stdout, also under the same lock.
    """

    def __init__(self, real_stdout):
        self._real = real_stdout
        self._local = threading.local()
        self._lock = threading.Lock()

    # -- file-like interface used by print() ----------------------------------

    def write(self, s):
        buf = getattr(self._local, "buffer", None)
        if buf is not None:
            buf.write(s)
        else:
            with self._lock:
                self._real.write(s)

    def flush(self):
        buf = getattr(self._local, "buffer", None)
        if buf is None:
            self._real.flush()

    # -- buffering control ----------------------------------------------------

    def start_buffering(self):
        """Begin capturing output for the calling thread."""
        self._local.buffer = io.StringIO()

    def flush_buffer(self):
        """Write all buffered output to real stdout in one atomic chunk."""
        buf = getattr(self._local, "buffer", None)
        if buf is not None:
            output = buf.getvalue()
            self._local.buffer = None
            if output:
                with self._lock:
                    self._real.write(output)
                    self._real.flush()

    # -- forward everything else to the real stdout ---------------------------

    def __getattr__(self, name):
        return getattr(self._real, name)


_result_lock = threading.Lock()


def _save_terminal_settings():
    """Save terminal settings and register atexit handler to restore them.

    OpenCV (cv2) and some FFmpeg backends modify termios settings (e.g. turn
    off echo) when imported or when VideoCapture is used.  If the process
    exits without restoring them the terminal appears "frozen" — you can type
    but characters are invisible.  Calling this once at program start ensures
    settings are always restored on exit.
    """
    try:
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, saved)
    except (termios.error, ValueError, OSError):
        pass  # not a real terminal (piped stdin, CI, etc.)


# ── Subtitle Directory Auto-Detection ────────────────────────────────────────

# Maps data path keywords to their subtitle directories.
# Order matters: first match wins. Use None for datasets without subtitles.
SUBTITLE_DIR_MAP = [
    ("Video-MME-v2", "data/Video-MME-v2/subtitle_srt"),
    ("videomme",     "data/videomme/subtitle"),
    ("LongVideoBench","data/LongVideoBench/subtitle_srt"),
    ("LVBench",      None),
    ("minerva",      None),
]


def infer_subtitle_dir(data_path: str) -> str | None:
    """Infer subtitle directory from data file path.

    Returns:
        Subtitle directory path, or None if the dataset has no subtitles.
        Empty string if no match found (fallback to video-side .srt detection).
    """
    for keyword, sub_dir in SUBTITLE_DIR_MAP:
        if keyword in data_path:
            return sub_dir
    return ""


# ── VideoCut Tool ─────────────────────────────────────────────────────────────

VIDECUT_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "VideoCut",
        "description": "Request high-density frame sampling for a specific time range in the video. Use this when low-frequency sampled frames miss crucial details (small objects, fast motion, text, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "timestamps": {
                    "type": "string",
                    "description": "Time range in format 'start_sec - end_sec' (e.g., '5 - 10' for 5-10 seconds)"
                },
                "fps": {
                    "type": "integer",
                    "description": "Sampling rate in frames per second (1-5 recommended)",
                    "minimum": 1,
                    "maximum": 5
                }
            },
            "required": ["timestamps", "fps"]
        }
    }
}

VIDECUT_TOOL_PROMPT = """**VideoCut Tool:**
You have access to the `VideoCut` tool for requesting high-density frame sampling.

**When to use:** ONLY when current frames are clearly insufficient for:
- Counting objects accurately
- Reading text/numbers on screen
- Detecting fast motion or brief events

**Important:** Do NOT call VideoCut if the current frames are sufficient to answer the question. Analyze what you have first, and only request more frames if truly needed.

**Parameters:**
- timestamps: Time range "start - end" (in seconds)
- fps: Sampling rate 1-5 (higher = more detail)"""


def parse_timestamps(ts_str: str) -> tuple[float, float]:
    """
    Parse timestamps string into (start_sec, end_sec).
    Handles formats: "10-20", "10 - 20", "10,20", "10-18, 90-95" (takes first range)
    """
    ts_str = ts_str.strip()

    # Try "start - end" format (with optional spaces)
    match = re.match(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)', ts_str)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Try comma-separated "start, end" format
    match = re.match(r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)', ts_str)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Fallback: extract first two numbers
    numbers = re.findall(r'\d+(?:\.\d+)?', ts_str)
    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])

    raise ValueError(f"Cannot parse timestamps: {ts_str}")


def extract_frames_in_range(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float = 3.0,
) -> list[tuple[float, str]]:

    """
    Extract frames from a specific time range with higher density.

    Returns:
        List of (timestamp, base64) tuples
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    # Clamp time range
    start_sec = max(0, min(start_sec, duration))
    end_sec = max(start_sec, min(end_sec, duration))

    # Calculate frame numbers
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)

    # Generate timestamps at desired FPS
    step = 1.0 / fps if fps > 0 else 1.0
    timestamps = []
    ts = start_sec
    while ts <= end_sec:
        timestamps.append(ts)
        ts += step

    frames: list[tuple[float, str]] = []
    for ts in timestamps:
        frame_num = int(ts * video_fps)
        frame_num = max(start_frame, min(frame_num, end_frame - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append((ts, frame_to_base64(frame)))

    cap.release()
    return frames


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://example.com"
DEFAULT_MODEL = "doubao-seed-2-0-pro-260215"
DEFAULT_DATA = "data/videomme/test/long_duration_data.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_api_url(url: str) -> str:
    """Ensure URL has protocol and ends with /v1/chat/completions."""
    if not url:
        raise ValueError("API URL is required")
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/v1/chat/completions"
    return url


def resize_frame(frame: np.ndarray, max_long_edge: int = 480) -> np.ndarray:
    """Resize frame so that its longer edge does not exceed max_long_edge."""
    h, w = frame.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return frame
    scale = max_long_edge / long_edge
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_to_base64(frame: np.ndarray, quality: int = 85, max_long_edge: int = 480) -> str:
    """Convert OpenCV frame to base64 encoded JPEG, resizing if too large."""
    frame = resize_frame(frame, max_long_edge)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return base64.b64encode(buffer).decode("utf-8")


def video_to_base64(video_path: str) -> str:
    """Read video file and convert to base64."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_video_format(video_path: str) -> str:
    """Get video format from file extension."""
    ext = Path(video_path).suffix.lower()
    format_map = {
        ".mp4": "mp4",
        ".avi": "avi",
        ".mov": "quicktime",
        ".mkv": "x-matroska",
        ".webm": "webm",
    }
    return format_map.get(ext, "mp4")


def reencode_video(
    video_path: str,
    fps: float = 1.0,
    max_long_edge: int = 480,
    max_payload_mb: float = 50.0,
) -> tuple[str, float, float]:
    """
    Re-encode video with target fps and resolution using ffmpeg.
    Dynamically calculates bitrate to keep base64 payload under max_payload_mb.

    Returns:
        (tmp_path, original_size_mb, new_size_mb)
    """
    # Probe original resolution and duration
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    cap.release()
    
    # Calculate target bitrate: base64 inflates by 4/3, leave 0.5MB for JSON/prompt overhead
    max_file_mb = max_payload_mb / 1.34 - 0.5
    if duration > 0:
        target_bitrate_kbps = int(max_file_mb * 1024 * 8 / duration)
        # Clamp: no less than 50k (too blurry), no more than 500k (diminishing returns at 480p)
        target_bitrate_kbps = max(50, min(target_bitrate_kbps, 500))
    else:
        target_bitrate_kbps = 200

    # Calculate scale filter: fit long edge to max_long_edge, keep aspect ratio, ensure even dimensions
    long_edge = max(w, h)
    if long_edge > max_long_edge:
        if w >= h:
            scale_filter = f"scale={max_long_edge}:-2"
        else:
            scale_filter = f"scale=-2:{max_long_edge}"
    else:
        scale_filter = None

    # Build ffmpeg command
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-r", str(fps),
    ]
    if scale_filter:
        cmd += ["-vf", scale_filter]
    cmd += [
        "-c:v", "libopenh264",
        "-b:v", f"{target_bitrate_kbps}k",
        "-an",               # drop audio
        "-movflags", "+faststart",
        tmp_path,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        os.unlink(tmp_path)
        stderr = result.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ffmpeg re-encode failed: {stderr}")

    original_size_mb = os.path.getsize(video_path) / 1024 / 1024
    new_size_mb = os.path.getsize(tmp_path) / 1024 / 1024
    print(f"[INFO] Re-encoded video: {w}x{h} → max_edge={max_long_edge}, fps={fps}, bitrate={target_bitrate_kbps}kbps")
    print(f"[INFO] Size: {original_size_mb:.2f} MB → {new_size_mb:.2f} MB (payload ≈ {new_size_mb * 1.34:.2f} MB)")
    return tmp_path, original_size_mb, new_size_mb


def extract_frames(
    video_path: str,
    fps: float = 1.0,
    max_long_edge: int = 480,
    max_payload_mb: float = 64.0,
    max_frames: int = 1024,
    sampling_strategy: str = "adaptive",
) -> tuple[list[tuple[float, str]], float, float]:
    """
    Extract frames from video at the given FPS and return (timestamp, base64) pairs.

    Automatically adjusts the number of frames to fit within max_payload_mb
    by sampling a frame to estimate per-frame size, then capping accordingly.

    Returns:
        Tuple of (frames, actual_fps, duration) where:
        - frames: List of (timestamp, base64) tuples
        - actual_fps: The actual sampling fps used
        - duration: Total video duration in seconds

    Args:
        sampling_strategy: "adaptive" or "uniform"
            - "adaptive": Reduce fps for long videos (>10min: 0.67fps, >30min: 0.33fps, >1hr: 0.2fps)
            - "uniform": Use 1fps by default; only uniformly sample max_frames when 1fps exceeds max_frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    if sampling_strategy == "uniform":
        # Uniform sampling: always use max_frames, fps is derived
        pass  # No fps adjustment, will use max_frames directly
    else:
        # Adaptive fps: reduce for long videos
        if duration > 3600:      # > 1 hr: 1 frame / 5s
            fps = min(fps, 0.2)
        elif duration > 1800:    # > 30 min: 1 frame / 3s
            fps = min(fps, 0.33)
        elif duration > 600:     # > 10 min: 1 frame / 1.5s
            fps = min(fps, 0.67)

    # Sample a frame from the middle to estimate per-frame base64 size
    mid_frame_num = min(total_frames // 2, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_num)
    ret, sample_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Cannot read sample frame from: {video_path}")

    sample_b64 = frame_to_base64(sample_frame, max_long_edge=max_long_edge)
    frame_size_bytes = len(sample_b64)  # base64 string length ≈ byte count in payload
    budget_bytes = int(max_payload_mb * 1024 * 1024)
    budget_frames = max(1, budget_bytes // frame_size_bytes)

    if sampling_strategy == "uniform":
        # Uniform: 1fps by default, only use max_frames when 1fps exceeds it
        fps_based_count = max(1, int(duration * fps))  # fps defaults to 1.0
        if fps_based_count > max_frames:
            # 1fps sampling exceeds max_frames, uniformly sample max_frames
            desired_count = max_frames
        else:
            # 1fps sampling is within max_frames, use 1fps
            desired_count = fps_based_count
        actual_count = min(desired_count, budget_frames)
    else:
        # Adaptive: fps-based count, constrained by budget and max_frames
        desired_count = max(1, int(duration * fps))
        actual_count = min(desired_count, budget_frames, max_frames)

    # Always print video info
    print(f"[INFO] Video duration: {duration:.1f}s, sampling {actual_count} frames "
          f"(1 frame every ~{duration/actual_count:.1f}s)")

    if actual_count < desired_count:
        print(f"[INFO] Frame budget: {frame_size_bytes/1024:.0f}KB/frame, "
              f"requested {desired_count} frames, capped to {actual_count}")

    actual_fps = actual_count / duration if duration > 0 else fps

    if actual_count == 1:
        timestamps = [duration / 2]
    else:
        step = duration / (actual_count - 1)
        timestamps = [i * step for i in range(actual_count)]

    frames: list[tuple[float, str]] = []
    for ts in timestamps:
        frame_num = int(ts * video_fps)
        frame_num = min(frame_num, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append((ts, frame_to_base64(frame, max_long_edge=max_long_edge)))

    cap.release()
    return frames, actual_fps, duration


def read_srt_subtitle(subtitle_path: str) -> str:
    """
    Parse SRT subtitle file and extract dialogue/text with timestamps.
    Automatically removes HTML tags (e.g., <font> tags) from subtitle text.
    Preserves timestamps for temporal context.

    Returns formatted subtitle text with timestamps: [HH:MM:SS,mmm] Text
    """
    if not os.path.isfile(subtitle_path):
        return ""

    try:
        with open(subtitle_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse SRT format
        lines = content.strip().split("\n")
        subtitle_items = []
        current_timestamp = None

        for line in lines:
            line = line.strip()

            # Extract timestamp line (format: HH:MM:SS,mmm --> HH:MM:SS,mmm)
            if "-->" in line:
                # Get start time (before the arrow)
                start_time = line.split("-->")[0].strip()
                current_timestamp = start_time
            # Skip empty lines and sequence numbers
            elif line and not line.isdigit() and current_timestamp:
                # Remove HTML tags
                cleaned_text = re.sub(r'<[^>]+>', '', line).strip()
                if cleaned_text:
                    # Append with timestamp: [HH:MM:SS,mmm] Text
                    subtitle_items.append(f"[{current_timestamp}] {cleaned_text}")
                    current_timestamp = None  # Reset for next subtitle

        return "\n".join(subtitle_items)
    except Exception as e:
        print(f"[WARN] Failed to read subtitle {subtitle_path}: {e}")
        return ""


# ── Prompt text constants (matching eval_seed_videommev2.py) ─────────────────

_LEGACY_PREFIX = 'These are the frames of a video.'

_THINK_PROMPT = (
    'Please perform a detailed reasoning based on the provided video frames '
    'to answer the following multiple-choice question selecting the best option '
    "from A through H and providing your final response strictly in the format: "
    "'Final Answer: <letter>."
)

_INSTRUCT_PROMPT = (
    'Select the best answer to the following multiple-choice question based on '
    'the video. Respond with only the letter (A, B, C, D, E, F, G, or H) of the '
    'correct option.'
)


def build_prompt(
    question: str,
    options: list[str],
    subtitle: str = "",
    with_videcut: bool = False,
    input_type: str = "frames",  # "frames" or "video"
    video_duration: float | None = None,  # Total video duration in seconds
    current_fps: float | None = None,  # Current sampling FPS (for frames mode)
    num_frames: int | None = None,  # Number of frames provided
    reasoning: bool = True,  # Whether to request detailed reasoning
    prompt_style: str = "legacy",  # "legacy" (flat, matching videommev2) or "structured" (## headers)
) -> str:
    """
    Build the text prompt for VLM evaluation.

    Args:
        question: The question to answer
        options: List of answer options
        subtitle: Optional video subtitle/dialogue
        with_videcut: Whether to include VideoCut tool instructions
        input_type: "frames" for frame-based input, "video" for video input
        video_duration: Total video duration in seconds
        current_fps: Current sampling FPS (helps model decide if VideoCut needed)
        num_frames: Number of frames provided
        reasoning: If True, use THINK_PROMPT for detailed reasoning;
                   if False, use INSTRUCT_PROMPT for letter-only response
        prompt_style: "legacy" = flat text style (matching eval_seed_videommev2.py,
                      default); "structured" = markdown ## headers style (original
                      eval_seed.py style)
    """
    # Remove existing prefix if present (e.g., "A. Apples." -> "Apples.", "E. E. xxx" -> "xxx")
    cleaned_options = []
    for opt in options:
        opt = opt.strip()
        # Strip letter prefix "A. "~"H. " or numeric prefix "1. "~"9. "
        opt = re.sub(r'^(?:[A-Z]|\d+)\.\s*', '', opt)
        cleaned_options.append(opt)

    options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(cleaned_options)])

    if prompt_style == "legacy":
        return _build_prompt_legacy(
            question, options_str, subtitle, with_videcut, reasoning)
    else:
        return _build_prompt_structured(
            question, options_str, subtitle, with_videcut, reasoning,
            input_type, video_duration, current_fps, num_frames)


def _build_prompt_legacy(
    question: str,
    options_str: str,
    subtitle: str = "",
    with_videcut: bool = False,
    reasoning: bool = True,
) -> str:
    """Build prompt in legacy flat style (matching eval_seed_videommev2.py)."""
    if subtitle:
        prefix = _LEGACY_PREFIX + '\n\n## Video Subtitle/Dialogue\n' + subtitle
    else:
        prefix = _LEGACY_PREFIX

    response_prompt = _THINK_PROMPT if reasoning else _INSTRUCT_PROMPT

    prompt = prefix + '\n'

    if with_videcut:
        prompt += '\n' + VIDECUT_TOOL_PROMPT + '\n\n'

    prompt += f'Question: {question}\n{options_str}\n{response_prompt}'

    return prompt


def _build_prompt_structured(
    question: str,
    options_str: str,
    subtitle: str = "",
    with_videcut: bool = False,
    reasoning: bool = True,
    input_type: str = "frames",
    video_duration: float | None = None,
    current_fps: float | None = None,
    num_frames: int | None = None,
) -> str:
    """Build prompt in structured markdown style (original eval_seed.py style)."""
    # Build sampling info
    sampling_info = ""
    if input_type == "frames" and (video_duration or current_fps or num_frames):
        parts = []
        if video_duration:
            parts.append(f"video duration: {video_duration:.0f}s")
        if current_fps:
            interval = round(1 / current_fps)
            parts.append(f"1 frame every ~{interval}s")
        if num_frames:
            parts.append(f"{num_frames} frames total")
        sampling_info = f" ({', '.join(parts)})"

    prompt = (
        "## System Role\n"
        "You are an expert video analyst with strong visual reasoning capabilities. "
        "Your task is to analyze video content and answer multiple-choice questions accurately.\n\n"
        "## Task\n"
        f"Based on the video content{sampling_info}, answer the following multiple-choice question.\n\n"
    )

    if subtitle:
        prompt += (
            "## Video Subtitle/Dialogue\n"
            f"{subtitle}\n\n"
        )

    if with_videcut:
        prompt += VIDECUT_TOOL_PROMPT + "\n\n"

    prompt += (
        "## Question\n"
        f"{question}\n\n"
        "## Options\n"
        f"{options_str}\n\n"
        "## Response Format\n"
    )

    if with_videcut:
        prompt += (
            "Use <think></think> tags to show your step-by-step reasoning and analysis, "
            "considering video evidence and any available text/audio information. "
            "If current frames are insufficient, call the VideoCut tool to get more frames, "
            "then continue your analysis.\n\n"
            "After your analysis, provide your final answer as: Final Answer: X"
        )
    elif reasoning:
        prompt += (
            "Use <think></think> tags to show your step-by-step reasoning and analysis, "
            "considering video evidence and any available text/audio information. "
            "Then provide your final answer.\n\n"
            "Final Answer: X"
        )
    else:
        prompt += (
            "Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.\n\n"
            "Final Answer: X"
        )

    return prompt


# Legacy wrapper for backward compatibility
def build_prompt_video(question: str, options: list[str], subtitle: str = "", with_videcut: bool = False) -> str:
    """Legacy wrapper: Build the text prompt for video input."""
    return build_prompt(question, options, subtitle, with_videcut, input_type="video")


def extract_answer(text: str) -> str | None:
    """Extract the single-letter answer (A-H) from model output."""
    # Optional English or localized option prefix, with optional Markdown bold
    _opt = r"(?:[Oo]ption|\u9009\u9879)?\s*"
    _letter = r"[A-Ha-h]"
    patterns = [
        rf"\*?\*?Final\s*Answer\*?\*?\s*[:\uff1a]\s*\*?\*?{_opt}({_letter})\*?\*?",
        rf"\*?\*?Answer\*?\*?\s*[:\uff1a]\s*\*?\*?{_opt}({_letter})\*?\*?",
        rf"\u7b54\u6848\s*[:\uff1a]\s*{_opt}({_letter})",
        rf"[Ff]inal [Aa]nswer\s*[:\uff1a]\s*{_opt}({_letter})",
        rf"\u9009\u62e9\s*[:\uff1a]?\s*{_opt}({_letter})",
        rf"\*\*({_letter})[\.\s]",
        rf"(?:^|\n)\s*\*?\*?({_letter})[\.\s]\s*[A-Z]",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()
    return None


# ── VLM API Calls ─────────────────────────────────────────────────────────────

def _make_api_request(
    api_url: str,
    api_key: str,
    payload: dict,
    timeout: float,
    max_retries: int = 5,
    verbose: bool = False,
) -> tuple[dict | None, str | None]:
    """
    Make API request with retry logic.

    Returns:
        (data, error): data is the JSON response, error is None on success.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    print(f"[INFO] Request payload size: {payload_bytes / 1024 / 1024:.2f} MB")

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                if verbose:
                    print(f"[API Call] Attempt {attempt + 1}/{max_retries}")
                resp = client.post(api_url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json(), None

        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"[WARN] Timeout. Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
                continue
            return None, "timeout"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = 2 ** attempt
                if attempt < max_retries - 1:
                    print(f"[WARN] Rate limited (429). Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            return None, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return None, str(e)[:200]

    return None, f"Failed after {max_retries} retries (HTTP 429)"


def _build_content_from_frames(
    frames_b64: list[tuple[float, str]],
    prompt_text: str,
) -> list[dict]:
    """Build message content from frames (image_url format)."""
    content = [{"type": "text", "text": prompt_text}]
    for _, img_b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })
    return content


def _build_content_from_video(
    video_path: str,
    prompt_text: str,
) -> list[dict]:
    """Build message content from video (video_url format)."""
    video_b64 = video_to_base64(video_path)
    video_format = get_video_format(video_path)
    return [
        {
            "type": "video_url",
            "video_url": {"url": f"data:video/{video_format};base64,{video_b64}"},
        },
        {"type": "text", "text": prompt_text},
    ]


def call_vlm(
    prompt_text: str,
    api_url: str,
    api_key: str,
    model: str,
    # Input: either frames or video (one must be provided)
    frames_b64: list[tuple[float, str]] | None = None,
    video_path: str | None = None,
    # Model parameters
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: float = 180.0,
    # Tool options
    use_videcut_tool: bool = False,
    fc_loop: bool = False,
    max_videcut_calls: int = 3,
) -> dict:
    """
    Unified VLM API call function supporting both frames and video input,
    with optional VideoCut function-call loop.

    Args:
        prompt_text: The prompt/question to send to the model
        api_url: API endpoint URL
        api_key: API key for authentication
        model: Model name/ID

        frames_b64: List of (timestamp, base64_image) tuples. Mutually exclusive with video_path.
        video_path: Path to video file. Mutually exclusive with frames_b64.

        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        timeout: Request timeout in seconds

        use_videcut_tool: If True, include VideoCut tool in request (single-call mode)
        fc_loop: If True, enable multi-turn function-call loop for VideoCut
        max_videcut_calls: Maximum number of VideoCut tool calls in fc_loop mode

    Returns:
        dict with keys: content, input_tokens, output_tokens, error, videcut_calls (if fc_loop)

    Examples:
        # Frames input, no tool
        call_vlm(prompt, url, key, model, frames_b64=frames)

        # Frames input, single-call tool mode
        call_vlm(prompt, url, key, model, frames_b64=frames, use_videcut_tool=True)

        # Frames input, function-call loop
        call_vlm(prompt, url, key, model, frames_b64=frames, video_path=video, fc_loop=True)

        # Video input, no tool
        call_vlm(prompt, url, key, model, video_path=video)

        # Video input, function-call loop
        call_vlm(prompt, url, key, model, video_path=video, fc_loop=True)
    """
    # Validate inputs
    if frames_b64 is None and video_path is None:
        raise ValueError("Either frames_b64 or video_path must be provided")

    if fc_loop and video_path is None:
        raise ValueError("video_path is required for fc_loop mode (needed for VideoCut)")

    # Build initial content
    if frames_b64 is not None:
        content = _build_content_from_frames(frames_b64, prompt_text)
    else:
        content = _build_content_from_video(video_path, prompt_text)

    # Initialize result
    result = {
        "content": "",           # Final combined content for extract_answer
        "rounds": [],            # Round-by-round details
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None,
    }
    if fc_loop:
        result["tool_calls"] = []

    # ─── Single-call mode (no loop) ───────────────────────────────────────────
    if not fc_loop:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            # "thinking": {"type": "auto"},
        }

        if use_videcut_tool:
            payload["tools"] = [VIDECUT_TOOL_DEFINITION]
        data, error = _make_api_request(api_url, api_key, payload, timeout)

        if error:
            result["error"] = error
            return result

        if "usage" in data:
            usage = data["usage"]
            result["input_tokens"] = usage.get("prompt_tokens", 0)
            result["output_tokens"] = usage.get("completion_tokens", 0)

        if "choices" in data and len(data["choices"]) > 0:
            msg = data["choices"][0]["message"]
            round_info = {"round": 1}
            if msg.get("reasoning_content"):
                round_info["reasoning"] = msg["reasoning_content"].strip()
            if msg.get("content"):
                round_info["answer"] = msg["content"].strip()
            result["rounds"].append(round_info)
            # content for extract_answer
            result["content"] = round_info.get("answer", "")
        else:
            result["error"] = "No choices in response"

        return result

    # ─── Function-call loop mode ──────────────────────────────────────────────
    messages = [{"role": "user", "content": content}]
    videcut_count = 0
    round_num = 0

    while videcut_count < max_videcut_calls:
        round_num += 1
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # "thinking": {"type": "auto"},
            "tools": [VIDECUT_TOOL_DEFINITION],
        }

        print(f"[API Call] Round {round_num}, messages count: {len(messages)}")
        data, error = _make_api_request(api_url, api_key, payload, timeout)

        if error:
            result["error"] = error
            break

        if "usage" in data:
            usage = data["usage"]
            result["input_tokens"] += usage.get("prompt_tokens", 0)
            result["output_tokens"] += usage.get("completion_tokens", 0)

        if "choices" not in data or len(data["choices"]) == 0:
            result["error"] = "No choices in response"
            print("[ERROR] No choices in response")
            break

        choice = data["choices"][0]
        resp_msg = choice["message"]
        finish_reason = choice.get("finish_reason", "")
        print(f"[API Response] finish_reason: {finish_reason}, has_content: {bool(resp_msg.get('content'))}, has_reasoning_content: {bool(resp_msg.get('reasoning_content'))}, has_tool_calls: {bool(resp_msg.get('tool_calls'))}")

        # Build round info
        round_info = {"round": round_num}
        if resp_msg.get("reasoning_content"):
            round_info["reasoning"] = resp_msg["reasoning_content"].strip()
        if resp_msg.get("content"):
            round_info["answer"] = resp_msg["content"].strip()

        # Check if model wants to call tools
        if finish_reason != "tool_calls" or not resp_msg.get("tool_calls"):
            print("[Done] No more tool calls, exiting loop")
            result["rounds"].append(round_info)
            break

        # Process tool calls
        tool_calls_info = []
        messages.append(resp_msg)

        for tool_call in resp_msg.get("tool_calls", []):
            tool_name = tool_call["function"]["name"]
            tool_call_id = tool_call["id"]

            if tool_name == "VideoCut":
                videcut_count += 1
                args = json.loads(tool_call["function"]["arguments"])

                try:
                    start_sec, end_sec = parse_timestamps(args["timestamps"])
                except ValueError:
                    messages.append({
                        "role": "tool",
                        "content": f"Invalid timestamps format: {args['timestamps']}. Use format 'start - end' (e.g., '10 - 20').",
                        "tool_call_id": tool_call_id,
                    })
                    continue

                fps = int(args.get("fps", 3))
                print(f"[VideoCut #{videcut_count}] Extracting frames: {start_sec}-{end_sec}s @ {fps}fps")

                try:
                    extra_frames, _ = extract_frames_in_range(video_path, start_sec, end_sec, fps)
                    tool_call_info = {
                        "name": "VideoCut",
                        "params": {"start": start_sec, "end": end_sec, "fps": fps},
                        "frames_extracted": len(extra_frames),
                    }
                    tool_calls_info.append(tool_call_info)
                    result["tool_calls"].append(tool_call_info)
                    print(f"[VideoCut #{videcut_count}] Extracted {len(extra_frames)} frames")

                    tool_content = [
                        {"type": "text", "text": f"[VideoCut Result] Extracted {len(extra_frames)} frames from {start_sec}-{end_sec}s. Use <think></think> tags to analyze these frames. If still insufficient, you can call VideoCut again with a different time range or higher fps. Otherwise, provide your final answer."}
                    ]
                    for _, img_b64 in extra_frames:
                        tool_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        })

                    messages.append({
                        "role": "tool",
                        "content": tool_content,
                        "tool_call_id": tool_call_id,
                    })

                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "content": f"Error extracting frames: {e}",
                        "tool_call_id": tool_call_id,
                    })
            else:
                messages.append({
                    "role": "tool",
                    "content": f"Unknown tool: {tool_name}",
                    "tool_call_id": tool_call_id,
                })

        if tool_calls_info:
            round_info["tool_calls"] = tool_calls_info
        result["rounds"].append(round_info)

    # Build content for extract_answer (use last round's answer)
    for r in reversed(result["rounds"]):
        if r.get("answer"):
            result["content"] = r["answer"]
            break

    return result


def _resolve_quota_url(cli_url: str | None, config: dict | None = None) -> str:
    """Resolve CLI > YAML quota_url > VLM_QUOTA_URL; an empty value disables queries."""
    config = config or {}
    value = cli_url if cli_url is not None else config.get(
        "quota_url", os.environ.get("VLM_QUOTA_URL", "")
    )
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("quota_url must be an HTTP(S) URL or an empty string")
    value = value.strip()
    if value:
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("quota_url must be an absolute HTTP(S) URL")
    return value


def _query_remain_quota(api_key: str, quota_url: str = "") -> float | None:
    """Query an explicitly configured endpoint; empty URLs make no HTTP request.

    The evaluator's API key is used as a Bearer token when present. Accept either
    {"remain_quota": number} or {"data": {"remain_quota": number}}. Unavailable
    or invalid responses return None rather than reporting a zero balance.
    """
    if not quota_url:
        return None
    try:
        resp = httpx.get(
            quota_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        value = data.get("remain_quota")
        if value is None and isinstance(data.get("data"), dict):
            value = data["data"].get("remain_quota")
        if value is None or isinstance(value, bool):
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def _start_quota_tracking(args, config: dict | None = None) -> float | None:
    """Resolve optional quota reporting after configuration has been loaded."""
    args.quota_url = _resolve_quota_url(getattr(args, "quota_url", None), config)
    api_key = args.api_key or os.environ.get("VLM_API_KEY", "")
    args.quota_before = _query_remain_quota(api_key, args.quota_url)
    args.quota_after = None
    if args.quota_before is not None:
        print(f"[QUOTA] before evaluation: {args.quota_before:.4f}")
    return args.quota_before


# ── Per-item Evaluation ───────────────────────────────────────────────────────

def evaluate_item(
    item: dict,
    api_url: str,
    api_key: str,
    model: str,
    input_mode: str,
    fps: float,
    max_frames: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    item_index: int,
    total_items: int,
    with_subtitle: bool = False,
    subtitle_dir: str = "",
    with_videcut: bool = False,
    max_long_edge: int = 480,
    max_payload_mb: float = 50.0,
    sampling_strategy: str = "adaptive",
    reasoning: bool = True,
    prompt_style: str = "legacy",
) -> dict:
    """Evaluate a single VideoMME item. Returns result dict."""
    qid = item["question_id"]
    video_path = item["video_path"]
    gt_answer = item["answer"]

    print(f"\n{'='*70}")
    print(f"[{item_index}/{total_items}] QID: {qid}")
    print(f"Video: {video_path}")
    print(f"GT Answer: {gt_answer}")
    print(f"Input Mode: {input_mode}")
    if with_subtitle:
        print(f"With Subtitle: Yes")
    if with_videcut:
        print(f"With VideoCut: Yes")
    print(f"{'='*70}")

    result = {
        "question_id": qid,
        "video_id": item.get("video_id", ""),
        "videoID": item.get("videoID", ""),
        "duration": item.get("duration", ""),
        "domain": item.get("domain", ""),
        "sub_category": item.get("sub_category", ""),
        "task_type": item.get("task_type", ""),
        "question": item["question"],
        "options": item["options"],
        "gt_answer": gt_answer,
        "model_answer": None,
        "is_correct": False,
        "model_output": "",
        "input_mode": input_mode,
        "num_frames": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None,
        "with_subtitle": with_subtitle,
        "with_videcut": with_videcut,
    }

    if not os.path.isfile(video_path):
        result["error"] = f"Video not found: {video_path}"
        print(f"[ERROR] {result['error']}")
        return result

    # Load subtitle if requested
    subtitle_text = ""
    if with_subtitle:
        # Try to find subtitle file
        if subtitle_dir and os.path.isdir(subtitle_dir):
            # Look for .srt file with same name in subtitle_dir
            video_stem = Path(video_path).stem
            subtitle_path = os.path.join(subtitle_dir, f"{video_stem}.srt")
        else:
            # Try to find .srt in same directory as video
            subtitle_path = str(Path(video_path).with_suffix(".srt"))

        if os.path.isfile(subtitle_path):
            subtitle_text = read_srt_subtitle(subtitle_path)
            print(f"[OK] Loaded subtitle: {subtitle_path}")
            print(f"[INFO] Subtitle length: {len(subtitle_text)} chars")
        else:
            print(f"[WARN] Subtitle not found: {subtitle_path}")

    video_path_for_api = video_path  # default, overridden in video mode

    if input_mode == "frames":
        # Frame extraction mode
        print(f"[INFO] Extracting frames (fps={fps}, strategy={sampling_strategy})...")
        try:
            frames_b64, actual_fps, video_duration = extract_frames(video_path, fps=fps, max_long_edge=max_long_edge, max_frames=max_frames, sampling_strategy=sampling_strategy)
            result["num_frames"] = len(frames_b64)
            result["actual_fps"] = actual_fps
            result["video_duration"] = video_duration
            print(f"[OK] Extracted {result['num_frames']} frames (actual fps={actual_fps:.2f})")
        except Exception as e:
            result["error"] = f"Frame extraction failed: {e}"
            print(f"[ERROR] {result['error']}")
            return result

        if not frames_b64:
            result["error"] = "No frames extracted"
            print(f"[ERROR] {result['error']}")
            return result

        prompt_text = build_prompt(
            item["question"],
            item["options"],
            subtitle_text,
            with_videcut=with_videcut,
            input_type="frames",
            video_duration=video_duration,
            current_fps=actual_fps,
            num_frames=len(frames_b64),
            reasoning=reasoning,
            prompt_style=prompt_style,
        )
        mode_info = f"frames (requested fps={fps}, actual fps={actual_fps:.2f}), {len(frames_b64)} frames"
        call_info = f"{len(frames_b64)} frames"

    else:
        # Video mode: re-encode with target fps and resolution
        try:
            reencoded_path, original_size_mb, new_size_mb = reencode_video(
                video_path, fps=fps, max_long_edge=max_long_edge, max_payload_mb=max_payload_mb,
            )
        except Exception as e:
            result["error"] = f"Video re-encode failed: {e}"
            print(f"[ERROR] {result['error']}")
            return result

        prompt_text = build_prompt(
            item["question"],
            item["options"],
            subtitle_text,
            with_videcut=with_videcut,
            input_type="video",
            reasoning=reasoning,
            prompt_style=prompt_style,
        )
        # Use re-encoded video for API call, keep original path for VideoCut
        video_path_for_api = reencoded_path
        result["num_frames"] = -1  # Mark as video mode
        mode_info = f"video (original: {original_size_mb:.2f} MB → re-encoded: {new_size_mb:.2f} MB, fps={fps}, max_edge={max_long_edge})"
        call_info = f"video (original: {original_size_mb:.2f} MB → re-encoded: {new_size_mb:.2f} MB)"

    # Log prompt (compact)
    sub_info = f", subtitle={len(subtitle_text)}chars" if subtitle_text else ""
    videcut_info = ", videcut=on" if with_videcut else ""
    print(f"\n[PROMPT] mode={mode_info}{sub_info}{videcut_info} | max_tokens={max_tokens}, temp={temperature}, timeout={timeout}s")
    prompt_preview = prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text
    print(f"[PROMPT TEXT] {prompt_preview}")
    print(f"[INFO] Calling API with {call_info}...")

    frames_arg = frames_b64 if input_mode == "frames" else None
    # For video mode: use re-encoded video for API content, original video for VideoCut
    api_video_path = video_path_for_api if input_mode == "video" else video_path
    if with_videcut:
        api_result = call_vlm(
            prompt_text, api_url, api_key, model,
            frames_b64=frames_arg, video_path=api_video_path,
            max_tokens=max_tokens, temperature=temperature, timeout=timeout,
            fc_loop=True,
        )
    else:
        api_result = call_vlm(
            prompt_text, api_url, api_key, model,
            frames_b64=frames_arg, video_path=api_video_path,
            max_tokens=max_tokens, temperature=temperature, timeout=timeout,
        )

    # Clean up re-encoded temp file
    if input_mode == "video":
        try:
            os.unlink(video_path_for_api)
        except OSError:
            pass

    # Store structured output
    result["prompt"] = prompt_text
    result["model_output"] = {
        "rounds": api_result.get("rounds", []),
        "tool_calls": api_result.get("tool_calls", []),
    }
    result["input_tokens"] = api_result["input_tokens"]
    result["output_tokens"] = api_result["output_tokens"]
    result["error"] = api_result["error"]

    if result["error"]:
        print(f"[ERROR] API call failed: {result['error']}")
        return result

    # Print structured output
    print(f"\n[MODEL OUTPUT]")
    print("-" * 50)
    for r in api_result.get("rounds", []):
        print(f"[Round {r['round']}]")
        if r.get("reasoning"):
            reasoning_preview = r["reasoning"][:500] + "..." if len(r.get("reasoning", "")) > 500 else r.get("reasoning", "")
            print(f"  Reasoning: {reasoning_preview}")
        if r.get("tool_calls"):
            for tc in r["tool_calls"]:
                print(f"  Tool Call: {tc['name']} -> {tc['params']}, extracted {tc['frames_extracted']} frames")
        if r.get("answer"):
            print(f"  Answer: {r['answer']}")
    print("-" * 50)

    # Extract answer from content
    if api_result["content"]:
        model_answer = extract_answer(api_result["content"])
        result["model_answer"] = model_answer
        if model_answer:
            result["is_correct"] = model_answer == gt_answer.upper()

    print(f"\n[RESULT]")
    print(f"  Model Answer: {result['model_answer']}")
    print(f"  GT Answer:    {gt_answer}")
    print(f"  Match:        {'✓ YES' if result['is_correct'] else '✗ NO'}")
    print(f"  Tokens:       {result['input_tokens']} in / {result['output_tokens']} out")

    return result


# ── Main Loop ─────────────────────────────────────────────────────────────────

def load_completed_ids_from_output(output_dir: str, model: str) -> set[str]:
    """
    Scan output_dir/model/ for existing result JSON files and collect
    question_ids that have been successfully completed.

    A result is considered completed when error is None.
    """
    completed = set()
    model_dir = Path(output_dir) / model
    if not model_dir.is_dir():
        return completed

    for fpath in model_dir.glob("*.json"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for rec in data.get("results", []):
                if rec.get("error") is None:
                    completed.add(rec["question_id"])
        except (json.JSONDecodeError, KeyError):
            continue

    return completed


def _save_output_file(
    output_file: Path,
    results: list[dict],
    summary: dict | None = None,
) -> None:
    """Save results and optional summary to a single JSON file."""
    output = {"results": results}
    if summary:
        output["summary"] = summary
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def run_evaluation(args, output_file: Path | None = None, shared_results: list[dict] | None = None):
    """Run evaluation on a single data file.

    Args:
        args: Parsed CLI arguments.
        output_file: If provided, use this path for output instead of auto-generating.
        shared_results: If provided, append results to this list (for multi-file mode).

    Returns:
        (summary_dict, results_list) tuple.
    """
    api_url = normalize_api_url(args.api_url)
    api_key = args.api_key or os.environ.get("VLM_API_KEY", "")

    sample_rate = getattr(args, "sample_rate", 1) or 1
    workers = getattr(args, "workers", 1) or 1
    skip_completed = getattr(args, "skip_completed", False)
    qids_file = getattr(args, "qids_file", "") or ""

    # Load qids filter if provided
    qids_filter: set[str] | None = None
    if qids_file:
        qids_filter_path = Path(qids_file)
        if qids_filter_path.exists():
            with open(qids_filter_path, "r", encoding="utf-8") as f:
                qids_filter = set(line.strip() for line in f if line.strip())
            print(f"[INFO] Loaded {len(qids_filter)} question IDs from {qids_file}")
        else:
            print(f"[WARN] QIDs file not found: {qids_file}, ignoring filter")

    # Load data
    # When qids_filter is provided, load all data first then filter (ignore limit/sample_rate for loading)
    items = []
    if qids_filter:
        # Load all data, filter by qids_filter
        with open(args.data, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if item.get("question_id") in qids_filter:
                        items.append(item)
        print(f"[INFO] Loaded {len(items)} items matching qids_filter from {args.data}")
    else:
        # Original logic: scan up to sample_rate * limit lines, take every sample_rate-th line
        total_line_nums = sample_rate * args.limit if args.limit else 0
        with open(args.data, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if total_line_nums and i >= total_line_nums:
                    break
                if i % sample_rate == 0:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        if args.limit:
            items = items[: args.limit]

    print(f"{'='*70}")
    print(f"Doubao Seed Evaluation")
    print(f"{'='*70}")
    print(f"Data: {args.data}")
    if qids_filter:
        print(f"QIDs Filter: {qids_file} ({len(qids_filter)} IDs)")
    print(f"Items: {len(items)}")
    if sample_rate > 1:
        print(f"Sample Rate: every {sample_rate}th item")
    if workers > 1:
        print(f"Workers: {workers}")
    print(f"API URL: {api_url}")
    print(f"Model: {args.model}")
    print(f"Input Mode: {args.input_mode}")
    if args.input_mode == "frames":
        print(f"FPS: {args.fps}")
        print(f"Sampling Strategy: {args.sampling_strategy}")
    if args.with_subtitle:
        subtitle_info = f"Subtitle Dir: {args.subtitle_dir}" if args.subtitle_dir else "Subtitle Dir: auto-detect"
        print(f"With Subtitle: Yes ({subtitle_info})")
    if args.with_videcut:
        print(f"With VideoCut: Yes")
    print()

    # Output file - with configuration tags
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_stem = Path(args.data).stem
        out_dir = Path(args.output_dir) / args.model
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build config string for filename
        config_parts = [args.input_mode]
        if args.input_mode == "frames":
            config_parts.append(args.sampling_strategy)
        if args.with_subtitle:
            config_parts.append("sub")
        if args.with_videcut:
            config_parts.append("videcut")

        config_str = "_".join(config_parts)
        output_file = out_dir / f"{data_stem}_{args.model}_{config_str}_{timestamp}.json"

    # Load completed question_ids and results for resume
    completed_ids: set[str] = set()
    all_results: list[dict] = shared_results if shared_results is not None else []
    # Track initial count for multi-file mode (to calculate per-file stats correctly)
    initial_results_count = len(all_results)

    # Extract completed_ids from shared_results (for multi-file resume)
    if shared_results:
        for rec in shared_results:
            completed_ids.add(rec["question_id"])
        if completed_ids:
            print(f"[INFO] {len(completed_ids)} items already completed from previous results")

    # Auto-skip completed items by scanning existing output files
    if skip_completed:
        loaded_ids = load_completed_ids_from_output(args.output_dir, args.model)
        if loaded_ids:
            completed_ids.update(loaded_ids)
            print(f"Skip-completed: found {len(loaded_ids)} completed items in {args.output_dir}/{args.model}/")

    if args.resume_file:
        resume_path = Path(args.resume_file)
        if resume_path.exists():
            with open(resume_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_records = data.get("results", [])
                old_summary = data.get("summary", None)  # Preserve the existing summary
                total_count = len(all_records)
                success_count = 0
                failed_count = 0
                skipped_dup = 0
                skipped_not_in_data = 0
                # Build set of valid question_ids from current items for filtering
                item_qids = {it["question_id"] for it in items} if shared_results is None else None
                seen_qids = set(completed_ids)  # already-seen qids (from shared_results or skip_completed)
                for rec in all_records:
                    has_error = rec.get("error") is not None
                    has_answer = rec.get("model_answer") is not None
                    qid = rec.get("question_id", "")
                    # Keep results that have no error AND have model_answer
                    if not has_error and has_answer:
                        # Dedup: skip if already in all_results
                        if qid in seen_qids:
                            skipped_dup += 1
                            continue
                        # Filter: skip if not in current data (single-file mode only)
                        if item_qids is not None and qid not in item_qids:
                            skipped_not_in_data += 1
                            continue
                        seen_qids.add(qid)
                        completed_ids.add(qid)
                        all_results.append(rec)
                        success_count += 1
                    else:
                        # Debug: print why this record is considered failed
                        print(f"  [DEBUG] Removing QID={qid}: error={has_error}, answer={has_answer}")
                        failed_count += 1
            output_file = resume_path
            print(f"Resuming from: {resume_path}")
            print(f"  - Total {total_count} cases")
            print(f"  - Keeping {success_count} successful (no error + has answer)")
            print(f"  - Removing {failed_count} failed/incomplete")
            if skipped_dup > 0:
                print(f"  - Skipped {skipped_dup} duplicate question_ids")
            if skipped_not_in_data > 0:
                print(f"  - Skipped {skipped_not_in_data} results not in current data")
            print(f"  - all_results now has {len(all_results)} items")

            # Immediately save cleaned results with recalculated summary
            if failed_count > 0 or skipped_dup > 0 or skipped_not_in_data > 0:
                new_summary = _recalculate_summary(all_results, old_summary)
                _save_output_file(output_file, all_results, new_summary)
                print(f"  - Cleaned file saved to: {output_file}")
                metrics = new_summary.get("metrics", new_summary)
                print(f"  - New accuracy: {metrics['accuracy']*100:.2f}% ({metrics['correct']}/{metrics['valid']})")
        else:
            print(f"Resume file not found: {resume_path}, starting fresh → {output_file}")
    else:
        print(f"Output: {output_file}")

    # Filter out completed items
    pending = [it for it in items if it["question_id"] not in completed_ids]
    print(f"Pending: {len(pending)} items\n")

    if not pending:
        print("All items already completed!")
        return _build_summary(args, 0, 0, 0, 0, 0, 0.0), all_results

    # Stats
    total = len(pending)
    done = 0
    correct = 0
    errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    # Common kwargs for evaluate_item
    eval_kwargs = dict(
        api_url=api_url, api_key=api_key, model=args.model,
        input_mode=args.input_mode,
        fps=args.fps,
        max_frames=getattr(args, "max_frames", 1024),
        sampling_strategy=getattr(args, "sampling_strategy", "adaptive"),
        max_tokens=args.max_tokens, temperature=args.temperature,
        timeout=args.timeout,
        with_subtitle=args.with_subtitle,
        subtitle_dir=args.subtitle_dir,
        with_videcut=args.with_videcut,
        max_long_edge=getattr(args, "max_long_edge", 480),
        max_payload_mb=getattr(args, "max_payload_mb", 50.0),
        reasoning=getattr(args, "reasoning", True),
        prompt_style=getattr(args, "prompt_style", "legacy"),
    )

    def _process_result(result):
        nonlocal done, correct, errors, total_input_tokens, total_output_tokens
        with _result_lock:
            done += 1
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            if result["error"]:
                errors += 1
            elif result["is_correct"]:
                correct += 1
            all_results.append(result)

            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            valid = done - errors
            acc = correct / valid * 100 if valid > 0 else 0
            print(f"\n[SUMMARY] Acc: {acc:.1f}% ({correct}/{valid}) | Errors: {errors} | Speed: {rate:.2f} it/s")

            if done % 5 == 0:
                _save_output_file(output_file, all_results, None)

            if done % 20 == 0:
                qb = getattr(args, "quota_before", None)
                if qb is not None:
                    api_key = args.api_key or os.environ.get("VLM_API_KEY", "")
                    qn = _query_remain_quota(api_key, getattr(args, "quota_url", ""))
                    if qn is not None:
                        cost = qb - qn
                        avg = cost / done
                        print(f"[QUOTA] used={cost:.4f}, avg/case={avg:.4f}, remain={qn:.4f}")

    if workers > 1:
        # Parallel processing with ThreadPoolExecutor (I/O-bound API calls)
        # Install thread-local stdout wrapper so each worker's output is
        # buffered and flushed atomically, preventing interleaved lines
        # that can overwhelm / freeze the terminal.
        print(f"[INFO] Using {workers} parallel workers")
        _tl_stdout = _ThreadLocalStdout(sys.stdout)
        sys.stdout = _tl_stdout

        def _worker_fn(item, idx):
            _tl_stdout.start_buffering()
            try:
                return evaluate_item(
                    item, **eval_kwargs,
                    item_index=idx, total_items=total,
                )
            finally:
                _tl_stdout.flush_buffer()

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for idx, item in enumerate(pending, 1):
                    future = executor.submit(_worker_fn, item, idx)
                    futures[future] = idx

                for future in as_completed(futures):
                    result = future.result()
                    _process_result(result)
        finally:
            # Always restore real stdout, even on Ctrl+C / exception
            sys.stdout = _tl_stdout._real
    else:
        # Sequential processing
        for idx, item in enumerate(pending, 1):
            result = evaluate_item(
                item, **eval_kwargs,
                item_index=idx, total_items=total,
            )
            _process_result(result)

    # Final summary - recalculate from results
    elapsed = time.time() - start_time

    # In multi-file mode, only count results from THIS file (not cumulative)
    if shared_results is not None:
        current_file_results = all_results[initial_results_count:]
    else:
        current_file_results = all_results

    final_total = len(current_file_results)
    final_correct = sum(1 for r in current_file_results if r.get("is_correct"))
    final_errors = sum(1 for r in current_file_results if r.get("error") is not None)
    final_input_tokens = sum(r.get("input_tokens", 0) for r in current_file_results)
    final_output_tokens = sum(r.get("output_tokens", 0) for r in current_file_results)

    summary = _build_summary(args, final_total, final_correct, final_errors, final_input_tokens, final_output_tokens, elapsed)

    # Inject quota info into summary
    quota_before = getattr(args, "quota_before", None)
    if quota_before is not None:
        api_key = args.api_key or os.environ.get("VLM_API_KEY", "")
        quota_after = _query_remain_quota(api_key, getattr(args, "quota_url", ""))
        args.quota_after = quota_after
        cost = round(quota_before - quota_after, 4) if quota_after is not None else None
        summary["quota"] = {"before": round(quota_before, 4), "after": round(quota_after, 4) if quota_after is not None else None, "cost": cost}

    _print_summary(args, summary)
    # Also print this-run stats for clarity
    if done < final_total:
        print(f"\n[This Run] Processed {done} items (total: {final_total})")

    # Write output file (skip in multi-file mode, handled by caller)
    if shared_results is None:
        _save_output_file(output_file, all_results, summary)
        print(f"\n[Output File]")
        print(f"  {output_file}")
    else:
        # Multi-file mode: periodic save for crash recovery
        _save_output_file(output_file, all_results, None)

    return summary, all_results


def _build_summary(args, total, correct, errors, total_input_tokens, total_output_tokens, elapsed):
    """Build summary dict from evaluation stats."""
    valid = total - errors
    acc = correct / valid if valid > 0 else 0  # Store as decimal (0.8922), not percentage
    api_url = normalize_api_url(args.api_url)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "timestamp": timestamp,
        "data_file": args.data,
        "model": args.model,
        "input_mode": args.input_mode,
        "metrics": {
            "total": total,
            "correct": correct,
            "valid": valid,
            "accuracy": round(acc, 4),
            "errors": errors,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "elapsed_seconds": round(elapsed, 1),
        },
        "config": {
            "api_url": api_url,
            "features": {
                "subtitle": {
                    "enabled": args.with_subtitle,
                    "directory": args.subtitle_dir if args.subtitle_dir else "auto-detect (same dir as video)",
                },
                "videcut": {
                    "enabled": args.with_videcut,
                },
            },
            "sampling": {
                "fps": args.fps if args.input_mode == "frames" else None,
                "strategy": args.sampling_strategy if args.input_mode == "frames" else None,
                "max_frames": args.max_frames if args.input_mode == "frames" else None,
            },
            "generation": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "timeout": args.timeout,
            },
        },
    }


def _recalculate_summary(results: list[dict], old_summary: dict | None = None) -> dict:
    """Recalculate summary from results list."""
    total = len(results)
    correct = 0
    errors = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for r in results:
        if r.get("error") is not None:
            errors += 1
        elif r.get("is_correct"):
            correct += 1
        total_input_tokens += r.get("input_tokens", 0)
        total_output_tokens += r.get("output_tokens", 0)

    valid = total - errors
    acc = correct / valid if valid > 0 else 0  # Store as decimal (0.8922), not percentage

    # Build new summary, preserving config from old summary
    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "metrics": {
            "total": total,
            "correct": correct,
            "valid": valid,
            "accuracy": round(acc, 4),
            "errors": errors,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
    }

    # Copy other fields from old summary
    if old_summary:
        for key in ["data_file", "model", "input_mode", "config"]:
            if key in old_summary:
                summary[key] = old_summary[key]
        # Preserve elapsed_seconds if exists
        if "metrics" in old_summary and "elapsed_seconds" in old_summary["metrics"]:
            summary["metrics"]["elapsed_seconds"] = old_summary["metrics"]["elapsed_seconds"]

    return summary


def _print_summary(args, summary):
    """Print evaluation summary to console."""
    m = summary["metrics"]
    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE")
    print(f"=" * 70)
    print(f"Data:      {summary['data_file']}")
    print(f"Total:     {m['total']}")
    print(f"Correct:   {m['correct']}/{m['valid']} = {m['accuracy']*100:.2f}%")
    print(f"Errors:    {m['errors']}")
    print(f"Tokens:    {m['total_input_tokens']:,} in / {m['total_output_tokens']:,} out")
    elapsed = m["elapsed_seconds"]
    total = m["total"]
    print(f"Time:      {elapsed:.1f}s ({total/elapsed:.2f} it/s)" if elapsed > 0 else f"Time:      {elapsed:.1f}s")
    print(f"\n[Configuration]")
    print(f"Input Mode:       {args.input_mode}")
    if args.input_mode == "frames":
        print(f"  FPS:            {args.fps}")
        print(f"  Sampling:       {args.sampling_strategy}")
    print(f"Subtitle:         {args.with_subtitle} {f'(dir: {args.subtitle_dir})' if args.subtitle_dir else '(auto-detect)'}")
    print(f"VideoCut:         {args.with_videcut}")


def _build_per_file_summaries(all_results: list[dict], data_files: list[str]) -> list[dict]:
    """Build per-file summaries by matching results to data files via question_id."""
    # Load question_ids for each data file
    file_qids = {}
    for data_file in data_files:
        qids = set()
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        qids.add(item.get("question_id", ""))
        except Exception:
            pass
        file_qids[data_file] = qids

    # Build per-file summaries
    summaries = []
    matched_qids = set()
    for data_file in data_files:
        qids = file_qids[data_file]
        file_results = [r for r in all_results if r.get("question_id") in qids]
        matched_qids.update(r.get("question_id") for r in file_results)

        total = len(file_results)
        correct = sum(1 for r in file_results if r.get("is_correct"))
        errors = sum(1 for r in file_results if r.get("error") is not None)
        valid = total - errors
        acc = correct / valid if valid > 0 else 0
        tokens_in = sum(r.get("input_tokens", 0) for r in file_results)
        tokens_out = sum(r.get("output_tokens", 0) for r in file_results)

        summaries.append({
            "data_file": data_file,
            "metrics": {
                "total": total,
                "correct": correct,
                "valid": valid,
                "accuracy": round(acc, 4),
                "errors": errors,
                "total_input_tokens": tokens_in,
                "total_output_tokens": tokens_out,
            },
        })

    # Handle results not matching any data file
    unmatched = [r for r in all_results if r.get("question_id") not in matched_qids]
    if unmatched:
        total = len(unmatched)
        correct = sum(1 for r in unmatched if r.get("is_correct"))
        errors = sum(1 for r in unmatched if r.get("error") is not None)
        valid = total - errors
        acc = correct / valid if valid > 0 else 0
        summaries.append({
            "data_file": "(unmatched)",
            "metrics": {
                "total": total, "correct": correct, "valid": valid,
                "accuracy": round(acc, 4), "errors": errors,
                "total_input_tokens": sum(r.get("input_tokens", 0) for r in unmatched),
                "total_output_tokens": sum(r.get("output_tokens", 0) for r in unmatched),
            },
        })

    return summaries


def run_multi_file_evaluation(args, data_files: list[str]):
    """Run evaluation across multiple data files, saving all results to a single output file."""
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    all_summaries = []
    completed_ids: set[str] = set()

    # Handle resume: use existing file instead of creating new one
    if args.resume_file:
        resume_path = Path(args.resume_file)
        if resume_path.exists():
            # Build set of valid qids from all data files for filtering
            all_valid_qids = set()
            qids_file = getattr(args, "qids_file", "") or ""
            qids_filter: set[str] | None = None
            if qids_file and Path(qids_file).exists():
                with open(qids_file, "r", encoding="utf-8") as f:
                    qids_filter = set(line.strip() for line in f if line.strip())
            for data_file in data_files:
                try:
                    with open(data_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                item = json.loads(line)
                                qid = item.get("question_id", "")
                                if qids_filter is None or qid in qids_filter:
                                    all_valid_qids.add(qid)
                except Exception:
                    pass

            with open(resume_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_records = data.get("results", [])
                old_summary = data.get("summary", None)
                success_count = 0
                failed_count = 0
                skipped_dup = 0
                skipped_not_in_data = 0
                seen_qids: set[str] = set()
                for rec in all_records:
                    has_error = rec.get("error") is not None
                    has_answer = rec.get("model_answer") is not None
                    qid = rec.get("question_id", "")
                    if not has_error and has_answer:
                        if qid in seen_qids:
                            skipped_dup += 1
                            continue
                        if all_valid_qids and qid not in all_valid_qids:
                            skipped_not_in_data += 1
                            continue
                        seen_qids.add(qid)
                        completed_ids.add(qid)
                        all_results.append(rec)
                        success_count += 1
                    else:
                        failed_count += 1
            output_file = resume_path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # For updated summary
            print(f"[MULTI] Resuming from: {resume_path}")
            print(f"  - Keeping {success_count} successful, removing {failed_count} failed")
            if skipped_dup > 0:
                print(f"  - Skipped {skipped_dup} duplicate question_ids")
            if skipped_not_in_data > 0:
                print(f"  - Skipped {skipped_not_in_data} results not in current data files")

            # Save cleaned results immediately
            if failed_count > 0 or skipped_dup > 0 or skipped_not_in_data > 0:
                new_summary = _recalculate_summary(all_results, old_summary)
                _save_output_file(output_file, all_results, new_summary)
                print(f"  - Cleaned file saved, new accuracy: {new_summary['metrics']['accuracy']*100:.2f}%")
        else:
            print(f"[WARN] Resume file not found: {resume_path}, creating new file")
            args.resume_file = ""  # Clear so run_evaluation doesn't try to resume again
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_stem = Path(args.config).stem if args.config else "multi"
            config_parts = [args.input_mode]
            if args.input_mode == "frames":
                config_parts.append(args.sampling_strategy)
            if args.with_subtitle:
                config_parts.append("sub")
            if args.with_videcut:
                config_parts.append("videcut")
            config_str = "_".join(config_parts)
            output_file = out_dir / f"{config_stem}_{args.model}_{config_str}_{timestamp}.json"
    else:
        # No resume - create new output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_stem = Path(args.config).stem if args.config else "multi"
        config_parts = [args.input_mode]
        if args.input_mode == "frames":
            config_parts.append(args.sampling_strategy)
        if args.with_subtitle:
            config_parts.append("sub")
        if args.with_videcut:
            config_parts.append("videcut")
        config_str = "_".join(config_parts)
        output_file = out_dir / f"{config_stem}_{args.model}_{config_str}_{timestamp}.json"

    # Clear resume_file in args so run_evaluation doesn't process it again
    original_resume_file = args.resume_file
    args.resume_file = ""

    print(f"[CONFIG] Output file: {output_file}")

    for i, data_file in enumerate(data_files, 1):
        print(f"\n{'#'*70}")
        print(f"# File {i}/{len(data_files)}: {data_file}")
        print(f"{'#'*70}\n")

        args.data = data_file
        summary, _ = run_evaluation(args, output_file=output_file, shared_results=all_results)
        if summary:
            all_summaries.append(summary)

    # Build combined summary — recalculate from all_results to include resumed results
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get("is_correct"))
    errors_total = sum(1 for r in all_results if r.get("error") is not None)
    valid = total - errors_total
    tokens_in = sum(r.get("input_tokens", 0) for r in all_results)
    tokens_out = sum(r.get("output_tokens", 0) for r in all_results)
    elapsed = sum(s["metrics"].get("elapsed_seconds", 0) for s in all_summaries)
    acc = correct / valid if valid > 0 else 0  # Store as decimal

    # Per-file summaries: rebuild from all_results to include resumed results
    per_file_summaries = _build_per_file_summaries(all_results, data_files)

    combined_summary = {
        "timestamp": timestamp,
        "model": args.model,
        "input_mode": args.input_mode,
        "data_files": data_files,
        "per_file_summaries": per_file_summaries,
        "metrics": {
            "total": total,
            "correct": correct,
            "valid": valid,
            "accuracy": round(acc, 4),
            "errors": errors_total,
            "total_input_tokens": tokens_in,
            "total_output_tokens": tokens_out,
            "elapsed_seconds": round(elapsed, 1),
        },
    }

    # Write final combined output
    _save_output_file(output_file, all_results, combined_summary)

    print(f"\n{'='*70}")
    print(f"COMBINED RESULTS ({len(data_files)} files)")
    print(f"{'='*70}")
    for s in per_file_summaries:
        m = s["metrics"]
        print(f"  {s['data_file']}: {m['accuracy']*100:.2f}% ({m['correct']}/{m['valid']})")
    print(f"  {'─'*50}")
    print(f"  Overall: {acc:.4f} ({correct}/{valid}) | Errors: {errors_total}")
    print(f"  Tokens:  {tokens_in:,} in / {tokens_out:,} out")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"\n[Output File]")
    print(f"  {output_file}")


def main():
    _save_terminal_settings()

    parser = argparse.ArgumentParser(
        description="VideoMME Benchmark Evaluation — Doubao Seed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file (iterates over data_files, uses max_samples/sample_rate/workers)
  python eval_mllms/eval_seed.py -c exp_configs/videomme_exp.yaml

  # Run with config + export data first
  python eval_mllms/eval_seed.py -c exp_configs/videomme_exp.yaml --export

  # Single data file (backward compatible)
  python eval_mllms/eval_seed.py --data data/videomme/test/short_duration_data.jsonl --limit 3

  # Filter by specific question IDs (ignores limit/sample_rate)
  python eval_mllms/eval_seed.py --data data.jsonl --qids-file data/sampled_qids/sampled_150.txt

  # Parallel workers + skip completed
  python eval_mllms/eval_seed.py --data data.jsonl -w 4 --skip-completed

  # Sample every 10th item
  python eval_mllms/eval_seed.py --data data.jsonl -r 10

  # Disable subtitles
  python eval_mllms/eval_seed.py --limit 3 --no-subtitle

  # Test with VideoCut tool
  python eval_mllms/eval_seed.py --input-mode video --limit 3 --with-videcut

  # Resume from previous run
  python eval_mllms/eval_seed.py --resume-file eval_MLLM/model/xxx.json

  # Optional quota endpoint (otherwise no quota requests are sent)
  python eval_mllms/eval_seed.py --data data.jsonl --quota-url https://quota.example/remaining

  # Disable quota reporting even when configured in YAML or the environment
  python eval_mllms/eval_seed.py --data data.jsonl --quota-url ""
        """,
    )

    # Config
    parser.add_argument("--config", "-c", default="", help="Path to experiment config YAML file")
    parser.add_argument("--export", action="store_true", help="Run export_commands from config before evaluation")

    # Data
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to VideoMME JSONL data file")
    parser.add_argument("--qids-file", default="", help="Path to file containing question IDs to evaluate (one per line, e.g., '602-1')")
    parser.add_argument("--limit", type=int, default=3, help="Limit number of items (default: 3, 0=all)")
    parser.add_argument("--sample-rate", "-r", type=int, default=1, help="Sample rate: evaluate every N-th item (default: 1)")
    parser.add_argument("--workers", "-w", type=int, default=6, help="Number of parallel workers (default: 1)")
    parser.add_argument("--skip-completed", action="store_true", help="Auto-skip completed items by scanning existing output files")

    # Input mode
    parser.add_argument(
        "--input-mode",
        choices=["video", "frames"],
        default="video",
        help="Input mode: 'frames' (extract frames as image_url, default) or 'video' (direct video_url)"
    )

    # API
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="VLM API base URL")
    parser.add_argument("--api-key", default="", help="API key (or set VLM_API_KEY env var)")
    parser.add_argument(
        "--quota-url", default=None,
        help="Optional quota endpoint (CLI > YAML quota_url > VLM_QUOTA_URL). "
             "Disabled when unset; pass an empty string to disable. Uses the evaluator API key.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")

    # Frame extraction (only used when input-mode=frames)
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample")
    parser.add_argument("--max-frames", type=int, default=1024, help="Maximum number of frames to extract (default: 1024)")
    parser.add_argument("--max-long-edge", type=int, default=448, help="Resize frames so long edge <= this value (default: 480)")
    parser.add_argument("--max-payload-mb", type=float, default=30.0, help="Max base64 payload size in MB for re-encoded video (default: 50.0)")
    parser.add_argument("--sampling-strategy", choices=["adaptive", "uniform"], default="uniform",
                        help="Frame sampling strategy: 'adaptive' reduces fps for long videos, 'uniform' uses 1fps by default and only uniformly samples max-frames when 1fps exceeds max-frames (default: uniform)")

    # Generation
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--timeout", type=float, default=300.0, help="API timeout in seconds")

    # Output
    parser.add_argument("--output-dir", default="eval_mllm", help="Output directory")

    # Subtitle (enabled by default)
    parser.add_argument(
        "--no-subtitle",
        action="store_true",
        help="Disable subtitle inclusion (subtitles are included by default)"
    )
    parser.add_argument(
        "--subtitle-dir",
        default="",
        help="Directory containing subtitle files (empty = auto-detect from data path)"
    )

    # VideoCut tool (optional)
    parser.add_argument(
        "--with-videcut",
        action="store_true",
        help="Enable VideoCut tool for high-density frame sampling (OpenAI-style function calling)"
    )

    # Prompt style
    parser.add_argument(
        "--prompt-style",
        choices=["legacy", "structured"],
        default="legacy",
        help="Prompt style: 'legacy' (flat text, matching videommev2, default) "
             "or 'structured' (markdown ## headers, original eval_seed style)"
    )

    # Reasoning (enabled by default)
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable detailed reasoning; model responds with only the answer letter "
             "(default: reasoning enabled)"
    )

    # Resume
    parser.add_argument("--resume-file", default="",
                        help="Resume from this result JSON file (re-run failed cases)")

    args = parser.parse_args()

    # Subtitle is enabled by default, --no-subtitle disables it
    args.with_subtitle = not args.no_subtitle

    # Reasoning is enabled by default, --no-reasoning disables it
    args.reasoning = not args.no_reasoning

    # Load config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {args.config}")
            return
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        print(f"[CONFIG] Loaded: {args.config}")

        # Map config fields to args (config overrides defaults, CLI overrides config)
        if "max_samples" in config:
            args.limit = config["max_samples"]
        if "sample_rate" in config:
            args.sample_rate = config["sample_rate"]
        if "workers" in config:
            args.workers = config["workers"]
        if "skip_completed" in config:
            args.skip_completed = config["skip_completed"]
        if "qids_file" in config and not args.qids_file:
            args.qids_file = config["qids_file"]
        if "with_subtitle" in config:
            args.with_subtitle = config["with_subtitle"]
        if "subtitle_dir" in config:
            args.subtitle_dir = config["subtitle_dir"]
        if "reasoning" in config:
            args.reasoning = config["reasoning"]
        if "prompt_style" in config:
            args.prompt_style = config["prompt_style"]

        try:
            quota_before = _start_quota_tracking(args, config)
        except ValueError as error:
            parser.error(str(error))

        # Auto-detect subtitle_dir from data_files when not explicitly set
        if not args.subtitle_dir:
            data_files_for_detect = config.get("data_files", [])
            detect_path = data_files_for_detect[0] if data_files_for_detect else args.data
            inferred = infer_subtitle_dir(detect_path)
            if inferred is None:
                args.with_subtitle = False
                print(f"[CONFIG] Auto-detected: no subtitles for this dataset")
            elif inferred:
                args.subtitle_dir = inferred
                print(f"[CONFIG] Auto-detected subtitle_dir: {inferred}")

        # Run export commands if --export flag is set
        if args.export and config.get("export_commands"):
            for cmd in config["export_commands"]:
                print(f"[EXPORT] Running: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
            print()

        # Multi-file or single-file evaluation
        data_files = config.get("data_files", [])
        if data_files:
            run_multi_file_evaluation(args, data_files)
        else:
            run_evaluation(args)
    else:
        try:
            quota_before = _start_quota_tracking(args)
        except ValueError as error:
            parser.error(str(error))
        # Auto-detect subtitle_dir from data path when not explicitly set
        if not args.subtitle_dir:
            inferred = infer_subtitle_dir(args.data)
            if inferred is None:
                args.with_subtitle = False
                print(f"[INFO] Auto-detected: no subtitles for this dataset")
            elif inferred:
                args.subtitle_dir = inferred
                print(f"[INFO] Auto-detected subtitle_dir: {inferred}")
        run_evaluation(args)

    # Print quota summary
    quota_after = getattr(args, "quota_after", None)
    if quota_before is not None and quota_after is not None:
        print(f"[QUOTA] after evaluation: {quota_after:.4f}, total cost: {quota_before - quota_after:.4f}")
    elif quota_after is not None:
        print(f"[QUOTA] after evaluation: {quota_after:.4f}")


if __name__ == "__main__":
    main()
