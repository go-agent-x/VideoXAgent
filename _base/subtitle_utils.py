"""
Shared subtitle utilities — unified subtitle parsing, discovery, and formatting.

Extracted from tools/text/subtitle-mcp/server.py (canonical source) to eliminate code duplication
across 5 MCP services. Supports SRT, JSON (LongVideoBench), and JSONL (Video-MME-v2) formats.
"""

import glob
import html
import json
import os
import re
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Dataset subtitle configuration
# ---------------------------------------------------------------------------
DATASET_SUBTITLE_CONFIG = {
    "videomme": {
        "video_path_pattern": r"data/videomme/videos?/",
        "subtitle_dir": "data/videomme/subtitle",
        "subtitle_ext": ".srt",
        "naming_pattern": "{video_id}.srt",
    },
    "longvideobench": {
        "video_path_pattern": r"data/LongVideoBench/videos?/",
        "subtitle_dir": "data/LongVideoBench/subtitles",
        "subtitle_ext": ".json",
        "naming_pattern": "{video_id}.srt", # Preferred naming pattern; fallback matching is available
    },
    "video-mme-v2": {
        "video_path_pattern": r"data/Video-MME-v2/videos?/",
        "subtitle_dir": "data/Video-MME-v2/subtitle",
        "subtitle_ext": ".jsonl",
        "naming_pattern": "{video_id}.jsonl",
    },
}


def register_dataset(name: str, config: dict) -> None:
    """Dynamically register a new dataset subtitle configuration.

    Args:
        name: Dataset name (e.g. "my-dataset")
        config: Dict with keys: video_path_pattern, subtitle_dir, subtitle_ext, naming_pattern
    """
    required_keys = {"video_path_pattern", "subtitle_dir", "subtitle_ext", "naming_pattern"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    DATASET_SUBTITLE_CONFIG[name] = config


def detect_dataset_from_video_path(video_path: str) -> Optional[str]:
    """Detect dataset name based on video path pattern."""
    for dataset_name, config in DATASET_SUBTITLE_CONFIG.items():
        if re.search(config["video_path_pattern"], video_path):
            return dataset_name
    return None


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------
def parse_timestamp(ts) -> float:
    """
    Parse a timestamp to seconds. Merges parse_srt_timestamp + _parse_time.

    Supports:
      - numeric (int/float) — returned directly
      - string HH:MM:SS,mmm / HH:MM:SS.mmm — SRT-style
      - string MM:SS.mmm — shorthand
      - string bare number — float conversion

    Args:
        ts: Timestamp as int, float, or string

    Returns:
        Time in seconds as float, or 0.0 on failure
    """
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        # Replace comma with dot for consistent parsing
        ts = ts.strip().replace(",", ".")
        parts = ts.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        try:
            return float(ts)
        except ValueError:
            return 0.0
    return 0.0


def format_seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def strip_html_tags(text: str) -> str:
    """Remove HTML/XML tags from text."""
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def clean_subtitle_text(text: str) -> str:
    """Clean subtitle text: HTML entity decode, tag strip, whitespace normalize."""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# File parsing (SRT / JSON / JSONL)
# ---------------------------------------------------------------------------
def parse_srt_file(file_path: str) -> list[dict]:
    """
    Parse an SRT subtitle file (UTF-8-sig + latin-1 fallback).

    Returns list of dicts with keys: index, start, end, text.
    """
    subtitles = []

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        timestamp_match = re.match(
            r'(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})',
            lines[1].strip()
        )
        if not timestamp_match:
            continue

        start_time = parse_timestamp(timestamp_match.group(1))
        end_time = parse_timestamp(timestamp_match.group(2))

        text_lines = lines[2:]
        text = ' '.join(strip_html_tags(line) for line in text_lines if line.strip())

        if text:
            subtitles.append({
                'index': index,
                'start': start_time,
                'end': end_time,
                'text': text
            })

    return subtitles


def parse_json_subtitle_file(file_path: str) -> list[dict]:
    """
    Parse a JSON subtitle file (LongVideoBench format).

    Expected format: [{"start": "HH:MM:SS.mmm", "end": "HH:MM:SS.mmm", "line": "text"}, ...]
    """
    subtitles = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to parse JSON subtitle file: {e}")

    if not isinstance(data, list):
        raise ValueError("JSON subtitle file must contain an array")

    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue

        start_str = entry.get('start', '')
        end_str = entry.get('end', '')
        line = entry.get('line', '')

        if not start_str or not end_str:
            continue

        try:
            start_time = parse_timestamp(start_str)
            end_time = parse_timestamp(end_str)
        except (ValueError, AttributeError):
            continue

        text = clean_subtitle_text(line)

        if text:
            subtitles.append({
                'index': idx + 1,
                'start': start_time,
                'end': end_time,
                'text': text
            })

    return subtitles


def parse_jsonl_subtitle_file(file_path: str) -> list[dict]:
    """
    Parse a JSONL subtitle file (Video-MME-v2 format).

    Expected format (one JSON object per line):
    {"text": "Hi,", "start_time": 115.6, "end_time": 116.0}
    """
    subtitles = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = entry.get('text', '').strip()
                start_time = entry.get('start_time', 0)
                end_time = entry.get('end_time', 0)

                if text and isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    subtitles.append({
                        'index': idx + 1,
                        'start': float(start_time),
                        'end': float(end_time),
                        'text': text
                    })
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to read JSONL subtitle file: {e}")

    return subtitles


def parse_subtitle_file(file_path: str) -> list[dict]:
    """
    Parse a subtitle file, auto-detecting format by extension.

    Supports .srt, .json, .jsonl.
    Returns list of dicts with keys: index, start, end, text.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.json':
        return parse_json_subtitle_file(file_path)
    elif ext == '.jsonl':
        return parse_jsonl_subtitle_file(file_path)
    else:
        return parse_srt_file(file_path)


# ---------------------------------------------------------------------------
# Video ID extraction
# ---------------------------------------------------------------------------
def get_video_id_from_path(video_path: str) -> str:
    """Extract video ID (filename without extension) from a video file path."""
    filename = os.path.basename(video_path)
    video_id = os.path.splitext(filename)[0]
    return video_id


# ---------------------------------------------------------------------------
# File discovery (4-level fallback)
# ---------------------------------------------------------------------------
def find_subtitle_file(video_path: str, subtitle_dir: str = "", workspace_root: str = "") -> Optional[str]:
    """
    Locate the subtitle file for a video using a 4-level fallback strategy.

    1. Explicit subtitle_dir (if provided)
    2. Dataset-specific lookup based on video path pattern
    3. Same directory as video file
    4. Broad recursive search under data/

    Supports SRT, JSON, JSONL formats at each level.

    Args:
        video_path: Path to video file
        subtitle_dir: Explicit subtitle directory (optional)
        workspace_root: Root directory of workspace (optional, defaults to WORKSPACE_ROOT env or project root)

    Returns:
        Path to subtitle file if found, None otherwise
    """
    video_id = get_video_id_from_path(video_path)
    ws = workspace_root or os.environ.get("WORKSPACE_ROOT", _PROJECT_ROOT)

    # 1. Explicit subtitle_dir
    if subtitle_dir:
        for ext in [".srt", ".json", ".jsonl"]:
            for pattern in [f"{video_id}{ext}", f"{video_id}_en{ext}"]:
                candidate = os.path.join(subtitle_dir, pattern)
                if os.path.isfile(candidate):
                    return candidate

    # 2. Dataset-specific lookup based on video path pattern
    dataset = detect_dataset_from_video_path(video_path)
    if dataset and dataset in DATASET_SUBTITLE_CONFIG:
        config = DATASET_SUBTITLE_CONFIG[dataset]
        sub_dir = os.path.join(ws, config["subtitle_dir"])
        # 2a. Try naming_pattern first
        filename = config["naming_pattern"].format(video_id=video_id)
        candidate = os.path.join(sub_dir, filename)
        if os.path.isfile(candidate):
            return candidate
        # 2b. Fallback: multi-format search in the same directory
        for ext in [".srt", ".json", ".jsonl"]:
            for pattern in [f"{video_id}{ext}", f"{video_id}_en{ext}"]:
                candidate = os.path.join(sub_dir, pattern)
                if os.path.isfile(candidate):
                    return candidate

    # 3. Fallback: search all known subtitle directories
    for config in DATASET_SUBTITLE_CONFIG.values():
        for ext in [".srt", ".json", ".jsonl"]:
            for pattern in [f"{video_id}{ext}", f"{video_id}_en{ext}"]:
                candidate = os.path.join(ws, config["subtitle_dir"], pattern)
                if os.path.isfile(candidate):
                    return candidate

    # 4. Same directory as video file
    video_dir = os.path.dirname(os.path.abspath(video_path))
    for ext in [".srt", ".json", ".jsonl"]:
        candidate = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.isfile(candidate):
            return candidate

    # 5. Broad search under data/
    data_dir = os.path.join(ws, "data")
    if os.path.isdir(data_dir):
        for ext in [".srt", ".json", ".jsonl"]:
            matches = glob.glob(os.path.join(data_dir, "**", f"{video_id}{ext}"), recursive=True)
            if matches:
                return matches[0]
            # Also try with _en suffix for json
            if ext == ".json":
                matches = glob.glob(os.path.join(data_dir, "**", f"{video_id}_en{ext}"), recursive=True)
                if matches:
                    return matches[0]

    return None


# ---------------------------------------------------------------------------
# Time-range filtering
# ---------------------------------------------------------------------------
def filter_subtitles_by_time_range(
    subtitles: list[dict],
    start: float,
    end: float,
) -> list[dict]:
    """
    Filter subtitles that overlap with a single time range.

    A subtitle overlaps if: sub.start < end AND sub.end > start.

    Args:
        subtitles: List of parsed subtitle dicts with 'start' and 'end' keys
        start: Start time in seconds
        end: End time in seconds

    Returns:
        List of overlapping subtitle dicts
    """
    return [s for s in subtitles if s['start'] < end and s['end'] > start]


def get_subtitles_for_segments(
    subtitles: list[dict],
    segments: list[dict]
) -> list[dict]:
    """
    Filter subtitles that fall within any of the given time segments, with deduplication.

    Args:
        subtitles: List of parsed subtitle entries with 'index', 'start', 'end' keys
        segments: List of time segments, each with 'start' and 'end' in seconds

    Returns:
        List of deduplicated subtitle entries sorted by start time
    """
    result = []
    seen_indices = set()

    for segment in segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', float('inf'))

        for sub in subtitles:
            if sub['start'] < seg_end and sub['end'] > seg_start:
                if sub['index'] not in seen_indices:
                    result.append(sub)
                    seen_indices.add(sub['index'])

    result.sort(key=lambda x: x['start'])
    return result


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------
def format_subtitles_as_text(
    subtitles: list[dict],
    include_timestamps: bool = False,
    max_length: Optional[int] = None,
) -> str:
    """
    Format subtitle entries as plain text.

    Args:
        subtitles: List of subtitle dicts with 'start', 'end', 'text' keys
        include_timestamps: Whether to include timestamps in output
        max_length: Optional max character length for truncation

    Returns:
        Formatted text string, optionally truncated
    """
    if not subtitles:
        return ""

    lines = []
    for sub in subtitles:
        if include_timestamps:
            start_fmt = format_seconds_to_timestamp(sub['start'])
            end_fmt = format_seconds_to_timestamp(sub['end'])
            lines.append(f"[{start_fmt} --> {end_fmt}] {sub['text']}")
        else:
            lines.append(sub['text'])

    text = '\n'.join(lines)

    if max_length is not None and len(text) > max_length:
        text = text[:max_length]

    return text


# ---------------------------------------------------------------------------
# Scene grouping
# ---------------------------------------------------------------------------
def group_subtitles_by_scenes(
    subtitles: list[dict],
    scenes: list[dict],
) -> dict[int, list[dict]]:
    """
    Map subtitles to scenes based on their midpoint falling within a scene's time range.

    Args:
        subtitles: List of subtitle dicts with 'start' and 'end' keys
        scenes: List of scene dicts with 'scene_id', 'start_time', 'end_time' keys

    Returns:
        Dict mapping scene_id to list of subtitle dicts
    """
    mapping: dict[int, list[dict]] = {s["scene_id"]: [] for s in scenes}
    for sub in subtitles:
        mid = (sub["start"] + sub["end"]) / 2
        for sc in scenes:
            if sc["start_time"] <= mid <= sc["end_time"]:
                mapping[sc["scene_id"]].append(sub)
                break
    return mapping
