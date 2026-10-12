#!/usr/bin/env python3
"""Audio Extract MCP Server — Extract audio track from video/audio files."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")

from _base.paths import PROJECT_ROOT

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response

# ── Business logic (from audio_extract_server.py) ──

import json
import os
import subprocess
import shutil

OUTPUT_DIR = PROJECT_ROOT / "output"

SUPPORTED_INPUT_FORMATS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".ts", ".mpg", ".mpeg", ".3gp",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"
}

# Output format to ffmpeg encoder mapping
FORMAT_CODEC_MAP = {
    "wav": ["-c:a", "pcm_s16le"],       # Lossless PCM
    "mp3": ["-c:a", "libmp3lame", "-q:a", "2"],  # High-quality VBR
    "aac": ["-c:a", "aac", "-b:a", "192k"],
    "m4a": ["-c:a", "aac", "-b:a", "192k"],
    "flac": ["-c:a", "flac"],            # Lossless
    "ogg": ["-c:a", "libvorbis", "-q:a", "6"],
}

# Formats supporting direct stream copy (input codec to compatible output formats)
COPY_COMPATIBLE = {
    "aac": {"aac", "m4a", "mp4"},
    "mp3": {"mp3"},
    "flac": {"flac"},
    "vorbis": {"ogg"},
    "opus": {"ogg", "webm"},
    "pcm_s16le": {"wav"},
    "pcm_s16be": {"wav"},
}


def _parse_time(t) -> float | None:
    """Convert time parameters to seconds. None means unspecified."""
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, str):
        t = t.strip()
        if not t:
            return None
        parts = t.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
        return float(t)
    return None


def _get_audio_info(file_path: str) -> dict | None:
    """Get audio stream information using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate,duration",
             "-of", "json", file_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                return streams[0]
    except Exception:
        pass
    return None


def _get_duration(file_path: str) -> float | None:
    """Get the total file duration using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def extract_audio(
    video_path: str,
    output_format: str = "wav",
    start_time=None,
    end_time=None,
    output_name: str = "",
    sample_rate: int | None = None,
    mono: bool = False,
) -> dict:
    """Extract the audio track from a video or audio file."""

    if not os.path.exists(video_path):
        return {"success": False, "error": f"File not found: {video_path}"}

    suffix = Path(video_path).suffix.lower()
    if suffix not in SUPPORTED_INPUT_FORMATS:
        return {"success": False, "error": f"Unsupported input format: {suffix}"}

    if not shutil.which("ffmpeg"):
        return {"success": False, "error": "ffmpeg not found in PATH"}

    output_format = output_format.lower().strip().lstrip(".")
    if output_format not in FORMAT_CODEC_MAP:
        return {
            "success": False,
            "error": f"Unsupported output format: {output_format}. "
                     f"Supported: {', '.join(sorted(FORMAT_CODEC_MAP.keys()))}"
        }

    start_sec = _parse_time(start_time)
    end_sec = _parse_time(end_time)

    if start_sec is not None and start_sec < 0:
        return {"success": False, "error": f"start_time must be >= 0, got {start_sec}"}
    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        return {"success": False, "error": f"end_time ({end_sec}) must be > start_time ({start_sec})"}

    # Check whether an audio stream exists.
    audio_info = _get_audio_info(video_path)
    if not audio_info:
        return {"success": False, "error": "No audio stream found in the input file"}

    # Output path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not output_name:
        video_name = Path(video_path).stem
        time_tag = ""
        if start_sec is not None or end_sec is not None:
            s = f"{start_sec:.1f}s" if start_sec is not None else "0s"
            e = f"{end_sec:.1f}s" if end_sec is not None else "end"
            time_tag = f"_{s}-{e}"
        output_name = f"{video_name}_audio{time_tag}.{output_format}"
    elif not Path(output_name).suffix:
        output_name = f"{output_name}.{output_format}"

    output_path = str((OUTPUT_DIR / output_name).resolve())

    # Choose between stream copy and re-encoding.
    input_codec = audio_info.get("codec_name", "")
    can_copy = (
        output_format in COPY_COMPATIBLE.get(input_codec, set())
        and sample_rate is None
        and not mono
    )

    # Build the ffmpeg command.
    cmd = ["ffmpeg", "-y"]

    if start_sec is not None:
        cmd.extend(["-ss", str(start_sec)])

    cmd.extend(["-i", video_path])

    if end_sec is not None:
        if start_sec is not None:
            cmd.extend(["-t", str(end_sec - start_sec)])
        else:
            cmd.extend(["-t", str(end_sec)])

    cmd.extend(["-vn"])  # Remove the video track.

    if can_copy:
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.extend(FORMAT_CODEC_MAP[output_format])

    if sample_rate is not None:
        cmd.extend(["-ar", str(sample_rate)])

    if mono:
        cmd.extend(["-ac", "1"])

    cmd.append(output_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "ffmpeg timed out (>600s)"}

    if result.returncode != 0:
        stderr = (result.stderr or "")[-500:]
        return {"success": False, "error": f"ffmpeg failed: {stderr}"}

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return {"success": False, "error": "Output file is empty or missing"}

    # Output information
    out_duration = _get_duration(output_path)
    out_audio_info = _get_audio_info(output_path)

    return {
        "success": True,
        "output_path": output_path,
        "output_format": output_format,
        "encoding_mode": "stream_copy" if can_copy else "re-encode",
        "start_time": start_sec,
        "end_time": end_sec,
        "duration": round(out_duration, 3) if out_duration else None,
        "sample_rate": out_audio_info.get("sample_rate") if out_audio_info else None,
        "channels": out_audio_info.get("channels") if out_audio_info else None,
        "codec": out_audio_info.get("codec_name") if out_audio_info else None,
        "file_size_bytes": os.path.getsize(output_path),
        "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2),
    }


async def handle_extract_audio(arguments: dict) -> dict:
    return extract_audio(
        video_path=arguments.get("video_path", ""),
        output_format=arguments.get("output_format", "wav"),
        start_time=arguments.get("start_time"),
        end_time=arguments.get("end_time"),
        output_name=arguments.get("output_name", ""),
        sample_rate=arguments.get("sample_rate"),
        mono=arguments.get("mono", False),
    )


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="audio-extract-mcp",
        tool_defs_source="_tool_defs/basic.yaml",
        handlers={"extract_audio": handle_extract_audio},
        is_file=True,
    )
    asyncio.run(run_server(app))
