#!/usr/bin/env python3
"""Video Duration MCP Server — Get video file duration and metadata."""
import asyncio
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, BinaryIO

sys.path.insert(0, str(Path(__file__).parent))
# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response

# ── Business logic ──



def read_uint32_be(f: BinaryIO) -> int:
    """Read 4 bytes as big-endian unsigned int."""
    data = f.read(4)
    if len(data) < 4:
        raise EOFError("Unexpected end of file")
    return struct.unpack(">I", data)[0]


def read_uint64_be(f: BinaryIO) -> int:
    """Read 8 bytes as big-endian unsigned int."""
    data = f.read(8)
    if len(data) < 8:
        raise EOFError("Unexpected end of file")
    return struct.unpack(">Q", data)[0]


def find_atom(f: BinaryIO, atom_type: bytes, end_pos: Optional[int] = None) -> Optional[tuple[int, int]]:
    """
    Find an atom in the file.
    Returns (start_position, size) or None if not found.
    """
    while True:
        if end_pos is not None and f.tell() >= end_pos:
            return None

        start_pos = f.tell()
        try:
            size = read_uint32_be(f)
            atype = f.read(4)
            if len(atype) < 4:
                return None
        except EOFError:
            return None

        if size == 1:
            # Extended size
            size = read_uint64_be(f)
        elif size == 0:
            # Atom extends to end of file
            current = f.tell()
            f.seek(0, 2)
            size = f.tell() - start_pos
            f.seek(current)

        if atype == atom_type:
            return (start_pos, size)

        # Skip to next atom
        next_pos = start_pos + size
        if next_pos < f.tell():
            return None
        f.seek(next_pos)


def parse_mvhd(f: BinaryIO, atom_size: int) -> Optional[dict]:
    """Parse mvhd (movie header) atom to get duration and timescale."""
    start = f.tell()
    version = struct.unpack(">B", f.read(1))[0]
    f.read(3)  # flags

    if version == 0:
        f.read(4)  # creation_time
        f.read(4)  # modification_time
        timescale = read_uint32_be(f)
        duration = read_uint32_be(f)
    elif version == 1:
        f.read(8)  # creation_time
        f.read(8)  # modification_time
        timescale = read_uint32_be(f)
        duration = read_uint64_be(f)
    else:
        return None

    if timescale == 0:
        return None

    return {
        "duration_ticks": duration,
        "timescale": timescale,
        "duration": duration / timescale,
    }


def parse_tkhd(f: BinaryIO) -> Optional[dict]:
    """Parse tkhd (track header) atom to get width and height."""
    version = struct.unpack(">B", f.read(1))[0]
    f.read(3)  # flags

    if version == 0:
        f.read(4 + 4 + 4 + 4 + 4)  # creation, modification, track_id, reserved, duration
    else:
        f.read(8 + 8 + 4 + 4 + 8)  # same but 64-bit times

    f.read(8)  # reserved
    f.read(2)  # layer
    f.read(2)  # alternate_group
    f.read(2)  # volume
    f.read(2)  # reserved
    f.read(36)  # matrix

    width_fixed = read_uint32_be(f)
    height_fixed = read_uint32_be(f)

    # Width and height are 16.16 fixed point
    width = width_fixed >> 16
    height = height_fixed >> 16

    if width > 0 and height > 0:
        return {"width": width, "height": height}
    return None


def parse_hdlr(f: BinaryIO) -> Optional[str]:
    """Parse hdlr (handler reference) atom to get the handler type.
    Returns handler type string like 'vide', 'soun', 'sbtl', or None.
    """
    try:
        f.read(1 + 3)  # version + flags
        f.read(4)  # pre_defined
        handler_type = f.read(4)
        if len(handler_type) < 4:
            return None
        return handler_type.decode("ascii", errors="replace")
    except Exception:
        return None


def parse_audio_stsd(f: BinaryIO, atom_size: int) -> Optional[dict]:
    """Parse stsd (sample description) atom for an audio track.
    Returns codec name, channel count, and sample rate.
    """
    try:
        f.read(1 + 3)  # version + flags
        entry_count = read_uint32_be(f)
        if entry_count == 0:
            return None

        # Parse first entry
        entry_size = read_uint32_be(f)
        codec_bytes = f.read(4)
        if len(codec_bytes) < 4:
            return None
        codec = codec_bytes.decode("ascii", errors="replace").strip()

        f.read(6)  # reserved
        f.read(2)  # data_reference_index
        f.read(8)  # reserved (2 x uint32 in Apple's spec)

        channel_count = struct.unpack(">H", f.read(2))[0]
        f.read(2)  # sample_size
        f.read(2)  # compression_id
        f.read(2)  # packet_size

        sample_rate_fixed = read_uint32_be(f)
        sample_rate = sample_rate_fixed >> 16  # 16.16 fixed point

        return {
            "audio_codec": codec,
            "audio_channels": channel_count,
            "audio_sample_rate": sample_rate,
        }
    except Exception:
        return None


MP4_FORMATS = {".mp4", ".m4v", ".mov", ".m4a", ".3gp", ".3g2"}
ALL_VIDEO_FORMATS = MP4_FORMATS | {
    ".mkv", ".avi", ".webm", ".flv", ".wmv", ".mpg", ".mpeg",
    ".ts", ".mts", ".m2ts", ".vob", ".ogv", ".rm", ".rmvb",
    ".asf", ".f4v", ".divx",
}


def get_ffprobe_duration(file_path: str) -> dict[str, Any]:
    """Get video metadata using ffprobe. Works for all formats ffmpeg supports."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                file_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        return {"error": "ffprobe not found — install ffmpeg to support non-MP4 formats"}
    except subprocess.TimeoutExpired:
        return {"error": "ffprobe timed out"}

    if result.returncode != 0:
        stderr = result.stderr.strip()
        return {"error": f"ffprobe failed (exit {result.returncode}): {stderr or 'unknown error'}"}

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "ffprobe returned invalid JSON"}

    # Extract duration from format section
    fmt = probe.get("format", {})
    duration_str = fmt.get("duration")
    if duration_str is None:
        return {"error": "ffprobe could not determine duration"}

    try:
        duration_seconds = float(duration_str)
    except (ValueError, TypeError):
        return {"error": f"ffprobe returned non-numeric duration: {duration_str}"}

    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = duration_seconds % 60
    formatted = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    metadata: dict[str, Any] = {
        "duration": duration_seconds,
        "duration_formatted": formatted,
    }

    # Scan streams for video/audio info
    has_video = False
    has_audio = False
    for stream in probe.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and not has_video:
            has_video = True
            w = stream.get("width")
            h = stream.get("height")
            if w and h:
                metadata["width"] = int(w)
                metadata["height"] = int(h)
        elif codec_type == "audio" and not has_audio:
            has_audio = True
            codec_name = stream.get("codec_name")
            channels = stream.get("channels")
            sample_rate = stream.get("sample_rate")
            if codec_name:
                metadata["audio_codec"] = codec_name
            if channels is not None:
                metadata["audio_channels"] = int(channels)
            if sample_rate is not None:
                metadata["audio_sample_rate"] = int(sample_rate)

    metadata["has_audio"] = has_audio
    metadata["has_video"] = has_video
    return metadata


def get_mp4_duration(file_path: str) -> dict[str, Any]:
    """
    Get video duration by parsing MP4/MOV container format.
    Works for MP4, M4V, MOV, and other ISO base media file format files.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    if not os.path.isfile(file_path):
        return {"error": f"Not a file: {file_path}"}

    suffix = Path(file_path).suffix.lower()
    if suffix not in MP4_FORMATS:
        return {"error": f"Not an MP4-family format: {suffix}"}

    try:
        with open(file_path, "rb") as f:
            # Find moov atom
            moov = find_atom(f, b"moov")
            if not moov:
                return {"error": "Could not find moov atom - file may be corrupted or not a valid MP4/MOV"}

            moov_start, moov_size = moov
            moov_end = moov_start + moov_size

            # Position after moov header
            f.seek(moov_start + 8)

            # Find mvhd atom inside moov
            mvhd = find_atom(f, b"mvhd", moov_end)
            if not mvhd:
                return {"error": "Could not find mvhd atom - file may be corrupted"}

            mvhd_start, mvhd_size = mvhd
            f.seek(mvhd_start + 8)  # Skip atom header

            mvhd_data = parse_mvhd(f, mvhd_size - 8)
            if not mvhd_data:
                return {"error": "Could not parse mvhd atom"}

            duration_seconds = mvhd_data["duration"]

            # Format duration
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            seconds = duration_seconds % 60
            formatted = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

            metadata = {
                "duration": duration_seconds,
                "duration_formatted": formatted,
            }

            # Scan all tracks for video dimensions and audio stream info
            has_video = False
            has_audio = False
            f.seek(moov_start + 8)
            while True:
                trak = find_atom(f, b"trak", moov_end)
                if not trak:
                    break

                trak_start, trak_size = trak
                trak_end = trak_start + trak_size

                # Determine track type via mdia -> hdlr
                handler_type = None
                f.seek(trak_start + 8)
                mdia = find_atom(f, b"mdia", trak_end)
                if mdia:
                    mdia_start, mdia_size = mdia
                    mdia_end = mdia_start + mdia_size

                    # Find hdlr inside mdia
                    f.seek(mdia_start + 8)
                    hdlr = find_atom(f, b"hdlr", mdia_end)
                    if hdlr:
                        hdlr_start, _ = hdlr
                        f.seek(hdlr_start + 8)
                        handler_type = parse_hdlr(f)

                # Video track: extract dimensions from tkhd
                if handler_type == "vide" and not has_video:
                    has_video = True
                    f.seek(trak_start + 8)
                    tkhd = find_atom(f, b"tkhd", trak_end)
                    if tkhd:
                        tkhd_start, _ = tkhd
                        f.seek(tkhd_start + 8)
                        tkhd_data = parse_tkhd(f)
                        if tkhd_data and tkhd_data.get("width") and tkhd_data.get("height"):
                            metadata["width"] = tkhd_data["width"]
                            metadata["height"] = tkhd_data["height"]

                # Audio track: extract codec, channels, sample_rate from stsd
                elif handler_type == "soun" and not has_audio:
                    has_audio = True
                    if mdia:
                        f.seek(mdia_start + 8)
                        minf = find_atom(f, b"minf", mdia_end)
                        if minf:
                            minf_start, minf_size = minf
                            minf_end = minf_start + minf_size
                            f.seek(minf_start + 8)
                            stbl = find_atom(f, b"stbl", minf_end)
                            if stbl:
                                stbl_start, stbl_size = stbl
                                stbl_end = stbl_start + stbl_size
                                f.seek(stbl_start + 8)
                                stsd = find_atom(f, b"stsd", stbl_end)
                                if stsd:
                                    stsd_start, stsd_size = stsd
                                    f.seek(stsd_start + 8)
                                    audio_info = parse_audio_stsd(f, stsd_size - 8)
                                    if audio_info:
                                        metadata.update(audio_info)

                # Move to next track
                f.seek(trak_end)

            metadata["has_audio"] = has_audio
            metadata["has_video"] = has_video

            # Add file size

            return metadata

    except Exception as e:
        return {"error": f"Failed to parse video file: {e}"}


def get_video_duration(file_path: str) -> dict[str, Any]:
    """Get video duration — tries native MP4 parser first, falls back to ffprobe."""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    if not os.path.isfile(file_path):
        return {"error": f"Not a file: {file_path}"}

    suffix = Path(file_path).suffix.lower()

    # For MP4-family files, try the fast native parser first
    if suffix in MP4_FORMATS:
        result = get_mp4_duration(file_path)
        if "error" not in result:
            return result
        # Native parser failed — fall through to ffprobe

    # ffprobe handles all formats
    return get_ffprobe_duration(file_path)


# Initialize MCP server


async def handle_get_video_duration(arguments: dict) -> dict:
    file_path = arguments.get("file_path", "")
    result = get_video_duration(file_path)
    if "error" in result:
        return make_error_response(result["error"])
    return make_success_response(result)


async def handle_get_videos_duration(arguments: dict) -> dict:
    file_paths = arguments.get("file_paths", [])
    results = {}
    total_duration = 0.0

    for fp in file_paths:
        result = get_video_duration(fp)
        results[fp] = result
        if "duration" in result:
            total_duration += result["duration"]

    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = total_duration % 60

    return make_success_response({
        "files": results,
        "total_files": len(file_paths),
        "total_duration": total_duration,
        "total_duration_formatted": f"{hours:02d}:{minutes:02d}:{seconds:06.3f}",
    })


async def handle_get_directory_videos_duration(arguments: dict) -> dict:
    directory_path = arguments.get("directory_path", "")
    recursive = arguments.get("recursive", False)

    if not os.path.isdir(directory_path):
        return make_error_response(f"Not a directory: {directory_path}")

    video_extensions = ALL_VIDEO_FORMATS
    video_files = []

    if recursive:
        for root, _, files in os.walk(directory_path):
            for f in files:
                if Path(f).suffix.lower() in video_extensions:
                    video_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(directory_path):
            full_path = os.path.join(directory_path, f)
            if os.path.isfile(full_path) and Path(f).suffix.lower() in video_extensions:
                video_files.append(full_path)

    results = {}
    total_duration = 0.0

    for fp in sorted(video_files):
        result = get_video_duration(fp)
        results[fp] = result
        if "duration" in result:
            total_duration += result["duration"]

    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = total_duration % 60

    return make_success_response({
        "directory": directory_path,
        "recursive": recursive,
        "files": results,
        "total_files": len(video_files),
        "total_duration": total_duration,
        "total_duration_formatted": f"{hours:02d}:{minutes:02d}:{seconds:06.3f}",
    })


def main():
    app = create_mcp_server(
        server_name="video-duration-mcp",
        tool_defs_source="_tool_defs/basic.yaml",
        handlers={
            "get_video_duration": handle_get_video_duration,
            "get_videos_duration": handle_get_videos_duration,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))


if __name__ == "__main__":
    main()
