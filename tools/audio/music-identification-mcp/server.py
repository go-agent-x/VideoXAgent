#!/usr/bin/env python3
"""Music Identification MCP Server — Identify music tracks from audio/video files."""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
sys.path.insert(0, str(Path(__file__).parent))

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response
from _base.config import get_env


# ── Constants ──────────────────────────────────────────────────────────────

SUPPORTED_FORMATS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".ts", ".mpg", ".mpeg", ".3gp",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac",
}

ACOUSTID_API_KEY = get_env("ACOUSTID_API_KEY", "")

# ── Lazy backend availability checks ────────────────────────────────────────────
_shazam_available: Optional[bool] = None
_acoustid_available: Optional[bool] = None


def _check_shazam() -> bool:
    global _shazam_available
    if _shazam_available is None:
        try:
            from shazamio import Shazam  # noqa: F401
            _shazam_available = True
        except ImportError:
            _shazam_available = False
    return _shazam_available


def _check_acoustid() -> bool:
    global _acoustid_available
    if _acoustid_available is None:
        try:
            import acoustid
            # pyacoustid supports two backends: libchromaprint (Python binding) or fpcalc (CLI).
            _acoustid_available = getattr(acoustid, "have_chromaprint", False) or shutil.which("fpcalc") is not None
        except ImportError:
            _acoustid_available = False
    return _acoustid_available


# ── Helper functions ──────────────────────────────────────────────────────────
def _parse_time(t) -> Optional[float]:
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


def _extract_audio_segment(
    file_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> str:
    """Extract an audio segment to a temporary WAV file using ffmpeg and return its path (the caller must delete it)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    cmd = ["ffmpeg", "-y"]
    if start_time is not None and start_time > 0:
        cmd.extend(["-ss", str(start_time)])
    cmd.extend(["-i", file_path])
    if end_time is not None:
        duration = end_time - (start_time or 0)
        if duration > 0:
            cmd.extend(["-t", str(duration)])
    # Mono 16kHz PCM, suitable for fingerprinting
    cmd.extend(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp_path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            os.unlink(tmp_path)
            raise RuntimeError(f"ffmpeg failed: {(result.stderr or '')[-500:]}")
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise RuntimeError("ffmpeg timed out (>120s)")

    return tmp_path


# ── Shazam backend ───────────────────────────────────────────────────────
def _get_proxy() -> Optional[str]:
    """Get the HTTPS proxy URL from the environment (aiohttp does not read it automatically)."""
    return os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or \
           os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY") or None


async def _identify_shazam(file_path: str, include_lyrics: bool = False, max_retries: int = 5) -> dict:
    """Identify music with shazamio and return structured results, retrying unstable proxy connections."""
    from shazamio import Shazam

    shazam = Shazam()
    proxy = _get_proxy()

    last_exc = None
    for attempt in range(max_retries):
        try:
            result = await shazam.recognize(file_path, proxy=proxy)
            break
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
    else:
        raise last_exc

    if not result or "track" not in result:
        return {"identified": False, "reason": "No match found via Shazam"}

    track = result["track"]
    metadata = {
        "identified": True,
        "title": track.get("title", "Unknown"),
        "artist": track.get("subtitle", "Unknown"),
        "genre": None,
        "album": None,
        "release_year": None,
        "label": None,
        "cover_art_url": None,
        "shazam_url": None,
        "isrc": None,
    }

    # Extract the album, year, label, and lyrics from sections.
    for section in track.get("sections", []):
        if section.get("type") == "SONG":
            for entry in section.get("metadata", []):
                key = entry.get("title", "").lower()
                val = entry.get("text")
                if "album" in key:
                    metadata["album"] = val
                elif "released" in key or "year" in key:
                    metadata["release_year"] = val
                elif "label" in key:
                    metadata["label"] = val
        elif section.get("type") == "LYRICS" and include_lyrics:
            lines = section.get("text", [])
            if lines:
                metadata["lyrics"] = "\n".join(lines) if isinstance(lines, list) else str(lines)

    # Genre
    genres = track.get("genres", {})
    if isinstance(genres, dict) and "primary" in genres:
        metadata["genre"] = genres["primary"]

    # Cover art
    images = track.get("images", {})
    if isinstance(images, dict):
        metadata["cover_art_url"] = images.get("coverarthq") or images.get("coverart")

    # URL
    metadata["shazam_url"] = track.get("url")
    metadata["isrc"] = track.get("isrc")

    # Streaming links
    providers = {}
    for provider in track.get("hub", {}).get("providers", []):
        ptype = provider.get("type", "").lower()
        for action in provider.get("actions", []):
            if action.get("uri"):
                providers[ptype] = action["uri"]
                break
    if providers:
        metadata["streaming_links"] = providers

    return metadata


# ── AcoustID backend ─────────────────────────────────────────────────────
def _identify_acoustid_sync(file_path: str) -> dict:
    """Identify music synchronously with pyacoustid, requesting recordings+releasegroups for detailed metadata."""
    import acoustid
    import json as json_lib
    import urllib.parse

    api_key = ACOUSTID_API_KEY
    if not api_key:
        return {"identified": False, "error": "ACOUSTID_API_KEY environment variable not set"}

    # Generate the fingerprint.
    try:
        duration, fingerprint = acoustid.fingerprint_file(file_path)
    except acoustid.NoBackendError:
        return {"identified": False, "error": "Chromaprint backend (fpcalc) not available"}
    except acoustid.FingerprintGenerationError:
        return {"identified": False, "error": "Failed to generate audio fingerprint"}

    # Call the API using curl (urllib3 2.x / aiohttp are incompatible with this proxy for acoustid.org).
    proxy_url = _get_proxy()
    post_data = urllib.parse.urlencode({
        "client": api_key,
        "duration": int(duration),
        "fingerprint": fingerprint,
        "meta": "recordings+releasegroups+compress",
    })

    cmd = ["curl", "-s", "--max-time", "10", "--retry", "5", "--retry-delay", "1",
           "--retry-all-errors", "--noproxy", ""]
    if proxy_url:
        cmd.extend(["-x", proxy_url])
    cmd.extend(["--data", post_data, "https://api.acoustid.org/v2/lookup"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if result.returncode != 0:
            return {"identified": False, "error": f"AcoustID API request failed: curl error {result.returncode}"}
        data = json_lib.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"identified": False, "error": "AcoustID API request timed out"}
    except (json_lib.JSONDecodeError, ValueError) as e:
        return {"identified": False, "error": f"AcoustID API returned invalid response: {e}"}

    if data.get("status") != "ok" or not data.get("results"):
        return {"identified": False, "reason": "No match found via AcoustID"}

    # Select the highest-scoring result.
    best = data["results"][0]
    score = best.get("score", 0)

    if not best.get("recordings"):
        return {"identified": False, "reason": "AcoustID matched but no recording metadata available", "score": round(score, 4)}

    recording = best["recordings"][0]
    title = recording.get("title", "Unknown")

    # Combine the artist names.
    artists = recording.get("artists", [])
    artist_name = " ".join(
        (a.get("name", "") + a.get("joinphrase", "")) for a in artists
    ).strip() if artists else "Unknown"

    # Extract album information.
    album = None
    album_type = None
    release_groups = recording.get("releasegroups", [])
    if release_groups:
        rg = release_groups[0]
        album = rg.get("title")
        album_type = rg.get("type")

    return {
        "identified": True,
        "title": title,
        "artist": artist_name,
        "album": album,
        "album_type": album_type,
        "score": round(float(score), 4),
        "duration": recording.get("duration"),
        "musicbrainz_recording_id": recording.get("id"),
        "acoustid": best.get("id"),
        "genre": None,
    }


async def _identify_acoustid(file_path: str) -> dict:
    return await asyncio.to_thread(_identify_acoustid_sync, file_path)



# ── Offline fingerprint matching ─────────────────────────────────────────────────
def _compare_fingerprints_sync(file_path_a: str, file_path_b: str) -> dict:
    """Compare the fingerprints of two audio files using chromaprint and return a similarity score."""
    import acoustid

    try:
        duration_a, fp_a = acoustid.fingerprint_file(file_path_a)
        duration_b, fp_b = acoustid.fingerprint_file(file_path_b)
    except acoustid.NoBackendError:
        return {"success": False, "error": "Chromaprint backend (fpcalc) not available"}
    except acoustid.FingerprintGenerationError as e:
        return {"success": False, "error": f"Failed to generate fingerprint: {e}"}

    # Decode the fingerprints and compute Jaccard similarity.
    try:
        from acoustid import chromaprint
        decoded_a = chromaprint.decode_fingerprint(fp_a)
        decoded_b = chromaprint.decode_fingerprint(fp_b)

        if decoded_a[0] and decoded_b[0]:
            set_a = set(decoded_a[0])
            set_b = set(decoded_b[0])
            if not set_a or not set_b:
                similarity = 0.0
            else:
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                similarity = intersection / union if union > 0 else 0.0
        else:
            similarity = 0.0
    except Exception:
        # decode_fingerprint failed — cannot compute meaningful similarity
        return {
            "success": True,
            "file_a": file_path_a,
            "file_b": file_path_b,
            "similarity": 0.0,
            "is_same_song": False,
            "duration_a": round(duration_a, 2) if duration_a else None,
            "duration_b": round(duration_b, 2) if duration_b else None,
            "warning": "chromaprint decode_fingerprint unavailable; similarity could not be computed",
        }

    return {
        "success": True,
        "file_a": file_path_a,
        "file_b": file_path_b,
        "similarity": round(similarity, 4),
        "is_same_song": similarity > 0.3,
        "duration_a": round(duration_a, 2) if duration_a else None,
        "duration_b": round(duration_b, 2) if duration_b else None,
    }


# ── Tool handlers ──────────────────────────────────────────────────────
async def _handle_identify_music(arguments: dict) -> dict:
    file_path = arguments.get("file_path", "")
    start_time = _parse_time(arguments.get("start_time"))
    end_time = _parse_time(arguments.get("end_time"))
    backend = arguments.get("backend", "auto").lower()
    include_lyrics = bool(arguments.get("include_lyrics", False))

    # Input validation
    if not file_path:
        return {"success": False, "error": "file_path is required"}
    if not os.path.isfile(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}
    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        return {"success": False, "error": f"Unsupported format: {suffix}. Supported: {sorted(SUPPORTED_FORMATS)}"}
    if not shutil.which("ffmpeg"):
        return {"success": False, "error": "ffmpeg not found in PATH"}
    if start_time is not None and start_time < 0:
        return {"success": False, "error": f"start_time must be >= 0, got {start_time}"}
    if start_time is not None and end_time is not None and end_time <= start_time:
        return {"success": False, "error": f"end_time ({end_time}) must be > start_time ({start_time})"}
    if backend not in ("shazam", "acoustid", "auto"):
        return {"success": False, "error": f"Unknown backend: {backend}. Choose from: shazam, acoustid, auto"}

    needs_extraction = start_time is not None or end_time is not None
    working_file = file_path
    tmp_path = None

    try:
        if needs_extraction:
            tmp_path = _extract_audio_segment(file_path, start_time, end_time)
            working_file = tmp_path

        used_backend = None
        result = None

        if backend == "shazam":
            if not _check_shazam():
                return {"success": False, "error": "shazamio not installed. From the repository root, run: bash scripts/install_profiles.sh music"}
            result = await _identify_shazam(working_file, include_lyrics=include_lyrics)
            used_backend = "shazam"

        elif backend == "acoustid":
            if not _check_acoustid():
                return {"success": False, "error": "pyacoustid not available or fpcalc not found. From the repository root, run: bash scripts/install_profiles.sh music. Also install the system fpcalc utility (Debian/Ubuntu: sudo apt install libchromaprint-tools)."}
            result = await _identify_acoustid(working_file)
            used_backend = "acoustid"

        else:  # auto uses shazam only.
            if _check_shazam():
                result = await _identify_shazam(working_file, include_lyrics=include_lyrics)
                used_backend = "shazam"
            else:
                return {
                    "success": False,
                    "error": "shazamio not installed. From the repository root, run: bash scripts/install_profiles.sh music",
                }

        # Assemble the final output.
        result["success"] = result.get("identified", False)
        result["backend"] = used_backend
        result["file_path"] = file_path
        if needs_extraction:
            result["analyzed_range"] = {
                "start_time": start_time or 0,
                "end_time": end_time,
            }

        return result

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _handle_compare_fingerprint(arguments: dict) -> dict:
    file_path_a = arguments.get("file_a", "")
    file_path_b = arguments.get("file_b", "")

    if not file_path_a or not file_path_b:
        return {"success": False, "error": "Both file_a and file_b are required"}
    for path in [file_path_a, file_path_b]:
        if not os.path.isfile(path):
            return {"success": False, "error": f"File not found: {path}"}
    if not _check_acoustid():
        return {"success": False, "error": "pyacoustid not available or fpcalc not found. From the repository root, run: bash scripts/install_profiles.sh music. Also install the system fpcalc utility (Debian/Ubuntu: sudo apt install libchromaprint-tools)."}

    try:
        return await asyncio.to_thread(_compare_fingerprints_sync, file_path_a, file_path_b)
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ── MCP registration ──────────────────────────────────────────────────────────
TOOL_HANDLERS = {
    "identify_music": _handle_identify_music,
    "compare_audio_fingerprint": _handle_compare_fingerprint,
}


# ── Entry point ──────────────────────────────────────────────────────────────


async def handle_identify_music(arguments: dict) -> dict:
    return await _handle_identify_music(arguments)


async def handle_compare_audio_fingerprint(arguments: dict) -> dict:
    return await _handle_compare_fingerprint(arguments)


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="music-identification",
        tool_defs_source="_tool_defs/audio.yaml",
        handlers={
            "identify_music": handle_identify_music,
            "compare_audio_fingerprint": handle_compare_audio_fingerprint,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
