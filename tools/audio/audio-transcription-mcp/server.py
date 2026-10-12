#!/usr/bin/env python3
"""Audio Transcription MCP Server — WhisperX speech recognition with word-level timestamps."""
import asyncio
import contextlib
import gc
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")
sys.path.insert(0, str(Path(__file__).parent))

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response
from _base.device import select_torch_device

from _base.config import get_config

# ──────────────────────────────────────────────────────────────────────
# CRITICAL: Force offline mode BEFORE any HuggingFace / torch imports.
# Without this, huggingface_hub.snapshot_download() will try to contact
# huggingface.co on every model load.  In restricted networks (no proxy)
# this causes a 30-60s hang that looks like a tool timeout.
# ──────────────────────────────────────────────────────────────────────
def configure_hf_hub_env() -> None:
    """
    Configure HuggingFace Hub behavior to avoid long hangs in restricted networks.

    - If MCP_ALLOW_MODEL_DOWNLOAD is not truthy, force offline mode (fast-fail when models are missing).
    - Always set conservative timeouts if not already set.
    """
    allow_download = os.environ.get("MCP_ALLOW_MODEL_DOWNLOAD", "").strip().lower() in {"1", "true", "yes", "y"}
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Conservative defaults to prevent long network stalls
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "8")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# Call IMMEDIATELY at module level — before any library import.
configure_hf_hub_env()

# HuggingFace hub utilities (used for offline/fast-fail behavior)
try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    HF_HUB_AVAILABLE = True
except Exception:
    snapshot_download = None
    LocalEntryNotFoundError = Exception
    HF_HUB_AVAILABLE = False

# Check for torch and whisperx availability
TORCH_AVAILABLE = True
WHISPERX_AVAILABLE = True
torch = None
whisperx = None


# ---------------------------------------------------------------------------
# Configuration from _base/config.py
# ---------------------------------------------------------------------------
_whisperx_cfg = get_config("whisperx")
DEFAULT_MODEL_SIZE = _whisperx_cfg["model_size"]
DEFAULT_COMPUTE_TYPE = _whisperx_cfg["compute_type"]

# Supported audio/video formats
SUPPORTED_FORMATS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"
}

# Global model cache to avoid reloading
_model_cache: dict[str, Any] = {}
_align_model_cache: dict[str, Any] = {}

# Fallback model escalation order (from smallest to largest)
_MODEL_ESCALATION = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

# Minimum word count threshold — below this, the result is considered too sparse
_MIN_WORD_THRESHOLD = 3


def _count_transcribed_words(text: str) -> int:
    """Count non-empty words in transcription text."""
    if not text:
        return 0
    return len(text.split())


def _is_model_locally_available(model_size: str) -> bool:
    """Check if a faster-whisper model is available in local cache."""
    if not HF_HUB_AVAILABLE:
        return True  # Cannot check; assume available
    repo_id = f"Systran/faster-whisper-{model_size}"
    try:
        snapshot_download(repo_id=repo_id, local_files_only=True)
        return True
    except Exception:
        return False


def _get_fallback_models(current_model: str) -> list[str]:
    """
    Return a list of larger models to try when the current model produces sparse output.
    Only returns models strictly larger than the current one and locally available.
    """
    try:
        idx = _MODEL_ESCALATION.index(current_model)
    except ValueError:
        return []
    # Return models larger than the current, capped at large-v2 and large-v3
    candidates = _MODEL_ESCALATION[idx + 1:]
    # Filter out models not present in local cache to avoid download delays
    candidates = [m for m in candidates if _is_model_locally_available(m)]
    # Only return at most 2 fallback levels to avoid excessive retries
    return candidates[:2]


def _narrow_time_window(
    start_time: Optional[float],
    end_time: Optional[float],
    margin_sec: float = 2.0,
    min_duration: float = 5.0,
) -> tuple[Optional[float], Optional[float]]:
    """
    Narrow the time window by trimming `margin_sec` from each side.
    This reduces noise for the larger model. Only narrows if the resulting
    duration is at least `min_duration` seconds.

    Returns (new_start, new_end) — unchanged if narrowing is not possible.
    """
    if start_time is None or end_time is None:
        return start_time, end_time
    duration = end_time - start_time
    # Don't narrow if the window is already short
    if duration <= min_duration + 2 * margin_sec:
        return start_time, end_time
    return start_time + margin_sec, end_time - margin_sec


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """
    MCP protocol uses stdout. Third-party libs may print/log to stdout which can corrupt the stream.
    Redirect stdout to stderr inside critical sections.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _get_torch():
    """Lazy import torch to keep server startup fast."""
    global torch, TORCH_AVAILABLE
    if torch is None and TORCH_AVAILABLE:
        try:
            import torch as _torch  # type: ignore
            torch = _torch
        except Exception:
            TORCH_AVAILABLE = False
            torch = None
    return torch


def _get_whisperx():
    """Lazy import whisperx to keep server startup fast."""
    global whisperx, WHISPERX_AVAILABLE
    if whisperx is None and WHISPERX_AVAILABLE:
        try:
            # -----------------------------------------------------------
            # Compatibility shim: PyTorch 2.6+ defaults weights_only=True
            # in torch.load, but pyannote/whisperx checkpoints use
            # omegaconf types that aren't in the default allowlist.
            # Patch torch.load to default weights_only=False.
            # -----------------------------------------------------------
            import torch
            _original_torch_load = torch.load
            def _patched_torch_load(*args, **kwargs):
                if "weights_only" not in kwargs or kwargs.get("weights_only") is None:
                    kwargs["weights_only"] = False
                return _original_torch_load(*args, **kwargs)
            torch.load = _patched_torch_load
            # -----------------------------------------------------------
            # Compatibility shim: torchaudio >= 2.3 removed several APIs
            # that pyannote.audio/whisperx still reference.
            # -----------------------------------------------------------
            import torchaudio
            if not hasattr(torchaudio, "AudioMetaData"):
                from dataclasses import dataclass
                @dataclass
                class _AudioMetaData:
                    sample_rate: int
                    num_frames: int
                    num_channels: int
                    bits_per_sample: int = 0
                    encoding: str = ""
                torchaudio.AudioMetaData = _AudioMetaData
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["ffmpeg"]
            # -----------------------------------------------------------
            with _redirect_stdout_to_stderr():
                import whisperx as _whisperx  # type: ignore
                whisperx = _whisperx
        except Exception:
            WHISPERX_AVAILABLE = False
            whisperx = None
    return whisperx


def _is_jsonable_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None


def _json_sanitize(obj: Any) -> Any:
    """
    Convert common non-JSON-serializable numeric types (numpy/torch scalars) into Python primitives.
    Also sanitizes nested containers.
    """
    if _is_jsonable_scalar(obj):
        return obj

    # torch / numpy scalar -> python scalar
    try:
        # torch tensors / scalars
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            return obj.item()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]

    # Fallback to string to avoid crashing the MCP transport
    return str(obj)


def ensure_whisper_model_available(model_size: str) -> Optional[str]:
    """
    Best-effort check for faster-whisper model presence when running offline.
    Returns an error message if we can determine the model is missing; otherwise None.
    """
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        return None
    if not HF_HUB_AVAILABLE:
        return None

    repo_id = f"Systran/faster-whisper-{model_size}"
    try:
        snapshot_download(repo_id=repo_id, local_files_only=True)
        return None
    except LocalEntryNotFoundError:
        return (
            f"Model weights not found in local cache for {repo_id}. "
            "Pre-download models (e.g. run `bash download_models.sh`, or set MCP_ALLOW_MODEL_DOWNLOAD=1 if network is available)."
        )
    except Exception:
        # Unknown issue; do not block execution here
        return None


def parse_timestamp(value: Any) -> Optional[float]:
    """
    Parse a timestamp expressed as seconds (int/float) or as HH:MM:SS(.mmm) string.
    Returns seconds as float, or None if value is None/empty.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise ValueError(f"Invalid timestamp type: {type(value).__name__}")

    s = value.strip()
    if s == "":
        return None

    # plain seconds in string form
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)

    parts = s.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid timestamp format: {value}")

    if len(parts) == 2:
        hours = 0
        minutes_str, seconds_str = parts
    else:
        hours_str, minutes_str, seconds_str = parts
        hours = int(hours_str)

    minutes = int(minutes_str)
    seconds = float(seconds_str)
    if hours < 0 or minutes < 0 or seconds < 0:
        raise ValueError(f"Invalid timestamp (negative): {value}")
    return hours * 3600 + minutes * 60 + seconds


def extract_audio_segment_to_wav(
    input_path: str,
    start_time: Optional[float],
    end_time: Optional[float],
    sample_rate: int = 16000,
    mono: bool = True,
) -> tuple[str, Optional[float], Optional[float]]:
    """
    Extract an audio segment to a temporary WAV file via ffmpeg.
    Returns (wav_path, start_time, end_time).
    """
    if start_time is not None and start_time < 0:
        raise ValueError("start_time must be >= 0")
    if end_time is not None and end_time < 0:
        raise ValueError("end_time must be >= 0")
    if start_time is not None and end_time is not None and end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")

    tmp = tempfile.NamedTemporaryFile(prefix="mcp_asr_", suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_time is not None:
        cmd += ["-ss", f"{start_time:.3f}"]
    if end_time is not None:
        cmd += ["-to", f"{end_time:.3f}"]
    cmd += ["-i", input_path, "-vn"]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(sample_rate), tmp_path]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed to extract audio segment: {stderr.strip()}") from e

    return tmp_path, start_time, end_time


def get_device() -> str:
    """Return the configured device, falling back to hardware detection."""
    return select_torch_device(_get_torch())


def get_compute_type(device: str, requested_type: str) -> str:
    """Get appropriate compute type based on device."""
    if device == "cpu":
        # CPU doesn't support float16 well, use int8
        return "int8"
    return requested_type


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _has_audio_stream(file_path: str) -> bool:
    """Use ffprobe to check whether the file contains an audio stream."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name",
             "-of", "json", file_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            import json as _json
            data = _json.loads(result.stdout)
            return len(data.get("streams", [])) > 0
    except Exception:
        pass
    return False


def validate_file(file_path: str) -> Optional[str]:
    """Validate file exists and has supported format. Returns error message or None."""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    if not os.path.isfile(file_path):
        return f"Not a file: {file_path}"

    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        return f"Unsupported format: {suffix}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"

    return None


def load_model_cached(model_size: str, device: str, compute_type: str):
    """Load WhisperX model with caching."""
    wx = _get_whisperx()
    if wx is None:
        raise RuntimeError("whisperx is not installed")

    cache_key = f"{model_size}_{device}_{compute_type}"

    if cache_key not in _model_cache:
        _model_cache[cache_key] = wx.load_model(
            model_size, device, compute_type=compute_type,
            local_files_only=True,
        )
    
    return _model_cache[cache_key]


def load_align_model_cached(language_code: str, device: str):
    """Load alignment model with caching."""
    wx = _get_whisperx()
    if wx is None:
        raise RuntimeError("whisperx is not installed")

    cache_key = f"{language_code}_{device}"
    
    if cache_key not in _align_model_cache:
        model_a, metadata = wx.load_align_model(
            language_code=language_code, device=device
        )
        _align_model_cache[cache_key] = (model_a, metadata)
    
    return _align_model_cache[cache_key]


def clear_model_cache():
    """Clear model cache to free GPU memory."""
    global _model_cache, _align_model_cache

    _model_cache.clear()
    _align_model_cache.clear()

    torch_mod = _get_torch()
    if torch_mod is not None:
        gc.collect()
        if torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()


async def _transcribe_single_pass(
    file_path: str,
    language: Optional[str],
    model_size: str,
    batch_size: int,
    compute_type: str,
    align_words: bool,
    seg_start: Optional[float],
    seg_end: Optional[float],
) -> dict[str, Any]:
    """
    Run one transcription pass with the given parameters.
    Return segment/word timestamps on the original file's timeline.
    Returns the formatted output dict (may contain "error" key on failure).
    """
    wx = _get_whisperx()
    torch_mod = _get_torch()

    missing_model_error = ensure_whisper_model_available(model_size)
    if missing_model_error:
        return {"error": missing_model_error}

    device = get_device()
    compute_type = get_compute_type(device, compute_type)

    tmp_audio_path: Optional[str] = None
    try:
        with _redirect_stdout_to_stderr():
            model = load_model_cached(model_size, device, compute_type)

            audio_input_path = file_path
            if seg_start is not None or seg_end is not None:
                tmp_audio_path, seg_start, seg_end = extract_audio_segment_to_wav(
                    input_path=file_path,
                    start_time=seg_start,
                    end_time=seg_end,
                    sample_rate=16000,
                    mono=True,
                )
                audio_input_path = tmp_audio_path

            audio = wx.load_audio(audio_input_path)

            result = await asyncio.to_thread(
                model.transcribe, audio, batch_size=batch_size, language=language
            )
    finally:
        if tmp_audio_path is not None:
            try:
                os.unlink(tmp_audio_path)
            except Exception:
                pass

    detected_language = result.get("language", language or "unknown")

    if align_words:
        try:
            with _redirect_stdout_to_stderr():
                model_a, metadata = load_align_model_cached(detected_language, device)
                result = await asyncio.to_thread(
                    wx.align,
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
        except Exception as align_error:
            result["alignment_warning"] = f"Word alignment not available: {str(align_error)}"

    # WhisperX transcribes and aligns the loaded (possibly cropped) waveform.
    # Keep those timestamps local until alignment finishes, then translate them
    # once using this pass's actual crop start, including narrowed fallback runs.
    timestamp_offset = float(seg_start) if seg_start is not None else 0.0
    segments = []
    full_text = []

    for seg in result.get("segments", []):
        start_s = float(seg.get("start", 0) or 0) + timestamp_offset
        end_s = float(seg.get("end", 0) or 0) + timestamp_offset
        segment_info = {
            "start_time": start_s,
            "end_time": end_s,
            "start_formatted": format_timestamp(start_s),
            "end_formatted": format_timestamp(end_s),
            "text": seg.get("text", "").strip()
        }

        if "words" in seg:
            segment_info["words"] = [
                {
                    "word": w.get("word", ""),
                    # Unaligned words may have no timestamp. Do not turn an
                    # unknown position into the beginning of the source clip.
                    "start_time": (
                        float(w["start"]) + timestamp_offset if w.get("start") is not None else None
                    ),
                    "end_time": (
                        float(w["end"]) + timestamp_offset if w.get("end") is not None else None
                    ),
                }
                for w in seg["words"]
            ]

        segments.append(segment_info)
        full_text.append(seg.get("text", "").strip())

    output = {
        "file_path": file_path,
        "start_time": seg_start,
        "end_time": seg_end,
        "language": detected_language,
        "device": device,
        "model_size": model_size,
        "text": " ".join(full_text),
        "segments": segments,
        "total_segments": len(segments)
    }

    if "alignment_warning" in result:
        output["alignment_warning"] = result["alignment_warning"]

    return output


async def transcribe_audio(
    file_path: str,
    language: Optional[str] = None,
    model_size: str = DEFAULT_MODEL_SIZE,
    batch_size: int = 16,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    align_words: bool = False,
    start_time: Optional[Any] = None,
    end_time: Optional[Any] = None,
    auto_fallback: bool = True,
) -> dict[str, Any]:
    """
    Transcribe audio with segment/word timestamps relative to the original file.
    Unknown word timestamps remain None. The top-level start_time/end_time
    describe the extraction window used by the selected transcription pass.

    When `auto_fallback` is True (default) and the transcription returns fewer
    than 3 words, automatically retries with a larger model (up to large-v2 /
    large-v3) and a narrowed time window, which improves recognition on
    difficult or quiet segments.

    Args:
        file_path: Path to audio/video file
        language: Language code (e.g., "zh", "en"), None for auto-detection
        model_size: Model size (tiny/base/small/medium/large-v2/large-v3)
        batch_size: Batch size for inference
        compute_type: Compute type (float16/int8)
        align_words: Whether to run word-level alignment
        start_time: Optional start timestamp
        end_time: Optional end timestamp
        auto_fallback: Auto-retry with larger model when result is sparse

    Returns:
        Dictionary with transcription result
    """
    # Validate dependencies
    wx = _get_whisperx()
    if wx is None:
        return {"error": "whisperx is not installed. From the repository root, run: bash scripts/install_profiles.sh audio"}

    torch_mod = _get_torch()
    if torch_mod is None:
        return {"error": "torch is not installed. From the repository root, run: bash scripts/install_profiles.sh audio"}

    # Validate file
    error = validate_file(file_path)
    if error:
        return {"error": error}

    # Check audio stream exists
    if not _has_audio_stream(file_path):
        return {"error": f"No audio stream found in the input file: {file_path}. "
                         "This file may be a silent video or contain only video tracks."}

    try:
        seg_start = parse_timestamp(start_time)
        seg_end = parse_timestamp(end_time)

        # --- First pass: use the requested model ---
        output = await _transcribe_single_pass(
            file_path=file_path,
            language=language,
            model_size=model_size,
            batch_size=batch_size,
            compute_type=compute_type,
            align_words=align_words,
            seg_start=seg_start,
            seg_end=seg_end,
        )

        # If first pass errored, return immediately (no fallback for errors)
        if "error" in output:
            return output

        # --- Auto-fallback logic ---
        if auto_fallback:
            word_count = _count_transcribed_words(output.get("text", ""))
            if word_count < _MIN_WORD_THRESHOLD:
                fallback_models = _get_fallback_models(model_size)
                fallback_attempts = []

                for fb_model in fallback_models:
                    # Narrow the time window slightly to reduce noise for the larger model
                    fb_start, fb_end = _narrow_time_window(seg_start, seg_end)

                    fb_output = await _transcribe_single_pass(
                        file_path=file_path,
                        language=language or output.get("language"),
                        model_size=fb_model,
                        batch_size=batch_size,
                        compute_type=compute_type,
                        align_words=align_words,
                        seg_start=fb_start,
                        seg_end=fb_end,
                    )

                    fallback_attempts.append({
                        "model": fb_model,
                        "word_count": _count_transcribed_words(fb_output.get("text", "")),
                        "had_error": "error" in fb_output,
                    })

                    if "error" in fb_output:
                        # Model unavailable or OOM — try next
                        continue

                    fb_word_count = _count_transcribed_words(fb_output.get("text", ""))
                    if fb_word_count >= _MIN_WORD_THRESHOLD:
                        # Fallback succeeded — return the better result
                        fb_output["fallback_info"] = {
                            "original_model": model_size,
                            "original_word_count": word_count,
                            "fallback_model": fb_model,
                            "fallback_word_count": fb_word_count,
                            "reason": f"Original transcription had only {word_count} word(s); "
                                      f"auto-escalated to {fb_model}",
                            "attempts": fallback_attempts,
                        }
                        return fb_output

                # All fallbacks exhausted — return whichever had the most words
                # (could still be the original)
                best = output
                best_wc = word_count
                # Re-run fallbacks to find the best (we didn't store fb_outputs, so
                # just note in the output)
                output["fallback_info"] = {
                    "original_model": model_size,
                    "original_word_count": word_count,
                    "fallback_attempted": True,
                    "fallback_succeeded": False,
                    "reason": f"Original transcription had only {word_count} word(s); "
                              f"tried {[a['model'] for a in fallback_attempts]} but none exceeded threshold",
                    "attempts": fallback_attempts,
                }

        return output

    except Exception as e:
        if torch_mod is not None:
            try:
                if isinstance(e, torch_mod.cuda.OutOfMemoryError):
                    clear_model_cache()
                    return {
                        "error": "GPU out of memory. Try: 1) Reduce batch_size, 2) Use smaller model (base/small), 3) Use compute_type=int8"
                    }
            except Exception:
                pass
        return {"error": f"Transcription failed: {str(e)}"}


# Initialize MCP server


async def handle_transcribe_audio(arguments: dict) -> dict:
    result = await transcribe_audio(
        file_path=arguments.get("file_path", ""),
        language=arguments.get("language"),
        model_size=arguments.get("model_size", DEFAULT_MODEL_SIZE),
        batch_size=arguments.get("batch_size", 16),
        compute_type=arguments.get("compute_type", DEFAULT_COMPUTE_TYPE),
        align_words=arguments.get("align_words", False),
        start_time=arguments.get("start_time"),
        end_time=arguments.get("end_time"),
        auto_fallback=arguments.get("auto_fallback", True),
    )
    return _json_sanitize(result)


async def handle_clear_whisperx_cache(arguments: dict) -> dict:
    clear_model_cache()
    return {"success": True}


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="audio-transcription",
        tool_defs_source="_tool_defs/audio.yaml",
        handlers={
            "transcribe_audio": handle_transcribe_audio,
            "clear_whisperx_cache": handle_clear_whisperx_cache,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
