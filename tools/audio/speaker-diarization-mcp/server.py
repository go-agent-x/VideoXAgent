#!/usr/bin/env python3
"""Speaker Diarization MCP Server — Transcription with speaker identification."""
import asyncio
import gc
import json
import os
import subprocess
import sys
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

# ---------------------------------------------------------------------------
# Compatibility shim: pyannote.audio 3.x uses the deprecated `use_auth_token`
# keyword in hf_hub_download(), which was removed in huggingface_hub >= 0.24.
# Monkey-patch hf_hub_download so that `use_auth_token` is silently mapped to
# `token`, allowing the two libraries to work together.
# ---------------------------------------------------------------------------
try:
    import huggingface_hub.file_download as _hf_dl
    _original_hf_hub_download = _hf_dl.hf_hub_download

    def _patched_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            token_val = kwargs.pop("use_auth_token")
            if token_val is not None and "token" not in kwargs:
                kwargs["token"] = token_val
        return _original_hf_hub_download(*args, **kwargs)

    _hf_dl.hf_hub_download = _patched_hf_hub_download
    import huggingface_hub
    huggingface_hub.hf_hub_download = _patched_hf_hub_download
except Exception:
    pass

# Check for torch and whisperx availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# ---------------------------------------------------------------------------
# Compatibility shim: torchaudio >= 2.3 removed several APIs that pyannote.audio
# and whisperx still reference. Inject stubs before importing those libraries.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Compatibility shim: PyTorch 2.6+ defaults weights_only=True in torch.load,
# but pyannote/whisperx checkpoints use omegaconf types that aren't in the
# default allowlist. Patch torch.load to default weights_only=False.
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE and torch is not None:
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs or kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

try:
    import torchaudio
    # AudioMetaData was removed in torchaudio 2.3+
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
    # list_audio_backends was removed in torchaudio 2.3+
    if not hasattr(torchaudio, "list_audio_backends"):
        def _list_audio_backends():
            return ["ffmpeg"]
        torchaudio.list_audio_backends = _list_audio_backends
except ImportError:
    pass

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    whisperx = None


# ---------------------------------------------------------------------------
# Configuration from _base/config.py
# ---------------------------------------------------------------------------
_whisperx_cfg = get_config("whisperx")
DEFAULT_MODEL_SIZE = _whisperx_cfg["model_size"]
DEFAULT_COMPUTE_TYPE = _whisperx_cfg["compute_type"]
HF_TOKEN = _whisperx_cfg["hf_token"]

# Supported audio/video formats
SUPPORTED_FORMATS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"
}

# Global model cache
_model_cache: dict[str, Any] = {}
_align_model_cache: dict[str, Any] = {}
_diarize_model_cache: Optional[Any] = None
_diarize_backend: Optional[str] = None  # "pyannote" or "fallback"


def get_device() -> str:
    return select_torch_device(torch if TORCH_AVAILABLE else None)


def get_compute_type(device: str, requested_type: str) -> str:
    if device == "cpu":
        return "int8"
    return requested_type


def format_timestamp(seconds: float) -> str:
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
            data = json.loads(result.stdout)
            return len(data.get("streams", [])) > 0
    except Exception:
        pass
    return False


def validate_file(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    if not os.path.isfile(file_path):
        return f"Not a file: {file_path}"
    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        return f"Unsupported format: {suffix}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
    return None


def load_model_cached(model_size: str, device: str, compute_type: str):
    cache_key = f"{model_size}_{device}_{compute_type}"
    if cache_key not in _model_cache:
        _model_cache[cache_key] = whisperx.load_model(
            model_size, device, compute_type=compute_type
        )
    return _model_cache[cache_key]


def load_align_model_cached(language_code: str, device: str):
    cache_key = f"{language_code}_{device}"
    if cache_key not in _align_model_cache:
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code, device=device
        )
        _align_model_cache[cache_key] = (model_a, metadata)
    return _align_model_cache[cache_key]


def _is_align_model_cached(language_code: str) -> bool:
    """Check if the wav2vec2 alignment model for this language is already cached locally.
    Returns False if the model would need to be downloaded."""
    try:
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF
        import torchaudio

        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
            if model_name in torchaudio.pipelines.__all__:
                bundle = torchaudio.pipelines.__dict__[model_name]
                cache_dir = os.path.join(
                    os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints"
                )
                return os.path.isfile(os.path.join(cache_dir, bundle._path))
        if language_code in DEFAULT_ALIGN_MODELS_HF:
            # HuggingFace models — conservatively assume not cached
            return False
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Diarization backend: pyannote (requires HF_TOKEN)
# ---------------------------------------------------------------------------
def _try_load_pyannote(hf_token: str, device: str):
    """Try to load pyannote diarization pipeline. Returns model or raises."""
    from whisperx.diarize import DiarizationPipeline
    return DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
        device=device,
    )


# ---------------------------------------------------------------------------
# Diarization backend: MFCC + AgglomerativeClustering (no external models)
# ---------------------------------------------------------------------------
class FallbackDiarizer:
    """Simple speaker diarization using MFCC features + clustering.

    Works entirely offline with librosa + scikit-learn (both already installed).
    Quality is lower than pyannote but functional for basic speaker separation.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def __call__(
        self,
        audio,  # numpy array (float32, mono, 16 kHz)
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        **kwargs,
    ):
        """Run diarization and return a DataFrame compatible with whisperx.assign_word_speakers.

        Matches pyannote semantics: num_speakers is an exact count hint,
        min_speakers/max_speakers are bounds.  When no hint is given the
        cluster count is determined automatically via distance_threshold.
        """
        import numpy as np
        import librosa
        import pandas as pd
        from sklearn.cluster import AgglomerativeClustering

        sr = self.sample_rate
        audio_np = np.asarray(audio, dtype=np.float32)

        # --- Step 1: Voice Activity Detection (energy-based) ---
        frame_length = int(0.025 * sr)  # 25 ms
        hop_length = int(0.010 * sr)    # 10 ms
        energy = librosa.feature.rms(
            y=audio_np, frame_length=frame_length, hop_length=hop_length
        )[0]
        energy_threshold = np.percentile(energy[energy > 0], 20) if np.any(energy > 0) else 0
        speech_mask = energy > energy_threshold

        # Convert frame mask to time segments
        segments = []
        in_speech = False
        seg_start = 0
        for i, is_speech in enumerate(speech_mask):
            t = i * hop_length / sr
            if is_speech and not in_speech:
                seg_start = t
                in_speech = True
            elif not is_speech and in_speech:
                if t - seg_start >= 0.3:  # minimum 300 ms
                    segments.append((seg_start, t))
                in_speech = False
        if in_speech:
            t = len(speech_mask) * hop_length / sr
            if t - seg_start >= 0.3:
                segments.append((seg_start, t))

        if not segments:
            return pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])

        # --- Step 2: Extract MFCC features per segment ---
        embeddings = []
        valid_segments = []
        for start, end in segments:
            s_idx = int(start * sr)
            e_idx = int(end * sr)
            chunk = audio_np[s_idx:e_idx]
            if len(chunk) < frame_length:
                continue
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
            # Use mean + std as a simple embedding
            emb = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
            embeddings.append(emb)
            valid_segments.append((start, end))

        if not valid_segments:
            return pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])

        embeddings = np.array(embeddings)

        # --- Step 3: Cluster segments by speaker ---
        n_seg = len(valid_segments)

        if n_seg <= 1:
            labels = list(range(n_seg))
        elif num_speakers is not None:
            # Exact count requested — use directly (matches pyannote semantics)
            n_clusters = max(2, min(num_speakers, n_seg))
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)
        else:
            # Use distance_threshold to auto-determine cluster count (like pyannote)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.7,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)

            n_found = len(set(labels))

            # Clamp to min_speakers/max_speakers bounds if specified
            if max_speakers is not None and n_found > max_speakers:
                # Re-cluster with fixed n_clusters = max_speakers
                clustering = AgglomerativeClustering(
                    n_clusters=max_speakers,
                    metric="cosine",
                    linkage="average",
                )
                labels = clustering.fit_predict(embeddings)
            elif min_speakers is not None and n_found < min_speakers:
                # Re-cluster with fixed n_clusters = min_speakers (if enough segments)
                target = min(min_speakers, len(valid_segments))
                clustering = AgglomerativeClustering(
                    n_clusters=target,
                    metric="cosine",
                    linkage="average",
                )
                labels = clustering.fit_predict(embeddings)

        # --- Step 4: Build DataFrame ---
        rows = []
        for (start, end), label in zip(valid_segments, labels):
            rows.append({
                "segment": None,
                "label": f"SPEAKER_{label:02d}",
                "speaker": f"SPEAKER_{label:02d}",
                "start": start,
                "end": end,
            })

        return pd.DataFrame(rows)


def get_diarize_model(hf_token: str, device: str):
    """Load diarization model: pyannote if possible, else MFCC fallback."""
    global _diarize_model_cache, _diarize_backend

    if _diarize_model_cache is not None:
        return _diarize_model_cache

    # Try pyannote first (best quality, requires HF_TOKEN)
    if hf_token:
        try:
            _diarize_model_cache = _try_load_pyannote(hf_token, device)
            _diarize_backend = "pyannote"
            print("[speaker-diarization] Using pyannote backend", file=sys.stderr)
            return _diarize_model_cache
        except Exception as e:
            print(f"[speaker-diarization] pyannote failed: {e}, using fallback", file=sys.stderr)

    # Fallback: MFCC + clustering
    _diarize_model_cache = FallbackDiarizer(sample_rate=16000)
    _diarize_backend = "fallback"
    print("[speaker-diarization] Using MFCC+clustering fallback backend", file=sys.stderr)
    return _diarize_model_cache


def clear_model_cache():
    global _model_cache, _align_model_cache, _diarize_model_cache, _diarize_backend
    _model_cache.clear()
    _align_model_cache.clear()
    _diarize_model_cache = None
    _diarize_backend = None
    if TORCH_AVAILABLE:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def transcribe_with_diarization(
    file_path: str,
    language: Optional[str] = None,
    model_size: str = DEFAULT_MODEL_SIZE,
    batch_size: int = 16,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> dict[str, Any]:
    if not WHISPERX_AVAILABLE:
        return {"error": "whisperx is not installed. From the repository root, run: bash scripts/install_profiles.sh audio"}
    if not TORCH_AVAILABLE:
        return {"error": "torch is not installed. From the repository root, run: bash scripts/install_profiles.sh audio"}

    error = validate_file(file_path)
    if error:
        return {"error": error}

    # Check audio stream exists
    if not _has_audio_stream(file_path):
        return {"error": f"No audio stream found in the input file: {file_path}. "
                         "This file may be a silent video or contain only video tracks."}

    try:
        device = get_device()
        compute_type = get_compute_type(device, compute_type)

        # Transcribe
        model = load_model_cached(model_size, device, compute_type)
        audio = whisperx.load_audio(file_path)
        result = model.transcribe(audio, batch_size=batch_size, language=language)
        detected_language = result.get("language", language or "unknown")

        # Align for word-level timestamps (only if model is already cached locally)
        if _is_align_model_cached(detected_language):
            try:
                model_a, metadata = load_align_model_cached(detected_language, device)
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, device,
                    return_char_alignments=False
                )
            except Exception as align_error:
                result["alignment_warning"] = f"Word alignment failed: {str(align_error)}"
        else:
            result["alignment_warning"] = (
                f"Word alignment skipped: model for '{detected_language}' is not cached locally. "
                f"Segment-level timestamps are still available."
            )

        # Speaker diarization
        diarization_note = None
        try:
            diarize_model = get_diarize_model(HF_TOKEN, device)

            diarize_kwargs = {}
            if num_speakers is not None:
                diarize_kwargs["num_speakers"] = num_speakers
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers

            diarize_segments = diarize_model(audio, **diarize_kwargs)

            # Assign speakers to transcript
            result = whisperx.assign_word_speakers(diarize_segments, result)

            if _diarize_backend == "fallback":
                diarization_note = (
                    "Using MFCC+clustering fallback (no HF_TOKEN). "
                    "For better accuracy, set HF_TOKEN environment variable with a "
                    "HuggingFace token that has accepted the pyannote license."
                )

        except Exception as diarize_error:
            result["diarization_warning"] = f"Speaker diarization failed: {str(diarize_error)}"

        # Format output
        segments = []
        full_text = []
        speakers_found = set()

        for seg in result.get("segments", []):
            segment_info = {
                "start_time": seg.get("start", 0),
                "end_time": seg.get("end", 0),
                "start_formatted": format_timestamp(seg.get("start", 0)),
                "end_formatted": format_timestamp(seg.get("end", 0)),
                "text": seg.get("text", "").strip(),
                "speaker": seg.get("speaker", "UNKNOWN")
            }

            if segment_info["speaker"] != "UNKNOWN":
                speakers_found.add(segment_info["speaker"])

            # NOT OUTPUT WORD LEVEL TIMESTAMPS for less context
            
            # if "words" in seg:
            #     segment_info["words"] = [
            #         {
            #             "word": w.get("word", ""),
            #             "start": w.get("start", 0),
            #             "end": w.get("end", 0),
            #             "speaker": w.get("speaker", seg.get("speaker", "UNKNOWN"))
            #         }
            #         for w in seg["words"]
            #     ]

            segments.append(segment_info)
            full_text.append(f"[{segment_info['speaker']}] {segment_info['text']}")

        output = {
            "file_path": file_path,
            "language": detected_language,
            "device": device,
            "model_size": model_size,
            "diarization_backend": _diarize_backend or "none",
            "text": "\n".join(full_text),
            "segments": segments,
            "total_segments": len(segments),
            "speakers": sorted(list(speakers_found)),
            "num_speakers": len(speakers_found)
        }

        if diarization_note:
            output["diarization_note"] = diarization_note
        if "alignment_warning" in result:
            output["alignment_warning"] = result["alignment_warning"]
        if "diarization_warning" in result:
            output["diarization_warning"] = result["diarization_warning"]

        return output

    except torch.cuda.OutOfMemoryError:
        clear_model_cache()
        return {
            "error": "GPU out of memory. Try: 1) Reduce batch_size, 2) Use smaller model, 3) Use compute_type=int8"
        }
    except Exception as e:
        return {"error": f"Transcription with diarization failed: {str(e)}"}


# Initialize MCP server


async def handle_speaker_diarize(arguments: dict) -> dict:
    return await transcribe_with_diarization(
        file_path=arguments.get("file_path", ""),
        language=arguments.get("language"),
        model_size=arguments.get("model_size", DEFAULT_MODEL_SIZE),
        batch_size=arguments.get("batch_size", 16),
        compute_type=arguments.get("compute_type", DEFAULT_COMPUTE_TYPE),
        num_speakers=arguments.get("num_speakers"),
        min_speakers=arguments.get("min_speakers"),
        max_speakers=arguments.get("max_speakers"),
    )


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="speaker-diarization",
        tool_defs_source="_tool_defs/audio.yaml",
        handlers={
            "speaker_diarize": handle_speaker_diarize,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
