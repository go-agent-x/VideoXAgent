#!/usr/bin/env python3
"""Audio Event Detection MCP Server — Detect non-speech audio events using PANNs."""
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

from _base.paths import PROJECT_ROOT
from _base.device import select_torch_device
from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response

# ── Business logic (from audio_event_detection_server.py) ──

import importlib.util
import json
import os
import subprocess
import tempfile
import threading
import traceback
from typing import Optional

import numpy as np


# =========================================================================
# Embedded AudioSet labels (offline-safe)
# =========================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from audioset_labels import AUDIOSET_LABELS, get_category_by_index  # noqa: E402

# =========================================================================
# Ensure panns_inference can import without network access.
# The package's config.py tries to download a CSV at import time.
# We pre-write the CSV from embedded data so it finds a valid file.
# =========================================================================
_PANNS_CSV_PATH = Path.home() / "panns_data" / "class_labels_indices.csv"

def _ensure_audioset_csv():
    """Write AudioSet labels CSV to the path panns_inference expects.

    The MID column uses sequential placeholders (/m/00000, /m/00001, ...) instead of
    real AudioSet MIDs (e.g. /m/09x0r). This is safe because panns_inference only uses
    the display_name column (via config.labels / config.classes_num) for inference.
    The MID-derived variables (ids, id_to_ix, ix_to_id) in config.py are populated
    but never imported by inference.py or by this server.
    """
    if _PANNS_CSV_PATH.is_file() and _PANNS_CSV_PATH.stat().st_size > 1000:
        return  # Already valid
    _PANNS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["index,mid,display_name"]
    for i, label in enumerate(AUDIOSET_LABELS):
        lines.append(f'{i},/m/{i:05d},"{label}"')
    _PANNS_CSV_PATH.write_text("\n".join(lines) + "\n")
    print(f"[audio-event-detection] Wrote {len(AUDIOSET_LABELS)} AudioSet labels to {_PANNS_CSV_PATH}",
          file=sys.stderr)

# Write CSV before any panns_inference import
_ensure_audioset_csv()

# =========================================================================
# Supported input formats
# =========================================================================
SUPPORTED_FORMATS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".ts", ".mpg", ".mpeg", ".3gp",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac",
}

# Speech-related class indices in AudioSet (for filtering)
_SPEECH_CLASS_INDICES = set(range(0, 8))  # Speech, Male/Female/Child speech, Conversation, Narration, Babbling, Speech synthesizer

# =========================================================================
# Backend state
# =========================================================================
_model_lock = threading.Lock()
_audio_tagger = None       # AudioTagging model wrapper or None
_sound_event_detector = None  # SoundEventDetection model wrapper or None
_backend = "uninitialized"  # "panns" | "spectral" | "uninitialized"


def _try_init_panns():
    """
    Attempt to load PANNs models. Tries panns_inference first,
    then falls back to direct model loading if import chain fails.
    Returns True on success, False on failure.
    """
    global _audio_tagger, _sound_event_detector

    # Determine checkpoint paths
    checkpoint_at = os.environ.get("PANNS_CHECKPOINT") or None
    checkpoint_sed = None

    if checkpoint_at and os.path.isfile(checkpoint_at):
        # User provided a specific checkpoint — use it for AudioTagging
        # Try to find SED checkpoint in the same directory
        cp_dir = os.path.dirname(checkpoint_at)
        sed_candidates = [
            os.path.join(cp_dir, "Cnn14_DecisionLevelMax.pth"),
            os.path.join(cp_dir, "Cnn14_DecisionLevelMax_mAP=0.385.pth"),
        ]
        for c in sed_candidates:
            if os.path.isfile(c):
                checkpoint_sed = c
                break
    else:
        # Try default paths
        default_dir = Path.home() / "panns_data"
        at_path = default_dir / "Cnn14_mAP=0.431.pth"
        sed_path = default_dir / "Cnn14_DecisionLevelMax.pth"

        if at_path.is_file() and at_path.stat().st_size > 3e8:
            checkpoint_at = str(at_path)
        if sed_path.is_file() and sed_path.stat().st_size > 3e8:
            checkpoint_sed = str(sed_path)

    if not checkpoint_at:
        print("[audio-event-detection] No PANNs checkpoint found. "
              "Set PANNS_CHECKPOINT env var or place Cnn14_mAP=0.431.pth in ~/panns_data/",
              file=sys.stderr)
        return False

    # --- Try loading via panns_inference (the normal way) ---
    try:
        from panns_inference import AudioTagging, SoundEventDetection

        import torch
        device = select_torch_device(torch)

        _audio_tagger = AudioTagging(checkpoint_path=checkpoint_at, device=device)
        if checkpoint_sed:
            _sound_event_detector = SoundEventDetection(checkpoint_path=checkpoint_sed, device=device)
        else:
            print("[audio-event-detection] SED checkpoint not found, timeline tool will use windowed tagging",
                  file=sys.stderr)

        print(f"[audio-event-detection] PANNs loaded via panns_inference on {device}", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"[audio-event-detection] panns_inference import failed ({exc}), trying direct loading...",
              file=sys.stderr)

    # --- Fallback: direct model loading via importlib (bypass __init__.py) ---
    try:
        import torch
        device = select_torch_device(torch)

        # Find panns_inference package directory
        panns_pkg_dir = None
        for p in sys.path:
            candidate = os.path.join(p, "panns_inference")
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "models.py")):
                panns_pkg_dir = candidate
                break
        # Also check the venv
        if not panns_pkg_dir:
            venv_candidate = str(PROJECT_ROOT / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "panns_inference")
            if os.path.isdir(venv_candidate):
                panns_pkg_dir = venv_candidate

        if not panns_pkg_dir:
            print("[audio-event-detection] Cannot find panns_inference package for direct loading",
                  file=sys.stderr)
            return False

        # Import models.py directly (no __init__.py, no config.py download)
        models_spec = importlib.util.spec_from_file_location(
            "panns_models", os.path.join(panns_pkg_dir, "models.py"))
        models_module = importlib.util.module_from_spec(models_spec)
        models_spec.loader.exec_module(models_module)

        utils_spec = importlib.util.spec_from_file_location(
            "panns_utils", os.path.join(panns_pkg_dir, "pytorch_utils.py"))
        utils_module = importlib.util.module_from_spec(utils_spec)
        utils_spec.loader.exec_module(utils_module)

        Cnn14 = models_module.Cnn14
        Cnn14_DecisionLevelMax = getattr(models_module, "Cnn14_DecisionLevelMax", None)
        move_data_to_device = utils_module.move_data_to_device

        # Create AudioTagging wrapper
        class _DirectAudioTagging:
            def __init__(self, checkpoint_path, device):
                self.device = device
                self.model = Cnn14(sample_rate=32000, window_size=1024,
                                   hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                                   classes_num=527)
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                self.model.load_state_dict(checkpoint['model'])
                if 'cuda' in str(device):
                    self.model.to(device)
                self.model.eval()

            def inference(self, audio):
                audio = move_data_to_device(audio, self.device)
                with torch.no_grad():
                    output_dict = self.model(audio, None)
                return (output_dict['clipwise_output'].data.cpu().numpy(),
                        output_dict['embedding'].data.cpu().numpy())

        _audio_tagger = _DirectAudioTagging(checkpoint_at, device)

        # Create SED wrapper if checkpoint exists
        if checkpoint_sed and Cnn14_DecisionLevelMax:
            class _DirectSED:
                def __init__(self, checkpoint_path, device):
                    self.device = device
                    self.model = Cnn14_DecisionLevelMax(
                        sample_rate=32000, window_size=1024, hop_size=320,
                        mel_bins=64, fmin=50, fmax=14000, classes_num=527)
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    self.model.load_state_dict(checkpoint['model'])
                    if 'cuda' in str(device):
                        self.model.to(device)
                    self.model.eval()

                def inference(self, audio):
                    audio = move_data_to_device(audio, self.device)
                    with torch.no_grad():
                        output_dict = self.model(input=audio, mixup_lambda=None)
                    return output_dict['framewise_output'].data.cpu().numpy()

            _sound_event_detector = _DirectSED(checkpoint_sed, device)

        print(f"[audio-event-detection] PANNs loaded via direct import on {device}", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"[audio-event-detection] Direct PANNs loading failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False


def _init_backend():
    """Initialise the backend (called once, thread-safe)."""
    global _backend
    if _backend != "uninitialized":
        return
    with _model_lock:
        if _backend != "uninitialized":
            return
        if _try_init_panns():
            _backend = "panns"
        else:
            _backend = "spectral"
            print("[audio-event-detection] Using spectral analysis fallback", file=sys.stderr)


# =========================================================================
# Audio loading via ffmpeg
# =========================================================================
def _load_audio_segment(file_path: str, start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        sample_rate: int = 32000) -> tuple[np.ndarray, float]:
    """
    Extract audio from a video/audio file using ffmpeg, return as numpy float32 at target sample rate.
    Returns (audio_array, duration_seconds).
    """
    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = ["ffmpeg", "-y"]
        if start_time is not None and start_time > 0:
            cmd.extend(["-ss", str(start_time)])
        cmd.extend(["-i", file_path])
        if end_time is not None:
            duration = end_time - (start_time or 0)
            if duration > 0:
                cmd.extend(["-t", str(duration)])
        cmd.extend([
            "-vn",                    # no video
            "-ac", "1",               # mono
            "-ar", str(sample_rate),  # target sample rate
            "-c:a", "pcm_s16le",     # 16-bit PCM
            tmp_path,
        ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {(result.stderr or '')[-300:]}")

        from scipy.io import wavfile
        sr, data = wavfile.read(tmp_path)

        # Convert to float32 in [-1, 1]
        audio = data.astype(np.float32)
        if data.dtype == np.int16:
            audio /= 32768.0
        elif data.dtype == np.int32:
            audio /= 2147483648.0

        duration_sec = len(audio) / sr
        return audio, duration_sec
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =========================================================================
# PANNs inference
# =========================================================================
def _panns_detect_events(audio: np.ndarray, top_k: int = 10,
                         min_confidence: float = 0.1,
                         include_speech: bool = False) -> list[dict]:
    """Run PANNs AudioTagging and return categorized events."""
    input_audio = audio[np.newaxis, :]  # shape: (1, num_samples)
    clipwise_output, _ = _audio_tagger.inference(input_audio)
    probs = clipwise_output[0]  # shape: (527,)

    events = []
    for idx in range(len(probs)):
        score = float(probs[idx])
        if score < min_confidence:
            continue
        if not include_speech and idx in _SPEECH_CLASS_INDICES:
            continue

        label = AUDIOSET_LABELS[idx] if idx < len(AUDIOSET_LABELS) else f"class_{idx}"
        category = get_category_by_index(idx)

        if not include_speech and category == "Speech":
            continue

        events.append({
            "label": label,
            "score": round(score, 4),
            "category": category,
            "class_index": idx,
        })

    events.sort(key=lambda x: x["score"], reverse=True)
    return events[:top_k]


def _panns_detect_timeline_sed(audio: np.ndarray, duration_sec: float,
                               time_resolution: float = 2.0,
                               top_k_per_segment: int = 5,
                               min_confidence: float = 0.2,
                               include_speech: bool = False,
                               offset: float = 0.0) -> dict:
    """Run PANNs SoundEventDetection (frame-level) and build a timeline."""
    input_audio = audio[np.newaxis, :]
    framewise_output = _sound_event_detector.inference(input_audio)
    frame_probs = framewise_output[0]  # shape: (time_frames, 527)

    num_frames = frame_probs.shape[0]
    frame_duration = duration_sec / num_frames if num_frames > 0 else 0.01
    frames_per_bin = max(1, int(time_resolution / frame_duration))

    timeline = []
    num_bins = (num_frames + frames_per_bin - 1) // frames_per_bin
    class_durations = {}

    for b in range(num_bins):
        start_frame = b * frames_per_bin
        end_frame = min((b + 1) * frames_per_bin, num_frames)
        avg_probs = np.mean(frame_probs[start_frame:end_frame], axis=0)

        bin_start = offset + start_frame * frame_duration
        bin_end = offset + end_frame * frame_duration

        events = []
        for idx in np.argsort(avg_probs)[::-1]:
            score = float(avg_probs[idx])
            if score < min_confidence:
                break
            if not include_speech and idx in _SPEECH_CLASS_INDICES:
                continue

            label = AUDIOSET_LABELS[idx] if idx < len(AUDIOSET_LABELS) else f"class_{idx}"
            category = get_category_by_index(idx)

            if not include_speech and category == "Speech":
                continue

            events.append({"label": label, "score": round(score, 4), "category": category})

            key = label
            if key not in class_durations:
                class_durations[key] = {"duration": 0.0, "category": category}
            class_durations[key]["duration"] += (bin_end - bin_start)

            if len(events) >= top_k_per_segment:
                break

        if events:
            timeline.append({
                "start_time": round(bin_start, 3),
                "end_time": round(bin_end, 3),
                "events": events,
            })

    dominant_events = sorted(
        [{"label": k, "total_duration": round(v["duration"], 3), "category": v["category"]}
         for k, v in class_durations.items()],
        key=lambda x: x["total_duration"], reverse=True
    )
    return {"timeline": timeline, "dominant_events": dominant_events[:20]}


def _panns_detect_timeline_windowed(audio: np.ndarray, duration_sec: float,
                                    sample_rate: int = 32000,
                                    time_resolution: float = 2.0,
                                    top_k_per_segment: int = 5,
                                    min_confidence: float = 0.2,
                                    include_speech: bool = False,
                                    offset: float = 0.0) -> dict:
    """Windowed AudioTagging as fallback when SED model is not available."""
    samples_per_bin = int(time_resolution * sample_rate)
    num_bins = max(1, (len(audio) + samples_per_bin - 1) // samples_per_bin)

    timeline = []
    class_durations = {}

    for b in range(num_bins):
        start_sample = b * samples_per_bin
        end_sample = min((b + 1) * samples_per_bin, len(audio))
        chunk = audio[start_sample:end_sample]

        if len(chunk) < sample_rate // 4:  # min 0.25s
            continue

        bin_start = offset + start_sample / sample_rate
        bin_end = offset + end_sample / sample_rate

        events = _panns_detect_events(chunk, top_k_per_segment, min_confidence, include_speech)

        for ev in events:
            key = ev["label"]
            if key not in class_durations:
                class_durations[key] = {"duration": 0.0, "category": ev["category"]}
            class_durations[key]["duration"] += (bin_end - bin_start)

        if events:
            # Strip class_index from timeline events for cleaner output
            timeline.append({
                "start_time": round(bin_start, 3),
                "end_time": round(bin_end, 3),
                "events": [{"label": e["label"], "score": e["score"], "category": e["category"]}
                           for e in events],
            })

    dominant_events = sorted(
        [{"label": k, "total_duration": round(v["duration"], 3), "category": v["category"]}
         for k, v in class_durations.items()],
        key=lambda x: x["total_duration"], reverse=True
    )
    return {"timeline": timeline, "dominant_events": dominant_events[:20]}


# =========================================================================
# Spectral analysis fallback
# =========================================================================
def _spectral_detect_events(audio: np.ndarray, sample_rate: int = 32000,
                            top_k: int = 10) -> list[dict]:
    """Simple spectral analysis fallback when PANNs is unavailable."""
    from scipy.signal import stft

    events = []

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.005:
        events.append({"label": "Silence", "score": round(1.0 - rms * 100, 4), "category": "Environment"})
    elif rms > 0.1:
        events.append({"label": "Loud sound", "score": round(min(rms * 5, 0.99), 4), "category": "Other"})

    zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))

    f, t, Zxx = stft(audio, fs=sample_rate, nperseg=2048, noverlap=1024)
    magnitude = np.abs(Zxx)
    power_spectrum = np.mean(magnitude ** 2, axis=1)
    total = float(np.sum(power_spectrum)) + 1e-10

    low_band = power_spectrum[f < 500]
    mid_band = power_spectrum[(f >= 500) & (f < 4000)]
    high_band = power_spectrum[f >= 4000]

    low_ratio = float(np.sum(low_band)) / total
    mid_ratio = float(np.sum(mid_band)) / total
    high_ratio = float(np.sum(high_band)) / total

    if low_ratio > 0.6 and zcr < 0.15:
        events.append({"label": "Music-like (bass-heavy)", "score": round(low_ratio, 4), "category": "Music"})
    if mid_ratio > 0.5 and 0.1 < zcr < 0.3:
        events.append({"label": "Speech-like", "score": round(mid_ratio, 4), "category": "Speech"})
    if high_ratio > 0.3:
        events.append({"label": "High-frequency content (hiss/noise)", "score": round(high_ratio, 4),
                        "category": "Environment"})

    if len(power_spectrum) > 0:
        peak_idx = np.argmax(power_spectrum)
        peak_freq = float(f[peak_idx])
        peak_ratio = float(power_spectrum[peak_idx]) / (total / len(power_spectrum))
        if peak_ratio > 5.0:
            events.append({
                "label": f"Tonal sound (dominant freq ~{peak_freq:.0f} Hz)",
                "score": round(min(peak_ratio / 20, 0.95), 4),
                "category": "Music" if 100 < peak_freq < 4000 else "Other",
            })

    frame_energies = np.sum(magnitude ** 2, axis=0)
    if len(frame_energies) > 1:
        energy_std = float(np.std(frame_energies))
        energy_mean = float(np.mean(frame_energies)) + 1e-10
        cv = energy_std / energy_mean
        if cv > 1.5:
            events.append({"label": "Impulsive/bursty sounds", "score": round(min(cv / 5, 0.9), 4),
                            "category": "Other"})
        elif cv < 0.3 and rms > 0.01:
            events.append({"label": "Steady/continuous sound", "score": round(1.0 - cv, 4),
                            "category": "Environment"})

    events.sort(key=lambda x: x["score"], reverse=True)
    return events[:top_k]


def _spectral_detect_timeline(audio: np.ndarray, duration_sec: float,
                              sample_rate: int = 32000,
                              time_resolution: float = 2.0,
                              offset: float = 0.0) -> dict:
    """Spectral analysis timeline fallback."""
    samples_per_bin = int(time_resolution * sample_rate)
    num_bins = max(1, (len(audio) + samples_per_bin - 1) // samples_per_bin)

    timeline = []
    class_durations = {}

    for b in range(num_bins):
        start_sample = b * samples_per_bin
        end_sample = min((b + 1) * samples_per_bin, len(audio))
        chunk = audio[start_sample:end_sample]

        if len(chunk) < 256:
            continue

        bin_start = offset + start_sample / sample_rate
        bin_end = offset + end_sample / sample_rate
        bin_dur = bin_end - bin_start

        events = _spectral_detect_events(chunk, sample_rate, top_k=5)

        for ev in events:
            key = ev["label"]
            if key not in class_durations:
                class_durations[key] = {"duration": 0.0, "category": ev["category"]}
            class_durations[key]["duration"] += bin_dur

        if events:
            timeline.append({
                "start_time": round(bin_start, 3),
                "end_time": round(bin_end, 3),
                "events": events,
            })

    dominant_events = sorted(
        [{"label": k, "total_duration": round(v["duration"], 3), "category": v["category"]}
         for k, v in class_durations.items()],
        key=lambda x: x["total_duration"], reverse=True
    )
    return {"timeline": timeline, "dominant_events": dominant_events[:20]}


# =========================================================================
# Core functions
# =========================================================================
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


def detect_audio_events(file_path: str, start_time: Optional[float] = None,
                        end_time: Optional[float] = None, top_k: int = 10,
                        min_confidence: float = 0.1,
                        include_speech: bool = False) -> dict:
    """Detect audio events in a segment of a video/audio file."""
    _init_backend()

    if not os.path.isfile(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    if not _has_audio_stream(file_path):
        return {"success": False, "error": f"No audio stream found in the input file: {file_path}. "
                "This file may be a silent video or contain only video tracks."}

    try:
        audio, dur = _load_audio_segment(file_path, start_time, end_time)
    except Exception as e:
        return {"success": False, "error": f"Failed to load audio: {e}"}

    if len(audio) < 100:
        return {"success": False, "error": "Audio segment too short"}

    try:
        if _backend == "panns":
            events = _panns_detect_events(audio, top_k, min_confidence, include_speech)
        else:
            events = _spectral_detect_events(audio, top_k=top_k)

        categories = {}
        for ev in events:
            cat = ev.get("category", "Other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ev["label"])

        return {
            "success": True,
            "backend": _backend,
            "duration": round(dur, 3),
            "analyzed_range": {
                "start_time": start_time or 0,
                "end_time": end_time if end_time else "file_end",
            },
            "events": events,
            "categories_summary": categories,
        }
    except Exception as e:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def detect_audio_event_timeline(file_path: str, start_time: Optional[float] = None,
                                end_time: Optional[float] = None,
                                time_resolution: float = 2.0,
                                top_k_per_segment: int = 5,
                                min_confidence: float = 0.2,
                                include_speech: bool = False) -> dict:
    """Detect audio events across a video/audio segment timeline."""
    _init_backend()

    if not os.path.isfile(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    if not _has_audio_stream(file_path):
        return {"success": False, "error": f"No audio stream found in the input file: {file_path}. "
                "This file may be a silent video or contain only video tracks."}

    if time_resolution < 0.5:
        time_resolution = 0.5

    try:
        audio, dur = _load_audio_segment(file_path, start_time, end_time)
    except Exception as e:
        return {"success": False, "error": f"Failed to load audio: {e}"}

    if len(audio) < 100:
        return {"success": False, "error": "Audio segment too short"}

    offset = start_time or 0.0

    try:
        if _backend == "panns":
            if _sound_event_detector is not None:
                result = _panns_detect_timeline_sed(
                    audio, dur, time_resolution, top_k_per_segment,
                    min_confidence, include_speech, offset
                )
            else:
                result = _panns_detect_timeline_windowed(
                    audio, dur, time_resolution=time_resolution,
                    top_k_per_segment=top_k_per_segment,
                    min_confidence=min_confidence,
                    include_speech=include_speech, offset=offset
                )
        else:
            result = _spectral_detect_timeline(
                audio, dur, time_resolution=time_resolution, offset=offset
            )

        categories_seen = set()
        for seg in result["timeline"]:
            for ev in seg.get("events", []):
                categories_seen.add(ev.get("category", "Other"))

        result["success"] = True
        result["backend"] = _backend
        result["duration"] = round(dur, 3)
        result["analyzed_range"] = {
            "start_time": start_time or 0,
            "end_time": end_time if end_time else "file_end",
        }
        result["time_resolution"] = time_resolution
        result["audio_summary"] = {
            "total_segments": len(result["timeline"]),
            "categories_detected": sorted(categories_seen),
        }
        return result
    except Exception as e:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# MCP Server
# =========================================================================


async def handle_detect_audio_events(arguments: dict) -> dict:
    return detect_audio_events(
        file_path=arguments.get("file_path", ""),
        start_time=arguments.get("start_time"),
        end_time=arguments.get("end_time"),
        top_k=int(arguments.get("top_k", 10)),
        min_confidence=float(arguments.get("min_confidence", 0.1)),
        include_speech=bool(arguments.get("include_speech", False)),
    )


async def handle_detect_audio_event_timeline(arguments: dict) -> dict:
    return detect_audio_event_timeline(
        file_path=arguments.get("file_path", ""),
        start_time=arguments.get("start_time"),
        end_time=arguments.get("end_time"),
        time_resolution=float(arguments.get("time_resolution", 2.0)),
        top_k_per_segment=int(arguments.get("top_k_per_segment", 5)),
        min_confidence=float(arguments.get("min_confidence", 0.2)),
        include_speech=bool(arguments.get("include_speech", False)),
    )


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="audio-event-detection",
        tool_defs_source="_tool_defs/audio.yaml",
        handlers={
            "detect_audio_events": handle_detect_audio_events,
            "detect_audio_event_timeline": handle_detect_audio_event_timeline,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
