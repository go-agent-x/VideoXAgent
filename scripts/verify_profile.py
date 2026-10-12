#!/usr/bin/env python3
"""Dependency smoke checks without downloading weights or running model inference."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tool_catalog import service_dir


PROFILE_IMPORTS = {
    "base": (
        "mcp", "numpy", "cv2", "PIL", "yaml", "httpx", "requests",
        "scenedetect", "scipy", "sklearn", "skimage", "shapely", "yt_dlp", "aiohttp",
        "anyio", "deepagents", "deepagents_cli.mcp_tools", "langchain",
        "langchain_anthropic", "langchain_google_genai", "langchain_openai", "rich",
    ),
    "vision": (
        "mcp", "numpy", "cv2", "torch", "torchvision", "transformers",
        "sentence_transformers", "insightface", "onnxruntime", "segment_anything",
        "skimage", "hydra", "iopath", "groundingdino", "timm", "sam2",
    ),
    "audio": (
        "mcp", "numpy", "cv2", "torch", "torchaudio", "transformers",
        "faster_whisper", "librosa", "soundfile", "speechbrain",
        "insightface", "onnxruntime", "whisperx", "pyannote.audio",
    ),
    "ocr": ("mcp", "numpy", "cv2", "paddle", "paddleocr", "paddlex"),
    "music": ("mcp", "yaml", "numpy", "shazamio", "acoustid", "pydub"),
}

# These packages perform downloads or write to the home directory merely when
# imported. Presence is checked without executing their module-level code.
PROFILE_SPEC_CHECKS = {"audio": ("panns_inference",)}


def verify_agent_runtime(repo_root: Path | None = None) -> None:
    """Import real Agent entry points without credentials, servers, or inference.

    The CLI imports the graph builder, provider adapters, and private MCP loader
    symbols also used by the demo's lazy imports. Running --help in a subprocess
    additionally exercises its argument parser while isolating signal handlers
    and atexit hooks. Importing the web modules does not create an application or
    execute startup hooks, so no MCP processes or HTTP server are started.
    """
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
    checks = (
        ("video-agent --help", [sys.executable, "-B", str(root / "video-agent.py"), "--help"]),
        ("web demo imports", [sys.executable, "-B", "-c",
                              "import demo.agent_runner; import demo.agent_server"]),
    )
    for label, command in checks:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"{label} exited with {result.returncode}: {detail}")


def verify_frame_diff() -> None:
    """Exercise the frame-diff implementation on tiny in-memory images."""
    source = service_dir("frame-diff") / "frame_diff_utils.py"
    spec = importlib.util.spec_from_file_location("_profile_frame_diff", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    numpy = importlib.import_module("numpy")
    original = numpy.zeros((16, 16), dtype=numpy.uint8)
    changed = original.copy()
    changed[4:12, 4:12] = 255
    same_score, _ = module.compute_ssim(original, original)
    changed_score, _ = module.compute_ssim(original, changed)
    if abs(same_score - 1.0) > 1e-6 or not numpy.isfinite(changed_score) or changed_score >= 1.0:
        raise RuntimeError(f"unexpected SSIM results: same={same_score}, changed={changed_score}")


def install_audio_compatibility_shim() -> None:
    """Mirror the compatibility shim used by the two audio MCP servers."""
    torchaudio = importlib.import_module("torchaudio")
    if not hasattr(torchaudio, "AudioMetaData"):
        @dataclass
        class AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int = 0
            encoding: str = ""

        torchaudio.AudioMetaData = AudioMetaData
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["ffmpeg"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=sorted(PROFILE_IMPORTS))
    parser.add_argument("--torch-backend", choices=("cpu", "cu126"))
    parser.add_argument("--skip-sam2", action="store_true")
    args = parser.parse_args(argv)

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    if args.profile == "audio":
        install_audio_compatibility_shim()
    if args.profile == "ocr":
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    failures: list[str] = []
    for module_name in PROFILE_SPEC_CHECKS.get(args.profile, ()):
        if importlib.util.find_spec(module_name) is None:
            failures.append(module_name)
            print(f"FAIL {module_name}: package not found", file=sys.stderr)
        else:
            print(f"OK   {module_name} (package present; import skipped to avoid side effects)")

    for module_name in PROFILE_IMPORTS[args.profile]:
        if module_name == "sam2" and args.skip_sam2:
            continue
        try:
            importlib.import_module(module_name)
            print(f"OK   {module_name}")
        except Exception as exc:  # import-time native errors matter here
            failures.append(module_name)
            print(f"FAIL {module_name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.profile == "base":
        try:
            verify_agent_runtime()
            print("OK   Agent CLI --help and web demo imports (no inference)")
        except Exception as exc:
            failures.append("Agent runtime")
            print(f"FAIL Agent runtime: {type(exc).__name__}: {exc}", file=sys.stderr)
        try:
            verify_frame_diff()
            print("OK   frame-diff SSIM (identical and changed images)")
        except Exception as exc:
            failures.append("frame-diff SSIM")
            print(f"FAIL frame-diff SSIM: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.torch_backend:
        try:
            torch = importlib.import_module("torch")
            expected_cuda = {"cpu": None, "cu126": "12.6"}[args.torch_backend]
            if torch.version.cuda != expected_cuda:
                raise RuntimeError(
                    f"requested {args.torch_backend}, found torch={torch.__version__}, "
                    f"CUDA build={torch.version.cuda}"
                )
            print(f"OK   torch backend {args.torch_backend}")
        except Exception as exc:
            failures.append("torch backend")
            print(f"FAIL torch backend: {exc}", file=sys.stderr)

    if failures:
        print(f"Profile {args.profile} failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"Profile {args.profile} dependency checks passed (model/API inference not tested)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
