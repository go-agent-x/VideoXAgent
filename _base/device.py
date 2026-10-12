"""Device choices shared by model backends without importing GPU libraries."""

import os


def select_torch_device(torch_module) -> str:
    """Honor the requested MCP device; otherwise keep CUDA/CPU auto-selection."""
    requested = os.environ.get("MCP_DEVICE", "").strip().lower()
    if requested == "cpu":
        return "cpu"
    if torch_module is not None and torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def insightface_device_options(configured_providers: str) -> tuple[list[str], int]:
    """Select installed ONNX providers and the matching InsightFace context."""
    import onnxruntime

    available = set(onnxruntime.get_available_providers())
    if os.environ.get("MCP_DEVICE", "").strip().lower() == "cpu":
        requested = ["CPUExecutionProvider"]
    else:
        requested = [provider.strip() for provider in configured_providers.split(",")]
    providers = [provider for provider in requested if provider in available]
    if not providers and "CPUExecutionProvider" in available:
        providers = ["CPUExecutionProvider"]
    if not providers:
        raise RuntimeError("No configured ONNX execution provider is available")
    context = 0 if "CUDAExecutionProvider" in providers else -1
    return providers, context


def ocr_device_options() -> dict[str, bool]:
    """Forward an explicit device to PaddleOCR 2.x, preserving its default."""
    requested = os.environ.get("OCR_USE_GPU")
    if requested is not None:
        return {"use_gpu": requested.strip().lower() in ("1", "true", "yes", "gpu", "cuda")}
    return {}
