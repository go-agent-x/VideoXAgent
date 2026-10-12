"""Local weight checks for model backends outside Hugging Face."""
import importlib
import os
from pathlib import Path


def allow_model_download():
    return os.environ.get('MCP_ALLOW_MODEL_DOWNLOAD', '0').lower() in ('1', 'true', 'yes', 'y')


def insightface_root(model):
    root = Path(os.environ.get('INSIGHTFACE_HOME', '~/.insightface')).expanduser()
    directory = root / 'models' / model
    if not allow_model_download() and not list(directory.glob('*.onnx')):
        raise FileNotFoundError(f'InsightFace weights missing: {directory}; run scripts/download_models.sh insightface')
    return str(root)


def require_local_weight(path):
    if not allow_model_download() and not Path(path).is_file():
        raise FileNotFoundError(f'Model weight missing in offline mode: {path}; pre-download it first')


def configure_paddle_downloads():
    """Guard PaddleOCR 2.x's download hook; preserve its normal cache layout."""
    module = importlib.import_module('paddleocr.paddleocr')
    if getattr(module.maybe_download, '_mcp_guard', False):
        return
    original = module.maybe_download

    def guarded(directory, url):
        if not allow_model_download():
            for filename in ('inference.pdmodel', 'inference.pdiparams'):
                require_local_weight(Path(directory) / filename)
            return
        return original(directory, url)

    guarded._mcp_guard = True
    module.maybe_download = guarded


def paddle_use_gpu():
    return os.environ.get('OCR_USE_GPU', '0').lower() in ('1', 'true', 'yes')
