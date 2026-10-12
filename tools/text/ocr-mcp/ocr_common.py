"""OCR Common — Shared PaddleOCR utilities for the consolidated OCR MCP server."""

import os
import sys
import threading

from _base.device import ocr_device_options
from contextlib import contextmanager

# Suppress PaddleX model source check at import time
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ---------------------------------------------------------------------------
# Monkey-patch: PaddleX 3.4.3 calls AnalysisConfig.set_optimization_level()
# which doesn't exist in Paddle 2.6.2. Add a no-op so it doesn't crash.
# ---------------------------------------------------------------------------
try:
    from paddle.base import libpaddle
    if not hasattr(libpaddle.AnalysisConfig, "set_optimization_level"):
        libpaddle.AnalysisConfig.set_optimization_level = lambda self, level: None
except Exception:
    pass

# ---------------------------------------------------------------------------
# Stdout suppression — PaddleOCR/PaddlePaddle writes debug info to stdout
# (C++ GLOG output) which corrupts the MCP JSON-RPC protocol over stdio.
# Redirect stdout → stderr during all PaddleOCR calls.
# ---------------------------------------------------------------------------
@contextmanager
def _suppress_stdout():
    """Redirect stdout to stderr so PaddleOCR debug output doesn't corrupt MCP stdio protocol."""
    old_stdout_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        yield
    finally:
        os.dup2(old_stdout_fd, sys.stdout.fileno())
        os.close(old_stdout_fd)


# ---------------------------------------------------------------------------
# PaddleOCR engine — lazy singleton (keyed by language)
# ---------------------------------------------------------------------------
_ocr_instances = {}
_ocr_lock = threading.Lock()
_ocr_infer_lock = threading.Lock()  # serialize inference to avoid ClosedResourceError


def init_ocr(lang: str = "ch"):
    """Get or create a PaddleOCR instance for the given language (PaddleOCR 2.x)."""
    if lang not in _ocr_instances:
        with _ocr_lock:
            if lang not in _ocr_instances:
                from paddleocr import PaddleOCR
                with _suppress_stdout():
                    _ocr_instances[lang] = PaddleOCR(
                        use_angle_cls=True,
                        lang=lang,
                        show_log=False,
                        **ocr_device_options(),
                    )
    return _ocr_instances[lang]


# ---------------------------------------------------------------------------
# Helper: 9-grid position label
# ---------------------------------------------------------------------------
def position_label(bbox, img_w, img_h):
    """Compute a 9-grid position label for a bbox within an image."""
    if img_w <= 0 or img_h <= 0:
        return ""
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    h_label = "left" if cx < img_w / 3.0 else ("center" if cx < 2.0 * img_w / 3.0 else "right")
    v_label = "upper" if cy < img_h / 3.0 else ("center" if cy < 2.0 * img_h / 3.0 else "lower")
    if v_label == "center" and h_label == "center":
        return "center"
    elif v_label == "center":
        return f"center-{h_label}"
    elif h_label == "center":
        return f"{v_label}-center"
    return f"{v_label}-{h_label}"


# ---------------------------------------------------------------------------
# Helper: run OCR on an image file
# ---------------------------------------------------------------------------
def run_ocr(image_path: str, lang: str = "ch", det_thresh: float = 0.3):
    """
    Run OCR on a single image file using PaddleOCR 2.x ocr API.

    Returns:
        (success, texts_list, error_message, img_w, img_h)
        texts_list items: {"text", "confidence", "bbox": [x1,y1,x2,y2], "position"}
    """
    if not os.path.isfile(image_path):
        return False, [], f"Image file not found: {image_path}", 0, 0

    try:
        ocr = init_ocr(lang)
        with _ocr_infer_lock, _suppress_stdout():
            results = ocr.ocr(image_path, cls=True)
    except Exception as e:
        return False, [], f"OCR engine error: {str(e)}", 0, 0

    import cv2
    img = cv2.imread(image_path)
    if img is not None:
        img_h, img_w = img.shape[:2]
    else:
        # Fallback: cv2.imread can fail on non-ASCII paths or certain formats,
        # but PaddleOCR (which loads images internally) may still succeed.
        # Use PIL to get dimensions so we never silently return 0x0.
        img_w, img_h = 0, 0
        try:
            from PIL import Image
            with Image.open(image_path) as pil_img:
                img_w, img_h = pil_img.size
        except Exception:
            pass
        if img_w == 0 or img_h == 0:
            import logging
            logging.getLogger(__name__).warning(
                "Could not determine image dimensions for %s; "
                "position labels will be unavailable.", image_path
            )

    texts = []
    # PaddleOCR 2.x returns [[line1, line2, ...]] where each line is (points, (text, confidence))
    if results and results[0]:
        for line in results[0]:
            points, (text_str, confidence) = line
            if confidence < det_thresh:
                continue

            # Convert 4-point polygon to axis-aligned bbox [x1, y1, x2, y2]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = [round(float(min(xs)), 1), round(float(min(ys)), 1),
                    round(float(max(xs)), 1), round(float(max(ys)), 1)]
            pos = position_label(bbox, img_w, img_h) if img_w > 0 else ""

            texts.append({
                "text": str(text_str),
                "confidence": round(float(confidence), 4),
                "bbox": bbox,
                "position": pos,
            })

    return True, texts, "", img_w, img_h


# ---------------------------------------------------------------------------
# Helper: run OCR on numpy array
# ---------------------------------------------------------------------------
def run_ocr_on_array(img_array, lang="ch", det_thresh=0.3):
    """
    Run OCR on an in-memory image (numpy array) using PaddleOCR 2.x ocr API.

    Returns:
        (success, texts_list, error_message)
    """
    if img_array is None or img_array.size == 0:
        return False, [], "Empty image array"

    try:
        ocr = init_ocr(lang)
        with _ocr_infer_lock, _suppress_stdout():
            results = ocr.ocr(img_array, cls=True)
    except Exception as e:
        return False, [], f"OCR engine error: {str(e)}"

    img_h, img_w = img_array.shape[:2]
    texts = []

    # PaddleOCR 2.x returns [[line1, line2, ...]] where each line is (points, (text, confidence))
    if results and results[0]:
        for line in results[0]:
            points, (text_str, confidence) = line
            if confidence < det_thresh:
                continue

            # Convert 4-point polygon to axis-aligned bbox [x1, y1, x2, y2]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = [round(float(min(xs)), 1), round(float(min(ys)), 1),
                    round(float(max(xs)), 1), round(float(max(ys)), 1)]
            pos = position_label(bbox, img_w, img_h)

            texts.append({
                "text": str(text_str),
                "confidence": round(float(confidence), 4),
                "bbox": bbox,
                "position": pos,
            })

    return True, texts, ""
