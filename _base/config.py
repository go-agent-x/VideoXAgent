"""Unified configuration layer with schema-based env resolution.

Priority: TOOL_{NAME}__{ENV_VAR} > {ENV_VAR} > schema default
For device fields: TOOL_{NAME}__{ENV_VAR} > MCP_DEVICE > {ENV_VAR} > default.

Usage:
    from _base.config import get_config, pick_api_key, get_env, normalize_vlm_url

    # Schema-based config (VLM server)
    cfg = get_config("vlm")
    url = normalize_vlm_url(cfg["api_url"])
    key = pick_api_key("VLM_API_KEY")

    # Schema-based config with tool-level override
    cfg = get_config("vlm", tool_name="inspect_image")
    # reads TOOL_INSPECT_IMAGE__VLM_API_KEY first, then VLM_API_KEY, then default

    # Single env var (for variables not in any schema)
    acoustid_key = get_env("ACOUSTID_API_KEY")
    depth_model = get_env("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------
# Each schema: { config_key: (ENV_VAR_NAME, default_value, cast_type) }

CONFIG_SCHEMAS: dict[str, dict[str, tuple[str, Any, type]]] = {
    "vlm": {
        "api_url":   ("VLM_API_URL", "", str),
        "api_key":   ("VLM_API_KEY", "", str),
        "model":     ("VLM_MODEL_NAME", "gpt-4.1", str),
        #"model":     ("VLM_MODEL_NAME", "gemini-3.1-pro-preview", str),
        "fallbacks": ("VLM_MODEL_FALLBACKS", "", str),
    },
    "grounding_dino": {
        "checkpoint":  ("GDINO_CHECKPOINT", "models/GroundingDINO/groundingdino_swinb_cogcoor.pth", str),
        "device":      ("GDINO_DEVICE", "cuda", str),
        "bert_path":   ("GDINO_BERT_PATH", "models/bert-base-uncased", str),
    },
    "sam": {
        "model_type": ("SAM_MODEL_TYPE", "vit_h", str),
        "checkpoint": ("SAM_CHECKPOINT", "models/sam_vit_h_4b8939.pth", str),
        "device":     ("SAM_DEVICE", "cuda", str),
    },
    "sam2": {
        "model_cfg":  ("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml", str),
        "checkpoint": ("SAM2_CHECKPOINT", "models/sam2.1_hiera_large.pt", str),
        "device":     ("SAM2_DEVICE", "cuda", str),
    },
    "insightface": {
        "model":     ("INSIGHTFACE_MODEL", "buffalo_l", str),
        "providers": ("INSIGHTFACE_PROVIDERS",
                      "CUDAExecutionProvider,CPUExecutionProvider", str),
    },
    "whisperx": {
        "model_size":   ("WHISPERX_MODEL_SIZE", "large-v3", str),
        "compute_type": ("WHISPERX_COMPUTE_TYPE", "float16", str),
        "hf_token":     ("HF_TOKEN", "", str),
    },
    "emotion": {
        "device":        ("EMOTION_DEVICE", "cuda", str),
        "fer_model":     ("FER_MODEL", "trpakov/vit-face-expression", str),
        "ser_model":     ("SER_MODEL",
                          "speechbrain/emotion-recognition-wav2vec2-IEMOCAP", str),
        "fusion_weight": ("FER_FUSION_WEIGHT", "0.6", float),
    },
    "depth": {
        "model":  ("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf", str),
        "device": ("DEPTH_DEVICE", "cuda", str),
    },
    "gaze": {
        "weights": ("L2CS_WEIGHTS", "models/L2CSNet_gaze360.pkl", str),
        "device":  ("GAZE_DEVICE", "cuda", str),
    },
    "moment": {
        "clip_model": ("CLIP_MODEL", "openai/clip-vit-base-patch32", str),
        # Optional: the service uses its public CLIP backend when no compatible
        # Moment-DETR checkpoint is explicitly configured.
        "checkpoint": ("MOMENT_DETR_CHECKPOINT", "", str),
    },
    "panns": {
        "checkpoint": ("PANNS_CHECKPOINT", "models/Cnn14_mAP=0.431.pth", str),
    },
    "action": {
        "model": ("VIDEOMAE_MODEL", "MCG-NJU/videomae-base-finetuned-kinetics", str),
    },
    "sbert": {
        "model": ("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2", str),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_config(schema_name: str, tool_name: str = "") -> dict[str, Any]:
    """Resolve config dict from a registered schema with tool overrides.

    Priority:
        L1: TOOL_{TOOL_NAME}__{ENV_VAR}  (tool-level override)
        L2: {ENV_VAR}                     (global env)
        L3: schema default

    For device fields, an explicit MCP_DEVICE takes priority over L2 but not L1.

    Args:
        schema_name: Registered schema name (e.g. "vlm", "grounding_dino").
        tool_name: Optional tool name for L1 override lookup.

    Returns:
        Config dict keyed by schema config names.

    Raises:
        KeyError: If schema_name is not registered.
    """
    schema = CONFIG_SCHEMAS[schema_name]
    prefix = f"TOOL_{tool_name.upper()}__" if tool_name else ""

    cfg: dict[str, Any] = {}
    for key, (env_var, default, cast) in schema.items():
        # L2: global env
        val = os.environ.get(env_var)
        # A CLI-selected device overrides stale saved per-backend settings.
        # Explicit TOOL_* overrides below remain authoritative.
        if key == "device" and os.environ.get("MCP_DEVICE") in ("cpu", "cuda"):
            val = os.environ["MCP_DEVICE"]
        # L1: tool-level override
        if prefix:
            tool_val = os.environ.get(f"{prefix}{env_var}")
            if tool_val is not None:
                val = tool_val
        # L3: default
        if val is None:
            try:
                cfg[key] = cast(default)
            except (ValueError, TypeError):
                cfg[key] = default
        else:
            try:
                cfg[key] = cast(val)
            except (ValueError, TypeError):
                cfg[key] = default

    if schema_name == "vlm":
        import logging as _logging
        _logging.getLogger(__name__).info(
            f"[VLM Config] model={cfg.get('model')}, api_url={cfg.get('api_url')}, "
            f"fallbacks={cfg.get('fallbacks', '')[:80]}"
        )

    return cfg


def pick_api_key(env_var: str = "VLM_API_KEY", tool_name: str = "") -> str:
    """Pick a random API key from a comma-separated env var value.

    Supports tool-level override: TOOL_{NAME}__{ENV_VAR} takes precedence.
    Falls back to CONFIG_SCHEMAS default if env var is not set.

    Args:
        env_var: The env var name holding the key(s).
        tool_name: Optional tool name for override lookup.

    Returns:
        A single API key string, or "" if none configured.
    """
    raw = ""
    if tool_name:
        raw = os.environ.get(f"TOOL_{tool_name.upper()}__{env_var}", "")
    if not raw:
        raw = os.environ.get(env_var, "")
    # Fall back to schema default
    if not raw:
        for schema in CONFIG_SCHEMAS.values():
            for _key, (var_name, default, _cast) in schema.items():
                if var_name == env_var and default:
                    raw = str(default)
                    break
            if raw:
                break
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return random.choice(keys) if keys else ""


def get_env(var: str, default: str = "", cast: type = str) -> Any:
    """Read a single env var with optional type cast.

    For env vars not belonging to any schema (e.g. ACOUSTID_API_KEY).

    Args:
        var: Environment variable name.
        default: Default value if not set.
        cast: Type to cast to (str, int, float, bool).

    Returns:
        The resolved value.
    """
    val = os.environ.get(var)
    if val is None:
        try:
            if cast is bool:
                return bool(default)
            return cast(default)
        except (ValueError, TypeError):
            return default
    try:
        if cast is bool:
            return val.lower() in ("1", "true", "yes")
        return cast(val)
    except (ValueError, TypeError):
        return default


def normalize_vlm_url(url: str) -> str:
    """Normalize VLM API URL: add http:// prefix and /v1/chat/completions suffix.

    Args:
        url: Raw URL string.

    Returns:
        Normalized URL ending with /v1/chat/completions.
    """
    if not url:
        return url
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url += "/chat/completions"
    elif not url.endswith("/v1/chat/completions"):
        url = url + "/v1/chat/completions"
    return url


def get_output_dir(env_var: str = "OUTPUT_DIR", default: str = "./output") -> Path:
    """Return the output directory, creating it if needed.

    Args:
        env_var: Environment variable name for the output directory.
        default: Default path if env var is not set.

    Returns:
        Path to the output directory.
    """
    d = Path(os.environ.get(env_var, default))
    d.mkdir(parents=True, exist_ok=True)
    return d
