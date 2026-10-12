"""
Shared helpers for VLM MCP servers.

Two responsibilities:

1) `resolve_max_tokens(tool_name, caller_arg)`:
   Decide the per-call ``max_tokens`` for a VLM request based on (in order):
     - explicit ``caller_arg`` from the tool implementation,
     - per-tool override from env ``VLM_TOOL_MAX_TOKENS`` (a JSON dict, optionally
       carrying a ``"default"`` key),
     - generic env ``VLM_MAX_TOKENS``,
     - hard default ``DEFAULT_VLM_MAX_TOKENS = 32768``.

2) `maybe_thinking_off_retry(...)`:
   Detect "thinking bloat" responses (empty content or oversized
   ``completion_tokens``) and, if the original payload had thinking enabled,
   redo the request once with ``enable_thinking=False``. All network errors
   raised by the retry are swallowed; the caller sees a structured result
   describing whether the retry happened and how its tokens should be merged.
"""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Optional

# ----- constants ------------------------------------------------------------

DEFAULT_VLM_MAX_TOKENS = 32768
"""Hard default upper bound for `max_tokens` across every VLM call site."""

THINKING_BLOAT_OUTPUT_TOKENS = 8192
"""``completion_tokens`` strictly greater than this triggers a thinking-off retry."""


# ----- per-tool max_tokens resolution ---------------------------------------

_TOOL_TOKENS_CACHE: Optional[dict] = None


def _load_tool_tokens() -> dict:
    """Parse and cache ``VLM_TOOL_MAX_TOKENS`` env (JSON dict)."""
    global _TOOL_TOKENS_CACHE
    if _TOOL_TOKENS_CACHE is not None:
        return _TOOL_TOKENS_CACHE
    raw = os.environ.get("VLM_TOOL_MAX_TOKENS", "")
    parsed: dict = {}
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                parsed = {str(k): int(v) for k, v in obj.items()
                          if isinstance(v, (int, float, str)) and str(v).strip()}
        except Exception as e:  # noqa: BLE001
            print(
                f"[vlm_thinking_retry] failed to parse VLM_TOOL_MAX_TOKENS={raw!r}: {e}",
                file=sys.stderr, flush=True,
            )
    _TOOL_TOKENS_CACHE = parsed
    return parsed


def resolve_max_tokens(tool_name: str, caller_arg: Optional[int] = None) -> int:
    """Resolve max_tokens for a VLM request. See module docstring for priority."""
    if caller_arg is not None:
        try:
            v = int(caller_arg)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass

    tool_tokens = _load_tool_tokens()
    if tool_name in tool_tokens:
        return int(tool_tokens[tool_name])
    if "default" in tool_tokens:
        return int(tool_tokens["default"])

    env_generic = os.environ.get("VLM_MAX_TOKENS", "").strip()
    if env_generic:
        try:
            v = int(env_generic)
            if v > 0:
                return v
        except ValueError:
            pass

    return DEFAULT_VLM_MAX_TOKENS


# ----- thinking-off retry ---------------------------------------------------

def _is_empty_content(content: Any) -> bool:
    if content is None:
        return True
    return not str(content).strip()


def _is_thinking_bloat(usage: Any, content: Any) -> Optional[str]:
    """Return ``"empty"``/``"bloat"`` if a retry is warranted, else ``None``."""
    if _is_empty_content(content):
        return "empty"
    if isinstance(usage, dict):
        try:
            ct = int(usage.get("completion_tokens", 0))
        except (TypeError, ValueError):
            ct = 0
        if ct > THINKING_BLOAT_OUTPUT_TOKENS:
            return "bloat"
    return None


def _payload_has_thinking_on(payload: dict) -> bool:
    ctk = payload.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        return False
    return bool(ctk.get("enable_thinking", False))


def _empty_retry_result(data: dict) -> dict:
    return {
        "data": data,
        "retried": False,
        "retry_succeeded": False,
        "extra_input_tokens": 0,
        "extra_output_tokens": 0,
        "reason": None,
    }


async def maybe_thinking_off_retry(
    *,
    data: dict,
    payload: dict,
    client: "httpx.AsyncClient",
    url: str,
    headers: dict,
    server_tag: str,
    semaphore: Any = None,
) -> dict:
    """
    Inspect ``data`` (parsed JSON of the first VLM response) and, if it shows
    thinking-bloat symptoms while ``payload`` had thinking enabled, redo the
    request once with ``enable_thinking=False``.

    Returns a dict (see module docstring for fields).
    """
    # Pre-conditions: choices present + thinking was actually requested.
    if not (isinstance(data, dict) and data.get("choices")):
        return _empty_retry_result(data)
    if not _payload_has_thinking_on(payload):
        return _empty_retry_result(data)

    try:
        first_msg = data["choices"][0].get("message", {}) or {}
        content = first_msg.get("content")
    except Exception:  # noqa: BLE001
        return _empty_retry_result(data)
    usage = data.get("usage", {}) or {}

    reason = _is_thinking_bloat(usage, content)
    if reason is None:
        return _empty_retry_result(data)

    # Build retry payload: same model/messages/max_tokens, thinking off.
    retry_payload = dict(payload)
    new_kwargs = dict(payload.get("chat_template_kwargs") or {})
    new_kwargs["enable_thinking"] = False
    retry_payload["chat_template_kwargs"] = new_kwargs

    print(
        f"[{server_tag}] thinking-off retry triggered "
        f"(reason={reason}, completion_tokens={usage.get('completion_tokens', 0)})",
        file=sys.stderr, flush=True,
    )

    extra_in = 0
    extra_out = 0
    final_data = data
    retry_succeeded = False
    final_reason = reason

    try:
        async def _do_post():
            return await client.post(url, json=retry_payload, headers=headers)

        if semaphore is not None:
            async with semaphore:
                resp = await _do_post()
        else:
            resp = await _do_post()
        resp.raise_for_status()
        data2 = resp.json()

        usage2 = data2.get("usage", {}) or {}
        extra_in = int(usage2.get("prompt_tokens", 0) or 0)
        extra_out = int(usage2.get("completion_tokens", 0) or 0)

        if isinstance(data2, dict) and data2.get("choices"):
            c2 = (data2["choices"][0].get("message", {}) or {}).get("content")
            if not _is_empty_content(c2):
                final_data = data2
                retry_succeeded = True
            else:
                # still empty: keep original data, mark reason
                final_reason = "empty"
        else:
            final_reason = "empty"
    except Exception as e:  # noqa: BLE001
        print(
            f"[{server_tag}] thinking-off retry failed: {type(e).__name__}: {e}",
            file=sys.stderr, flush=True,
        )
        final_reason = "error"

    return {
        "data": final_data,
        "retried": True,
        "retry_succeeded": retry_succeeded,
        "extra_input_tokens": extra_in,
        "extra_output_tokens": extra_out,
        "reason": final_reason,
    }


# ----- chat_template_kwargs gating -----------------------------------------

# OpenAI / Anthropic / Google official series reject the vLLM/SGLang-only
# ``chat_template_kwargs`` extension with HTTP 400 ``Unknown parameter``.
# Operational guideline: if a new model is observed returning the same 400,
# add its lowercase prefix to this tuple.
_OPENAI_LIKE_PREFIXES: tuple[str, ...] = (
    "o1", "o3", "o4",
    "gpt-", "gpt4", "gpt5",
    "chatgpt",
    "claude-", "claude3", "claude4",
    "gemini-", "gemini1", "gemini2", "gemini3",
)


def _model_supports_chat_template_kwargs(model_name: str) -> bool:
    """Return False for OpenAI/Anthropic/Google official series."""
    if not model_name:
        return False
    name = model_name.strip().lower()
    return not name.startswith(_OPENAI_LIKE_PREFIXES)


def build_chat_template_kwargs(model_name: str) -> Optional[dict]:
    """Return the payload-ready ``chat_template_kwargs`` dict, or ``None`` if
    the field must be omitted from the request entirely.

    Rules:
      * If ``VLM_ENABLE_THINKING`` is not truthy -> return ``None`` (omit field).
        ``enable_thinking=False`` is the default behaviour of supported open
        backends, so omitting the field is functionally equivalent and avoids
        provoking 400 responses from gateways that don't whitelist it.
      * If the model is OpenAI-like (see ``_OPENAI_LIKE_PREFIXES``) -> return
        ``None`` (omit field) regardless of the env var.
      * Otherwise -> ``{"enable_thinking": True}``.
    """
    enable = os.environ.get("VLM_ENABLE_THINKING", "false").strip().lower() == "true"
    if not enable:
        return None
    if not _model_supports_chat_template_kwargs(model_name):
        return None
    return {"enable_thinking": True}


# ----- shared HTTP retry policy --------------------------------------------

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
"""HTTP status codes that warrant an inner-loop retry across all VLM MCP servers."""

MAX_VLM_RETRIES_DEFAULT = 6
"""Default per-call retry budget. Designed to ride out o3 429 windows
(~30s) while staying under the 240s httpx client timeout."""


def get_max_vlm_retries() -> int:
    """Read ``VLM_MAX_RETRIES`` env with positive-int validation; else default."""
    raw = os.environ.get("VLM_MAX_RETRIES", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return MAX_VLM_RETRIES_DEFAULT


def _parse_retry_after(value: str) -> Optional[float]:
    """Parse the numeric-seconds form of ``Retry-After``.

    HTTP-date form is ignored (the project's gateway returns numeric seconds).
    """
    if not value:
        return None
    s = value.strip()
    try:
        v = float(s)
        if v >= 0:
            return v
    except ValueError:
        pass
    return None


def compute_retry_sleep(
    attempt: int,
    response: Any = None,
    *,
    base: float = 1.0,
    cap: float = 30.0,
) -> float:
    """Backoff in seconds for a retry decision.

    Strategy:
      * If ``response`` carries a parseable ``Retry-After`` header, use it
        (plus small jitter) capped at ``cap``.
      * Otherwise exponential backoff: ``base * 2 ** (attempt-1)``, capped at
        ``cap``, plus uniform jitter ``[0, 0.5)``.

    With the defaults and ``MAX_VLM_RETRIES_DEFAULT=6`` the worst-case wait
    sequence is roughly 1, 2, 4, 8, 16, 30 (seconds) — total ~61s — well
    within a 240s client timeout.

    The ``cap`` can be overridden at runtime via the ``VLM_RETRY_SLEEP_CAP``
    env var (positive float, in seconds). This is useful against backends
    whose ``Retry-After`` window exceeds 30s (e.g. o3 commonly returns
    60–180s); when unset the caller-supplied ``cap`` is preserved exactly.
    """
    raw_cap = os.environ.get("VLM_RETRY_SLEEP_CAP", "").strip()
    if raw_cap:
        try:
            v = float(raw_cap)
            if v > 0:
                cap = v
        except ValueError:
            pass
    retry_after: Optional[float] = None
    if response is not None:
        try:
            headers = getattr(response, "headers", None)
            if headers is not None:
                retry_after = _parse_retry_after(headers.get("Retry-After", "") or "")
        except Exception:  # noqa: BLE001
            retry_after = None
    if retry_after is not None:
        return min(cap, retry_after + random.uniform(0, 0.5))
    backoff = min(cap, base * (2 ** max(0, attempt - 1)))
    return backoff + random.uniform(0, 0.5)


# ----- shared VLM concurrency cap ------------------------------------------

DEFAULT_MAX_CONCURRENT_VLM = 5
"""Default per-process concurrent in-flight VLM HTTP calls.

Combined with ``experiment_config.params.workers`` and the number of VLM MCP
tools, this caps how many concurrent requests a single MCP-server subprocess
will issue. There is no cross-process coordination — set this lower (e.g. 1
or 2) when running against tight RPM/TPM gateways like o3.
"""


def get_max_concurrent_vlm() -> int:
    """Read ``VLM_MAX_CONCURRENT`` env with positive-int validation; else default.

    Resolved once at module import time by each MCP server so the semaphore
    capacity is fixed for the life of the subprocess.
    """
    raw = os.environ.get("VLM_MAX_CONCURRENT", "").strip()
    if raw:
        try:
            v = int(raw)
            if v >= 1:
                return v
        except ValueError:
            pass
    return DEFAULT_MAX_CONCURRENT_VLM
