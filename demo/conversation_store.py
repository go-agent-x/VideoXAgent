#!/usr/bin/env python3
"""Server-side persistence for demo conversations (Recents).

Each conversation is stored as a single JSON file under ``conversations/demo/``
so the demo's chat history survives page refreshes AND server restarts, and is
shared across browsers (unlike the previous browser-only localStorage).

Record shape:
    {
        "id": "c1_1699999999",
        "title": "...",
        "pinned": false,
        "updated_at": "2026-09-15T12:00:00",
        "html": "<...rendered transcript...>",
        "agent_id": "abcd1234"   # runtime hint only; not valid across restarts
    }
"""

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

# Directory holding one JSON file per demo conversation.
REPO_ROOT = Path(__file__).parent.parent.resolve()
STORE_DIR = Path(os.environ.get("CONVERSATIONS_DIR", REPO_ROOT / "conversations")) / "demo"
# Raw agent transcripts (message history + build config) keyed by agent_id.
# Kept separate from the client-facing conversation records so that a
# conversation can be resumed (context replayed) after a server restart.
AGENTS_DIR = STORE_DIR / "_agents"

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_id(conv_id: str) -> str:
    """Validate a conversation id to prevent path traversal."""
    if not conv_id or not _ID_RE.match(conv_id):
        raise ValueError(f"Invalid conversation id: {conv_id!r}")
    return conv_id


def _path_for(conv_id: str) -> Path:
    return STORE_DIR / f"{_safe_id(conv_id)}.json"


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically (temp file + rename) to avoid partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def list_conversations() -> list:
    """Return conversation metadata (no html), newest first.

    Pinned conversations are surfaced first, then by ``updated_at`` descending.
    """
    if not STORE_DIR.exists():
        return []

    items = []
    for fp in STORE_DIR.glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        items.append({
            "id": rec.get("id", fp.stem),
            "title": rec.get("title", "Untitled"),
            "pinned": bool(rec.get("pinned", False)),
            "updated_at": rec.get("updated_at", ""),
        })

    items.sort(key=lambda x: ((1 if x["pinned"] else 0), x["updated_at"]), reverse=True)
    return items


def get_conversation(conv_id: str):
    """Return the full record (incl. html) or None if missing."""
    path = _path_for(conv_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_conversation(conv_id: str, payload: dict) -> dict:
    """Create or update a conversation from an incoming payload."""
    conv_id = _safe_id(conv_id)
    existing = get_conversation(conv_id) or {}

    record = {
        "id": conv_id,
        "title": payload.get("title", existing.get("title", "New chat")),
        "pinned": bool(payload.get("pinned", existing.get("pinned", False))),
        "html": payload.get("html", existing.get("html", "")),
        "agent_id": payload.get("agent_id", existing.get("agent_id")),
        "updated_at": datetime.now().isoformat(),
    }
    _atomic_write(_path_for(conv_id), record)
    return record


def delete_conversation(conv_id: str) -> bool:
    """Delete a conversation file. Returns True if it existed."""
    path = _path_for(conv_id)
    if path.exists():
        path.unlink()
        return True
    return False


# ----- Raw agent transcripts (for resume-after-restart) -----

def save_agent_transcript(agent_id: str, payload: dict) -> None:
    """Persist an agent's message history + build config, keyed by agent_id.

    ``payload`` = {config_path, system_prompt_path, system_prompt_text, messages}
    where ``messages`` is a list of {"role", "content"} dicts.
    """
    agent_id = _safe_id(agent_id)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "agent_id": agent_id,
        "config_path": payload.get("config_path"),
        "system_prompt_path": payload.get("system_prompt_path"),
        "system_prompt_text": payload.get("system_prompt_text"),
        "messages": payload.get("messages", []),
        "updated_at": datetime.now().isoformat(),
    }
    _atomic_write(AGENTS_DIR / f"{agent_id}.json", record)


def load_agent_transcript(agent_id: str):
    """Return a stored agent transcript or None if missing."""
    try:
        path = AGENTS_DIR / f"{_safe_id(agent_id)}.json"
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
