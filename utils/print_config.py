#!/usr/bin/env python3
"""Print the selected experiment's MCP configuration without credentials."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
REDACTED = "[REDACTED]"
_SENSITIVE_NAME = re.compile(
    r"api[_-]?key|token|secret|password|passwd|pwd|authorization|"
    r"credential|cookie|private[_-]?key|access[_-]?key|signature",
    re.IGNORECASE,
)
_URL = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>\"']+")
_SAFE_PATH = re.compile(r"[\w ./+-]+\Z")
_PUBLIC_ENV_NAMES = {"MAX_TOKENS", "VLM_MAX_TOKENS", "MAX_COMPLETION_TOKENS", "TOKENIZERS_PARALLELISM"}


class ConfigError(ValueError):
    """A configuration error whose message is safe to display."""


def _redact_urls(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        try:
            parts = urlsplit(match.group())
            host = parts.netloc.rsplit("@", 1)[-1]
            if "@" in parts.netloc:
                host = REDACTED + "@" + host
            # Any query parameter can carry a provider-specific credential.
            query = REDACTED if parts.query else ""
            return urlunsplit((parts.scheme, host, parts.path, query,
                               REDACTED if parts.fragment else ""))
        except ValueError:
            return REDACTED

    return _URL.sub(replace, value)


def _display_command(command: str) -> str:
    # A command should be an executable, not a shell expression. Do not expose
    # inline assignments, shell scripts, or credentials embedded in it.
    if _SAFE_PATH.fullmatch(command) and not re.search(r"\s", Path(command).name):
        return shlex.quote(command)
    return REDACTED


def _display_args(args: list[str]) -> str:
    """Keep runtime paths useful; omit arbitrary flags, values and code payloads."""
    displayed = []
    payload_started = False
    for arg in args:
        if not payload_started and arg in ("-u", "-B"):
            displayed.append(arg)
        elif not payload_started and arg.endswith(".py") and _SAFE_PATH.fullmatch(arg):
            displayed.append(shlex.quote(arg))
        else:
            # After the first non-path argument, even a value ending in .py
            # may be a password or part of an inline shell/Python command.
            payload_started = True
            if not displayed or displayed[-1] != REDACTED:
                displayed.append(REDACTED)
    return " ".join(displayed) or "(none)"


def _read_text(path: Path, kind: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ConfigError(f"Cannot read {kind}: {path}") from None


def _experiment_mcp_path(config_path: str) -> Path:
    try:
        import yaml
    except ImportError:
        raise ConfigError(
            "PyYAML is required; run bash scripts/install_profiles.sh base."
        ) from None

    path = Path(config_path).expanduser()
    contents = _read_text(path, "experiment config")
    try:
        config = yaml.safe_load(contents)
    except yaml.YAMLError:
        # Parser errors include the source line, which may contain a secret.
        raise ConfigError(f"Invalid YAML in experiment config: {path}") from None
    if not isinstance(config, dict):
        raise ConfigError(f"Experiment config must be a mapping: {path}")
    selected = config.get("mcp_json_file", ".mcp.json")
    if not isinstance(selected, str) or not selected.strip():
        raise ConfigError("mcp_json_file must be a nonempty path string.")
    mcp_path = Path(selected).expanduser()
    # Match the Agent loader: MCP paths are relative to the repository root,
    # while the --config argument is relative to the caller's working directory.
    return mcp_path if mcp_path.is_absolute() else ROOT / mcp_path


def print_mcp_json(config_path: str | Path | None = None) -> None:
    """Print a redacted MCP summary; raise ConfigError for invalid input."""
    path = Path(config_path) if config_path is not None else ROOT / ".mcp.json"
    contents = _read_text(path, "MCP config")
    try:
        config = json.loads(contents)
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"Invalid JSON in MCP config: {path} (line {error.lineno}, column {error.colno})"
        ) from None
    servers = config.get("mcpServers") if isinstance(config, dict) else None
    if not isinstance(servers, dict):
        raise ConfigError("MCP config must contain an mcpServers mapping.")
    # Validate all entries before printing any of them.
    for info in servers.values():
        if not isinstance(info, dict):
            raise ConfigError("Each MCP server must be a mapping.")
        if not isinstance(info.get("command", ""), str):
            raise ConfigError("MCP command must be a string.")
        args = info.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ConfigError("MCP args must be a list of strings.")
        if not isinstance(info.get("env", {}), dict):
            raise ConfigError("MCP env must be a mapping.")
        if "url" in info and not isinstance(info["url"], str):
            raise ConfigError("MCP url must be a string.")

    print("=" * 60)
    print(f"MCP server configuration: {path}")
    print("=" * 60)
    for name, info in servers.items():
        print(f"\n{name}")
        print(f"   Command: {_display_command(info.get('command', 'N/A'))}")
        print(f"   Arguments: {_display_args(info.get('args', []))}")
        if "url" in info:
            print(f"   URL: {_redact_urls(info['url'])}")
        for key, value in info.get("env", {}).items():
            if key.upper() not in _PUBLIC_ENV_NAMES and _SENSITIVE_NAME.search(key):
                shown = REDACTED
            elif isinstance(value, (dict, list)):
                # env values are normally scalars; never dump nested payloads.
                shown = REDACTED
            else:
                shown = _redact_urls(str(value)) if value is not None and value != "" else "(empty)"
            print(f"   {key}={shown}")
    print("\nAdditional command arguments and credentials are hidden.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Experiment YAML; defaults to the root .mcp.json when omitted")
    args = parser.parse_args(argv)
    try:
        selected = _experiment_mcp_path(args.config) if args.config else None
        print_mcp_json(selected)
    except ConfigError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
