#!/usr/bin/env python3
"""Rewrite MCP commands so each server uses its pre-installed profile venv."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


# This module is also invoked directly by install_profiles.sh.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tool_catalog import (
    profile_for_server as catalog_profile_for_server,
    service_entrypoint,
    service_ids,
)


PROFILES = ("base", "vision", "audio", "ocr", "music")

# Keep previously supported configurations for services whose implementations
# are not in this checkout. They are not catalog entries or migration targets.
HISTORICAL_SERVER_PROFILES = {
    "video-event-timeline": "base",
}

DEVICE_ENV_VARS = {
    "GDINO_DEVICE", "SAM_DEVICE", "SAM2_DEVICE", "DEPTH_DEVICE",
    "EMOTION_DEVICE", "GAZE_DEVICE",
}
SERVICE_DEVICE_VARS = {
    "grounding-dino": {"GDINO_DEVICE"},
    "auto-segment": {"SAM_DEVICE"},
    "prompt-segment": {"SAM_DEVICE"},
    "object-track": {"SAM2_DEVICE"},
    "obj-count-sequence": {"SAM2_DEVICE", "GDINO_DEVICE"},
    "motion-analyze": {"SAM2_DEVICE"},
    "spatial-analyze": {"SAM2_DEVICE", "GDINO_DEVICE"},
    "interaction-recognition": {"GDINO_DEVICE"},
    "segment-deep-query": {"GDINO_DEVICE"},
    "depth-estimation": {"DEPTH_DEVICE"},
    "emotion-analysis": {"EMOTION_DEVICE"},
    "gaze-estimation": {"GAZE_DEVICE"},
}
INSIGHTFACE_SERVERS = {
    "face-cluster", "gaze-estimation", "emotion-analysis", "active-speaker-detection",
}
OCR_SERVERS = {"ocr", "segment-deep-query"}


def profile_for_server(server_name: str) -> str:
    """Return the runtime profile for an MCP server name."""
    name = server_name.removesuffix("-mcp")
    if name in service_ids():
        return catalog_profile_for_server(name)
    if name in HISTORICAL_SERVER_PROFILES:
        return HISTORICAL_SERVER_PROFILES[name]
    raise ValueError(
        f"MCP server {server_name!r} has no dependency profile; "
        "add it to tools/catalog.json"
    )


def _display_path(path: Path, root: Path) -> str:
    """Prefer a portable path relative to the repository root."""
    # Keep venv interpreter symlinks: resolving them selects the base Python
    # and bypasses the environment's installed dependencies.
    absolute_path = Path(os.path.abspath(path))
    absolute_root = Path(os.path.abspath(root))
    try:
        value = absolute_path.relative_to(absolute_root).as_posix()
        return f"./{value}"
    except ValueError:
        return str(absolute_path)


def _script_argument_index(args: list[Any]) -> int | None:
    """Find Python's script slot without treating option values as scripts."""
    index = 0
    while index < len(args):
        argument = args[index]
        if not isinstance(argument, str):
            return None
        if argument == "--":
            return index + 1 if index + 1 < len(args) else None
        if argument == "-" or argument.startswith(("-m", "-c")):
            return None
        if argument in ("-W", "-X", "--check-hash-based-pycs"):
            if index + 1 >= len(args) or not isinstance(args[index + 1], str):
                return None
            index += 2
            continue
        if argument.startswith(("-W", "-X", "--check-hash-based-pycs=")):
            index += 1
            continue
        if argument.startswith("-"):
            # Unknown options, help/version and custom launcher modes are not
            # safe to reinterpret. Ordinary Python flags may be combined.
            if len(argument) < 2 or any(flag not in "bBdEiIOPqRsSuvx" for flag in argument[1:]):
                return None
            index += 1
            continue
        return index
    return None


def _rewrite_entrypoint(args: list[Any], service_name: str, repo_root: Path) -> None:
    """Wrap only a recognized local server in Python's actual script slot."""
    index = _script_argument_index(args)
    if index is None or not isinstance(args[index], str):
        return

    def absolute(argument: str) -> str:
        path = Path(argument)
        return os.path.abspath(path if path.is_absolute() else repo_root / path)

    runtime_path = repo_root / "scripts" / "mcp_runtime.py"
    wrapped = absolute(args[index]) == os.path.abspath(runtime_path)
    server_index = index + 1 if wrapped else index
    if server_index >= len(args) or not isinstance(args[server_index], str):
        return
    current_path = service_entrypoint(service_name, repo_root)
    legacy_path = repo_root / f"{service_name}-mcp" / "server.py"
    known_entrypoints = {os.path.abspath(path) for path in (legacy_path, current_path)}
    if absolute(args[server_index]) not in known_entrypoints:
        return
    args[server_index] = _display_path(current_path, repo_root)
    if not wrapped:
        args.insert(index, _display_path(runtime_path, repo_root))


def rewrite_config(
    config: dict[str, Any],
    venv_root: Path,
    repo_root: Path,
    *,
    device: str | None = None,
    ocr_device: str | None = None,
    disable_servers: Iterable[str] | None = None,
    device_profiles: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a copied MCP config with profile-specific Python commands."""
    if device not in (None, "cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if ocr_device not in (None, "cpu", "gpu", "cuda"):
        raise ValueError("ocr_device must be cpu or gpu (cuda is an alias for gpu)")
    scopes = set(device_profiles) if device_profiles is not None else {"vision", "audio"}
    if not scopes <= {"vision", "audio"}:
        raise ValueError("device_profiles may contain only vision or audio")
    if device_profiles is not None and device is None:
        raise ValueError("device_profiles requires device")
    known_services = set(service_ids())
    disabled = {name.removesuffix("-mcp") for name in (disable_servers or ())}
    unknown_disabled = disabled - known_services - set(HISTORICAL_SERVER_PROFILES)
    if unknown_disabled:
        raise ValueError(f"Unknown MCP service(s) to disable: {', '.join(sorted(unknown_disabled))}")

    rewritten = json.loads(json.dumps(config))
    servers = rewritten.get("mcpServers")
    if not isinstance(servers, dict):
        raise ValueError("configuration must contain an object named 'mcpServers'")

    for server_name in list(servers):
        if server_name.removesuffix("-mcp") in disabled:
            del servers[server_name]

    for server_name, server in servers.items():
        if not isinstance(server, dict):
            raise ValueError(f"mcpServers.{server_name} must be an object")
        profile = profile_for_server(server_name)
        previous_bin = Path(server.get("command", "python")).parent
        python_path = venv_root / profile / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        server["command"] = _display_path(python_path, repo_root)

        # Historical missing services retain their configured entrypoint paths.
        args = server.get("args")
        service_name = server_name.removesuffix("-mcp")
        if service_name in known_services and isinstance(args, list):
            _rewrite_entrypoint(args, service_name, repo_root)

        if service_name == "segment-deep-query":
            # The base-profile composite can invoke detection in its isolated
            # vision environment without importing Torch/Paddle into base.
            env = server.setdefault("env", {})
            if not isinstance(env, dict):
                raise ValueError(f"mcpServers.{server_name}.env must be an object")
            vision_python = venv_root / "vision" / python_path.parent.name / python_path.name
            existing = env.get("GDINO_PYTHON")
            if existing is not None and not isinstance(existing, str):
                raise ValueError("segment-deep-query GDINO_PYTHON must be an executable path string")
            old_roots = (repo_root / ".venvs", venv_root)
            if previous_bin.parent.name in PROFILES:
                old_roots += (previous_bin.parent.parent,)
            managed_paths = {
                os.path.abspath(path if path.is_absolute() else repo_root / path)
                for path in (old / "vision" / python_path.parent.name / python_path.name for old in old_roots)
            }
            existing_path = Path(existing) if isinstance(existing, str) and existing else None
            if existing_path is not None and not existing_path.is_absolute():
                existing_path = repo_root / existing_path
            if not existing or (existing_path is not None and os.path.abspath(existing_path) in managed_paths):
                env["GDINO_PYTHON"] = _display_path(vision_python, repo_root)

        # segment-deep-query uses local vision and OCR backends even though its
        # launcher is assigned to base. OCR's Paddle device is independent.
        device_profile = "vision" if service_name == "segment-deep-query" else profile
        apply_device = device is not None and device_profile in scopes and service_name in known_services
        apply_ocr = ocr_device is not None and service_name in OCR_SERVERS
        if apply_device or apply_ocr:
            env = server.setdefault("env", {})
            if not isinstance(env, dict):
                raise ValueError(f"mcpServers.{server_name}.env must be an object")
            if apply_device:
                env["MCP_DEVICE"] = device
                keys = SERVICE_DEVICE_VARS.get(service_name, set()) | (DEVICE_ENV_VARS & env.keys())
                for key in keys:
                    env[key] = device
                if device == "cpu" and (service_name in INSIGHTFACE_SERVERS or "INSIGHTFACE_PROVIDERS" in env):
                    env["INSIGHTFACE_PROVIDERS"] = "CPUExecutionProvider"
            if apply_ocr:
                env["OCR_USE_GPU"] = "0" if ocr_device == "cpu" else "1"

        env = server.get("env")
        if isinstance(env, dict) and isinstance(env.get("PATH"), str):
            profile_bin = python_path.parent
            replacement = _display_path(profile_bin, repo_root)
            # Replace whole PATH entries, including an already-generated profile.
            # Keep unrelated system/custom paths and never resolve Python symlinks.
            known_bins = {repo_root / "venv" / "bin"}
            for root in (repo_root / ".venvs", venv_root):
                known_bins.update(root / name / profile_bin.name for name in PROFILES)
            if previous_bin.parent.name in PROFILES:
                known_bins.add(previous_bin)

            def normalized(path: Path) -> str:
                return os.path.abspath(path if path.is_absolute() else repo_root / path)

            known_paths = {normalized(path) for path in known_bins}
            remaining = [
                entry for entry in env["PATH"].split(os.pathsep)
                if not entry or normalized(Path(entry)) not in known_paths
            ]
            env["PATH"] = os.pathsep.join([replacement, *remaining])

    return rewritten


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an MCP config that uses the pre-installed profile environments."
    )
    parser.add_argument(
        "--source",
        help="source MCP JSON (default: existing output, then repository .mcp.profiles.json)",
    )
    parser.add_argument("--output", default=".mcp.profiles.json", help="generated MCP JSON")
    parser.add_argument("--venv-root", default=".venvs", help="directory containing profile venvs")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="set local model runtime devices")
    parser.add_argument(
        "--device-profile", action="append", choices=("vision", "audio"),
        help="limit --device to selected profiles (repeatable; default: vision and audio)",
    )
    parser.add_argument(
        "--ocr-device", choices=("cpu", "gpu", "cuda"),
        help="set PaddleOCR runtime device (cuda is an alias for gpu)",
    )
    parser.add_argument(
        "--disable-server", action="append", default=[], metavar="NAME",
        help="remove a known service from the output (repeatable)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate and print the mapping without writing the output file",
    )
    return parser


def resolve_source(source: str | None, output: Path, repo_root: Path) -> Path:
    """Select an existing configuration without replacing invalid user output."""
    if source:
        selected = Path(source)
        if not selected.is_absolute():
            selected = repo_root / selected
    elif output.exists() or output.is_symlink():
        selected = output
    else:
        selected = repo_root / ".mcp.profiles.json"
    if not selected.is_file():
        raise ValueError(
            f"MCP source is not an existing file: {selected}. "
            "Provide an existing configuration with --source PATH."
        )
    return selected


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    venv_root = Path(args.venv_root)
    if not venv_root.is_absolute():
        venv_root = repo_root / venv_root

    try:
        source = resolve_source(args.source, output, repo_root)
        with source.open(encoding="utf-8") as handle:
            config = json.load(handle)
        rewritten = rewrite_config(
            config, venv_root, repo_root,
            device=args.device, ocr_device=args.ocr_device,
            disable_servers=args.disable_server, device_profiles=args.device_profile,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"configure_mcp_profiles: {exc}", file=sys.stderr)
        return 2

    servers = rewritten["mcpServers"]
    counts = {profile: 0 for profile in PROFILES}
    for server_name in servers:
        counts[profile_for_server(server_name)] += 1

    if args.check:
        for profile in PROFILES:
            print(f"{profile}: {counts[profile]} server(s)")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(rewritten, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {output}")
    for profile in PROFILES:
        print(f"  {profile}: {counts[profile]} server(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
