"""Dependency-free access to MCP service paths and runtime profiles."""

from functools import lru_cache
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "tools" / "catalog.json"


@lru_cache(maxsize=1)
def load_catalog():
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    if catalog.get("version") != 1 or not isinstance(catalog.get("services"), dict):
        raise ValueError(f"Invalid MCP service catalog: {CATALOG_PATH}")
    return catalog["services"]


def service_info(name):
    name = name.removesuffix("-mcp")
    try:
        return load_catalog()[name]
    except KeyError:
        raise ValueError(f"Unknown MCP service: {name}") from None


def service_entrypoint(name, root=PROJECT_ROOT):
    return Path(root) / service_info(name)["entrypoint"]


def service_dir(name, root=PROJECT_ROOT):
    return service_entrypoint(name, root).parent


def profile_for_server(name):
    return service_info(name)["profile"]


def service_ids():
    return tuple(load_catalog())
