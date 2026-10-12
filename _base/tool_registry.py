"""Tool Registry - Centralized tool definition and routing.

This module provides a centralized registry for tool definitions and handlers,
enabling decoupling of tool descriptions (YAML) from implementations (Python).

Each ``ToolRegistry`` instance holds its own isolated state, so multiple MCP
servers can coexist without overwriting each other's registrations.  The
module-level helper functions delegate to a shared ``_default_registry``
instance, preserving full backward compatibility with existing callers.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from mcp.types import Tool


def _check_type(value: Any, expected_type: str) -> bool:
    """Check if value matches expected JSON Schema type."""
    if expected_type == "null":
        return value is None

    # bool is a subclass of int in Python, so isinstance(True, int) is True.
    # Check bool first to prevent booleans from passing integer/number checks.
    if expected_type in ("integer", "number"):
        if isinstance(value, bool):
            return False
        if expected_type == "integer":
            return isinstance(value, int)
        return isinstance(value, (int, float))

    if expected_type == "boolean":
        return isinstance(value, bool)

    type_map = {
        "string": str,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(expected_type)
    if expected:
        return isinstance(value, expected)

    return True


class ToolRegistry:
    """Isolated registry for tool definitions and handlers.

    Each instance maintains its own ``_tool_defs`` and ``_tool_handlers``
    dicts, so concurrent MCP servers can each own a separate registry.
    """

    def __init__(self) -> None:
        self._tool_defs: Dict[str, Dict] = {}
        self._tool_handlers: Dict[str, Callable] = {}

    # -- loading ----------------------------------------------------------

    def load_tool_defs(self, defs_dir: Path) -> Dict[str, Dict]:
        """Load all YAML tool definitions from *defs_dir*.

        Clears any previously loaded definitions in this registry first.

        Raises:
            FileNotFoundError: If *defs_dir* does not exist.
            yaml.YAMLError: If YAML parsing fails.
        """
        self._tool_defs = {}

        if not defs_dir.exists():
            raise FileNotFoundError(
                f"Tool definitions directory not found: {defs_dir}"
            )

        for yaml_file in sorted(defs_dir.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    defs = yaml.safe_load(f)
                    if defs and isinstance(defs, dict):
                        self._tool_defs.update(defs)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Failed to parse {yaml_file}: {e}")

        return self._tool_defs

    def load_tool_defs_from_file(self, yaml_path: Path) -> Dict[str, Dict]:
        """Load tool definitions from a single YAML file (additive)."""
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Tool definitions file not found: {yaml_path}"
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            defs = yaml.safe_load(f)
            if defs and isinstance(defs, dict):
                self._tool_defs.update(defs)

        return self._tool_defs

    # -- MCP helpers ------------------------------------------------------

    def build_mcp_tools(self) -> List[Tool]:
        """Convert tool definitions to MCP ``Tool`` objects."""
        tools = []
        for name, tdef in self._tool_defs.items():
            if not isinstance(tdef, dict):
                continue

            description = tdef.get("description", "")
            input_schema = tdef.get("input_schema", {"type": "object"})

            backend = tdef.get("backend", "unknown")
            full_description = f"[{backend}] {description}"

            tools.append(
                Tool(
                    name=name,
                    description=full_description,
                    inputSchema=input_schema,
                )
            )
        return tools

    # -- handler registration ---------------------------------------------

    def register_handler(self, tool_name: str, handler: Callable) -> None:
        """Register a handler function for *tool_name*."""
        self._tool_handlers[tool_name] = handler

    def route_tool_call(self, tool_name: str) -> Callable:
        """Return the handler for *tool_name*.

        Raises:
            ValueError: If no handler is registered.
        """
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            raise ValueError(f"No handler registered for tool: {tool_name}")
        return handler

    # -- queries ----------------------------------------------------------

    def filter_tool_defs(self, keep_names: set) -> None:
        """Remove definitions whose names are not in *keep_names*."""
        self._tool_defs = {
            k: v for k, v in self._tool_defs.items() if k in keep_names
        }

    def get_tool_def(self, tool_name: str) -> Optional[Dict]:
        """Return the definition for *tool_name*, or ``None``."""
        return self._tool_defs.get(tool_name)

    def validate_arguments(
        self, tool_name: str, arguments: dict
    ) -> Tuple[bool, Optional[str]]:
        """Basic JSON-Schema validation (required fields + types)."""
        tdef = self.get_tool_def(tool_name)
        if not tdef:
            return False, f"Unknown tool: {tool_name}"

        schema = tdef.get("input_schema", {})

        required = schema.get("required", [])
        for field in required:
            if field not in arguments:
                return False, f"Missing required field: {field}"

        properties = schema.get("properties", {})
        for field, value in arguments.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type and not _check_type(value, expected_type):
                    return False, (
                        f"Field '{field}' has wrong type, "
                        f"expected {expected_type}"
                    )

        return True, None

    def list_all_tools(self) -> Dict[str, Dict]:
        """Return a shallow copy of all registered definitions."""
        return self._tool_defs.copy()

    def get_tool_backend(self, tool_name: str) -> str:
        """Return the backend type string for *tool_name*."""
        tdef = self.get_tool_def(tool_name)
        return tdef.get("backend", "unknown") if tdef else "unknown"

    def get_tool_media_type(self, tool_name: str) -> str:
        """Return the media type string for *tool_name*."""
        tdef = self.get_tool_def(tool_name)
        return tdef.get("media_type", "none") if tdef else "none"

    def clear(self) -> None:
        """Clear all definitions and handlers (useful for testing)."""
        self._tool_defs = {}
        self._tool_handlers = {}


# ── Default instance & backward-compatible module-level functions ────────

_default_registry = ToolRegistry()


def load_tool_defs(defs_dir: Path) -> Dict[str, Dict]:
    """Load all YAML tool definitions from directory (default registry)."""
    return _default_registry.load_tool_defs(defs_dir)


def load_tool_defs_from_file(yaml_path: Path) -> Dict[str, Dict]:
    """Load tool definitions from a single YAML file (default registry)."""
    return _default_registry.load_tool_defs_from_file(yaml_path)


def build_mcp_tools() -> List[Tool]:
    """Convert tool definitions to MCP Tool objects (default registry)."""
    return _default_registry.build_mcp_tools()


def register_handler(tool_name: str, handler: Callable) -> None:
    """Register a handler function for a tool (default registry)."""
    _default_registry.register_handler(tool_name, handler)


def filter_tool_defs(keep_names: set) -> None:
    """Remove tool definitions not in *keep_names* (default registry)."""
    _default_registry.filter_tool_defs(keep_names)


def get_tool_def(tool_name: str) -> Optional[Dict]:
    """Get tool definition by name (default registry)."""
    return _default_registry.get_tool_def(tool_name)


def route_tool_call(tool_name: str) -> Callable:
    """Get the handler function for a tool (default registry)."""
    return _default_registry.route_tool_call(tool_name)


def validate_arguments(tool_name: str, arguments: dict) -> Tuple[bool, Optional[str]]:
    """Validate arguments against tool's input_schema (default registry)."""
    return _default_registry.validate_arguments(tool_name, arguments)


def list_all_tools() -> Dict[str, Dict]:
    """Get all registered tool definitions (default registry)."""
    return _default_registry.list_all_tools()


def clear_registry() -> None:
    """Clear all registered tools and handlers (default registry)."""
    _default_registry.clear()


def get_tool_backend(tool_name: str) -> str:
    """Get the backend type for a tool (default registry)."""
    return _default_registry.get_tool_backend(tool_name)


def get_tool_media_type(tool_name: str) -> str:
    """Get the media type for a tool (default registry)."""
    return _default_registry.get_tool_media_type(tool_name)
