"""Convert tool inputs to JSON-safe UI events without inspecting runtime state."""

import json


def make_tool_start_event(name: str, tool_input: object) -> dict:
    # Framework-injected objects (e.g. ToolRuntime) are execution context.
    # Represent only their type; str/repr could expose state or recurse.
    safe_input = json.loads(json.dumps(
        tool_input,
        ensure_ascii=False,
        default=lambda value: f"<{type(value).__name__}>",
    ))
    input_str = (
        json.dumps(safe_input, ensure_ascii=False)
        if isinstance(safe_input, (dict, list)) else str(safe_input)
    )
    input_display = input_str[:500] + "..." if len(input_str) > 500 else input_str
    return {
        "type": "tool_start",
        "data": {"name": name, "input": input_display, "input_raw": safe_input},
    }
