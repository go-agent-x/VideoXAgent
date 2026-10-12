# _base — Shared Base Utilities for MCP Servers

`_base` provides configuration management, the tool registry, the server framework,
and shared path helpers. Service implementations live under
`tools/<category>/<service>-mcp/`; shared code and tool schemas remain at the root.

## Modules

### `paths.py` — Repository and Service Paths

Resolve shared assets and cross-service imports without assuming that tool
directories are siblings:

```python
from _base.paths import PROJECT_ROOT, service_dir, service_entrypoint

schema_path = PROJECT_ROOT / "_tool_defs" / "temporal.yaml"
detector_dir = service_dir("grounding-dino")
detector_server = service_entrypoint("grounding-dino")
```

The paths and dependency profiles come from [the service catalog](../tools/catalog.json)
through the dependency-free `scripts.tool_catalog` module. `models/`, `data/`, and
`output/` remain at the repository root.

### `config.py` — Configuration Management

Unified, schema-based environment variable resolution with 3-tier priority:

```
TOOL_{TOOL_NAME}__{ENV_VAR}  >  {ENV_VAR}  >  schema default
```

For device fields, `MCP_DEVICE` takes precedence over saved global backend values;
explicit `TOOL_*` overrides remain first. Shared device helpers in `device.py`
also cover automatic Torch selection, available ONNX providers, and the separate
`OCR_USE_GPU` option for PaddleOCR.

**Key functions:**

| Function | Description |
|----------|-------------|
| `get_config(schema, tool_name?)` | Resolve a full config dict from a registered schema (e.g. `"vlm"`, `"grounding_dino"`, `"sam"`) |
| `pick_api_key(env_var, tool_name?)` | Randomly pick one API key from a comma-separated env var (supports load balancing across keys) |
| `get_env(var, default?, cast?)` | Read a single env var with optional type cast, for vars not in any schema |
| `normalize_vlm_url(url)` | Normalize a VLM API URL (add `http://` prefix and `/v1/chat/completions` suffix) |
| `get_output_dir(env_var?, default?)` | Return (and create) the output directory |

**Built-in schemas:** `vlm`, `grounding_dino`, `sam`, `sam2`, `insightface`, `whisperx`, `emotion`

**Example:**

```python
from _base.config import get_config, pick_api_key, normalize_vlm_url

cfg = get_config("vlm", tool_name="inspect_image")
url = normalize_vlm_url(cfg["api_url"])
key = pick_api_key("VLM_API_KEY")
```

### `tool_registry.py` — Tool Definition & Routing

Centralized registry that decouples tool descriptions (YAML) from implementations (Python).

**Key functions:**

| Function | Description |
|----------|-------------|
| `load_tool_defs(defs_dir)` | Load all YAML tool definitions from a directory |
| `build_mcp_tools()` | Convert loaded definitions into MCP `Tool` objects |
| `register_handler(tool_name, handler)` | Register an async handler function for a tool |
| `route_tool_call(tool_name)` | Look up the handler for a tool call |
| `validate_arguments(tool_name, arguments)` | Validate arguments against the tool's input schema |
| `list_all_tools()` | List all registered tool definitions |
| `clear_registry()` | Clear all tools and handlers (useful for testing) |

**YAML tool definition format:**

```yaml
tool_name:
  description: "What the tool does"
  backend: "grounding_dino"          # backend identifier
  media_type: "image"       # image / video / audio / none
  input_schema:
    type: object
    properties:
      image_path:
        type: string
        description: "Absolute path to the image file"
    required:
      - image_path
```

### `server_framework.py` — MCP Server Bootstrap

Reusable framework for creating and running MCP servers over stdio transport.

**Key functions:**

| Function | Description |
|----------|-------------|
| `create_mcp_server(server_name, tool_defs_source, handlers, is_file?)` | Create a fully configured MCP `Server` with tool listing, argument validation, and call routing |
| `run_server(app)` | Run the server with stdio transport |
| `make_success_response(data?, **kwargs)` | Create a standardized `{"success": true, ...}` response |
| `make_error_response(error, details?)` | Create a standardized `{"success": false, "error": ...}` response |

## Usage

A minimal MCP server using `_base`, launched through `scripts/mcp_runtime.py`
to put the repository and service directories on the import path:

```python
import asyncio
from _base.paths import PROJECT_ROOT
from _base.server_framework import create_mcp_server, run_server

async def handle_my_tool(arguments: dict) -> dict:
    # ... tool logic ...
    return {"success": True, "result": "done"}

if __name__ == "__main__":
    app = create_mcp_server(
        server_name="my-tool-mcp",
        tool_defs_source=str(PROJECT_ROOT / "_tool_defs" / "my_tools.yaml"),
        handlers={"my_tool": handle_my_tool},
        is_file=True,
    )
    asyncio.run(run_server(app))
```

## Tests

Run the installation-related tests from the repository root:

```bash
bash scripts/install_profiles.sh base --dev
.venvs/base/bin/python -m pytest \
  _base/tests/test_dependency_profiles.py _base/tests/test_profile_agent_runtime.py
```
