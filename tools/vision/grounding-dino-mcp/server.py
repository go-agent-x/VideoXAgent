#!/usr/bin/env python3
"""Grounding DINO MCP Server — local inference replacement for DINO-X API."""

import asyncio
import sys
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))
# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")

from _base.server_framework import create_mcp_server, run_server

from handler_detect import handle_detect
from handler_count import handle_count
from handler_scene_scan import handle_scene_scan
from handler_locate import handle_locate
from handler_inst_segment import handle_instance_segment
from handler_spatial_relation import handle_spatial_relation


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="grounding-dino-mcp",
        tool_defs_source="_tool_defs/grounding_dino.yaml",
        handlers={
            "detect": handle_detect,
            "count": handle_count,
            "scene_scan": handle_scene_scan,
            "locate": handle_locate,
            "instance_segment": handle_instance_segment,
            "spatial_relation": handle_spatial_relation,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
