#!/usr/bin/env python3
"""Consolidated OCR MCP Server — single process for all 5 OCR tools."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")

from _base.server_framework import create_mcp_server, run_server

from handler_frame_text import handle_ocr_frame_text
from handler_region_read import handle_ocr_region_read
from handler_table_extract import handle_ocr_table_extract
from handler_layout_analysis import handle_ocr_layout_analysis
from handler_video_timeline import handle_ocr_video_text_timeline

if __name__ == "__main__":
    app = create_mcp_server(
        server_name="ocr-mcp",
        tool_defs_source="_tool_defs/ocr.yaml",
        handlers={
            "ocr_frame_text": handle_ocr_frame_text,
            "ocr_region_read": handle_ocr_region_read,
            "ocr_table_extract": handle_ocr_table_extract,
            "ocr_layout_analysis": handle_ocr_layout_analysis,
            "ocr_video_text_timeline": handle_ocr_video_text_timeline,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
