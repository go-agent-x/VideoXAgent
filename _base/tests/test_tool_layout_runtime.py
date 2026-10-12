"""MCP stdio checks for relocated lightweight services; no model or API calls."""

import asyncio
from datetime import timedelta
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tool_catalog import service_entrypoint, service_info


class ToolLayoutRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def call_service(self, service, tool, arguments, wrapped):
        args = [str(service_entrypoint(service))]
        if wrapped:
            args.insert(0, str(ROOT / "scripts/mcp_runtime.py"))
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile(mode="w+") as errors:
            parameters = StdioServerParameters(
                command=sys.executable, args=["-B", *args], cwd=directory,
                env={"MCP_ALLOW_MODEL_DOWNLOAD": "0"},
            )
            async with asyncio.timeout(30):
                async with stdio_client(parameters, errlog=errors) as (read, write):
                    async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=10)) as client:
                        await client.initialize()
                        listing = await client.list_tools()
                        self.assertEqual({tool.name for tool in listing.tools}, set(service_info(service)["tools"]))
                        result = await client.call_tool(tool, arguments)
                        return json.loads(result.content[0].text)

    async def test_geometry_roundtrip_with_direct_and_wrapped_entrypoints(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                result = await self.call_service("spatial-calc", "calc_point_in_region", {
                    "point": [5, 5], "region": [[0, 0], [10, 0], [10, 10], [0, 10]],
                }, wrapped)
                self.assertTrue(result["success"])
                self.assertTrue(result["is_inside"])
                self.assertEqual(result["region_area_px"], 100.0)

    async def test_subtitle_roundtrip_from_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = await self.call_service("subtitle", "check_subtitle_exists", {
                "video_path": str(Path(directory) / "unavailable-video.mp4"),
            }, True)
        self.assertFalse(result["subtitle_exists"])

    async def test_duration_missing_file_reports_tool_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result = await self.call_service("video-duration", "get_video_duration", {
                "file_path": str(Path(directory) / "unavailable-video.mp4"),
            }, False)
        self.assertFalse(result["success"])
        self.assertIn("File not found", result["error"])

    async def test_vlm_sequence_registration_and_missing_video_without_api(self):
        # Invalid local input must be rejected before frame decoding or VLM calls.
        with tempfile.TemporaryDirectory() as directory:
            for wrapped in (False, True):
                with self.subTest(wrapped=wrapped):
                    result = await self.call_service("vlm-query-sequence", "query_segment_detail", {
                        "video_path": str(Path(directory) / "unavailable-video.mp4"),
                        "query": "Describe this interval",
                        "start_time": 0,
                        "end_time": 1,
                    }, wrapped)
                    self.assertFalse(result["success"])
                    self.assertIn("not found", result["error"].lower())

    async def test_duration_roundtrip_reads_container_metadata(self):
        # A small ISO-BMFF metadata fixture exercises the native duration parser
        # without video codecs, ffmpeg, or a downloaded sample video.
        def atom(kind, payload):
            return struct.pack(">I4s", len(payload) + 8, kind) + payload

        movie_header = struct.pack(">IIIII", 0, 0, 0, 1000, 2500) + bytes(80)
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "metadata.mp4"
            video.write_bytes(atom(b"moov", atom(b"mvhd", movie_header)))
            result = await self.call_service("video-duration", "get_video_duration", {
                "file_path": str(video),
            }, True)
        self.assertTrue(result["success"])
        self.assertEqual(result["duration"], 2.5)


if __name__ == "__main__":
    unittest.main()
