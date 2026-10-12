"""Frame detector routing and cleanup checks without models or network access."""

import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "segment_detection_backend_tested",
    ROOT / "tools/composite/segment-deep-query-mcp/detection_backend.py",
)
BACKEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND)


def write_fake_jpeg(path, frame):
    Path(path).write_bytes(b"fake-jpeg")
    return True


def text_result(payload, *, is_error=False):
    return SimpleNamespace(
        isError=is_error,
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
    )


class FakeMCP:
    def __init__(self, response=None):
        self.calls = []
        self.parameters = []
        self.initializations = 0
        self.exits = 0
        self.paths_existed_on_exit = []
        self.response = response
        self.started = asyncio.Event()
        self.block_calls = False
        self.block_startup = False
        self.startup_error = None
        self.clients_active = 0
        self.peak_clients = 0

        fake = self

        @asynccontextmanager
        async def stdio(parameters):
            fake.parameters.append(parameters)
            fake.clients_active += 1
            fake.peak_clients = max(fake.peak_clients, fake.clients_active)
            try:
                yield "read", "write"
            finally:
                fake.paths_existed_on_exit.extend(
                    Path(arguments["image_path"]).exists() for _, arguments in fake.calls
                    if Path(arguments["image_path"]).parent.exists()
                )
                fake.clients_active -= 1
                fake.exits += 1

        class Client:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def initialize(self):
                fake.initializations += 1
                if fake.startup_error:
                    raise fake.startup_error
                if fake.block_startup:
                    await asyncio.Event().wait()

            async def call_tool(self, name, arguments):
                fake.calls.append((name, arguments))
                fake.started.set()
                self.assert_image_present(arguments)
                if fake.block_calls:
                    await asyncio.Event().wait()
                await asyncio.sleep(0)
                if callable(fake.response):
                    return fake.response(len(fake.calls))
                return fake.response or text_result({"success": True, "objects": []})

            @staticmethod
            def assert_image_present(arguments):
                assert Path(arguments["image_path"]).read_bytes() == b"fake-jpeg"

        mcp = ModuleType("mcp")
        mcp.__path__ = []
        mcp.ClientSession = Client
        mcp.StdioServerParameters = lambda **kwargs: SimpleNamespace(**kwargs)
        client = ModuleType("mcp.client")
        client.__path__ = []
        stdio_module = ModuleType("mcp.client.stdio")
        stdio_module.stdio_client = stdio
        self.modules = {"mcp": mcp, "mcp.client": client, "mcp.client.stdio": stdio_module}


class DetectionBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.python = self.root / ".venvs/vision/bin/python"
        self.python.parent.mkdir(parents=True)
        # A symlink models a real venv. The command must retain this path.
        self.python.symlink_to(sys.executable)
        self.cv2 = SimpleNamespace(imwrite=write_fake_jpeg)
        self.enterContext(patch.dict(sys.modules, {"cv2": self.cv2}))
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(patch.object(BACKEND, "_VISION_MCP_SEMAPHORE", asyncio.Semaphore(1)))

    async def test_local_detects_sequentially_and_preserves_metadata(self):
        paths = []
        seen = []

        def detector(**kwargs):
            path = Path(kwargs["image_source"])
            paths.append(path)
            seen.append(kwargs)
            self.assertEqual(path.read_bytes(), b"fake-jpeg")
            return {
                "success": True, "objects": [{"category": "traffic light"}],
                "partial": True, "warnings": ["one category batch failed"],
                "categories_used": ["traffic light"], "detection_mode": "category_sweep",
            }

        results = await BACKEND.detect_frames(
            [object(), object()], ["traffic light"], local_detector=detector,
        )
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(call["prompt"] == "<prompt_free>" for call in seen))
        self.assertEqual(seen[0]["categories"], ["traffic light"])
        self.assertEqual(seen[0]["max_objects"], 50)
        self.assertTrue(all(not path.exists() for path in paths))
        for result in results:
            self.assertEqual(result["backend"], "in_process")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["object_count"], 1)
            self.assertEqual(result["categories_used"], ["traffic light"])
            self.assertEqual(result["warnings"], ["one category batch failed"])

    async def test_local_zero_objects_is_success_and_detector_errors_are_visible(self):
        result = await BACKEND.detect_frames(
            [object()], local_detector=lambda **kwargs: {"success": True, "objects": []},
        )
        self.assertEqual(result[0]["status"], "no_detections")
        self.assertTrue(result[0]["success"])

        paths = []

        def broken(**kwargs):
            paths.append(Path(kwargs["image_source"]))
            raise RuntimeError("checkpoint missing")

        result = await BACKEND.detect_frames([object()], local_detector=broken)
        self.assertFalse(result[0]["success"])
        self.assertIn("checkpoint missing", result[0]["error"])
        self.assertFalse(paths[0].exists())

    async def test_concurrent_local_batches_do_not_overlap_inference(self):
        active = 0
        peak = 0

        def detector(**kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            time.sleep(0.015)
            active -= 1
            return {"success": True, "objects": []}

        await asyncio.gather(*(
            BACKEND.detect_frames([object(), object()], local_detector=detector) for _ in range(2)
        ))
        self.assertEqual(peak, 1)

    async def test_cancelling_local_wait_keeps_image_until_worker_finishes(self):
        started = threading.Event()
        release = threading.Event()
        paths = []

        def detector(**kwargs):
            paths.append(Path(kwargs["image_source"]))
            started.set()
            release.wait(2)
            self.assertTrue(paths[0].is_file())
            return {"success": True, "objects": []}

        task = asyncio.create_task(BACKEND.detect_frames([object()], local_detector=detector))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(paths[0].is_file())
        finally:
            release.set()
        for _ in range(100):
            if not paths[0].exists():
                break
            await asyncio.sleep(0.01)
        self.assertFalse(paths[0].exists())

    async def test_remote_batch_reuses_session_preserves_env_and_cleans_files(self):
        fake = FakeMCP(text_result({
            "success": True, "objects": [{"category": "person"}],
            "categories_used": ["person", "traffic light"], "partial": True,
            "errors": ["one batch unavailable"], "warnings": ["partial evidence"],
        }))
        with patch.dict(sys.modules, fake.modules), patch.dict(os.environ, {
            "GDINO_DEVICE": "cpu", "MODEL_DIR": "/custom/models", "PATH": "/usr/bin",
        }):
            results = await BACKEND.detect_frames(
                [object(), object()], ["person", "traffic light"], root=self.root,
            )
        self.assertEqual(len(fake.parameters), 1)
        self.assertEqual(fake.initializations, 1)
        self.assertEqual(fake.exits, 1)
        params = fake.parameters[0]
        self.assertEqual(params.command, str(self.python))
        self.assertNotEqual(params.command, str(self.python.resolve()))
        self.assertEqual(params.cwd, str(self.root))
        self.assertEqual(params.args[1], str(self.root / "scripts/mcp_runtime.py"))
        self.assertEqual(params.args[2], str(self.root / "tools/vision/grounding-dino-mcp/server.py"))
        self.assertEqual(params.env["GDINO_DEVICE"], "cpu")
        self.assertEqual(params.env["MODEL_DIR"], "/custom/models")
        self.assertEqual(params.env["PATH"], str(self.python.parent) + os.pathsep + "/usr/bin")
        self.assertEqual([name for name, _ in fake.calls], ["scene_scan", "scene_scan"])
        self.assertEqual(fake.calls[0][1]["categories"], ["person", "traffic light"])
        self.assertTrue(all(fake.paths_existed_on_exit))
        self.assertTrue(all(not Path(args["image_path"]).exists() for _, args in fake.calls))
        self.assertTrue(all(result["backend"] == "vision_mcp" for result in results))
        self.assertTrue(all(result["status"] == "partial" for result in results))
        self.assertEqual(results[0]["errors"], ["one batch unavailable"])

    async def test_remote_batches_do_not_load_concurrent_children(self):
        fake = FakeMCP()
        with patch.dict(sys.modules, fake.modules):
            await asyncio.gather(*(
                BACKEND.detect_frames([object()], root=self.root) for _ in range(2)
            ))
        self.assertEqual(fake.peak_clients, 1)
        self.assertEqual(len(fake.parameters), 2)

    async def test_missing_vision_interpreter_is_explicitly_unavailable(self):
        self.python.unlink()
        results = await BACKEND.detect_frames([object(), object()], root=self.root)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["unavailable"] for result in results))
        self.assertTrue(all(result["status"] == "unavailable" for result in results))
        self.assertIn("Install the vision profile", results[0]["error"])

    async def test_relative_override_keeps_venv_path(self):
        fake = FakeMCP()
        custom = self.root / "custom env/bin/python"
        custom.parent.mkdir(parents=True)
        custom.symlink_to(sys.executable)
        with patch.dict(sys.modules, fake.modules), patch.dict(os.environ, {
            "GDINO_PYTHON": "custom env/bin/python",
        }):
            await BACKEND.detect_frames([object()], root=self.root)
        self.assertEqual(fake.parameters[0].command, str(custom))

    async def test_imwrite_failure_skips_detector_and_child(self):
        self.cv2.imwrite = lambda *args: False
        fake = FakeMCP()

        def never_call(**kwargs):
            self.fail("must not infer with a failed JPEG write")

        result = await BACKEND.detect_frames([object()], local_detector=never_call)
        self.assertFalse(result[0]["success"])
        with patch.dict(sys.modules, fake.modules):
            result = await BACKEND.detect_frames([object()], root=self.root)
        self.assertFalse(result[0]["success"])
        self.assertEqual(fake.parameters, [])
        self.assertIn("write", result[0]["error"])

    async def test_remote_successful_empty_and_failed_results_remain_distinct(self):
        fake = FakeMCP(lambda count: text_result(
            {"success": True, "objects": []} if count == 1 else
            {"success": False, "objects": [], "error": "missing weights"}
        ))
        with patch.dict(sys.modules, fake.modules):
            results = await BACKEND.detect_frames([object(), object()], root=self.root)
        self.assertEqual(results[0]["status"], "no_detections")
        self.assertEqual(results[1]["status"], "failed")
        self.assertIn("missing weights", results[1]["error"])

    async def test_remote_rejects_mcp_error_and_invalid_json(self):
        responses = [
            text_result({"error": "tool not found"}, is_error=True),
            SimpleNamespace(isError=False, content=[SimpleNamespace(type="text", text="not JSON")]),
            text_result({"success": True, "objects": "not a list"}),
        ]
        for response in responses:
            with self.subTest(response=response):
                fake = FakeMCP(response)
                with patch.dict(sys.modules, fake.modules):
                    result = await BACKEND.detect_frames([object()], root=self.root)
                self.assertFalse(result[0]["success"])
                self.assertIn("error", result[0])

    async def test_remote_startup_failure_and_timeout_close_child(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                fake = FakeMCP()
                fake.block_startup = timeout
                fake.startup_error = None if timeout else RuntimeError("vision dependencies missing")
                with patch.dict(sys.modules, fake.modules), patch.dict(os.environ, {
                    "GDINO_TIMEOUT_SECONDS": "0.02",
                }):
                    result = await BACKEND.detect_frames([object()], root=self.root)
                self.assertEqual(result[0]["status"], "unavailable")
                self.assertEqual(fake.exits, 1)
                self.assertIn("timed out" if timeout else "dependencies missing", result[0]["error"])

    async def test_remote_call_timeout_aborts_remaining_frames(self):
        fake = FakeMCP()
        fake.block_calls = True
        with patch.dict(sys.modules, fake.modules), patch.dict(os.environ, {
            "GDINO_TIMEOUT_SECONDS": "0.02",
        }):
            results = await BACKEND.detect_frames([object(), object()], root=self.root)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.exits, 1)
        self.assertTrue(all(not result["success"] for result in results))
        self.assertTrue(all("timed out" in result["error"] for result in results))
        self.assertFalse(Path(fake.calls[0][1]["image_path"]).exists())

    async def test_remote_cancellation_closes_child_before_removing_images(self):
        fake = FakeMCP()
        fake.block_calls = True
        with patch.dict(sys.modules, fake.modules):
            task = asyncio.create_task(BACKEND.detect_frames([object()], root=self.root))
            await fake.started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(fake.exits, 1)
        self.assertEqual(fake.paths_existed_on_exit, [True])
        self.assertFalse(Path(fake.calls[0][1]["image_path"]).exists())

    async def test_invalid_timeout_and_categories_do_not_start_child(self):
        for value in ("0", "-1", "nan", "inf", "bad"):
            with self.subTest(value=value), patch.dict(os.environ, {"GDINO_TIMEOUT_SECONDS": value}):
                result = await BACKEND.detect_frames([object()], root=self.root)
                self.assertFalse(result[0]["success"])
                self.assertIn("GDINO_TIMEOUT_SECONDS", result[0]["error"])
        for categories in ("person", [], ["person"] * 65, ["<prompt_free>"], ["x" * 81]):
            with self.subTest(categories=categories):
                result = await BACKEND.detect_frames([object()], categories, root=self.root)
                self.assertFalse(result[0]["success"])
                self.assertIn("categories", result[0]["error"])

    async def test_actual_mcp_stdio_roundtrip_with_fake_model_worker(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("MCP package not installed in this test environment")
        wrapper = self.root / "scripts/mcp_runtime.py"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text(
            "import runpy, sys\nrunpy.run_path(sys.argv[1], run_name='__main__')\n",
        )
        server = self.root / "tools/vision/grounding-dino-mcp/server.py"
        server.parent.mkdir(parents=True)
        # A stdlib protocol fixture exercises the actual MCP client process,
        # initialize handshake and context cleanup without importing a model.
        server.write_text('''\
import json
from pathlib import Path
import sys
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    if request["method"] == "initialize":
        result = {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-detector", "version": "1"},
        }
    elif request["method"] == "tools/list":
        result = {"tools": [{"name": "scene_scan", "inputSchema": {"type": "object"}}]}
    elif request["method"] == "tools/call":
        assert request["params"]["name"] == "scene_scan"
        image = Path(request["params"]["arguments"]["image_path"])
        assert image.read_bytes() == b"fake-jpeg"
        result = {"content": [{"type": "text", "text": json.dumps({
            "success": True, "objects": [{"category": "person"}],
        })}], "isError": False}
    else:
        raise RuntimeError(request["method"])
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
''')
        with patch.dict(os.environ, {"GDINO_PYTHON": sys.executable}):
            results = await BACKEND.detect_frames([object(), object()], root=self.root)
        self.assertEqual([result["status"] for result in results], ["detected", "detected"])
        self.assertTrue(all(result["backend"] == "vision_mcp" for result in results))


if __name__ == "__main__":
    unittest.main()
