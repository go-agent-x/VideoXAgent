"""Composite detection evidence regressions without models, GPUs, or API calls."""

import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = ROOT / "tools/composite/segment-deep-query-mcp/server.py"


def load_server():
    """Import the real service while isolating optional native backends."""
    spec = importlib.util.spec_from_file_location("segment_detection_test_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    optional_modules = {
        "cv2": SimpleNamespace(),
        "gdino_api": SimpleNamespace(call_gdino_api=Mock()),
        "ocr_common": SimpleNamespace(run_ocr_on_array=Mock()),
        "detection_backend": SimpleNamespace(detect_frames=AsyncMock()),
    }
    with patch.dict(sys.modules, optional_modules), \
            patch.dict(os.environ, {"MCP_ALLOW_MODEL_DOWNLOAD": "0"}), \
            patch.object(sys, "path", list(sys.path)):
        spec.loader.exec_module(module)
    return module


def detection_result(*, objects=None, success=True, **extra):
    return {
        "success": success,
        "objects": copy.deepcopy(objects or []),
        "mode": "category_sweep",
        "categories_used": ["person", "traffic light"],
        "backend": "in_process",
        **extra,
    }


class SegmentDetectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server = load_server()
        temporary = tempfile.TemporaryDirectory(prefix="videoxagent-segment-detection-")
        self.addCleanup(temporary.cleanup)
        self.video = Path(temporary.name) / "synthetic.mp4"
        self.video.touch()
        self.frames = [
            (60.0, np.zeros((2, 2, 3), dtype=np.uint8)),
            (65.0, np.ones((2, 2, 3), dtype=np.uint8)),
        ]
        self.extract = self.patch_server("extract_frames", return_value=self.frames)
        self.patch_server("frame_to_base64", return_value="synthetic-jpeg")
        self.detector = self.patch_server(
            "detect_frames", new=AsyncMock(return_value=[detection_result(), detection_result()]),
        )
        self.ocr = self.patch_server("_ocr_text", new=AsyncMock(return_value={
            "success": True, "texts": [{"text": "EXIT", "confidence": 0.95}],
        }))
        self.vlm = self.patch_server("_call_vlm", new=AsyncMock(return_value={
            "response": "A person is near the traffic light.",
            "frame_timestamps": ["00:01:00.000", "00:01:05.000"],
        }))

    def patch_server(self, name, **kwargs):
        patcher = patch.object(self.server, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    async def run_query(self, **extra):
        return await self.server.handle_segment_deep_query({
            "video_path": str(self.video), "start_time": 60, "end_time": 70,
            "query": "What is the person doing?", "num_frames": 2, **extra,
        })

    def evidence_frames(self):
        return self.vlm.call_args.args[0]

    async def test_batch_detection_reaches_vlm_and_preserves_ocr(self):
        obj = {"category": "person", "score": 0.9, "bbox": [0, 0, 1, 1]}
        self.detector.return_value = [detection_result(objects=[obj]), detection_result()]
        result = await self.run_query()
        self.assertTrue(result["success"])
        self.detector.assert_awaited_once()
        sent_frames = self.detector.call_args.args[0]
        self.assertEqual(len(sent_frames), 2)
        for sent, (_, original) in zip(sent_frames, self.frames):
            self.assertIs(sent, original)
        self.assertEqual(self.ocr.await_count, 2)
        frames = self.evidence_frames()
        self.assertEqual([frame["timestamp"] for frame in frames], [60.0, 65.0])
        self.assertEqual(frames[0]["detections"], [obj])
        self.assertEqual(frames[0]["ocr_texts"], [{"text": "EXIT", "confidence": 0.95}])
        self.assertEqual(frames[1]["ocr_texts"], frames[0]["ocr_texts"])
        summary = result["evidence_summary"]
        self.assertTrue(summary["detection_available"])
        self.assertEqual(summary["total_objects_detected"], 1)
        self.assertEqual(summary["total_text_regions"], 2)
        self.assertEqual(summary["detection_status"], "ok")
        self.assertEqual(summary["detection_frames_succeeded"], 2)
        self.assertEqual(summary["detection_frames_failed"], 0)
        self.assertEqual(
            [frame["status"] for frame in summary["detection_frame_status"]],
            ["detected", "no_detections"],
        )

    async def test_explicit_categories_keep_multiword_phrases(self):
        categories = ["traffic light", "tennis racket"]
        self.detector.return_value = [
            detection_result(categories_used=categories), detection_result(categories_used=categories),
        ]
        result = await self.run_query(detection_categories=categories)
        self.assertTrue(result["success"])
        call = self.detector.call_args
        passed = call.args[1] if len(call.args) > 1 else call.kwargs.get("categories")
        self.assertEqual(passed, categories)
        self.assertEqual(result["evidence_summary"]["detection_categories"], categories)

    async def test_detect_objects_wrapper_uses_local_backend_when_imported(self):
        obj = {"category": "person", "score": 0.8}
        self.detector.return_value = [detection_result(objects=[obj])]
        result = await self.server._detect_objects(self.frames[0][1], ["person"])
        self.assertEqual(result["objects"], [obj])
        self.detector.assert_awaited_once()
        self.assertIs(self.detector.call_args.kwargs["local_detector"], self.server.call_gdino_api)

    async def test_detect_objects_wrapper_can_use_profile_bridge(self):
        self.detector.return_value = [detection_result()]
        with patch.object(self.server, "_HAS_DETECTION", False):
            result = await self.server._detect_objects(self.frames[0][1])
        self.assertTrue(result["success"])
        self.assertIsNone(self.detector.call_args.kwargs["local_detector"])

    async def test_profile_bridge_success_counts_as_available_without_local_import(self):
        with patch.object(self.server, "_HAS_DETECTION", False):
            result = await self.run_query()
        self.assertTrue(result["success"])
        self.assertTrue(result["evidence_summary"]["detection_available"])
        self.assertEqual(result["evidence_summary"]["detection_frames_succeeded"], 2)
        self.assertIsNone(self.detector.call_args.kwargs["local_detector"])

    async def test_invalid_category_scope_fails_before_video_or_model_work(self):
        for categories in (
            "traffic light", [], [""], ["  "], [3], ["person"] * 65,
            ["x" * 81], ["one two three four five six seven eight nine"], ["<prompt_free>"],
        ):
            with self.subTest(categories=categories):
                result = await self.run_query(detection_categories=categories)
                self.assertFalse(result["success"])
                self.assertIn("categor", result["error"].lower())
                self.extract.assert_not_called()
                self.detector.assert_not_awaited()
                self.vlm.assert_not_awaited()

    async def test_failed_detection_is_reported_while_ocr_and_vlm_continue(self):
        self.detector.return_value = [
            detection_result(success=False, error="checkpoint not found"),
            detection_result(success=False, error="checkpoint not found"),
        ]
        result = await self.run_query()
        self.assertTrue(result["success"])
        summary = result["evidence_summary"]
        self.assertFalse(summary["detection_available"])
        self.assertEqual(summary["detection_status"], "failed")
        self.assertEqual(summary["detection_frames_failed"], 2)
        self.assertEqual(summary["detection_frames_succeeded"], 0)
        self.assertEqual(summary["total_objects_detected"], 0)
        self.assertEqual(summary["total_text_regions"], 2)
        for frame in self.evidence_frames():
            self.assertEqual(frame["detection_info"]["status"], "failed")
            self.assertIn("checkpoint not found", frame["detection_info"]["error"])
        self.vlm.assert_awaited_once()

    async def test_successful_empty_detection_has_no_failure_status(self):
        result = await self.run_query()
        summary = result["evidence_summary"]
        self.assertTrue(summary["detection_available"])
        self.assertEqual(summary["detection_status"], "ok")
        self.assertEqual(summary["detection_frames_succeeded"], 2)
        self.assertEqual(summary["detection_frames_failed"], 0)
        self.assertEqual(summary["total_objects_detected"], 0)
        for frame in self.evidence_frames():
            self.assertEqual(frame["detection_info"]["status"], "no_detections")
            self.assertEqual(frame["detection_info"]["categories_used"], ["person", "traffic light"])

    async def test_frame_failure_keeps_successful_frame_evidence(self):
        obj = {"category": "person", "score": 0.9}
        self.detector.return_value = [
            detection_result(objects=[obj]),
            detection_result(success=False, error="inference failed", objects=[{"category": "invalid"}]),
        ]
        result = await self.run_query()
        summary = result["evidence_summary"]
        self.assertEqual(summary["detection_status"], "partial")
        self.assertTrue(summary["detection_available"])
        self.assertEqual(summary["detection_frames_succeeded"], 1)
        self.assertEqual(summary["detection_frames_failed"], 1)
        self.assertEqual(self.evidence_frames()[0]["detections"], [obj])
        self.assertEqual(self.evidence_frames()[1]["detections"], [])

    async def test_partial_category_scan_keeps_objects_and_warning(self):
        obj = {"category": "person", "score": 0.9}
        self.detector.return_value = [
            detection_result(objects=[obj], partial=True, warnings=["traffic light batch failed"]),
            detection_result(),
        ]
        result = await self.run_query()
        self.assertEqual(result["evidence_summary"]["detection_status"], "partial")
        first = self.evidence_frames()[0]
        self.assertEqual(first["detections"], [obj])
        self.assertEqual(first["detection_info"]["status"], "partial")
        self.assertIn("traffic light batch failed", first["detection_info"]["warnings"])

    async def test_unavailable_backend_is_not_reported_as_empty_detection(self):
        self.detector.return_value = [
            detection_result(success=False, unavailable=True, error="vision interpreter missing"),
            detection_result(success=False, unavailable=True, error="vision interpreter missing"),
        ]
        result = await self.run_query()
        self.assertTrue(result["success"])
        summary = result["evidence_summary"]
        self.assertFalse(summary["detection_available"])
        self.assertEqual(summary["detection_status"], "unavailable")
        self.assertTrue(all(frame["status"] == "unavailable" for frame in summary["detection_frame_status"]))

    async def test_vlm_error_still_propagates_to_tool_result(self):
        self.vlm.return_value = {"response": "", "error": "synthetic timeout"}
        result = await self.run_query()
        self.assertFalse(result["success"])
        self.assertIn("synthetic timeout", result["error"])

    def test_optional_categories_are_visible_in_mcp_schema(self):
        schema = yaml.safe_load((ROOT / "_tool_defs/vlm_query.yaml").read_text())["segment_deep_query"]["input_schema"]
        categories = schema["properties"]["detection_categories"]
        self.assertEqual(categories["type"], "array")
        self.assertEqual(categories["items"]["type"], "string")
        self.assertEqual(categories["maxItems"], 64)
        self.assertNotIn("detection_categories", schema["required"])


class SegmentDetectionPromptTests(unittest.IsolatedAsyncioTestCase):
    async def prompt_text(self, detection_info, detections=None):
        server = load_server()
        body = {"choices": [{"message": {"content": "mock answer"}}]}
        response = Mock()
        response.json.return_value = body
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__.return_value = client
        with patch.object(server.httpx, "AsyncClient", return_value=client), \
                patch.object(server, "maybe_thinking_off_retry", new=AsyncMock(return_value={"retried": False, "data": body})):
            result = await server._call_vlm([{
                "timestamp": 60.0, "base64": "synthetic-jpeg",
                "detections": detections or [],
                "ocr_texts": [{"text": "EXIT", "confidence": 0.95}],
                "detection_info": detection_info,
            }], 60.0, 70.0, "What is happening?", 128)
        self.assertEqual(result["response"], "mock answer")
        contents = client.post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertTrue(any(item["type"] == "image_url" for item in contents))
        text = "\n".join(item["text"] for item in contents if item["type"] == "text")
        self.assertIn("EXIT", text)
        self.assertIn("00:01:00.000", text)
        return text

    async def test_actual_vlm_payload_contains_nonempty_detection_evidence(self):
        text = await self.prompt_text(
            {"status": "detected", "categories_used": ["traffic light"]},
            [{"category": "traffic light", "score": 0.87}],
        )
        self.assertIn("traffic light", text)
        self.assertIn("0.87", text)
        self.assertIn("object detection results", text)

    async def test_actual_vlm_payload_preserves_detection_failure(self):
        text = await self.prompt_text({"status": "failed", "error": "checkpoint not found"})
        self.assertIn("checkpoint not found", text)
        self.assertIn("fail", text.lower())

    async def test_actual_vlm_payload_scopes_empty_result_to_scanned_categories(self):
        text = await self.prompt_text({
            "status": "no_detections", "categories_used": ["person", "traffic light"],
        })
        self.assertIn("traffic light", text)
        self.assertRegex(text.lower(), r"(scanned|searched|scope|categor|vocab)")
        self.assertNotIn("No objects exist", text)

    async def test_actual_vlm_payload_reports_partial_scan_warning(self):
        text = await self.prompt_text({
            "status": "partial", "warnings": ["traffic light batch failed"],
            "categories_used": ["person", "traffic light"],
        }, [{"category": "person", "score": 0.8}])
        self.assertIn("traffic light batch failed", text)
        self.assertIn("partial", text.lower())
        self.assertIn("person", text)


if __name__ == "__main__":
    unittest.main()
