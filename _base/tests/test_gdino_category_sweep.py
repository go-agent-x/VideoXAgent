"""Detector regressions using the real sweep and API routing without model imports."""

import ast
import asyncio
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
GDINO = ROOT / "tools/vision/grounding-dino-mcp"
spec = importlib.util.spec_from_file_location("gdino_sweep_under_test", GDINO / "gdino_fallback.py")
fallback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fallback)


def load_function(filename, function_name, namespace):
    """Execute a production function without importing heavyweight module setup."""
    tree = ast.parse((GDINO / filename).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(GDINO / filename), "exec"), namespace)
    return namespace[function_name]


def detection(category="person", score=0.8, bbox=None):
    return {"category": category, "score": score, "bbox": bbox or [0, 0, 10, 10]}


class CategorySweepTests(unittest.TestCase):
    def test_default_scan_calls_real_text_prompts_and_deduplicates(self):
        backend = Mock(return_value={"success": True, "objects": [detection()]})
        result = fallback.run_category_sweep(backend, "frame.jpg")
        self.assertTrue(result["success"])
        self.assertEqual(result["detection_mode"], "category_sweep")
        self.assertEqual(result["categories_used"], fallback._SWEEP_CATEGORIES)
        self.assertEqual(result["object_count"], 1)
        self.assertEqual(result["batches_succeeded"], 6)
        self.assertEqual(backend.call_count, 6)
        for call in backend.call_args_list:
            self.assertNotIn("<prompt_free>", call.kwargs["prompt"])
            self.assertLessEqual(len(call.kwargs["prompt"].split(" . ")), 8)

    def test_multiword_vocabulary_is_preserved_and_duplicates_removed(self):
        backend = Mock(return_value={"success": True, "objects": []})
        result = fallback.run_category_sweep(
            backend, "frame.jpg", categories=["traffic light", " fire  hydrant ", "Traffic Light"]
        )
        self.assertEqual(backend.call_args.kwargs["prompt"], "traffic light . fire hydrant")
        self.assertEqual(result["categories_used"], ["traffic light", "fire hydrant"])
        self.assertTrue(result["vocabulary_limited"])

    def test_successful_no_detections_is_not_a_backend_failure(self):
        backend = Mock(return_value={"success": True, "objects": []})
        result = fallback.run_category_sweep(backend, "frame.jpg")
        self.assertTrue(result["success"])
        self.assertFalse(result["partial"])
        self.assertEqual(result["objects"], [])
        self.assertEqual(result["errors"], [])
        self.assertNotIn("error", result)

    def test_all_backend_failures_do_not_masquerade_as_an_empty_scene(self):
        backend = Mock(return_value={"success": False, "error": "Missing checkpoint"})
        result = fallback.run_category_sweep(backend, "frame.jpg")
        self.assertFalse(result["success"])
        self.assertFalse(result["partial"])
        self.assertEqual(result["batches_failed"], 6)
        self.assertEqual(result["batches_succeeded"], 0)
        self.assertIn("Missing checkpoint", result["error"])
        self.assertEqual(result["object_count"], 0)

    def test_partial_backend_failure_preserves_detections_and_diagnostics(self):
        backend = Mock(side_effect=[
            {"success": False, "error": "Inference failed"},
            {"success": True, "objects": [detection("clock")]},
        ])
        result = fallback.run_category_sweep(backend, "frame.jpg", categories=[f"class {i}" for i in range(9)])
        self.assertTrue(result["success"])
        self.assertTrue(result["partial"])
        self.assertEqual(result["batches_attempted"], 2)
        self.assertEqual(result["batches_failed"], 1)
        self.assertEqual(result["batches_succeeded"], 1)
        self.assertEqual(result["objects"][0]["category"], "clock")
        self.assertIn("Inference failed", result["warnings"][0])

    def test_partial_empty_scan_is_marked_incomplete(self):
        backend = Mock(side_effect=[RuntimeError("Model error"), {"success": True, "objects": []}])
        result = fallback.run_category_sweep(backend, "frame.jpg", categories=[f"class {i}" for i in range(9)])
        self.assertTrue(result["success"])
        self.assertTrue(result["partial"])
        self.assertEqual(result["object_count"], 0)
        self.assertIn("Model error", result["errors"][0])

    def test_invalid_requests_are_rejected_before_any_inference(self):
        invalid = [
            {"categories": []}, {"categories": "person"}, {"categories": [1]},
            {"categories": [" "]}, {"categories": ["<prompt_free>"]},
            {"categories": ["x" * 81]}, {"categories": ["word " * 9]},
            {"categories": ["person"] * 65},
            {"max_objects": 0}, {"max_objects": 51}, {"max_objects": 1.5}, {"max_objects": True},
            {"box_threshold": -0.1}, {"box_threshold": 1.1}, {"box_threshold": float("nan")},
            {"box_threshold": "0.5"}, {"box_threshold": True},
            {"iou_merge_threshold": float("inf")}, {"iou_merge_threshold": -1},
        ]
        for options in invalid:
            with self.subTest(options=options):
                backend = Mock()
                result = fallback.run_category_sweep(backend, "frame.jpg", **options)
                backend.assert_not_called()
                self.assertFalse(result["success"])
                self.assertEqual(result["batches_attempted"], 0)
                self.assertIn("error", result)

    def test_object_cap_keeps_highest_scores_but_scans_all_categories(self):
        objects = [detection(f"item {i}", i / 100, [i * 20, 0, i * 20 + 10, 10]) for i in range(60)]
        backend = Mock(return_value={"success": True, "objects": objects})
        result = fallback.run_category_sweep(backend, "frame.jpg", categories=[f"class {i}" for i in range(64)])
        self.assertEqual(backend.call_count, 8)
        self.assertEqual(len(result["categories_used"]), 64)
        self.assertEqual(result["object_count"], 50)
        self.assertEqual(result["objects"][0]["category"], "item 59")
        self.assertEqual(result["objects"][-1]["category"], "item 10")


class LegacySentinelTests(unittest.TestCase):
    def test_production_api_sentinel_reaches_text_inference_without_model_imports(self):
        predict = Mock(return_value=(
            [types.SimpleNamespace(tolist=lambda: [0.5, 0.5, 0.2, 0.2])], [0.9], ["traffic light"]
        ))
        inference = types.ModuleType("groundingdino.util.inference")
        inference.load_image = lambda image: (image, "image_tensor")
        inference.predict = predict
        namespace = {
            "Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "os": os,
            "_get_model": lambda: "fake_model",
            "_load_image": lambda source: types.SimpleNamespace(size=(100, 200)),
            "GDINO_DEVICE": "cpu",
        }
        api = load_function("gdino_api.py", "call_gdino_api", namespace)
        with tempfile.NamedTemporaryFile() as image, patch.dict(sys.modules, {
            "gdino_fallback": fallback,
            "groundingdino": types.ModuleType("groundingdino"),
            "groundingdino.util": types.ModuleType("groundingdino.util"),
            "groundingdino.util.inference": inference,
            "numpy": types.ModuleType("numpy"),
        }):
            result = api(image.name, "<prompt_free>", ["bbox"], categories=["traffic light", "fire hydrant"])
        self.assertTrue(result["success"])
        self.assertEqual(result["detection_mode"], "category_sweep")
        self.assertEqual(predict.call_args.kwargs["caption"], "traffic light . fire hydrant.")
        self.assertEqual(result["objects"][0]["bbox"], [40.0, 80.0, 60.0, 120.0])

    def test_missing_image_is_reported_as_a_failure_through_the_real_api(self):
        api = load_function("gdino_api.py", "call_gdino_api", {
            "Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "os": os,
        })
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"gdino_fallback": fallback}):
            result = api(str(Path(tmp) / "missing.jpg"), "<prompt_free>", ["bbox"])
        self.assertFalse(result["success"])
        self.assertIn("Image not found", result["error"])
        self.assertEqual(result["batches_failed"], 6)


class ModelInitializationTests(unittest.TestCase):
    def test_failed_initialization_is_not_cached_and_next_attempt_reloads(self):
        for stage in ("checkpoint", "load_state_dict", "eval", "device", "cuda"):
            with self.subTest(stage=stage):
                first, second = Mock(), Mock()
                build_model = Mock(side_effect=[first, second])
                torch_stub = types.ModuleType("torch")
                torch_stub.load = Mock(return_value={"model": "weights"})
                if stage == "checkpoint":
                    torch_stub.load.side_effect = [RuntimeError("checkpoint failed"), {"model": "weights"}]
                elif stage == "device":
                    first.to.side_effect = RuntimeError("device failed")
                else:
                    getattr(first, stage).side_effect = RuntimeError(f"{stage} failed")
                transformers_stub = types.ModuleType("transformers")
                transformers_stub.BertModel = type("BertModel", (), {
                    "get_head_mask": lambda *args: None,
                    "get_extended_attention_mask": lambda *args: None,
                })
                config_stub = types.ModuleType("groundingdino.util.slconfig")
                config_stub.SLConfig = types.SimpleNamespace(
                    fromfile=lambda path: types.SimpleNamespace(text_encoder_type="bert-base-uncased")
                )
                models_stub = types.ModuleType("groundingdino.models")
                models_stub.build_model = build_model
                namespace = {
                    "_model": None, "_model_lock": threading.Lock(), "os": os,
                    "contextlib": contextlib, "io": io, "_gdino_cfg": {},
                    "GDINO_CONFIG": "test_config.py", "GDINO_CHECKPOINT": "test_weights.pth",
                    "GDINO_DEVICE": "cuda" if stage == "cuda" else "cpu",
                }
                get_model = load_function("gdino_api.py", "_get_model", namespace)
                with patch.dict(os.environ), patch.dict(sys.modules, {
                    "torch": torch_stub, "transformers": transformers_stub,
                    "groundingdino": types.ModuleType("groundingdino"),
                    "groundingdino.util": types.ModuleType("groundingdino.util"),
                    "groundingdino.util.slconfig": config_stub,
                    "groundingdino.models": models_stub,
                }):
                    with self.assertRaisesRegex(RuntimeError, "failed"):
                        get_model()
                    self.assertIsNone(namespace["_model"])
                    self.assertIs(get_model(), second)
                    self.assertIs(get_model(), second)
                self.assertEqual(build_model.call_count, 2)
                self.assertEqual(torch_stub.load.call_count, 2)
                self.assertIs(namespace["_model"], second)
                second.load_state_dict.assert_called_once_with("weights", strict=False)
                second.eval.assert_called_once()


class SceneScanTests(unittest.TestCase):
    def test_zero_and_one_detection_do_not_trigger_a_second_scan(self):
        for objects in ([], [detection()]):
            with self.subTest(objects=objects):
                result = fallback.run_category_sweep(
                    lambda **kwargs: {"success": True, "objects": objects},
                    "frame.jpg", categories=["traffic light"],
                )
                backend = Mock(return_value=result)
                handler = load_function("handler_scene_scan.py", "handle_scene_scan", {"call_gdino_api": backend})
                output = asyncio.run(handler({"image_path": "frame.jpg", "categories": ["traffic light"], "max_objects": 2}))
                backend.assert_called_once()
                self.assertEqual(backend.call_args.kwargs["categories"], ["traffic light"])
                self.assertEqual(backend.call_args.kwargs["max_objects"], 2)
                self.assertEqual(output["detection_mode"], "category_sweep")
                self.assertEqual(output["categories_used"], ["traffic light"])
                self.assertEqual(output["total_objects"], len(objects))
                if not objects:
                    self.assertNotEqual(output["scene_summary"], "Empty scene")

    def test_scene_scan_preserves_partial_and_failed_statuses(self):
        for outcomes in (
            [{"success": False, "error": "bad batch"}, {"success": True, "objects": [detection()]}],
            [{"success": False, "error": "bad batch"}] * 2,
        ):
            with self.subTest(outcomes=outcomes):
                result = fallback.run_category_sweep(Mock(side_effect=outcomes), "frame.jpg", categories=[f"class {i}" for i in range(9)])
                backend = Mock(return_value=result)
                handler = load_function("handler_scene_scan.py", "handle_scene_scan", {"call_gdino_api": backend})
                output = asyncio.run(handler({"image_path": "frame.jpg"}))
                for key in ("success", "partial", "errors", "warnings", "batches_failed"):
                    self.assertEqual(output[key], result[key])


if __name__ == "__main__":
    unittest.main()
