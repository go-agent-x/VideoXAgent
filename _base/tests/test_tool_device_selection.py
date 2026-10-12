"""Device propagation into real backend constructors, with model libraries mocked."""

import ast
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _base.config import CONFIG_SCHEMAS, get_config
from _base.device import insightface_device_options, ocr_device_options, select_torch_device
from scripts.tool_catalog import service_dir, service_entrypoint


def backend_function(path, name, **state):
    """Execute the actual initializer without importing unrelated model stacks.

    Some service modules perform cache setup at import time. Loading just the
    requested function keeps these unit tests free of downloads and home writes.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {
        "os": os, "sys": sys, "Path": Path,
        "select_torch_device": select_torch_device,
        "insightface_device_options": insightface_device_options,
        "ocr_device_options": ocr_device_options,
        **state,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name], namespace


class DeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(return_value=True)))

    def test_global_cpu_overrides_saved_schema_devices_and_preserves_tool_override(self):
        os.environ["MCP_DEVICE"] = "cpu"
        for schema in ("grounding_dino", "sam", "sam2", "emotion", "depth", "gaze"):
            variable = CONFIG_SCHEMAS[schema]["device"][0]
            with self.subTest(schema=schema), patch.dict(os.environ, {variable: "cuda"}):
                self.assertEqual(get_config(schema)["device"], "cpu")
                with patch.dict(os.environ, {f"TOOL_SPECIAL__{variable}": "cuda"}):
                    self.assertEqual(get_config(schema, "special")["device"], "cuda")

    def test_absent_global_option_keeps_existing_schema_device(self):
        self.assertEqual(get_config("sam")["device"], "cuda")
        with patch.dict(os.environ, {"SAM_DEVICE": "cpu"}):
            self.assertEqual(get_config("sam")["device"], "cpu")

    def test_cpu_selection_ignores_available_cuda_without_changing_worker_assignment(self):
        os.environ.update(MCP_DEVICE="cpu", CUDA_VISIBLE_DEVICES="3,5")
        self.assertEqual(select_torch_device(self.torch), "cpu")
        self.torch.cuda.is_available.assert_not_called()
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3,5")

    def test_automatic_and_requested_cuda_keep_unavailable_gpu_fallback(self):
        for environment in ({}, {"MCP_DEVICE": "cuda"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                self.torch.cuda.is_available.return_value = True
                self.assertEqual(select_torch_device(self.torch), "cuda")
                self.torch.cuda.is_available.return_value = False
                self.assertEqual(select_torch_device(self.torch), "cpu")
                self.assertEqual(select_torch_device(None), "cpu")

    def test_onnx_cpu_request_filters_cuda_and_sets_cpu_context(self):
        runtime = SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        with patch.dict(sys.modules, {"onnxruntime": runtime}), patch.dict(os.environ, {"MCP_DEVICE": "cpu"}):
            self.assertEqual(insightface_device_options("CUDAExecutionProvider,CPUExecutionProvider"),
                             (["CPUExecutionProvider"], -1))

    def test_cuda_request_does_not_invent_uninstalled_onnx_provider(self):
        runtime = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        with patch.dict(sys.modules, {"onnxruntime": runtime}), patch.dict(os.environ, {"MCP_DEVICE": "cuda"}):
            self.assertEqual(insightface_device_options("CUDAExecutionProvider,CPUExecutionProvider"),
                             (["CPUExecutionProvider"], -1))

    def test_installed_onnx_providers_preserve_configured_preference(self):
        runtime = SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        with patch.dict(sys.modules, {"onnxruntime": runtime}):
            self.assertEqual(insightface_device_options("CUDAExecutionProvider,CPUExecutionProvider"),
                             (["CUDAExecutionProvider", "CPUExecutionProvider"], 0))
            self.assertEqual(insightface_device_options("CPUExecutionProvider"), (["CPUExecutionProvider"], -1))

    def test_ocr_selection_is_independent_from_torch_device(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device), patch.dict(os.environ, {"MCP_DEVICE": device}):
                self.assertEqual(ocr_device_options(), {})
                for setting in ("0", "1"):
                    with patch.dict(os.environ, {"OCR_USE_GPU": setting}):
                        self.assertEqual(ocr_device_options(), {"use_gpu": setting == "1"})

    def test_whisper_services_use_cpu_and_cpu_compatible_compute_type(self):
        os.environ["MCP_DEVICE"] = "cpu"
        for service in ("audio-transcription", "speaker-diarization"):
            with self.subTest(service=service):
                get_device, _ = backend_function(service_entrypoint(service), "get_device",
                                               _get_torch=lambda: self.torch,
                                               torch=self.torch, TORCH_AVAILABLE=True)
                compute_type, _ = backend_function(service_entrypoint(service), "get_compute_type")
                self.assertEqual(get_device(), "cpu")
                self.assertEqual(compute_type(get_device(), "float16"), "int8")

    def test_action_pipeline_receives_cpu_even_when_cuda_is_available(self):
        pipeline = Mock(return_value=SimpleNamespace(model=object()))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text("{}")
            with patch.dict(os.environ, {"MCP_DEVICE": "cpu", "VIDEOMAE_MODEL": directory}), \
                 patch.dict(sys.modules, {"torch": self.torch, "transformers": SimpleNamespace(pipeline=pipeline)}):
                initialize, _ = backend_function(service_dir("action-classify") / "action_utils.py", "_try_init_videomae")
                self.assertIsNotNone(initialize())
        self.assertEqual(pipeline.call_args.kwargs["device"], -1)

    def test_raft_model_is_moved_to_cpu(self):
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        optical_flow = SimpleNamespace(raft_large=Mock(return_value=model),
                                       Raft_Large_Weights=SimpleNamespace(DEFAULT=Mock()))
        with patch.dict(os.environ, {"MCP_DEVICE": "cpu"}), \
             patch.dict(sys.modules, {"torch": self.torch, "torchvision.models.optical_flow": optical_flow}):
            initialize, _ = backend_function(service_entrypoint("optical-flow"), "_try_init_raft")
            self.assertIs(initialize()[0], model)
        model.to.assert_called_once_with("cpu")

    def test_clip_model_is_moved_to_cpu(self):
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        transformers = SimpleNamespace(CLIPModel=SimpleNamespace(from_pretrained=Mock(return_value=model)),
                                       CLIPProcessor=SimpleNamespace(from_pretrained=Mock()))
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"MCP_DEVICE": "cpu"}), \
             patch.dict(sys.modules, {"torch": self.torch, "transformers": transformers}):
            initialize, _ = backend_function(service_entrypoint("moment-retrieval"), "_try_init_clip",
                                            CLIP_MODEL_NAME=directory)
            self.assertIs(initialize()[0], model)
        model.to.assert_called_once_with("cpu")

    def test_sbert_receives_explicit_cpu_but_keeps_library_default_when_unset(self):
        constructor = Mock()
        with tempfile.TemporaryDirectory() as directory:
            for device in (None, "cpu"):
                environment = {"SBERT_MODEL": directory}
                if device:
                    environment["MCP_DEVICE"] = device
                with self.subTest(device=device), patch.dict(os.environ, environment, clear=True), \
                     patch.dict(sys.modules, {"torch": self.torch,
                                             "sentence_transformers": SimpleNamespace(SentenceTransformer=constructor)}):
                    constructor.reset_mock()
                    initialize, _ = backend_function(service_dir("temporal-ground") / "temporal_utils.py", "_init_sbert")
                    initialize()
                    self.assertEqual(constructor.call_args.kwargs, {"device": "cpu"} if device else {})

    def test_panns_constructors_receive_cpu(self):
        tagging = Mock()
        detection = Mock()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "Cnn14.pth"
            checkpoint.touch()
            (Path(directory) / "Cnn14_DecisionLevelMax.pth").touch()
            with patch.dict(os.environ, {"MCP_DEVICE": "cpu", "PANNS_CHECKPOINT": str(checkpoint)}), \
                 patch.dict(sys.modules, {"torch": self.torch, "panns_inference": SimpleNamespace(
                     AudioTagging=tagging, SoundEventDetection=detection)}):
                initialize, _ = backend_function(service_entrypoint("audio-event-detection"), "_try_init_panns")
                self.assertTrue(initialize())
        self.assertEqual(tagging.call_args.kwargs["device"], "cpu")
        self.assertEqual(detection.call_args.kwargs["device"], "cpu")

    def test_direct_panns_fallback_loads_checkpoints_on_cpu(self):
        self.torch.load = Mock(return_value={"model": {}})
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "panns_inference"
            package.mkdir()
            (package / "models.py").write_text(
                "class Cnn14:\n"
                "    def __init__(self, **kwargs): pass\n"
                "    def load_state_dict(self, state): pass\n"
                "    def eval(self): pass\n"
                "    def to(self, device): raise AssertionError('CPU model must not move to CUDA')\n"
                "Cnn14_DecisionLevelMax = Cnn14\n"
            )
            (package / "pytorch_utils.py").write_text("def move_data_to_device(data, device): return data\n")
            checkpoint = Path(directory) / "Cnn14.pth"
            checkpoint.touch()
            (Path(directory) / "Cnn14_DecisionLevelMax.pth").touch()
            with patch.dict(os.environ, {"MCP_DEVICE": "cpu", "PANNS_CHECKPOINT": str(checkpoint)}), \
                 patch.dict(sys.modules, {"torch": self.torch, "panns_inference": SimpleNamespace()}), \
                 patch.object(sys, "path", [directory, *sys.path]):
                initialize, state = backend_function(service_entrypoint("audio-event-detection"), "_try_init_panns",
                                                     importlib=importlib, traceback=traceback, PROJECT_ROOT=ROOT)
                self.assertTrue(initialize())
            self.assertEqual(state["_audio_tagger"].device, "cpu")
            self.assertEqual(state["_sound_event_detector"].device, "cpu")
            self.assertEqual(self.torch.load.call_count, 2)
            for call in self.torch.load.call_args_list:
                self.assertEqual(call.kwargs["map_location"], "cpu")

    def test_insightface_consumers_pass_cpu_provider_and_context(self):
        constructor = Mock()
        runtime = SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        services = {
            "face-cluster": service_dir("face-cluster") / "face_utils.py",
            **{name: service_entrypoint(name) for name in ("gaze-estimation", "emotion-analysis", "active-speaker-detection")},
        }
        with patch.dict(os.environ, {"MCP_DEVICE": "cpu"}), patch.dict(sys.modules, {
            "onnxruntime": runtime, "insightface": SimpleNamespace(),
            "insightface.app": SimpleNamespace(FaceAnalysis=constructor),
        }):
            for name, path in services.items():
                with self.subTest(service=name):
                    constructor.reset_mock()
                    initialize, _ = backend_function(path, "_try_init_insightface",
                                                    _INSIGHTFACE_MODEL="buffalo_l",
                                                    _INSIGHTFACE_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider")
                    app = initialize()
                    self.assertIs(app, constructor.return_value)
                    self.assertEqual(constructor.call_args.kwargs["providers"], ["CPUExecutionProvider"])
                    app.prepare.assert_called_once_with(ctx_id=-1, det_size=(640, 640))

    def test_all_ocr_constructors_receive_cpu_gpu_or_unchanged_default(self):
        functions = [("ocr_common.py", "init_ocr"), ("handler_layout_analysis.py", "_init_ocr"),
                     ("handler_layout_analysis.py", "_init_ppstructure")]
        for setting, expected in ((None, {}), ("0", {"use_gpu": False}), ("1", {"use_gpu": True})):
            for filename, function in functions:
                with self.subTest(setting=setting, function=function), \
                     patch.dict(os.environ, {"OCR_USE_GPU": setting} if setting else {}, clear=True):
                    constructor = Mock()
                    with patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PaddleOCR=constructor, PPStructure=constructor)}):
                        initialize, _ = backend_function(service_dir("ocr") / filename, function,
                            _ocr_instances={}, _ppstructure_instances={}, _ocr_lock=threading.Lock(),
                            _engine_lock=threading.Lock(), _suppress_stdout=nullcontext)
                        initialize("ch")
                    actual = constructor.call_args.kwargs
                    self.assertEqual({key: value for key, value in actual.items() if key == "use_gpu"}, expected)
                    self.assertEqual(actual["lang"], "ch")


if __name__ == "__main__":
    unittest.main()
