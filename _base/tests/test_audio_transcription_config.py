"""WhisperX defaults through the real MCP handler; no weights or APIs required."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "tools/audio/audio-transcription-mcp/server.py"


def load_server(settings=None):
    environment = dict(os.environ)
    for key in ("WHISPERX_MODEL_SIZE", "WHISPERX_COMPUTE_TYPE"):
        environment.pop(key, None)
    environment.update(settings or {})
    environment["MCP_ALLOW_MODEL_DOWNLOAD"] = "0"
    spec = importlib.util.spec_from_file_location("asr_config_test_server", SERVER)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, environment, clear=True), patch.object(sys, "path", list(sys.path)):
        spec.loader.exec_module(module)
    return module


class AudioTranscriptionConfigTests(unittest.IsolatedAsyncioTestCase):
    async def selected_options(self, server, **arguments):
        with patch.object(server, "transcribe_audio", new=AsyncMock(return_value={})) as transcribe:
            await server.handle_transcribe_audio({"file_path": "unused.wav", **arguments})
        return transcribe.call_args.kwargs

    async def test_omitted_arguments_use_service_environment(self):
        server = load_server({"WHISPERX_MODEL_SIZE": "base", "WHISPERX_COMPUTE_TYPE": "int8"})
        options = await self.selected_options(server)
        self.assertEqual(options["model_size"], "base")
        self.assertEqual(options["compute_type"], "int8")

    async def test_explicit_arguments_override_environment(self):
        server = load_server({"WHISPERX_MODEL_SIZE": "base", "WHISPERX_COMPUTE_TYPE": "int8"})
        options = await self.selected_options(server, model_size="small", compute_type="float32")
        self.assertEqual(options["model_size"], "small")
        self.assertEqual(options["compute_type"], "float32")

    async def test_each_omitted_option_independently_uses_environment(self):
        server = load_server({"WHISPERX_MODEL_SIZE": "base", "WHISPERX_COMPUTE_TYPE": "int8"})
        for arguments, expected in (
            ({"model_size": "small"}, ("small", "int8")),
            ({"compute_type": "float32"}, ("base", "float32")),
        ):
            with self.subTest(arguments=arguments):
                options = await self.selected_options(server, **arguments)
                self.assertEqual((options["model_size"], options["compute_type"]), expected)

    async def test_unset_environment_keeps_schema_defaults(self):
        options = await self.selected_options(load_server())
        self.assertEqual((options["model_size"], options["compute_type"]), ("large-v3", "float16"))

    async def test_defaults_are_read_at_service_startup(self):
        first = load_server({"WHISPERX_MODEL_SIZE": "base", "WHISPERX_COMPUTE_TYPE": "int8"})
        restarted = load_server({"WHISPERX_MODEL_SIZE": "small", "WHISPERX_COMPUTE_TYPE": "float32"})
        for server, expected in ((first, ("base", "int8")), (restarted, ("small", "float32"))):
            with self.subTest(expected=expected):
                options = await self.selected_options(server)
                self.assertEqual((options["model_size"], options["compute_type"]), expected)

    async def test_environment_selection_reaches_model_loader(self):
        for device, configured_precision, expected_precision in (
            ("cuda", "int8", "int8"),
            ("cuda", "float32", "float32"),
            ("cpu", "float16", "int8"),
        ):
            with self.subTest(device=device, precision=configured_precision):
                server = load_server({"WHISPERX_MODEL_SIZE": "base", "WHISPERX_COMPUTE_TYPE": configured_precision})
                model = SimpleNamespace(transcribe=Mock(return_value={
                    "language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "one two three"}],
                }))
                whisperx = SimpleNamespace(load_audio=Mock(return_value=object()), load_model=Mock(return_value=model))
                torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: device == "cuda", OutOfMemoryError=MemoryError))
                with tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / "synthetic.wav"
                    source.touch()
                    with patch.dict(os.environ, {"MCP_DEVICE": device}), \
                            patch.object(server, "_get_whisperx", return_value=whisperx), \
                            patch.object(server, "_get_torch", return_value=torch), \
                            patch.object(server, "ensure_whisper_model_available", return_value=None), \
                            patch.object(server, "_has_audio_stream", return_value=True):
                        result = await server.handle_transcribe_audio({"file_path": str(source), "auto_fallback": False})
                self.assertNotIn("error", result)
                self.assertEqual(result["model_size"], "base")
                self.assertEqual(result["device"], device)
                whisperx.load_model.assert_called_once_with(
                    "base", device, compute_type=expected_precision, local_files_only=True,
                )

    def test_model_options_are_advertised_without_fixed_defaults(self):
        schema = yaml.safe_load((ROOT / "_tool_defs/audio.yaml").read_text())["transcribe_audio"]["input_schema"]
        for key in ("model_size", "compute_type"):
            with self.subTest(option=key):
                self.assertIn(key, schema["properties"])
                self.assertEqual(schema["properties"][key]["type"], "string")
                self.assertNotIn("default", schema["properties"][key])
                self.assertNotIn(key, schema["required"])


if __name__ == "__main__":
    unittest.main()
