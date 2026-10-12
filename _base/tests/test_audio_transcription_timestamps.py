"""Source-timeline ASR regressions; no model weights, model APIs, or GPU required."""

import copy
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave


ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = ROOT / "tools/audio/audio-transcription-mcp/server.py"


def load_server():
    """Import the real service without retaining its environment/path changes."""
    spec = importlib.util.spec_from_file_location("audio_timestamp_test_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"MCP_ALLOW_MODEL_DOWNLOAD": "0"}), \
            patch.object(sys, "path", list(sys.path)):
        spec.loader.exec_module(module)
    return module


class AudioTranscriptionTimestampTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = load_server()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="videoxagent-asr-timestamps-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.input_file = self.directory / "source.wav"
        self.input_file.touch()
        self.audio = object()
        self.raw_result = {
            "language": "en",
            "segments": [{
                "start": 1.0,
                "end": 2.5,
                "text": " one two three ",
                "words": [{"word": "one", "start": 1.0, "end": 1.4}],
            }],
        }
        self.model = SimpleNamespace(
            transcribe=Mock(side_effect=lambda *args, **kwargs: copy.deepcopy(self.raw_result))
        )
        self.whisperx = SimpleNamespace(
            load_audio=Mock(return_value=self.audio),
            align=Mock(side_effect=lambda segments, *args, **kwargs: {
                "segments": copy.deepcopy(segments),
            }),
        )
        self.torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=Mock(return_value=False), OutOfMemoryError=MemoryError,
        ))
        self.align_model = object()
        self.align_metadata = {"language": "en"}
        self.patch_server("_get_whisperx", return_value=self.whisperx)
        self.patch_server("_get_torch", return_value=self.torch)
        self.patch_server("ensure_whisper_model_available", return_value=None)
        self.load_model = self.patch_server("load_model_cached", return_value=self.model)
        self.patch_server("load_align_model_cached", return_value=(self.align_model, self.align_metadata))
        self.patch_server("get_device", return_value="cpu")
        self.patch_server("_has_audio_stream", return_value=True)
        self.patch_server("_get_fallback_models", return_value=["small"])
        self.real_extract = self.server.extract_audio_segment_to_wav
        self.extract = self.patch_server("extract_audio_segment_to_wav", side_effect=self.extract_stub)

    def patch_server(self, name, **kwargs):
        patcher = patch.object(self.server, name, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def extract_stub(self, *, input_path, start_time, end_time, sample_rate, mono):
        cropped = self.directory / "cropped.wav"
        cropped.touch()
        return str(cropped), start_time, end_time

    async def transcribe(self, **kwargs):
        arguments = {
            "file_path": str(self.input_file),
            "model_size": "base",
            "auto_fallback": False,
        }
        arguments.update(kwargs)
        output = await self.server.transcribe_audio(**arguments)
        self.assertNotIn("error", output)
        return output

    async def test_cropped_segments_and_words_use_original_file_time(self):
        output = await self.transcribe(start_time=60, end_time=70)
        segment = output["segments"][0]
        self.assertEqual((segment["start_time"], segment["end_time"]), (61.0, 62.5))
        self.assertEqual(segment["start_formatted"], "00:01:01.000")
        self.assertEqual(segment["end_formatted"], "00:01:02.500")
        self.assertEqual(segment["words"], [{"word": "one", "start_time": 61.0, "end_time": 61.4}])
        self.assertEqual((output["start_time"], output["end_time"]), (60.0, 70.0))
        self.assertEqual(output["file_path"], str(self.input_file))
        self.assertEqual(output["text"], "one two three")
        self.assertFalse((self.directory / "cropped.wav").exists())

    async def test_alignment_receives_local_times_and_only_output_is_shifted(self):
        aligned_segments = [{
            "start": 1.25, "end": 2.75, "text": "one two three",
            "words": [{"word": "one", "start": 1.25, "end": 1.5}],
        }]
        self.whisperx.align.side_effect = None
        self.whisperx.align.return_value = {"segments": aligned_segments}
        output = await self.transcribe(start_time=60, end_time=70, align_words=True)
        self.whisperx.align.assert_called_once_with(
            self.raw_result["segments"], self.align_model, self.align_metadata,
            self.audio, "cpu", return_char_alignments=False,
        )
        segment = output["segments"][0]
        self.assertEqual((segment["start_time"], segment["end_time"]), (61.25, 62.75))
        self.assertEqual(segment["start_formatted"], "00:01:01.250")
        self.assertEqual(segment["words"][0]["start_time"], 61.25)
        self.assertEqual(aligned_segments[0]["start"], 1.25)
        self.assertEqual(aligned_segments[0]["words"][0]["start"], 1.25)

    async def test_alignment_failure_still_shifts_unaligned_result(self):
        self.whisperx.align.side_effect = RuntimeError("alignment unavailable in test")
        output = await self.transcribe(start_time=60, end_time=70, align_words=True)
        self.assertIn("alignment unavailable in test", output["alignment_warning"])
        self.assertEqual(output["segments"][0]["start_time"], 61.0)
        self.assertEqual(output["segments"][0]["words"][0]["start_time"], 61.0)

    async def test_fractional_start_and_timestamp_string_preserve_precision(self):
        output = await self.transcribe(start_time="00:01:00.125", end_time="00:01:10.125")
        segment = output["segments"][0]
        self.assertEqual(segment["start_time"], 61.125)
        self.assertEqual(segment["end_time"], 62.625)
        self.assertEqual(segment["start_formatted"], "00:01:01.125")
        self.assertAlmostEqual(segment["words"][0]["end_time"], 61.525)

    async def test_whole_file_zero_start_and_end_only_keep_zero_offset(self):
        for arguments in ({}, {"start_time": 0, "end_time": 10}, {"end_time": 10}):
            with self.subTest(arguments=arguments):
                self.extract.reset_mock()
                output = await self.transcribe(**arguments)
                segment = output["segments"][0]
                self.assertEqual((segment["start_time"], segment["end_time"]), (1.0, 2.5))
                self.assertEqual(segment["start_formatted"], "00:00:01.000")
                self.assertEqual(segment["words"][0]["start_time"], 1.0)
                if arguments:
                    self.extract.assert_called_once()
                else:
                    self.extract.assert_not_called()
                    self.assertIsNone(output["start_time"])
                    self.assertIsNone(output["end_time"])

    async def test_start_only_open_ended_crop_uses_start_offset(self):
        output = await self.transcribe(start_time=60)
        self.assertEqual(output["segments"][0]["start_time"], 61.0)
        self.assertEqual(output["segments"][0]["words"][0]["end_time"], 61.4)
        self.assertIsNone(output["end_time"])

    async def test_missing_word_endpoints_remain_unknown_but_zero_is_shifted(self):
        self.raw_result["segments"][0]["words"] = [
            {"word": "missing"},
            {"word": "null", "start": None, "end": None},
            {"word": "zero", "start": 0.0, "end": 0.25},
            {"word": "start_only", "start": 1.0},
            {"word": "end_only", "end": 2.0},
        ]
        for start in (None, 60):
            with self.subTest(start=start):
                output = await self.transcribe(start_time=start)
                offset = start or 0
                self.assertEqual(output["segments"][0]["words"], [
                    {"word": "missing", "start_time": None, "end_time": None},
                    {"word": "null", "start_time": None, "end_time": None},
                    {"word": "zero", "start_time": offset, "end_time": offset + 0.25},
                    {"word": "start_only", "start_time": offset + 1.0, "end_time": None},
                    {"word": "end_only", "start_time": None, "end_time": offset + 2.0},
                ])

    async def test_offset_uses_effective_start_returned_by_extractor(self):
        def normalized_extract(**kwargs):
            path, _, end = self.extract_stub(**kwargs)
            return path, 60.25, end

        self.extract.side_effect = normalized_extract
        output = await self.transcribe(start_time=60, end_time=70)
        self.assertEqual(output["start_time"], 60.25)
        self.assertEqual(output["segments"][0]["start_time"], 61.25)
        self.assertEqual(output["segments"][0]["words"][0]["start_time"], 61.25)

    async def test_successful_fallback_uses_its_narrowed_crop_offset(self):
        initial = copy.deepcopy(self.raw_result)
        initial["segments"][0]["text"] = "one"
        self.model.transcribe.side_effect = [initial, copy.deepcopy(self.raw_result)]
        output = await self.transcribe(start_time=60, end_time=70, auto_fallback=True)
        self.assertEqual(
            [(call.kwargs["start_time"], call.kwargs["end_time"]) for call in self.extract.call_args_list],
            [(60.0, 70.0), (62.0, 68.0)],
        )
        self.assertEqual((output["start_time"], output["end_time"]), (62.0, 68.0))
        self.assertEqual(output["segments"][0]["start_time"], 63.0)
        self.assertEqual(output["segments"][0]["start_formatted"], "00:01:03.000")
        self.assertEqual(output["segments"][0]["words"][0]["start_time"], 63.0)
        self.assertEqual(output["fallback_info"]["fallback_model"], "small")

    async def test_unsuccessful_fallback_keeps_original_offset_exactly_once(self):
        self.raw_result["segments"][0]["text"] = "one"
        output = await self.transcribe(start_time=60, end_time=70, auto_fallback=True)
        self.assertEqual(self.model.transcribe.call_count, 2)
        self.assertFalse(output["fallback_info"]["fallback_succeeded"])
        self.assertEqual((output["start_time"], output["end_time"]), (60.0, 70.0))
        self.assertEqual(output["segments"][0]["start_time"], 61.0)
        self.assertEqual(output["segments"][0]["words"][0]["start_time"], 61.0)

    async def test_empty_transcription_remains_empty(self):
        self.raw_result["segments"] = []
        output = await self.transcribe(start_time=60, end_time=70)
        self.assertEqual(output["segments"], [])
        self.assertEqual(output["total_segments"], 0)
        self.assertEqual(output["text"], "")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    async def test_real_ffmpeg_crop_and_mock_asr_agree_on_source_timeline(self):
        """Exercise actual extraction/loading boundaries with generated silent audio."""
        sample_rate = 16000
        with wave.open(str(self.input_file), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(sample_rate)
            audio_file.writeframes(b"\x00\x00" * (72 * sample_rate))

        loaded = []

        def load_cropped_audio(path):
            with wave.open(path, "rb") as audio_file:
                loaded.append((path, audio_file.getnframes() / audio_file.getframerate()))
            return self.audio

        self.extract.side_effect = self.real_extract
        self.whisperx.load_audio.side_effect = load_cropped_audio
        output = await self.transcribe(start_time=60, end_time=70)
        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(loaded[0][1], 10.0, places=2)
        self.assertNotEqual(loaded[0][0], str(self.input_file))
        self.assertFalse(Path(loaded[0][0]).exists())
        self.assertEqual(output["segments"][0]["start_time"], 61.0)
        self.assertEqual(output["segments"][0]["words"][0]["start_time"], 61.0)


if __name__ == "__main__":
    unittest.main()
