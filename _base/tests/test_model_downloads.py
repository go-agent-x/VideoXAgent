"""Exercise downloader routing with fake binaries; never fetch model weights."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "download_models.sh"


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="model-download-tests-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.profiles = self.folder / "profiles with spaces"
        self.path_bin = self.folder / "path commands"
        self.log = self.folder / "calls.jsonl"
        self.model_dir = self.folder / "model output"
        self.token = "test-token-must-not-be-printed"
        self.env = os.environ.copy()
        self.env.update({
            "PATH": str(self.path_bin) + os.pathsep + self.env.get("PATH", ""),
            "DOWNLOAD_TEST_LOG": str(self.log),
            "MODEL_DIR": str(self.model_dir),
            "HF_HOME": str(self.folder / "huggingface cache"),
            "TORCH_HOME": str(self.folder / "torch cache"),
            "INSIGHTFACE_HOME": str(self.folder / "insightface cache"),
            "HF_TOKEN": self.token,
            "WHISPER_MODEL_SIZE": "large-v3",
            "L2CS_WEIGHTS_URL": "https://example.invalid/gaze-weights",
        })
        for command in ("hf", "python", "curl", "unzip"):
            self.make_binary(self.path_bin / command)

    def make_binary(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#!" + sys.executable + "\n"
            "import json, os, sys\n"
            "with open(os.environ['DOWNLOAD_TEST_LOG'], 'a') as handle:\n"
            "    handle.write(json.dumps({'binary': sys.argv[0], 'args': sys.argv[1:], "
            "'paddle_source_check': os.environ.get('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK')}) + '\\n')\n",
            encoding="utf-8",
        )
        path.chmod(0o755)

    def install_fake_profiles(self):
        for profile, commands in {
            "vision": ("hf",),
            "audio": ("hf", "python"),
            "ocr": ("python",),
        }.items():
            for command in commands:
                self.make_binary(self.profiles / profile / "bin" / command)

    def run_script(self, *args):
        return subprocess.run(
            ["bash", str(SCRIPT), *map(str, args)],
            cwd=self.folder,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(self.token, result.stdout + result.stderr)

    def test_dry_run_uses_profiles_without_creating_or_executing_anything(self):
        profiles = self.folder / "absent-profiles"
        before = sorted(str(p.relative_to(self.folder)) for p in self.folder.rglob("*"))
        result = self.run_script("--dry-run", "--venv-root", profiles, "all")
        self.assert_success(result)
        for profile, command in (("vision", "hf"), ("audio", "hf"),
                                 ("audio", "python"), ("ocr", "python")):
            self.assertIn(str(profiles / profile / "bin" / command), result.stdout)
        self.assertIn("REDACTED", result.stdout)
        self.assertEqual(self.calls(), [])
        after = sorted(str(p.relative_to(self.folder)) for p in self.folder.rglob("*"))
        self.assertEqual(after, before)

    def test_all_routes_each_model_to_its_profile_and_keeps_system_downloaders(self):
        self.install_fake_profiles()
        result = self.run_script("--venv-root", self.profiles, "all")
        self.assert_success(result)
        calls = self.calls()
        repositories = {
            "sentence-transformers/all-MiniLM-L6-v2": "vision",
            "depth-anything/Depth-Anything-V2-Small-hf": "vision",
            "MCG-NJU/videomae-base-finetuned-kinetics": "vision",
            "openai/clip-vit-base-patch32": "vision",
            "google-bert/bert-base-uncased": "vision",
            "Systran/faster-whisper-large-v3": "audio",
            "pyannote/speaker-diarization-3.1": "audio",
            "pyannote/segmentation-3.0": "audio",
            "trpakov/vit-face-expression": "audio",
            "speechbrain/emotion-recognition-wav2vec2-IEMOCAP": "audio",
            "facebook/wav2vec2-base": "audio",
        }
        hf_calls = [call for call in calls if Path(call["binary"]).name == "hf"]
        self.assertEqual({call["args"][1] for call in hf_calls}, set(repositories))
        self.assertEqual(len(hf_calls), len(repositories))
        for call in hf_calls:
            repository = call["args"][1]
            self.assertEqual(call["binary"], str(self.profiles / repositories[repository] / "bin/hf"))
            self.assertEqual(call["args"][0], "download")
        python_calls = [call for call in calls if Path(call["binary"]).name == "python"]
        self.assertEqual(len(python_calls), 2)
        audio_call = next(call for call in python_calls if "torchaudio" in call["args"][1])
        ocr_call = next(call for call in python_calls if "PaddleOCR" in call["args"][1])
        self.assertEqual(audio_call["binary"], str(self.profiles / "audio/bin/python"))
        self.assertEqual(ocr_call["binary"], str(self.profiles / "ocr/bin/python"))
        self.assertEqual(ocr_call["paddle_source_check"], "True")
        for command in ("curl", "unzip"):
            selected = [call for call in calls if Path(call["binary"]).name == command]
            self.assertTrue(selected)
            self.assertTrue(all(call["binary"] == str(self.path_bin / command) for call in selected))
        for call in hf_calls:
            if "--token" in call["args"]:
                self.assertEqual(call["args"][call["args"].index("--token") + 1], self.token)
        bert = next(call for call in hf_calls if call["args"][1] == "google-bert/bert-base-uncased")
        self.assertEqual(bert["args"][bert["args"].index("--local-dir") + 1],
                         str(self.model_dir / "bert-base-uncased"))

    def test_omitting_venv_root_preserves_path_commands(self):
        result = self.run_script("hf-core", "audio", "emotion", "diarization", "ocr")
        self.assert_success(result)
        calls = self.calls()
        self.assertTrue(calls)
        self.assertTrue(all(call["binary"] in (str(self.path_bin / "hf"),
                                               str(self.path_bin / "python")) for call in calls))

    def test_relative_venv_root_is_relative_to_repository_not_caller(self):
        relative = "profile examples"
        result = self.run_script("--dry-run", "--venv-root=" + relative, "audio")
        self.assert_success(result)
        escaped = str(ROOT / relative / "audio/bin/hf").replace(" ", "\\ ")
        self.assertIn(escaped, result.stdout)
        self.assertFalse((self.folder / relative).exists())

    def test_invalid_arguments_fail_before_any_component_runs(self):
        for args in (("audio", "--unknown"), ("audio", "no-such-component"),
                     ("audio", "--venv-root"), ("--venv-root", ""),
                     ("--venv-root", "--dry-run"), ("--venv-root=",)):
            with self.subTest(args=args):
                result = self.run_script(*args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("error:", result.stderr)
                self.assertEqual(self.calls(), [])
                self.assertFalse(self.model_dir.exists())

    def test_missing_profile_binary_does_not_fall_back_to_path(self):
        result = self.run_script("--venv-root", self.profiles, "audio")
        self.assertEqual(result.returncode, 1)
        self.assertIn("required command not found", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_unconfigured_gated_component_is_skipped_without_executing(self):
        self.env.pop("HF_TOKEN", None)
        result = self.run_script("--venv-root", self.profiles, "diarization")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("skipped:", result.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
