"""Verify backend choices reach MCP configuration without installing packages."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class InstallerDeploymentOptionsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="installer options with spaces ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts/install_profiles.sh"
        shutil.copyfile(ROOT / "scripts/install_profiles.sh", self.script)
        (self.root / ".mcp.profiles.json").write_text(json.dumps({"mcpServers": {}}))

    def run_installer(self, *args):
        return subprocess.run(["bash", str(self.script), *args, "--dry-run", "--skip-sam2"],
                              capture_output=True, text=True, cwd=self.root)

    def generated_options(self, *args):
        result = self.run_installer(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]
        configs = [command for command in commands if any(item.endswith("/configure_mcp_profiles.py") for item in command)]
        self.assertEqual(len(configs), 1)
        self.assertFalse((self.root / ".venvs").exists())
        return configs[0]

    def option_values(self, command, flag):
        return [command[i + 1] for i, arg in enumerate(command) if arg == flag]

    def test_all_cpu_backends_configure_both_model_profiles_and_ocr(self):
        command = self.generated_options("all", "--torch-backend", "cpu", "--paddle-backend", "cpu")
        self.assertEqual(self.option_values(command, "--device"), ["cpu"])
        self.assertEqual(self.option_values(command, "--device-profile"), ["vision", "audio"])
        self.assertEqual(self.option_values(command, "--ocr-device"), ["cpu"])

    def test_vision_cuda_does_not_change_audio_or_ocr_device_settings(self):
        command = self.generated_options("vision", "--torch-backend", "cu126")
        self.assertEqual(self.option_values(command, "--device"), ["cuda"])
        self.assertEqual(self.option_values(command, "--device-profile"), ["vision"])
        self.assertNotIn("--ocr-device", command)

    def test_audio_cpu_only_selects_audio_scope(self):
        command = self.generated_options("audio", "--torch-backend", "cpu")
        self.assertEqual(self.option_values(command, "--device-profile"), ["audio"])
        self.assertEqual(self.option_values(command, "--device"), ["cpu"])

    def test_ocr_gpu_does_not_change_torch_device_settings(self):
        command = self.generated_options("ocr", "--paddle-backend", "gpu")
        self.assertEqual(self.option_values(command, "--ocr-device"), ["gpu"])
        self.assertNotIn("--device", command)

    def test_base_install_preserves_model_device_settings(self):
        command = self.generated_options("base")
        self.assertNotIn("--device", command)
        self.assertNotIn("--device-profile", command)
        self.assertNotIn("--ocr-device", command)

    def test_service_exclusions_are_forwarded(self):
        command = self.generated_options("base", "--disable-server", "music-identification",
                                         "--disable-server", "vlm-query-sequence")
        self.assertEqual(self.option_values(command, "--disable-server"),
                         ["music-identification", "vlm-query-sequence"])

    def test_service_exclusion_cannot_be_silently_ignored(self):
        for options in [("--disable-server",), ("--disable-server", "music-identification", "--skip-config")]:
            with self.subTest(options=options):
                result = self.run_installer("base", *options)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--disable-server", result.stderr)
                self.assertNotIn("Installing", result.stdout)


if __name__ == "__main__":
    unittest.main()
