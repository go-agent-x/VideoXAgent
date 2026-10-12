from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "requirements" / "profiles"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.configure_mcp_profiles import PROFILES, profile_for_server, rewrite_config  # noqa: E402
from scripts import verify_profile
from scripts.tool_catalog import service_dir, service_entrypoint, service_ids


class DependencyProfileTests(unittest.TestCase):
    def test_base_resolves_project_and_profile_together(self) -> None:
        for options, suffix in [
            ([], ""),
            (["--dev"], "[dev]"),
            (["--evaluation"], "[evaluation]"),
            (["--dev", "--evaluation"], "[dev,evaluation]"),
        ]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as folder:
                venv_root = Path(folder) / "profiles with spaces"
                result = subprocess.run([
                    "bash", str(ROOT / "scripts/install_profiles.sh"), "base",
                    *options, "--dry-run", "--skip-config", "--venv-root", str(venv_root),
                ], check=True, capture_output=True, text=True)
                commands = [shlex.split(line[2:]) for line in result.stdout.splitlines()
                            if line.startswith("+ ")]
                installs = [cmd for cmd in commands if "-r" in cmd or "-e" in cmd]
                self.assertEqual(installs, [[
                    str(venv_root / "base/bin/python"), "-m", "pip", "install",
                    "-r", str(PROFILE_DIR / "base.txt"), "-e", str(ROOT) + suffix,
                ]])
                self.assertFalse(venv_root.exists())
                self.assertNotIn("passed dependency/smoke checks", result.stdout)

    def test_project_extras_require_base(self) -> None:
        for extra in ("--dev", "--evaluation"):
            with self.subTest(extra=extra):
                result = subprocess.run([
                    "bash", str(ROOT / "scripts/install_profiles.sh"), "audio", extra,
                    "--dry-run", "--skip-config",
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("require the base profile", result.stderr)
                self.assertNotIn("pip install", result.stdout)

    def test_python_path_with_spaces_is_used_for_version_check(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            interpreter = Path(folder) / "python with spaces"
            interpreter.write_text("#!/bin/sh\necho 3.10\n")
            interpreter.chmod(0o755)
            result = subprocess.run([
                "bash", str(ROOT / "scripts/install_profiles.sh"), "base",
                "--python", str(interpreter), "--skip-config",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("currently support Python 3.11; found 3.10", result.stderr)
            self.assertNotIn("Installing", result.stdout)

    def test_all_profiles_install_project_extras_only_in_base(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            venv_root = Path(folder) / "profiles"
            result = subprocess.run([
                "bash", str(ROOT / "scripts/install_profiles.sh"), "all", "--dev", "--evaluation",
                "--dry-run", "--skip-config", "--skip-sam2", "--venv-root", str(venv_root),
            ], check=True, capture_output=True, text=True)
            commands = [shlex.split(line[2:]) for line in result.stdout.splitlines()
                        if line.startswith("+ ")]
            installs = [cmd for cmd in commands if "-r" in cmd]
            self.assertEqual(len(installs), len(PROFILES))
            for profile, command in zip(PROFILES, installs):
                self.assertEqual(command[0], str(venv_root / profile / "bin/python"))
                self.assertEqual(command[command.index("-r") + 1], str(PROFILE_DIR / f"{profile}.txt"))
                self.assertEqual("-e" in command, profile == "base")
            self.assertEqual(installs[0][-1], str(ROOT) + "[dev,evaluation]")
            self.assertFalse(venv_root.exists())

    def test_catalog_fixture_includes_all_repository_mcp_servers(self) -> None:
        servers = {p.parent.name.removesuffix("-mcp") for p in (ROOT / "tools").glob("*/*-mcp/server.py")}
        self.assertEqual(set(self.source["mcpServers"]), servers)
        self.assertEqual(profile_for_server("camera-motion"), "base")
        self.assertEqual(profile_for_server("concat-images"), "base")

    def test_base_check_fails_when_frame_diff_runtime_is_broken(self) -> None:
        with patch.dict(verify_profile.PROFILE_IMPORTS, {"base": ()}), \
             patch.object(verify_profile, "verify_agent_runtime"), \
             patch.object(verify_profile, "verify_frame_diff", side_effect=ModuleNotFoundError("No module named 'skimage'")), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(verify_profile.main(["base"]), 1)
            self.assertIn("FAIL frame-diff SSIM", errors.getvalue())

    def test_rewrite_moves_path_to_new_profile_and_is_idempotent(self) -> None:
        source = {"mcpServers": {"interaction-recognition": {
            "command": "./.venvs/base/bin/python",
            "args": ["./interaction-recognition-mcp/server.py"],
            "env": {"PATH": "./.venvs/base/bin:/usr/bin:/custom/venv/bin-extra",
                    "HTTPS_PROXY": "http://proxy.example:8080"},
        }}}
        result = rewrite_config(source, ROOT / ".venvs", ROOT)
        server = result["mcpServers"]["interaction-recognition"]
        self.assertEqual(server["env"]["PATH"],
                         "./.venvs/vision/bin:/usr/bin:/custom/venv/bin-extra")
        self.assertEqual(server["env"]["HTTPS_PROXY"], "http://proxy.example:8080")
        self.assertEqual(rewrite_config(result, ROOT / ".venvs", ROOT), result)

    def test_cli_defaults_to_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "mcp.json"
            config = {"mcpServers": {"interaction-recognition": {
                "command": "python", "env": {"HTTPS_PROXY": "http://proxy.example:8080"},
            }}}
            output.write_text(json.dumps(config))
            subprocess.run([sys.executable, str(ROOT / "scripts/configure_mcp_profiles.py"),
                            "--output", str(output)], check=True, capture_output=True)
            result = json.loads(output.read_text())
            self.assertEqual(set(result["mcpServers"]), {"interaction-recognition"})
            self.assertEqual(result["mcpServers"]["interaction-recognition"]["env"],
                             config["mcpServers"]["interaction-recognition"]["env"])

    def test_installer_preserves_existing_config_and_selects_cuda_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "custom.json"
            original = '{"mcpServers": {}}\n'
            output.write_text(original)
            result = subprocess.run([
                "bash", str(ROOT / "scripts/install_profiles.sh"), "vision", "audio",
                "--dry-run", "--skip-sam2", "--torch-backend", "cu126",
                "--venv-root", str(Path(temp_dir) / "profiles"),
                "--config-output", str(output),
            ], check=True, capture_output=True, text=True)
            self.assertIn("torch==2.11.0+cu126", result.stdout)
            self.assertIn("torchaudio==2.11.0+cu126", result.stdout)
            self.assertIn(f"--source {output}", result.stdout)
            self.assertIn("vision --torch-backend cu126 --skip-sam2", result.stdout)
            self.assertEqual(output.read_text(), original)

    def test_backend_check_rejects_cpu_build_for_cuda_request(self) -> None:
        torch = SimpleNamespace(version=SimpleNamespace(cuda=None), __version__="2.11.0+cpu")
        with patch.dict(verify_profile.PROFILE_IMPORTS, {"vision": ()}), \
             patch.object(verify_profile.importlib, "import_module", return_value=torch), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(verify_profile.main(["vision", "--torch-backend", "cu126"]), 1)
            self.assertEqual(verify_profile.main(["vision", "--torch-backend", "cpu"]), 0)

    def test_interaction_recognition_uses_grounding_dino_environment(self) -> None:
        self.assertEqual(profile_for_server("interaction-recognition"), "vision")
        requirements = (service_dir("interaction-recognition") / "requirements.txt").read_text()
        self.assertIn("-r ../../../requirements/profiles/vision.txt", requirements)

    def test_existing_venv_interpreter_symlink_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            venv_root = root / ".venvs"
            interpreter = venv_root / "vision" / "bin" / "python"
            interpreter.parent.mkdir(parents=True)
            interpreter.symlink_to(sys.executable)
            source = {"mcpServers": {"depth-estimation": {"command": "python"}}}
            for repo_root, expected in [
                (root, "./.venvs/vision/bin/python"),
                (root / "other-repo", str(interpreter)),
            ]:
                rewritten = rewrite_config(source, venv_root, repo_root)
                self.assertEqual(
                    rewritten["mcpServers"]["depth-estimation"]["command"], expected
                )

    @classmethod
    def setUpClass(cls) -> None:
        # Generate portable service paths without depending on user settings.
        cls.source = {"mcpServers": {
            name: {
                "command": "python",
                "args": ["./" + service_entrypoint(name).relative_to(ROOT).as_posix()],
                "env": {"HTTPS_PROXY": "http://proxy.example:8080", "CUSTOM_SETTING": name},
            }
            for name in service_ids()
        }}
        directory = tempfile.TemporaryDirectory(prefix="mcp-profile-fixture-")
        cls.addClassCleanup(directory.cleanup)
        cls.source_path = Path(directory.name) / "services.json"
        cls.source_path.write_text(json.dumps(cls.source), encoding="utf-8")

    def test_catalog_config_is_valid_and_rewritable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_root = Path(temp_dir) / ".venvs"
            rewritten = rewrite_config(self.source, venv_root, ROOT)

        servers = rewritten["mcpServers"]
        self.assertTrue(servers["video-duration"]["command"].endswith("/base/bin/python"))
        self.assertTrue(servers["depth-estimation"]["command"].endswith("/vision/bin/python"))
        self.assertTrue(servers["audio-transcription"]["command"].endswith("/audio/bin/python"))
        self.assertTrue(servers["ocr"]["command"].endswith("/ocr/bin/python"))
        self.assertTrue(servers["music-identification"]["command"].endswith("/music/bin/python"))
        for name, server in servers.items():
            with self.subTest(service=name):
                expected_env = dict(self.source["mcpServers"][name]["env"])
                if name == "segment-deep-query":
                    expected_env["GDINO_PYTHON"] = str(venv_root / "vision/bin/python")
                self.assertEqual(server["env"], expected_env)

    def test_every_configured_server_has_a_known_profile(self) -> None:
        assigned = {profile_for_server(name) for name in self.source["mcpServers"]}
        self.assertEqual(assigned, set(PROFILES))
        for server_name in self.source["mcpServers"]:
            profile = profile_for_server(server_name)
            self.assertTrue((PROFILE_DIR / f"{profile}.txt").is_file())

    def test_configured_server_scripts_exist(self) -> None:
        for server_name, server in self.source["mcpServers"].items():
            script = server.get("args", [None])[0]
            self.assertIsInstance(script, str, server_name)
            self.assertTrue((ROOT / script).is_file(), f"{server_name}: {script}")

    def test_unknown_server_is_not_silently_assigned_to_base(self) -> None:
        with self.assertRaises(ValueError):
            profile_for_server("new-unmapped-model-server")

    def test_tool_requirements_only_delegate_to_profiles(self) -> None:
        for requirements in (ROOT / "tools").glob("*/*-mcp/requirements.txt"):
            active_lines = [
                line.strip()
                for line in requirements.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            self.assertEqual(len(active_lines), 1, requirements)
            self.assertTrue(
                active_lines[0].startswith("-r ../../../requirements/profiles/"), requirements
            )

    def test_numpy_conflict_is_isolated(self) -> None:
        ocr = (PROFILE_DIR / "ocr.txt").read_text(encoding="utf-8")
        music = (PROFILE_DIR / "music.txt").read_text(encoding="utf-8")
        common = (PROFILE_DIR / "common.txt").read_text(encoding="utf-8")

        self.assertIn("numpy==1.26.4", ocr)
        self.assertIn("numpy==1.26.4", common)
        self.assertIn("numpy==2.2.6", music)
        self.assertNotIn(
            "--no-deps",
            "".join(path.read_text(encoding="utf-8") for path in PROFILE_DIR.glob("*.txt")),
        )

    def test_installer_dry_run_has_no_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_root = Path(temp_dir) / "profiles"
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts" / "install_profiles.sh"),
                    "all",
                    "--dry-run",
                    "--skip-config",
                    "--skip-sam2",
                    "--venv-root",
                    str(venv_root),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Installing base profile", result.stdout)
            self.assertIn("Installing music profile", result.stdout)
            self.assertFalse(venv_root.exists())

    def test_config_cli_check_mode_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "mcp.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "configure_mcp_profiles.py"),
                    "--source",
                    str(self.source_path),
                    "--output",
                    str(output),
                    "--check",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(output.exists())


class ConfigSourcePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="mcp-source-policy-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "scripts").mkdir()
        (self.root / "tools").mkdir()
        for name in ("configure_mcp_profiles.py", "tool_catalog.py"):
            shutil.copyfile(ROOT / "scripts" / name, self.root / "scripts" / name)
        shutil.copyfile(ROOT / "tools/catalog.json", self.root / "tools/catalog.json")
        self.output = self.root / "generated" / "mcp.json"
        self.default_source = self.root / ".mcp.profiles.json"

    def write_config(self, path: Path, marker: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mcpServers": {"ocr": {
            "command": "python",
            "args": ["./" + service_entrypoint("ocr").relative_to(ROOT).as_posix()],
            "env": {"CUSTOM_SOURCE": marker, "HTTPS_PROXY": "http://proxy.example:8080"},
        }}}), encoding="utf-8")

    def run_config(self, *options: str) -> subprocess.CompletedProcess:
        return subprocess.run([
            sys.executable, str(self.root / "scripts/configure_mcp_profiles.py"),
            "--output", str(self.output), *options,
        ], cwd=self.root, capture_output=True, text=True)

    def assert_source_marker(self, expected: str) -> None:
        server = json.loads(self.output.read_text())["mcpServers"]["ocr"]
        self.assertEqual(server["env"], {
            "CUSTOM_SOURCE": expected, "HTTPS_PROXY": "http://proxy.example:8080",
        })
        self.assertEqual(server["command"], "./.venvs/ocr/bin/python")

    def test_explicit_source_overrides_existing_output_and_default(self) -> None:
        source = self.root / "custom.json"
        self.write_config(source, "explicit")
        self.write_config(self.output, "output")
        self.write_config(self.default_source, "default")
        original = source.read_text()
        result = self.run_config("--source", str(source))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_source_marker("explicit")
        self.assertEqual(source.read_text(), original)

    def test_existing_output_overrides_default(self) -> None:
        self.write_config(self.output, "output")
        self.write_config(self.default_source, "default")
        result = self.run_config()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_source_marker("output")

    def test_default_config_is_used_for_a_new_output(self) -> None:
        self.write_config(self.default_source, "default")
        result = self.run_config()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_source_marker("default")

    def test_missing_sources_fail_with_actionable_error(self) -> None:
        result = self.run_config()
        self.assertEqual(result.returncode, 2)
        self.assertIn("--source", result.stderr)
        self.assertFalse(self.output.exists())

    def test_malformed_existing_output_is_not_bypassed(self) -> None:
        self.write_config(self.default_source, "default")
        self.output.parent.mkdir(parents=True)
        original = '{"mcpServers": '
        self.output.write_text(original)
        result = self.run_config()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.output.read_text(), original)

    def test_missing_explicit_source_does_not_fall_back(self) -> None:
        self.write_config(self.default_source, "default")
        self.write_config(self.output, "output")
        original = self.output.read_text()
        result = self.run_config("--source", str(self.root / "missing.json"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("--source", result.stderr)
        self.assertEqual(self.output.read_text(), original)

    def test_broken_output_symlink_is_not_replaced_by_default(self) -> None:
        self.write_config(self.default_source, "default")
        self.output.parent.mkdir(parents=True)
        target = self.root / "missing-target.json"
        self.output.symlink_to(target)
        result = self.run_config()
        self.assertEqual(result.returncode, 2)
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(self.output.readlink(), target)
        self.assertFalse(target.exists())


class InstallerConfigSourcePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="installer-source-policy-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "scripts").mkdir()
        self.installer = self.root / "scripts/install_profiles.sh"
        shutil.copyfile(ROOT / "scripts/install_profiles.sh", self.installer)
        self.output = self.root / "generated" / "mcp.json"
        self.venv_root = self.root / "profiles"

    def run_installer(self, *options: str) -> subprocess.CompletedProcess:
        return subprocess.run([
            "bash", str(self.installer), "base", "--dry-run",
            "--config-output", str(self.output), "--venv-root", str(self.venv_root),
            *options,
        ], cwd=self.root, capture_output=True, text=True)

    def test_new_output_uses_repository_config_without_writing(self) -> None:
        source = self.root / ".mcp.profiles.json"
        original = '{"mcpServers": {}}\n'
        source.write_text(original)
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line[2:]) for line in result.stdout.splitlines()
                    if line.startswith("+ ")]
        configured = [cmd for cmd in commands if "--source" in cmd]
        self.assertEqual(len(configured), 1)
        self.assertEqual(configured[0][configured[0].index("--source") + 1], str(source))
        self.assertEqual(source.read_text(), original)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.venv_root.exists())

    def test_bad_explicit_source_fails_before_installing(self) -> None:
        (self.root / ".mcp.profiles.json").write_text('{"mcpServers": {}}')
        result = self.run_installer("--config-source", str(self.root / "missing.json"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("--config-source", result.stderr)
        self.assertNotIn("Installing", result.stdout)
        self.assertFalse(self.venv_root.exists())

    def test_missing_sources_fail_with_instructions_before_installing(self) -> None:
        result = self.run_installer()
        self.assertEqual(result.returncode, 2)
        self.assertIn("--config-source", result.stderr)
        self.assertIn("--skip-config", result.stderr)
        self.assertNotIn("Installing", result.stdout)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.venv_root.exists())

    def test_skip_config_does_not_require_a_source(self) -> None:
        result = self.run_installer("--skip-config", "--config-source", str(self.root / "missing.json"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Installing base profile", result.stdout)
        self.assertNotIn("configure_mcp_profiles.py", result.stdout)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.venv_root.exists())


if __name__ == "__main__":
    unittest.main()
