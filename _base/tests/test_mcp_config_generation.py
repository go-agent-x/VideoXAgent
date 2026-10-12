"""Configuration generation regressions without model imports or downloads."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.configure_mcp_profiles import rewrite_config
from scripts.tool_catalog import service_entrypoint


class MCPConfigurationGenerationTests(unittest.TestCase):
    def test_composite_uses_the_selected_vision_environment(self):
        config = {"mcpServers": {"segment-deep-query": {
            "command": "./.venvs/base/bin/python",
            "args": ["./tools/composite/segment-deep-query-mcp/server.py"],
            "env": {"CUSTOM_SETTING": "preserved"},
        }}}
        first = rewrite_config(config, ROOT / ".venvs", ROOT)
        self.assertEqual(first["mcpServers"]["segment-deep-query"]["env"], {
            "CUSTOM_SETTING": "preserved", "GDINO_PYTHON": "./.venvs/vision/bin/python",
        })
        changed = rewrite_config(first, ROOT / "custom profiles", ROOT)
        self.assertEqual(changed["mcpServers"]["segment-deep-query"]["env"]["GDINO_PYTHON"],
                         "./custom profiles/vision/bin/python")
        self.assertEqual(rewrite_config(changed, ROOT / "custom profiles", ROOT), changed)
        first["mcpServers"]["segment-deep-query"]["env"]["GDINO_PYTHON"] = "/explicit/detector/bin/python"
        custom = rewrite_config(first, ROOT / "custom profiles", ROOT)
        self.assertEqual(custom["mcpServers"]["segment-deep-query"]["env"]["GDINO_PYTHON"],
                         "/explicit/detector/bin/python")

    def configuration(self, name="ocr", args=None, env=None):
        server = {
            "command": "python",
            "args": args if args is not None else [f"./{name}-mcp/server.py"],
        }
        if env is not None:
            server["env"] = copy.deepcopy(env)
        return {"mcpServers": {name: server}}

    def rewrite(self, config, **kwargs):
        return rewrite_config(config, ROOT / ".venvs", ROOT, **kwargs)

    def test_known_entrypoints_are_wrapped_and_normalized_once(self):
        current = "./" + service_entrypoint("ocr").relative_to(ROOT).as_posix()
        for entrypoint in (
            "./ocr-mcp/server.py", current, str(ROOT / "ocr-mcp/server.py"),
            str(service_entrypoint("ocr")),
        ):
            for flags in ([], ["-u", "-B"], ["-OO", "-X", "dev"], ["--"]):
                with self.subTest(entrypoint=entrypoint, flags=flags):
                    source = self.configuration(args=[*flags, entrypoint, "--custom", "value"])
                    before = copy.deepcopy(source)
                    output = self.rewrite(source)
                    self.assertEqual(
                        output["mcpServers"]["ocr"]["args"],
                        [*flags, "./scripts/mcp_runtime.py", current, "--custom", "value"],
                    )
                    self.assertEqual(self.rewrite(output), output)
                    self.assertEqual(source, before)

    def test_interpreter_option_values_are_never_treated_as_script_slots(self):
        current = "./" + service_entrypoint("ocr").relative_to(ROOT).as_posix()
        options = [
            ["-W", "./ocr-mcp/server.py"],
            ["-X", "./ocr-mcp/server.py"],
            ["--check-hash-based-pycs", "always"],
            ["-Wignore", "-Xdev", "--check-hash-based-pycs=always"],
        ]
        for flags in options:
            with self.subTest(flags=flags):
                source = self.configuration(args=[*flags, "--", "./ocr-mcp/server.py"])
                output = self.rewrite(source)
                self.assertEqual(
                    output["mcpServers"]["ocr"]["args"],
                    [*flags, "--", "./scripts/mcp_runtime.py", current],
                )

    def test_existing_wrapper_is_preserved_without_duplication(self):
        current = "./" + service_entrypoint("ocr").relative_to(ROOT).as_posix()
        for wrapper in ("./scripts/mcp_runtime.py", str(ROOT / "scripts/mcp_runtime.py")):
            with self.subTest(wrapper=wrapper):
                source = self.configuration(args=["-u", wrapper, "./ocr-mcp/server.py", "--custom"])
                output = self.rewrite(source)
                self.assertEqual(
                    output["mcpServers"]["ocr"]["args"],
                    ["-u", wrapper, current, "--custom"],
                )
                self.assertEqual(self.rewrite(output), output)

    def test_wrapper_in_server_arguments_does_not_suppress_real_wrapper(self):
        source = self.configuration(args=[
            "./ocr-mcp/server.py", "--config", "./scripts/mcp_runtime.py",
        ])
        output = self.rewrite(source)
        self.assertEqual(output["mcpServers"]["ocr"]["args"], [
            "./scripts/mcp_runtime.py", "./tools/text/ocr-mcp/server.py",
            "--config", "./scripts/mcp_runtime.py",
        ])
        self.assertEqual(self.rewrite(output), output)

    def test_custom_launch_modes_and_non_script_values_are_unchanged(self):
        cases = (
            ["-m", "custom_launcher", "./ocr-mcp/server.py"],
            ["-mcustom_launcher", "./ocr-mcp/server.py"],
            ["-c", "print('custom')", "./ocr-mcp/server.py"],
            ["-cprint('custom')", "./ocr-mcp/server.py"],
            ["./launcher.py", "--config", "./ocr-mcp/server.py"],
            ["--config", "./ocr-mcp/server.py"],
            ["/custom/ocr-mcp/server.py", "--config", "./ocr-mcp/server.py"],
            ["./scripts/mcp_runtime.py", "/custom/server.py", "./ocr-mcp/server.py"],
            ["-", "./ocr-mcp/server.py"],
            ["-W", "./ocr-mcp/server.py"],
            ["-X"],
            ["--"],
            [],
        )
        for args in cases:
            with self.subTest(args=args):
                output = self.rewrite(self.configuration(args=args))
                self.assertEqual(output["mcpServers"]["ocr"]["args"], args)

    def test_absent_historical_services_are_not_wrapped(self):
        for name in ("video-event-timeline",):
            with self.subTest(name=name):
                source = self.configuration(name)
                output = self.rewrite(source)
                self.assertEqual(output["mcpServers"][name]["args"], source["mcpServers"][name]["args"])
                self.assertEqual(output["mcpServers"][name]["command"], "./.venvs/base/bin/python")

    def test_default_device_options_preserve_environment_and_metadata(self):
        env = {
            "MCP_DEVICE": "cuda", "SAM_DEVICE": "cuda", "SAM2_DEVICE": "cuda",
            "OCR_USE_GPU": "1", "VLM_MODEL_NAME": "custom-model",
            "https_proxy": "http://proxy.example:8080", "CUDA_VISIBLE_DEVICES": "2",
        }
        source = self.configuration("object-track", env=env)
        source["mcpServers"]["object-track"]["custom_metadata"] = {"keep": True}
        output = self.rewrite(source)
        self.assertEqual(output["mcpServers"]["object-track"]["env"], env)
        self.assertEqual(output["mcpServers"]["object-track"]["custom_metadata"], {"keep": True})

    def test_cpu_updates_schema_devices_and_cross_model_consumers(self):
        expected = {
            "grounding-dino": {"GDINO_DEVICE"},
            "auto-segment": {"SAM_DEVICE"},
            "object-track": {"SAM2_DEVICE"},
            "obj-count-sequence": {"SAM2_DEVICE", "GDINO_DEVICE"},
            "motion-analyze": {"SAM2_DEVICE"},
            "spatial-analyze": {"SAM2_DEVICE", "GDINO_DEVICE"},
            "interaction-recognition": {"GDINO_DEVICE"},
            "segment-deep-query": {"GDINO_DEVICE"},
            "depth-estimation": {"DEPTH_DEVICE"},
            "emotion-analysis": {"EMOTION_DEVICE"},
            "gaze-estimation": {"GAZE_DEVICE"},
            "audio-transcription": set(),
            "optical-flow": set(),
            "action-classify": set(),
            "temporal-ground": set(),
            "moment-retrieval": set(),
        }
        for name, keys in expected.items():
            with self.subTest(name=name):
                source = self.configuration(name, env={
                    "SAM_DEVICE": "cuda", "CUDA_VISIBLE_DEVICES": "3",
                    "TOOL_CUSTOM__SAM_DEVICE": "cuda",
                })
                output = self.rewrite(source, device="cpu")
                env = output["mcpServers"][name]["env"]
                self.assertEqual(env["MCP_DEVICE"], "cpu")
                self.assertEqual(env["SAM_DEVICE"], "cpu")
                for key in keys:
                    self.assertEqual(env[key], "cpu")
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
                self.assertEqual(env["TOOL_CUSTOM__SAM_DEVICE"], "cuda")
                self.assertEqual(self.rewrite(output, device="cpu"), output)

    def test_cpu_insightface_and_cuda_provider_preservation(self):
        for name in ("face-cluster", "gaze-estimation", "emotion-analysis", "active-speaker-detection"):
            with self.subTest(name=name):
                source = self.configuration(name, env={"INSIGHTFACE_PROVIDERS": "CustomProvider,CPUExecutionProvider"})
                cpu = self.rewrite(source, device="cpu")["mcpServers"][name]["env"]
                cuda = self.rewrite(source, device="cuda")["mcpServers"][name]["env"]
                self.assertEqual(cpu["INSIGHTFACE_PROVIDERS"], "CPUExecutionProvider")
                self.assertEqual(cuda["INSIGHTFACE_PROVIDERS"], "CustomProvider,CPUExecutionProvider")
                self.assertEqual(cuda["MCP_DEVICE"], "cuda")
        env = self.rewrite(self.configuration("face-cluster"), device="cuda")["mcpServers"]["face-cluster"]["env"]
        self.assertNotIn("INSIGHTFACE_PROVIDERS", env)

    def test_device_profile_selection_preserves_other_profiles(self):
        config = {"mcpServers": {}}
        for name in ("object-track", "audio-transcription", "segment-deep-query", "video-duration"):
            config["mcpServers"].update(self.configuration(name, env={"MCP_DEVICE": "cuda"})["mcpServers"])
        output = self.rewrite(config, device="cpu", device_profiles=["vision"])
        for name in ("object-track", "segment-deep-query"):
            self.assertEqual(output["mcpServers"][name]["env"]["MCP_DEVICE"], "cpu")
        for name in ("audio-transcription", "video-duration"):
            self.assertEqual(output["mcpServers"][name]["env"], config["mcpServers"][name]["env"])
        audio = self.rewrite(config, device="cpu", device_profiles=["audio"])
        self.assertEqual(audio["mcpServers"]["audio-transcription"]["env"]["MCP_DEVICE"], "cpu")
        self.assertEqual(audio["mcpServers"]["segment-deep-query"]["env"], {
            **config["mcpServers"]["segment-deep-query"]["env"],
            "GDINO_PYTHON": "./.venvs/vision/bin/python",
        })

    def test_ocr_device_updates_both_consumers_independently(self):
        for name in ("ocr", "segment-deep-query"):
            for device, expected in (("cpu", "0"), ("gpu", "1"), ("cuda", "1")):
                with self.subTest(name=name, device=device):
                    source = self.configuration(name, env={"MCP_DEVICE": "cuda", "OCR_USE_GPU": "previous"})
                    env = self.rewrite(source, ocr_device=device)["mcpServers"][name]["env"]
                    self.assertEqual(env["OCR_USE_GPU"], expected)
                    self.assertEqual(env["MCP_DEVICE"], "cuda")
        source = self.configuration("object-track", env={"OCR_USE_GPU": "keep"})
        self.assertEqual(self.rewrite(source, ocr_device="cpu")["mcpServers"]["object-track"]["env"], {"OCR_USE_GPU": "keep"})

    def test_disabling_known_absent_and_repeated_names_is_idempotent(self):
        source = self.configuration("music-identification")
        source["mcpServers"].update(self.configuration("ocr")["mcpServers"])
        disabled = ["music-identification", "music-identification-mcp", "vlm-query-sequence", "video-event-timeline"]
        output = self.rewrite(source, disable_servers=disabled)
        self.assertEqual(set(output["mcpServers"]), {"ocr"})
        self.assertEqual(self.rewrite(output, disable_servers=disabled), output)
        self.assertEqual(set(source["mcpServers"]), {"music-identification", "ocr"})
        with self.assertRaisesRegex(ValueError, "Unknown MCP service.*music-typo"):
            self.rewrite(source, disable_servers=["music-typo"])

    def test_invalid_device_options_fail_without_mutating_input(self):
        source = self.configuration()
        before = copy.deepcopy(source)
        for kwargs in (
            {"device": "gpu"}, {"ocr_device": "invalid"},
            {"device": "cpu", "device_profiles": ["base"]},
            {"device_profiles": ["vision"]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.rewrite(source, **kwargs)
        self.assertEqual(source, before)

    def test_cli_combines_options_and_check_does_not_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.json"
            output = root / "output.json"
            config = {"mcpServers": {}}
            for name in ("music-identification", "object-track", "audio-transcription", "ocr"):
                config["mcpServers"].update(self.configuration(name)["mcpServers"])
            source.write_text(json.dumps(config))
            args = [
                sys.executable, "-B", str(ROOT / "scripts/configure_mcp_profiles.py"),
                "--source", str(source), "--output", str(output),
                "--device", "cpu", "--device-profile", "vision", "--ocr-device", "cpu",
                "--disable-server", "music-identification", "--disable-server", "music-identification",
            ]
            subprocess.run([*args, "--check"], check=True, capture_output=True, text=True)
            self.assertFalse(output.exists())
            subprocess.run(args, check=True, capture_output=True, text=True)
            servers = json.loads(output.read_text())["mcpServers"]
            self.assertNotIn("music-identification", servers)
            self.assertEqual(servers["object-track"]["env"]["SAM2_DEVICE"], "cpu")
            self.assertNotIn("env", servers["audio-transcription"])
            self.assertEqual(servers["ocr"]["env"]["OCR_USE_GPU"], "0")


if __name__ == "__main__":
    unittest.main()
