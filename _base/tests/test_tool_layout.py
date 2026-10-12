"""Structural regression checks for the categorized MCP service tree.

These checks do not import model backends, download weights, or require GPUs.
"""
from __future__ import annotations

import ast
from collections import Counter
import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.configure_mcp_profiles import rewrite_config
from scripts.tool_catalog import (
    load_catalog,
    profile_for_server,
    service_dir,
    service_entrypoint,
    service_ids,
)


class ToolLayoutTests(unittest.TestCase):
    def test_catalog_covers_exactly_the_relocated_services(self):
        catalog = load_catalog()
        self.assertEqual(len(catalog), 43)
        self.assertEqual(Counter(info["category"] for info in catalog.values()), {
            "media": 8, "vision": 11, "audio": 4, "text": 2,
            "motion": 5, "spatial": 3, "temporal": 3, "composite": 7,
        })
        actual = set((ROOT / "tools").glob("*/*-mcp/server.py"))
        self.assertEqual(actual, {service_entrypoint(name) for name in service_ids()})
        self.assertEqual(list(ROOT.glob("*-mcp/server.py")), [])
        for name, info in catalog.items():
            with self.subTest(service=name):
                path = service_entrypoint(name)
                self.assertEqual(path.parent.name, f"{name}-mcp")
                self.assertEqual(path.parent.parent.name, info["category"])
                self.assertEqual(profile_for_server(name), info["profile"])
                self.assertTrue((ROOT / "requirements/profiles" / f"{info['profile']}.txt").is_file())

    def test_registered_tools_match_pre_move_catalog(self):
        registered = []
        for name, info in load_catalog().items():
            with self.subTest(service=name):
                tree = ast.parse(service_entrypoint(name).read_text(encoding="utf-8"))
                calls = [node for node in ast.walk(tree)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                         and node.func.id == "create_mcp_server"]
                self.assertEqual(len(calls), 1)
                kwargs = {arg.arg: arg.value for arg in calls[0].keywords}
                handlers = kwargs["handlers"]
                self.assertIsInstance(handlers, ast.Dict)
                tool_ids = [ast.literal_eval(key) for key in handlers.keys]
                self.assertCountEqual(tool_ids, info["tools"])
                registered.extend(tool_ids)
                definitions = ROOT / ast.literal_eval(kwargs["tool_defs_source"])
                self.assertTrue(definitions.is_file(), definitions)
        self.assertEqual(len(registered), 71)
        self.assertEqual(len(set(registered)), 71)

    def test_all_service_requirement_includes_resolve_to_assigned_profile(self):
        requirement_files = list((ROOT / "tools").glob("*/*-mcp/requirements.txt"))
        self.assertTrue(requirement_files)
        for requirement in requirement_files:
            with self.subTest(requirements=requirement):
                lines = [line.strip() for line in requirement.read_text().splitlines()
                         if line.strip() and not line.lstrip().startswith("#")]
                self.assertEqual(len(lines), 1)
                self.assertTrue(lines[0].startswith("-r "))
                resolved = (requirement.parent / lines[0][3:]).resolve()
                profile = profile_for_server(requirement.parent.name)
                self.assertEqual(resolved, ROOT / "requirements/profiles" / f"{profile}.txt")
                self.assertTrue(resolved.is_file())

    def test_cross_service_backends_resolve_through_catalog(self):
        dependencies = {
            "interaction-recognition": {"grounding-dino": "gdino_api.py"},
            "segment-deep-query": {"grounding-dino": "gdino_api.py", "ocr": "ocr_common.py"},
            "spatial-analyze": {"grounding-dino": "gdino_api.py"},
            "obj-count-sequence": {"grounding-dino": "gdino_api.py"},
        }
        for caller, backends in dependencies.items():
            tree = ast.parse(service_entrypoint(caller).read_text())
            referenced = {
                ast.literal_eval(node.args[0]) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "service_dir" and node.args
            }
            for backend, filename in backends.items():
                with self.subTest(caller=caller, backend=backend):
                    self.assertIn(backend, referenced)
                    self.assertTrue((service_dir(backend) / filename).is_file())

    def test_config_rewrites_legacy_and_current_entrypoints_without_losing_options(self):
        for name, info in load_catalog().items():
            for entrypoint in (f"./{name}-mcp/server.py", info["entrypoint"],
                               str(ROOT / f"{name}-mcp/server.py")):
                for wrapper in ([], ["./scripts/mcp_runtime.py"]):
                    with self.subTest(service=name, entrypoint=entrypoint, wrapper=wrapper):
                        config = {"mcpServers": {name: {
                            "command": "python", "args": ["-u", *wrapper, entrypoint, "--custom", "value"],
                            "env": {"HTTPS_PROXY": "http://proxy.example:8080", "CUSTOM_SETTING": "kept"},
                            "custom_metadata": {"enabled": True},
                        }}}
                        before = copy.deepcopy(config)
                        rewritten = rewrite_config(config, ROOT / ".venvs", ROOT)
                        server = rewritten["mcpServers"][name]
                        self.assertEqual(server["command"], f"./.venvs/{info['profile']}/bin/python")
                        self.assertEqual(server["args"], ["-u", "./scripts/mcp_runtime.py", f"./{info['entrypoint']}", "--custom", "value"])
                        expected_env = dict(before["mcpServers"][name]["env"])
                        if name == "segment-deep-query":
                            expected_env["GDINO_PYTHON"] = "./.venvs/vision/bin/python"
                        self.assertEqual(server["env"], expected_env)
                        self.assertEqual(server["custom_metadata"], {"enabled": True})
                        self.assertEqual(config, before)
                        self.assertEqual(rewrite_config(rewritten, ROOT / ".venvs", ROOT), rewritten)

    def test_config_preserves_custom_entrypoints(self):
        config = {"mcpServers": {"ocr": {
            "command": "python", "args": ["/custom/ocr-mcp/server.py", "--verbose"],
        }}}
        rewritten = rewrite_config(config, ROOT / ".venvs", ROOT)
        self.assertEqual(rewritten["mcpServers"]["ocr"]["args"], config["mcpServers"]["ocr"]["args"])

    def test_backlog_services_keep_their_existing_profile_mapping(self):
        # These configured services predate this migration but have no local
        # implementation, so they must not become false entries in the catalog.
        for name in ("video-event-timeline",):
            with self.subTest(service=name):
                self.assertNotIn(name, load_catalog())
                config = {"mcpServers": {name: {
                    "command": "python", "args": [f"./{name}-mcp/server.py"],
                }}}
                rewritten = rewrite_config(config, ROOT / ".venvs", ROOT)
                self.assertEqual(rewritten["mcpServers"][name]["command"], "./.venvs/base/bin/python")
                self.assertEqual(rewritten["mcpServers"][name]["args"], config["mcpServers"][name]["args"])

    def test_catalog_paths_can_be_rebased_without_reading_target_tree(self):
        with tempfile.TemporaryDirectory(prefix="tool catalog with spaces ") as temporary:
            root = Path(temporary)
            for name, info in load_catalog().items():
                self.assertEqual(service_entrypoint(name, root), root / info["entrypoint"])
                self.assertEqual(service_dir(name, root), (root / info["entrypoint"]).parent)


if __name__ == "__main__":
    unittest.main()
