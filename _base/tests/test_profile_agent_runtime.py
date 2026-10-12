from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import verify_profile  # noqa: E402


class AgentRuntimeVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo_root = Path(self.temp_dir.name)
        # A valid CLI exits at --help before any runtime action takes place.
        (self.repo_root / "video-agent.py").write_text(
            "import argparse\n"
            "argparse.ArgumentParser().parse_args()\n"
            "raise RuntimeError('Agent execution must not start during verification')\n"
        )
        demo = self.repo_root / "demo"
        demo.mkdir()
        (demo / "__init__.py").write_text("")
        for module in ("agent_runner", "agent_server"):
            (demo / f"{module}.py").write_text(
                "if __name__ == '__main__':\n"
                "    raise RuntimeError('Demo execution must not start during verification')\n"
            )

    def test_entry_points_are_checked_without_executing_agents(self) -> None:
        verify_profile.verify_agent_runtime(self.repo_root)
        self.assertFalse(list(self.repo_root.rglob("__pycache__")))

    def test_incompatible_mcp_loader_api_fails_cli_validation(self) -> None:
        # A package can exist and import successfully while required symbols do
        # not exist. The real CLI smoke check must detect that incompatibility.
        (self.repo_root / "example_loader.py").write_text("load_mcp_config = None\n")
        (self.repo_root / "video-agent.py").write_text(
            "from example_loader import load_mcp_config, _load_tools_from_config\n"
        )
        with self.assertRaises(RuntimeError) as error:
            verify_profile.verify_agent_runtime(self.repo_root)
        self.assertIn("video-agent --help", str(error.exception))
        self.assertIn("ImportError", str(error.exception))

    def test_missing_demo_dependency_is_not_silently_skipped(self) -> None:
        (self.repo_root / "demo" / "agent_runner.py").write_text(
            "import missing_profile_smoke_test_dependency\n"
        )
        with self.assertRaises(RuntimeError) as error:
            verify_profile.verify_agent_runtime(self.repo_root)
        self.assertIn("web demo imports", str(error.exception))
        self.assertIn("missing_profile_smoke_test_dependency", str(error.exception))

    def test_agent_runtime_failure_makes_base_verification_fail(self) -> None:
        with patch.dict(verify_profile.PROFILE_IMPORTS, {"base": ()}), \
             patch.object(verify_profile, "verify_frame_diff"), \
             patch.object(verify_profile, "verify_agent_runtime",
                          side_effect=ImportError("incompatible Agent runtime")), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(verify_profile.main(["base"]), 1)
        self.assertIn("FAIL Agent runtime: ImportError", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
