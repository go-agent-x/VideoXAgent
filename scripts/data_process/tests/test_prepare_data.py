"""Offline integration checks for dataset preparation in temporary checkouts."""

import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]
EXPORTERS = (
    "export_longvideobench.py", "export_lvbench.py", "export_minerva_data.py",
    "export_videomme_data.py", "export_videomme_v2_data.py",
)


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.root = self.work / "checkout with a quote's"
        self.scripts = self.root / "scripts/data_process"
        self.scripts.mkdir(parents=True)
        self.launcher = self.scripts / "prepare_all_data.sh"
        shutil.copyfile(SCRIPTS / "prepare_all_data.sh", self.launcher)
        # Parquet converters are unchanged. Stubs exercise the shell's dispatch
        # and failure propagation without requiring pandas or a real dataset.
        stub = (
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "with open('export-calls.jsonl', 'a') as handle:\n"
            "    handle.write(json.dumps([Path(__file__).name, *sys.argv[1:]]) + '\\n')\n"
            "if os.environ.get('FAIL_EXPORT') == Path(__file__).name:\n"
            "    raise SystemExit(17)\n"
        )
        for name in EXPORTERS:
            (self.scripts / name).write_text(stub)
        self.source = self.work / "source data"
        self.source.mkdir()
        self.env = os.environ.copy()
        for name in ("DATA_SRC", "PYTHON_BIN", "FAIL_EXPORT"):
            self.env.pop(name, None)
        self.env["PREPARE_PYTHON_LOG"] = str(self.work / "python-calls.log")

    def python_wrapper(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '#!/bin/sh\n'
            'printf "%s\\n" "$*" >> "$PREPARE_PYTHON_LOG"\n'
            f'exec {shlex.quote(sys.executable)} "$@"\n'
        )
        path.chmod(0o755)

    def archive(self, relative, entries):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w") as archive:
            for name, contents in entries.items():
                payload = contents.encode()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    def inputs(self):
        self.archive("longvideobench/videos.tar", {"videos/lvb.mp4": "fixture"})
        self.archive("longvideobench/subtitles.tar", {"subtitles/lvb.json": "[]"})
        (self.source / "longvideobench/lvb_val.json").write_text("[]")
        self.archive("longvideo/LVBench.tar", {"LVBench/video_info.meta.jsonl": ""})
        self.archive("videomme/videomme.tar", {"videomme/test-00000-of-00001.parquet": "fixture"})
        self.archive("videomme/Video-MME-v2.tar", {"Video-MME-v2/test.parquet": "fixture"})
        videos = self.source / "minerva/youtube_videos/youtube_cache"
        videos.mkdir(parents=True)
        (videos / "clip1.mp4").write_text("fixture, not a decodable video")
        row = {
            "key": "sample-1", "video_id": "clip1", "question": "What is visible?",
            "answer_id": 0, "answer": "A cup", "question_type": "recognition",
            "category": "objects", "split": "STEM", "reasoning": "A cup is shown.",
            **{f"answer_choice_{i}": text for i, text in enumerate(("A cup", "A dog", "A car", "A tree", "A boat"))},
        }
        (self.source / "minerva/minerva.json").write_text(json.dumps([row]))

    def run_prepare(self, *args):
        return subprocess.run(
            ["bash", str(self.launcher), *map(str, args)], cwd=self.work,
            env=self.env, capture_output=True, text=True, timeout=30,
        )

    def calls(self):
        path = self.root / "export-calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_default_all_uses_base_python_and_expected_exporters(self):
        self.inputs()
        self.python_wrapper(self.root / ".venvs/base/bin/python")
        self.env["DATA_SRC"] = str(self.source)
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(self.calls(), [
            ["export_longvideobench.py", "short"], ["export_longvideobench.py", "medium"],
            ["export_longvideobench.py", "long"], ["export_lvbench.py", "all"],
            ["export_minerva_data.py"], ["export_videomme_data.py", "short"],
            ["export_videomme_data.py", "medium"], ["export_videomme_data.py", "long"],
            ["export_videomme_v2_data.py"],
        ])
        self.assertEqual(len((self.work / "python-calls.log").read_text().splitlines()), 9)
        for relative in (
            "data/LongVideoBench/videos/lvb.mp4", "data/LongVideoBench/subtitles/lvb.json",
            "data/LVBench/video_info.meta.jsonl", "data/videomme/test-00000-of-00001.parquet",
            "data/Video-MME-v2/test.parquet", "data/youtube_cache/clip1.mp4",
        ):
            self.assertTrue((self.root / relative).is_file(), relative)
        self.assertFalse((self.work / "data").exists())

    def test_minerva_real_export_relative_paths_and_cli_source_override(self):
        self.inputs()
        shutil.copyfile(SCRIPTS / "export_minerva_data.py", self.scripts / "export_minerva_data.py")
        self.python_wrapper(self.work / "custom runtime/python")
        self.env["PYTHON_BIN"] = "custom runtime/python"
        self.env["DATA_SRC"] = str(self.work / "wrong source")
        result = self.run_prepare("minerva", "--data-src", self.source.name)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        output = self.root / "data/minerva/test/minerva_all_data.jsonl"
        item = json.loads(output.read_text())
        self.assertEqual(item["question_id"], "clip1-1")
        self.assertEqual(Path(item["video_path"]).resolve(), (self.root / "data/youtube_cache/clip1.mp4").resolve())
        self.assertIn("Given a video: " + item["video_path"], item["prompt"])

    def test_skip_needs_no_source_or_python_and_force_reexports(self):
        output = self.root / "data/minerva/test/minerva_all_data.jsonl"
        output.parent.mkdir(parents=True)
        output.write_text("previous export\n")
        result = self.run_prepare("minerva")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.read_text(), "previous export\n")
        self.assertEqual(self.calls(), [])
        self.inputs()
        self.python_wrapper(self.root / ".venvs/base/bin/python")
        shutil.copyfile(SCRIPTS / "export_minerva_data.py", self.scripts / "export_minerva_data.py")
        result = self.run_prepare("minerva", "--force", "--data-src", self.source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())["question_id"], "clip1-1")

    def test_preflight_checks_every_selected_source_before_copying(self):
        self.inputs()
        self.python_wrapper(self.root / ".venvs/base/bin/python")
        (self.source / "videomme/Video-MME-v2.tar").unlink()
        result = self.run_prepare("--data-src", self.source, "minerva", "videomme_v2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Video-MME-v2.tar", result.stderr)
        self.assertFalse((self.root / "data").exists())
        self.assertEqual(self.calls(), [])

    def test_export_failure_stops_remaining_datasets(self):
        self.inputs()
        self.python_wrapper(self.root / ".venvs/base/bin/python")
        self.env["FAIL_EXPORT"] = "export_longvideobench.py"
        result = self.run_prepare("--data-src", self.source, "longvideobench", "lvbench")
        self.assertEqual(result.returncode, 17)
        self.assertEqual(self.calls(), [["export_longvideobench.py", "short"]])
        self.assertFalse((self.root / "data/LVBench").exists())
        self.assertNotIn("prepared successfully", result.stdout)

    def test_missing_source_python_and_invalid_arguments_fail(self):
        for args, message in (
            (("minerva",), "set DATA_SRC"),
            (("--data-src",), "requires a directory"),
            (("unknown",), "unknown dataset"),
            (("--bad-option",), "unknown option"),
            (("--data-src", self.source, "minerva"), "base Python not found"),
        ):
            with self.subTest(args=args):
                result = self.run_prepare(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
        self.assertFalse((self.root / "data").exists())

    def test_failed_export_is_retried_even_when_all_output_paths_exist(self):
        self.inputs()
        self.python_wrapper(self.root / ".venvs/base/bin/python")
        (self.scripts / "export_longvideobench.py").write_text(
            "import os, sys\n"
            "from pathlib import Path\n"
            "output = Path('data/LongVideoBench/test')\n"
            "output.mkdir(parents=True, exist_ok=True)\n"
            "(output / (sys.argv[1] + '_duration_data.jsonl')).write_text('partial\\n')\n"
            "if sys.argv[1] == 'long' and os.environ.get('FAIL_CATEGORY'):\n"
            "    raise SystemExit(17)\n"
        )
        self.env["FAIL_CATEGORY"] = "long"
        args = ("--data-src", self.source, "longvideobench")
        first = self.run_prepare(*args)
        self.assertEqual(first.returncode, 17)
        for category in ("short", "medium", "long"):
            self.assertTrue((self.root / f"data/LongVideoBench/test/{category}_duration_data.jsonl").exists())
        marker = self.root / "data/.prepare_state/longvideobench.incomplete"
        self.assertTrue(marker.exists())
        self.env.pop("FAIL_CATEGORY")
        second = self.run_prepare(*args)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("Preparing dataset: longvideobench", second.stdout)
        self.assertNotIn("Already exported", second.stdout)
        self.assertFalse(marker.exists())
        third = self.run_prepare("longvideobench")
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertIn("Already exported", third.stdout)

    def test_broken_archive_stops_before_export(self):
        self.inputs()
        self.python_wrapper(self.root / ".venvs/base/bin/python")
        (self.source / "videomme/videomme.tar").write_text("not a tar archive")
        result = self.run_prepare("--data-src", self.source, "videomme")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])
        self.assertNotIn("prepared successfully", result.stdout)


if __name__ == "__main__":
    unittest.main()
