"""Tests for recorded times, source precedence, and metadata allowlisting."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_metadata import collect_run_metadata


class RunMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_path = self.root / "benchmark.json"
        self.data_path.write_text("{}")

    def transcript(self, records, name="trace.jsonl"):
        path = self.root / name
        path.write_text("\n".join(json.dumps(record) for record in records))
        return path

    def fields(self, data, paths=()):
        return dict(collect_run_metadata(data, self.data_path, paths)["fields"])

    def test_explicit_run_times_override_sessions_transcripts_and_mtime(self):
        data = {
            "run_metadata": {"started_at": "2026-04-23T09:00:00+08:00", "ended_at": "2026-04-23T10:00:00+08:00"},
            "sessions": [{"timestamp": "2026-01-01T00:00:00"}],
        }
        trace = self.transcript([{"timestamp": "2026-05-01T00:00:00Z"}])
        fields = self.fields(data, [trace])
        self.assertEqual(fields["Run started"], "2026-04-23 09:00:00+08:00")
        self.assertEqual(fields["Run ended"], "2026-04-23 10:00:00+08:00")
        self.assertNotIn("Data modified", fields)
        self.assertNotIn("First event", fields)

    def test_transcript_bounds_are_observations_not_fabricated_run_boundaries(self):
        trace = self.transcript([
            {"timestamp": "2026-04-23T18:00:00"},
            {"timestamp": "2026-04-23T17:00:00"},
        ])
        data = {"sessions": [{"timestamp": "2020-01-01T00:00:00"}], "experiment_config": {"timestamp": "2026-04-23T16:00:00"}}
        fields = self.fields(data, [trace])
        self.assertEqual(fields["First event"], "2026-04-23 17:00:00 (timezone not recorded)")
        self.assertEqual(fields["Last event"], "2026-04-23 18:00:00 (timezone not recorded)")
        self.assertEqual(fields["Time source"], "Transcript timestamps")
        self.assertEqual(fields["Config recorded"], "2026-04-23 16:00:00 (timezone not recorded)")
        self.assertNotIn("Run started", fields)

    def test_sessions_are_fallback_when_transcript_cannot_be_read(self):
        fields = self.fields({"sessions": [{"timestamp": "2026-04-23T17:00:00"}]}, [self.root / "missing.jsonl"])
        self.assertEqual(fields["Time source"], "Session timestamps")
        self.assertIn("timezone not recorded", fields["First session"])

    def test_mtime_is_never_called_run_time(self):
        fields = self.fields({"sessions": [], "timestamp": "not a timestamp"})
        self.assertIn("Data modified", fields)
        self.assertTrue(fields["Data modified"].endswith("+00:00"))
        self.assertNotIn("Run started", fields)
        self.assertNotIn("Run ended", fields)

    def test_mixed_timezones_keep_separate_ranges(self):
        trace = self.transcript([
            {"timestamp": "2026-04-23T17:00:00+08:00"},
            {"timestamp": "2026-04-23T10:00:00Z"},
            {"timestamp": "2026-04-23T08:00:00"},
        ])
        fields = self.fields({}, [trace])
        self.assertIn("17:00:00+08:00", fields["First event"])
        self.assertIn("10:00:00+00:00", fields["Last event"])
        self.assertIn("08:00:00 (timezone not recorded)", fields["First event"])

    def test_multiple_models_and_unknown_values(self):
        trace = self.transcript([
            {"message": {"model": "unknown"}},
            {"message": {"model": "model-a"}},
            {"message": {"model": "model-b"}},
            {"message": {"model": "model-a"}},
        ])
        self.assertEqual(self.fields({}, [trace])["Model"], "model-a, model-b")
        data = {"experiment_config": {"models": ["configured-a", "unknown", "configured-b"]}}
        self.assertEqual(self.fields(data, [trace])["Model"], "configured-a, configured-b")

    def test_allowlist_excludes_prompts_keys_config_contents_and_machine_paths(self):
        data = {
            "run_metadata": {"benchmark": "Video-MME", "run_id": "run-42", "config_path": "/private/machine/config.yaml"},
            "experiment_config": {
                "model": {"model_name": "model-a", "api_key": "SECRET_MODEL_KEY"},
                "system_prompt": "SECRET_SYSTEM_PROMPT",
                "system_prompt_file": "/private/machine/secret-prompt.txt",
                "api_key": "SECRET_API_KEY",
                "params": {"token": "SECRET_PARAM"},
            },
        }
        fields = self.fields(data)
        rendered = json.dumps(fields)
        self.assertEqual(fields["Model"], "model-a")
        self.assertEqual(fields["Config"], "config.yaml")
        self.assertEqual(fields["Benchmark"], "Video-MME")
        self.assertEqual(fields["Run ID"], "run-42")
        self.assertNotIn("SECRET", rendered)
        self.assertNotIn("/private", rendered)
        self.assertNotIn("system_prompt", rendered)

    def test_malformed_lines_do_not_hide_valid_events(self):
        trace = self.transcript([[], {"timestamp": "bad"}, {"timestamp": "2026-04-23T17:00:00Z"}])
        with trace.open("a") as stream:
            stream.write("\n{unfinished")
        fields = self.fields({}, [trace])
        self.assertEqual(fields["First event"], "2026-04-23 17:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
