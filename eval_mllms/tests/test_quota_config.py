"""Quota regressions with real evaluator orchestration and no network or models.

OpenCV is optional for these tests: no frame processing or model API is invoked.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import httpx
import yaml


EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))
try:
    import cv2  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != "cv2":
        raise
    # The quota paths do not touch this module. Keep a missing optional video
    # dependency from preventing imports of the actual evaluator modules.
    with patch.dict(sys.modules, {"cv2": types.ModuleType("cv2")}):
        seed = importlib.import_module("eval_seed")
        gemini = importlib.import_module("eval_gemini")
else:
    seed = importlib.import_module("eval_seed")
    gemini = importlib.import_module("eval_gemini")


CLI_URL = "https://cli.example.test/quota"
CONFIG_URL = "https://config.example.test/quota"
ENV_URL = "https://env.example.test/quota"
ENV_KEY = "test-env-api-key"
CLI_KEY = "test-cli-api-key"


def response(payload, status=200):
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("GET", CONFIG_URL),
    )


class QuotaConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="eval-quota-tests-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.env_patch = patch.dict(os.environ, {
            "VLM_QUOTA_URL": ENV_URL,
            "VLM_API_KEY": ENV_KEY,
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def invoke_stubbed_main(self, module, cli=(), config=None, quota_value=12):
        arguments = [str(EVAL_DIR / (module.__name__ + ".py")), *cli]
        if config is not None:
            config_path = self.folder / (module.__name__ + ".yaml")
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            arguments.extend(["--config", str(config_path)])
        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", arguments))
            stack.enter_context(patch.object(module, "_save_terminal_settings"))
            stack.enter_context(patch.object(module, "infer_subtitle_dir", return_value=None))
            evaluate = stack.enter_context(patch.object(module, "run_evaluation"))
            multi = stack.enter_context(patch.object(module, "run_multi_file_evaluation"))
            get = stack.enter_context(patch.object(seed.httpx, "get", return_value=response({"remain_quota": quota_value})))
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(redirect_stderr(output))
            module.main()
        self.assertNotIn(ENV_KEY, output.getvalue())
        self.assertNotIn(CLI_KEY, output.getvalue())
        return evaluate, multi, get, output.getvalue()

    def test_url_precedence_and_explicit_empty_values(self):
        cases = [
            (CLI_URL, {"quota_url": CONFIG_URL}, CLI_URL),
            (None, {"quota_url": CONFIG_URL}, CONFIG_URL),
            (None, {}, ENV_URL),
            ("", {"quota_url": CONFIG_URL}, ""),
            ("   ", {"quota_url": CONFIG_URL}, ""),
            (None, {"quota_url": ""}, ""),
            (None, {"quota_url": "  "}, ""),
            (None, {"quota_url": None}, ""),
            ("  " + CLI_URL + "  ", {"quota_url": CONFIG_URL}, CLI_URL),
        ]
        for cli, config, expected in cases:
            with self.subTest(cli=cli, config=config):
                self.assertEqual(seed._resolve_quota_url(cli, config), expected)
        with patch.dict(os.environ, {"VLM_QUOTA_URL": ""}):
            self.assertEqual(seed._resolve_quota_url(None), "")

    def test_invalid_urls_are_rejected_without_http(self):
        with patch.object(seed.httpx, "get") as get:
            for value in ("/quota", "ftp://quota.example.test", "https:///quota", 7, True, []):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    seed._resolve_quota_url(None, {"quota_url": value})
            get.assert_not_called()

    def test_empty_query_has_no_implicit_endpoint_even_with_environment_set(self):
        with patch.object(seed.httpx, "get") as get:
            self.assertIsNone(seed._query_remain_quota(ENV_KEY))
            self.assertIsNone(seed._query_remain_quota(ENV_KEY, ""))
            get.assert_not_called()

    def test_valid_zero_and_nested_quota_responses(self):
        for payload, expected in (
            ({"remain_quota": 0}, 0.0),
            ({"remain_quota": "12.5"}, 12.5),
            ({"data": {"remain_quota": 7}}, 7.0),
            ({"remain_quota": 0, "data": {"remain_quota": 99}}, 0.0),
        ):
            with self.subTest(payload=payload), patch.object(seed.httpx, "get", return_value=response(payload)) as get:
                self.assertEqual(seed._query_remain_quota(CLI_KEY, CONFIG_URL), expected)
                get.assert_called_once_with(CONFIG_URL, headers={"Authorization": "Bearer " + CLI_KEY}, timeout=5)
        with patch.object(seed.httpx, "get", return_value=response({"remain_quota": 1})) as get:
            self.assertEqual(seed._query_remain_quota("", CONFIG_URL), 1.0)
            self.assertEqual(get.call_args.kwargs["headers"], {})

    def test_malformed_or_unavailable_quota_does_not_fabricate_zero(self):
        payloads = ({}, {"data": {}}, {"remain_quota": None}, {"remain_quota": True},
                    {"remain_quota": "invalid"}, {"remain_quota": "NaN"},
                    {"remain_quota": "Infinity"}, {"data": None}, [], None)
        for payload in payloads:
            with self.subTest(payload=payload), patch.object(seed.httpx, "get", return_value=response(payload)) as get:
                self.assertIsNone(seed._query_remain_quota(CLI_KEY, CONFIG_URL))
                self.assertEqual(get.call_count, 1)
        for status in (401, 403, 500):
            with self.subTest(status=status), patch.object(seed.httpx, "get", return_value=response({"remain_quota": 0}, status)) as get:
                self.assertIsNone(seed._query_remain_quota(CLI_KEY, CONFIG_URL))
                self.assertEqual(get.call_count, 1)
        invalid_json = httpx.Response(200, text="not JSON", request=httpx.Request("GET", CONFIG_URL))
        with patch.object(seed.httpx, "get", return_value=invalid_json):
            self.assertIsNone(seed._query_remain_quota(CLI_KEY, CONFIG_URL))
        with patch.object(seed.httpx, "get", side_effect=httpx.ReadTimeout("synthetic timeout")) as get:
            self.assertIsNone(seed._query_remain_quota(CLI_KEY, CONFIG_URL))
            self.assertEqual(get.call_count, 1)

    def test_both_main_functions_load_config_before_initial_query(self):
        for module in (seed, gemini):
            with self.subTest(module=module.__name__):
                evaluate, multi, get, output = self.invoke_stubbed_main(
                    module, config={"quota_url": CONFIG_URL, "max_samples": 7}, quota_value=0,
                )
                multi.assert_not_called()
                evaluate.assert_called_once()
                args = evaluate.call_args.args[0]
                self.assertEqual(args.limit, 7)
                self.assertEqual(args.quota_url, CONFIG_URL)
                self.assertEqual(args.quota_before, 0.0)
                self.assertIsNone(args.quota_after)
                get.assert_called_once_with(CONFIG_URL, headers={"Authorization": "Bearer " + ENV_KEY}, timeout=5)
                self.assertLess(output.index("[CONFIG] Loaded:"), output.index("[QUOTA] before evaluation:"))

    def test_both_main_functions_honor_cli_url_and_cli_key(self):
        for module in (seed, gemini):
            with self.subTest(module=module.__name__):
                evaluate, _, get, _ = self.invoke_stubbed_main(
                    module, cli=("--quota-url", CLI_URL, "--api-key", CLI_KEY),
                    config={"quota_url": CONFIG_URL},
                )
                self.assertEqual(evaluate.call_args.args[0].quota_url, CLI_URL)
                get.assert_called_once_with(CLI_URL, headers={"Authorization": "Bearer " + CLI_KEY}, timeout=5)

    def test_both_main_functions_use_environment_without_a_config(self):
        for module in (seed, gemini):
            with self.subTest(module=module.__name__):
                evaluate, _, get, _ = self.invoke_stubbed_main(module)
                self.assertEqual(evaluate.call_args.args[0].quota_url, ENV_URL)
                get.assert_called_once_with(ENV_URL, headers={"Authorization": "Bearer " + ENV_KEY}, timeout=5)

    def test_both_main_functions_disable_queries_for_explicit_empty_or_null(self):
        for module in (seed, gemini):
            for cli, config in ((('--quota-url', ''), {"quota_url": CONFIG_URL}),
                                ((), {"quota_url": ""}), ((), {"quota_url": None})):
                with self.subTest(module=module.__name__, cli=cli, config=config):
                    evaluate, _, get, _ = self.invoke_stubbed_main(module, cli, config)
                    get.assert_not_called()
                    self.assertEqual(evaluate.call_args.args[0].quota_url, "")
                    self.assertIsNone(evaluate.call_args.args[0].quota_before)
            with self.subTest(module=module.__name__, no_default=True), patch.dict(os.environ, {"VLM_QUOTA_URL": ""}):
                evaluate, _, get, _ = self.invoke_stubbed_main(module)
                evaluate.assert_called_once()
                get.assert_not_called()

    def test_missing_config_does_not_query_or_evaluate(self):
        for module in (seed, gemini):
            with self.subTest(module=module.__name__):
                evaluate, multi, get, output = self.invoke_stubbed_main(
                    module, cli=("--config", str(self.folder / "missing.yaml")),
                )
                self.assertIn("Config file not found", output)
                evaluate.assert_not_called()
                multi.assert_not_called()
                get.assert_not_called()

    def test_both_evaluators_keep_same_endpoint_and_auth_for_progress_and_final(self):
        data = self.folder / "synthetic.jsonl"
        items = [{"question_id": str(index), "video_path": "unused.mp4", "answer": "A"}
                 for index in range(20)]
        data.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")

        def fake_evaluate(item, **kwargs):
            return {**item, "model_answer": "A", "is_correct": True, "error": None,
                    "input_tokens": 2, "output_tokens": 1}

        for module in (seed, gemini):
            with self.subTest(module=module.__name__), ExitStack() as stack:
                destination = self.folder / module.__name__
                argv = [module.__name__, "--data", str(data), "--limit", "20", "--workers", "1",
                        "--output-dir", str(destination), "--model", "test-model", "--no-subtitle",
                        "--api-url", "https://api.example.test/v1", "--quota-url", CLI_URL,
                        "--api-key", CLI_KEY]
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(patch.object(module, "_save_terminal_settings"))
                stack.enter_context(patch.object(module, "infer_subtitle_dir", return_value=None))
                evaluate = stack.enter_context(patch.object(module, "evaluate_item", side_effect=fake_evaluate))
                get = stack.enter_context(patch.object(seed.httpx, "get", side_effect=[
                    response({"remain_quota": 100}), response({"remain_quota": 90}),
                    response({"remain_quota": 80}),
                ]))
                output = io.StringIO()
                stack.enter_context(redirect_stdout(output))
                stack.enter_context(redirect_stderr(output))
                module.main()
                self.assertEqual(evaluate.call_count, 20)
                self.assertEqual(get.call_count, 3)
                for call in get.call_args_list:
                    self.assertEqual(call.args, (CLI_URL,))
                    self.assertEqual(call.kwargs, {"headers": {"Authorization": "Bearer " + CLI_KEY}, "timeout": 5})
                saved = list(destination.rglob("*.json"))
                self.assertEqual(len(saved), 1)
                result = json.loads(saved[0].read_text())
                self.assertEqual(len(result["results"]), 20)
                self.assertEqual(result["summary"]["quota"], {"before": 100.0, "after": 80.0, "cost": 20.0})
                self.assertIn("[QUOTA] used=10.0000", output.getvalue())
                self.assertNotIn(CLI_KEY, output.getvalue())


if __name__ == "__main__":
    unittest.main()
