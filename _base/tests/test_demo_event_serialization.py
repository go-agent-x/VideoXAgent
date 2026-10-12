import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demo.event_serialization import make_tool_start_event


class ToolRuntime:
    def __repr__(self):
        raise AssertionError("Runtime state must not be rendered")


class ToolEventTests(unittest.TestCase):
    def test_runtime_input_can_be_sent_over_websocket_without_mutation(self):
        runtime = ToolRuntime()
        runtime.state = runtime
        tool_input = {"query": "circuit — schematic", "runtime": runtime, "nested": [runtime]}
        event = make_tool_start_event("search", tool_input)
        sent = json.loads(json.dumps(event, ensure_ascii=False))
        self.assertEqual(sent["data"]["input_raw"], {
            "query": "circuit — schematic", "runtime": "<ToolRuntime>", "nested": ["<ToolRuntime>"],
        })
        self.assertIs(tool_input["runtime"], runtime)
        self.assertIs(tool_input["nested"][0], runtime)

    def test_regular_json_input_is_preserved(self):
        for value in [{"video_id": "fiAyZ5UcN4c", "enabled": True}, [1, None], "text"]:
            event = make_tool_start_event("tool", value)
            self.assertEqual(event["data"]["input_raw"], value)
            json.dumps(event)

    def test_display_truncation_keeps_full_raw_input(self):
        value = {"query": "x" * 1000}
        event = make_tool_start_event("tool", value)
        self.assertEqual(len(event["data"]["input"]), 503)
        self.assertEqual(event["data"]["input_raw"], value)


if __name__ == "__main__":
    unittest.main()
