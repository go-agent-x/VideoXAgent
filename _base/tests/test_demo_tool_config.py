import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from demo import agent_server


class ToolConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_experiment_controls_tool_list(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            configs = root / 'exp_configs'
            configs.mkdir()
            (configs / 'selected.yaml').write_text('mcp_json_file: custom.json\n')
            (root / 'custom.json').write_text(json.dumps({'mcpServers': {'scene-query': {}, 'inspect-image': {}}}))
            (root / '.mcp.json').write_text('{"mcpServers":{"wrong":{}}}')
            with patch.object(agent_server, 'REPO_ROOT', root), patch.object(agent_server, 'EXP_CONFIGS_DIR', configs):
                response = await agent_server.list_tools_handler(SimpleNamespace(query={'config': 'exp_configs/selected.yaml'}))
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.text)['tools'], ['inspect-image', 'scene-query'])

    async def test_startup_selected_experiment_and_default_mcp_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'default.yaml').write_text('model: {}\n')
            (root / '.mcp.profiles.json').write_text('{"mcpServers":{"expected":{}}}')
            with patch.object(agent_server, 'REPO_ROOT', root), patch.object(agent_server, 'DEFAULT_EXP_CONFIG', 'default.yaml'):
                response = await agent_server.list_tools_handler(SimpleNamespace(query={}))
            self.assertEqual(json.loads(response.text)['tools'], ['expected'])

    async def test_missing_config_is_reported(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(agent_server, 'REPO_ROOT', Path(folder)):
            response = await agent_server.list_tools_handler(SimpleNamespace(query={'config': 'missing.yaml'}))
        self.assertEqual(response.status, 400)
        self.assertIn('error', json.loads(response.text))

    async def test_arbitrary_yaml_outside_experiment_configs_is_denied(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'key.yaml').write_text('mcp_json_file: private.json\n')
            (root / 'private.json').write_text('{"mcpServers":{"secret-server":{}}}')
            with patch.object(agent_server, 'REPO_ROOT', root), patch.object(agent_server, 'EXP_CONFIGS_DIR', root / 'exp_configs'):
                response = await agent_server.list_tools_handler(SimpleNamespace(query={'config': 'key.yaml'}))
        self.assertEqual(response.status, 400)
        self.assertNotIn('secret-server', response.text)
