import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mcp_runtime import prepare_environment
from scripts.configure_mcp_profiles import rewrite_config
from scripts.tool_catalog import service_entrypoint, service_ids


class DeploymentRuntimeTests(unittest.TestCase):
    def test_legacy_weights_and_custom_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / 'models/SAM/sam_vit_h_4b8939.pth'
            legacy.parent.mkdir(parents=True)
            legacy.touch()
            env = {'SAM_CHECKPOINT': 'models/sam_vit_h_4b8939.pth',
                   'SAM2_CHECKPOINT': '/custom/sam2.pt', 'HF_HUB_OFFLINE': '0'}
            prepare_environment(env, root)
            self.assertEqual(env['SAM_CHECKPOINT'], str(legacy))
            self.assertEqual(env['SAM2_CHECKPOINT'], '/custom/sam2.pt')
            self.assertEqual(env['HF_HUB_OFFLINE'], '1')
            self.assertEqual(env['TRANSFORMERS_OFFLINE'], '1')

    def test_custom_model_and_cache_roots(self):
        env = {'MODEL_DIR': 'weights', 'HF_HOME': 'cache/hf', 'MCP_ALLOW_MODEL_DOWNLOAD': '1'}
        prepare_environment(env, Path('/repo'))
        self.assertEqual(env['SAM_CHECKPOINT'], '/repo/weights/sam_vit_h_4b8939.pth')
        self.assertEqual(env['HF_HOME'], '/repo/cache/hf')
        self.assertEqual(env['HF_HUB_OFFLINE'], '0')

    def test_wrapper_applies_policy_before_server_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / 'server.py'
            probe.write_text('import os, json; print(json.dumps([os.environ["HF_HUB_OFFLINE"], os.getcwd()]))')
            env = os.environ.copy()
            env['MCP_ALLOW_MODEL_DOWNLOAD'] = '0'
            result = subprocess.run([sys.executable, str(ROOT/'scripts/mcp_runtime.py'), str(probe)],
                                    env=env, cwd=tmp, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), ['1', str(ROOT)])

    def test_wrapper_idempotence_and_proxy_preservation(self):
        config = {'mcpServers': {'ocr': {'command':'python', 'args':['./ocr-mcp/server.py'],
                  'env': {'https_proxy':'http://example:8080'}}}}
        rewritten = rewrite_config(config, ROOT/'.venvs', ROOT)
        self.assertEqual(rewritten, rewrite_config(rewritten, ROOT/'.venvs', ROOT))
        self.assertEqual(rewritten['mcpServers']['ocr']['args'], ['./scripts/mcp_runtime.py','./tools/text/ocr-mcp/server.py'])
        self.assertEqual(rewritten['mcpServers']['ocr']['env'], config['mcpServers']['ocr']['env'])

    def test_generated_wrapper_prepares_environment_before_service_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / 'scripts').mkdir()
            shutil.copyfile(ROOT / 'scripts/mcp_runtime.py', root / 'scripts/mcp_runtime.py')
            probe = root / 'tools/text/ocr-mcp/server.py'
            probe.parent.mkdir(parents=True)
            probe.write_text(
                'import json, os, sys\n'
                'print(json.dumps([os.environ["HF_HUB_OFFLINE"], '
                'os.environ["SAM_CHECKPOINT"], os.getcwd(), sys.argv[1:]]))\n'
            )
            config = {'mcpServers': {'ocr': {
                'command': 'python', 'args': ['-u', './ocr-mcp/server.py', '--probe', 'with spaces'],
            }}}
            args = rewrite_config(config, root / '.venvs', root)['mcpServers']['ocr']['args']
            env = dict(os.environ, MCP_ALLOW_MODEL_DOWNLOAD='0', HF_HUB_OFFLINE='0',
                       SAM_CHECKPOINT='models/sam_vit_h_4b8939.pth', MODEL_DIR='models')
            result = subprocess.run([sys.executable, *args], cwd=root, env=env,
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), [
                '1', str(root / 'models/sam_vit_h_4b8939.pth'), str(root), ['--probe', 'with spaces'],
            ])

    def test_download_commands_select_profiles_without_leaking_token(self):
        env = os.environ.copy()
        env['HF_TOKEN'] = 'test-secret-not-for-output'
        result = subprocess.run(['bash', str(ROOT/'scripts/download_models.sh'), '--dry-run',
                                 '--venv-root','/custom/profiles','audio','ocr','diarization'],
                                env=env, capture_output=True, text=True, check=True)
        self.assertIn('/custom/profiles/audio/bin/hf', result.stdout)
        self.assertIn('/custom/profiles/audio/bin/python', result.stdout)
        self.assertIn('/custom/profiles/ocr/bin/python', result.stdout)
        self.assertNotIn(env['HF_TOKEN'],result.stdout)

    def test_disable_music_and_choose_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'mcp.json'
            source = Path(tmp)/'services.json'
            source.write_text(json.dumps({'mcpServers': {
                name: {
                    'command': 'python',
                    'args': ['./' + service_entrypoint(name).relative_to(ROOT).as_posix()],
                }
                for name in service_ids()
            }}), encoding='utf-8')
            subprocess.run([sys.executable,str(ROOT/'scripts/configure_mcp_profiles.py'),
                            '--source',str(source),'--output',str(out),
                            '--disable-server','music-identification','--device','cpu','--ocr-device','cpu'],
                           capture_output=True,check=True)
            servers=json.loads(out.read_text())['mcpServers']
            self.assertNotIn('music-identification',servers)
            self.assertEqual(servers['object-track']['env']['SAM2_DEVICE'],'cpu')
            self.assertEqual(servers['ocr']['env']['OCR_USE_GPU'],'0')

if __name__ == '__main__':
    unittest.main()
