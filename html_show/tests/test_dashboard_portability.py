"""Move an exported benchmark bundle and exercise the real generator/server."""

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener


HTML_DIR = Path(__file__).resolve().parents[1]
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aSAAAAABJRU5ErkJggg==')


class PortableDashboardTests(unittest.TestCase):
    def test_exported_page_and_assets_survive_directory_move(self):
        with tempfile.TemporaryDirectory(prefix='benchmark move ') as temporary:
            home = Path(temporary)
            original = home / 'original results'
            (original / 'transcripts').mkdir(parents=True)
            (original / 'output' / 'frames').mkdir(parents=True)
            image_rel = 'output/frames/frame café #1.png'
            (original / image_rel).write_bytes(PNG)
            (original / 'transcripts' / 'nearby.png').write_bytes(PNG)
            old_prefix = '/retired/machine/project'
            trace = original / 'transcripts' / 'session.jsonl'
            records = [
                {'type': 'user', 'timestamp': '2026-01-02T03:04:05',
                 'message': {'content': 'Question: What happens?'}},
                {'type': 'tool', 'timestamp': '2026-01-02T03:04:06+08:00',
                 'message': {'name': 'inspect', 'input': {},
                             'output': json.dumps({'image': old_prefix + '/' + image_rel})}},
            ]
            trace.write_text('\n'.join(json.dumps(e) for e in records) + '\n')
            data = original / 'results.json'
            data.write_text(json.dumps({
                'experiment_config': {'model': 'fixture-model', 'system_prompt': 'DO NOT DISPLAY'},
                'sessions': [{'session_id': 'session', 'question_id': 'q1',
                              'transcript_path': old_prefix + '/transcripts/session.jsonl',
                              'init_user_query': 'Question: What happens?',
                              'model_answer': 'A', 'gt_answer': 'A', 'is_correct': True}],
            }))
            before = {p.relative_to(original).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in original.rglob('*') if p.is_file()}
            out = original / 'index.html'
            subprocess.run([sys.executable, str(HTML_DIR / 'gen_dashboard.py'),
                            '--data', str(data), '--data-root', str(original),
                            '--out', str(out)], check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
            page = out.read_text()
            sessions = json.loads(re.search(r'var evalSessions=(.*?);\n', page, re.S).group(1))
            self.assertEqual(sessions[0]['transcript_path'], 'transcripts/session.jsonl')
            self.assertNotIn(str(original), page)
            self.assertNotIn(old_prefix, page)
            self.assertNotIn('EVAL_REPO_ROOT', page)
            self.assertNotIn('DO NOT DISPLAY', page)
            self.assertIn('fixture-model', page)
            self.assertIn('timezone not recorded', page)
            self.assertNotIn('Updated: 2026-03-11', page)
            self.assert_js_valid_and_media_urls_portable(page)

            moved = home / 'downloaded copy café'
            shutil.move(str(original), str(moved))
            self.assertFalse(original.exists())
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = os.environ.copy()
            for key in ('VIDEOX_DATA_ROOT', 'VIDEOX_DATA_FILE', 'VIDEOX_PATH_MAPS',
                        'VIDEOX_WEB_HOST', 'VIDEOX_WEB_TOKEN', 'VIDEOX_DASHBOARD_FILE'):
                env.pop(key, None)
            process = subprocess.Popen([
                sys.executable, str(HTML_DIR / 'server.py'), str(port), str(moved),
                '--data-root', str(moved), '--data', str(moved / 'results.json'),
                '--dashboard', str(moved / 'index.html'),
            ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                opener = build_opener(ProxyHandler({}))
                origin = 'http://127.0.0.1:{}'.format(port)
                deadline = time.monotonic() + 5
                while True:
                    try:
                        with opener.open(origin + '/index.html', timeout=0.5) as response:
                            self.assertEqual(response.read().decode(), page)
                        break
                    except OSError:
                        if time.monotonic() >= deadline or process.poll() is not None:
                            self.fail('Relocated fixture server failed to start')
                        time.sleep(0.05)
                with opener.open(origin + '/api/read?' + urlencode({
                    'path': sessions[0]['transcript_path']})) as response:
                    self.assertEqual(response.read(), (moved / 'transcripts/session.jsonl').read_bytes())
                for path in (image_rel, old_prefix + '/' + image_rel, 'nearby.png'):
                    with opener.open(origin + '/api/file?' + urlencode({
                        'path': path, 'base': sessions[0]['transcript_path']})) as response:
                        self.assertEqual(response.headers.get_content_type(), 'image/png')
                        self.assertEqual(response.read(), PNG)
                for relative, digest in before.items():
                    self.assertEqual(hashlib.sha256((moved / relative).read_bytes()).hexdigest(), digest)
            finally:
                process.terminate()
                process.wait(timeout=5)

    def assert_js_valid_and_media_urls_portable(self, page):
        node = shutil.which('node')
        if not node:
            self.skipTest('Node.js is needed for the generated JavaScript check')
        scripts = re.findall(r'<script\b[^>]*>(.*?)</script>', page, re.S)
        for script in scripts:
            result = subprocess.run([node, '--check'], input=script, universal_newlines=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr)
        session_script = next(script for script in scripts if 'var evalSessions=' in script)
        # Exercise the generated renderer, including its actual media URL chain.
        harness = r'''
const vm=require('vm'), assert=require('assert');
const nodes={};
const node=()=>({value:'all', dataset:{}, style:{}, classList:{add(){},remove(){}},
  innerHTML:'', scrollIntoView(){}});
const ctx={URL, URLSearchParams, window:{location:{search:'?api_base=http%3A%2F%2Fexample.test%3A9000',
  origin:'http://example.test:9000',href:'http://example.test:9000/index.html'}},
  document:{getElementById(id){return nodes[id]||(nodes[id]=node());},
    createElement(){const n=node();Object.defineProperty(n,'textContent',{set(v){n.innerHTML=String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}});return n;}}};
vm.createContext(ctx);
vm.runInContext(SCRIPT,ctx);
const path='output/frames/frame café #1.png', base='transcripts/session.jsonl';
const rendered=ctx.evalRenderInteractions([{userMessage:'test',trace:[{type:'tool_use', name:'inspect', input:{image:path}, output:''}]}],base);
assert(rendered.includes('http://example.test:9000/api/file?path='));
assert(rendered.includes('&amp;base=transcripts%2Fsession.jsonl'));
const url=new URL(ctx.evalSafeUrl(ctx.evalPathToUrl(path,base)));
assert.strictEqual(url.searchParams.get('path'),path);
assert.strictEqual(url.searchParams.get('base'),base);
'''.replace('SCRIPT', json.dumps(session_script))
        result = subprocess.run([node], input=harness, universal_newlines=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
