"""Lifecycle checks for the dashboard launcher, using isolated mock repositories.

Run with: python3 -m unittest discover -s html_show/tests -p 'test_start_session_server_tool.py' -v
The tests bind ephemeral localhost ports; they never touch the real dashboard server.
"""

import os
import json
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


LAUNCHER = Path(__file__).resolve().parents[1] / "start_session_server_tool.sh"

GENERATOR = r'''
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--tag', default='')
parser.add_argument('--data-root')
parser.add_argument('--path-map', action='append', default=[])
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
(root / 'generator_args.json').write_text(json.dumps(vars(args)))
if (root / 'fail_generation').exists():
    print('Intentional generation failure', file=sys.stderr)
    sys.exit(7)
name = 'tool_workflow' + ('_' + args.tag if args.tag else '') + '.html'
(root / 'html_show' / name).write_text('<html><body>Fixture dashboard</body></html>')
'''

SERVER = r'''
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import base64
import os
import sys
import json
from pathlib import Path

os.chdir(sys.argv[2])
Path('server_environment.json').write_text(json.dumps({k: os.environ.get(k) for k in ('VIDEOX_DATA_ROOT', 'VIDEOX_DATA_FILE', 'VIDEOX_PATH_MAPS', 'VIDEOX_WEB_HOST')}))
class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        token = os.environ.get('VIDEOX_WEB_TOKEN', '')
        basic = 'Basic ' + base64.b64encode(('videoxagent:' + token).encode()).decode()
        if token and self.headers.get('Authorization') not in ('Bearer ' + token, basic):
            self.send_error(401, 'Authentication required')
            return
        if os.path.exists('unhealthy_server'):
            self.send_error(503, 'Intentional unhealthy fixture')
            return
        if self.path == '/healthz':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        super().do_GET()

ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
'''


class DashboardLauncherLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dashboard lifecycle ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repository with spaces"
        self.html = self.root / "html_show"
        self.html.mkdir(parents=True)
        shutil.copy2(str(LAUNCHER), str(self.html / LAUNCHER.name))
        utilities = self.root / "utils"
        utilities.mkdir()
        shutil.copy2(str(LAUNCHER.parents[1] / "utils" / "web_security.py"),
                     str(utilities / "web_security.py"))
        (self.html / "gen_dashboard.py").write_text(GENERATOR)
        (self.html / "add_gt_answer.py").write_text("# Intentionally no-op fixture.\n")
        (self.html / "server.py").write_text(SERVER)
        self.data = self.root / "fixture data.json"
        self.data.write_text('{"sessions": []}')
        self.env = os.environ.copy()
        for key in ("VIDEOX_WEB_HOST", "VIDEOX_WEB_TOKEN", "VIDEOX_DASHBOARD_FILE"):
            self.env.pop(key, None)
        self.env["PYTHON_BIN"] = sys.executable
        self.ports = set()
        self.children = []
        self.opener = build_opener(ProxyHandler({}))
        self.addCleanup(self.cleanup_processes)

    def free_port(self):
        while True:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            if port not in self.ports:
                self.ports.add(port)
                return port

    def launch(self, port, *extra, expect_success=True):
        result = subprocess.run(
            ["sh", str(self.html / LAUNCHER.name), "--port", str(port),
             "--data", str(self.data), "--tag", "lifecycle"] + list(extra),
            cwd=str(self.root), env=self.env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True, timeout=30,
        )
        if expect_success:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def server_pid(self, port):
        return int((self.html / ("server_{}.pid".format(port))).read_text().strip())

    @staticmethod
    def process_running(pid):
        # kill(pid, 0) alone considers an exited-but-unreaped process alive.
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True,
        )
        state = result.stdout.strip()
        return bool(state) and not state.startswith("Z")

    def assert_stopped(self, pid):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not self.process_running(pid):
                return
            time.sleep(0.05)
        self.fail("Process {} still running after shutdown".format(pid))

    def assert_http_ready(self, port, path="/"):
        deadline = time.monotonic() + 5
        last_error = None
        while time.monotonic() < deadline:
            try:
                with self.opener.open(
                    "http://127.0.0.1:{}{}".format(port, path), timeout=0.5
                ) as response:
                    self.assertEqual(response.status, 200)
                    return
            except (OSError, URLError) as error:
                last_error = error
                time.sleep(0.05)
        self.fail("Fixture server {} did not start: {}".format(port, last_error))

    def start_direct_server(self, port, unrelated=False):
        if unrelated:
            server_path = Path(self.temporary.name) / "unrelated server.py"
            server_path.write_text(SERVER)
        else:
            server_path = self.html / "server.py"
        child = subprocess.Popen(
            [sys.executable, str(server_path), str(port), str(self.root)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.children.append(child)
        self.assert_http_ready(port)
        if not unrelated:
            (self.html / "server_{}.pid".format(port)).write_text(str(child.pid))
        return child

    def fixture_server_pids(self):
        # Only consider services carrying this test's unique absolute script/root.
        result = subprocess.run(
            ["ps", "-axww", "-o", "pid=", "-o", "command="],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, check=True,
        )
        script = " " + str(self.html / "server.py") + " "
        suffix = " " + str(self.root)
        found = set()
        for line in result.stdout.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) != 2:
                continue
            pid, command = fields
            if script in command and command.endswith(suffix):
                found.add(int(pid))
        return found

    def cleanup_processes(self):
        # Do not trust PID files: one regression deliberately points one elsewhere.
        pids = self.fixture_server_pids()
        pids.update(child.pid for child in self.children if child.poll() is None)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if not any(self.process_running(pid) for pid in pids):
                break
            time.sleep(0.05)
        for pid in pids:
            if self.process_running(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for child in self.children:
            child.wait(timeout=5)

    def test_repeated_launch_replaces_pid_and_keeps_one_server(self):
        port = self.free_port()
        self.launch(port)
        old_pid = self.server_pid(port)
        self.launch(port)
        new_pid = self.server_pid(port)
        self.assertNotEqual(new_pid, old_pid)
        self.assert_stopped(old_pid)
        self.assert_http_ready(port, "/html_show/tool_workflow_lifecycle.html")
        self.assertEqual(self.fixture_server_pids(), {new_pid})

    def test_data_root_and_maps_reach_generator_and_server_without_enrichment(self):
        data_root = self.root / 'downloaded results'
        data_root.mkdir()
        mapping = '/old/machine=.'
        (self.html / 'add_gt_answer.py').write_text(
            'from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text("mutated")\n'
        )
        before = self.data.read_bytes()
        port = self.free_port()
        self.launch(port, '--data-root', str(data_root), '--path-map', mapping)
        self.assertEqual(self.data.read_bytes(), before)
        gen = json.loads((self.root / 'generator_args.json').read_text())
        env = json.loads((self.root / 'server_environment.json').read_text())
        self.assertEqual(gen['data_root'], str(data_root.resolve()))
        self.assertEqual(gen['path_map'], [mapping])
        self.assertEqual(env['VIDEOX_DATA_ROOT'], str(data_root.resolve()))
        self.assertEqual(env['VIDEOX_DATA_FILE'], str(self.data.resolve()))
        self.assertEqual(json.loads(env['VIDEOX_PATH_MAPS']), [mapping])
        self.assertEqual(env['VIDEOX_WEB_HOST'], '127.0.0.1')

    def test_remote_binding_requires_token_before_replacing_existing_service(self):
        port = self.free_port()
        self.launch(port)
        previous = self.server_pid(port)
        result = self.launch(port, '--ip', '0.0.0.0', expect_success=False)
        self.assertIn('VIDEOX_WEB_TOKEN', result.stdout)
        self.assertEqual(self.server_pid(port), previous)
        self.assert_http_ready(port, '/html_show/tool_workflow_lifecycle.html')

    def test_remote_bind_host_reaches_server_without_printing_token(self):
        self.env['VIDEOX_WEB_TOKEN'] = 'fixture-token-never-log-this'
        result = self.launch(self.free_port(), '--ip', '0.0.0.0')
        environment = json.loads((self.root / 'server_environment.json').read_text())
        self.assertEqual(environment['VIDEOX_WEB_HOST'], '0.0.0.0')
        self.assertNotIn(self.env['VIDEOX_WEB_TOKEN'], result.stdout)

    def test_authenticated_healthcheck_handles_non_ascii_token_without_disclosing_it(self):
        self.env['VIDEOX_WEB_TOKEN'] = 'fixture-secret-key🔑\r\nsecond-line'
        result = self.launch(self.free_port(), '--ip', '0.0.0.0')
        for token_fragment in ('fixture-secret', 'key🔑', 'second-line'):
            self.assertNotIn(token_fragment, result.stdout)

    def test_enrichment_requires_explicit_flag(self):
        (self.html / 'add_gt_answer.py').write_text(
            'from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text("enriched")\n'
        )
        self.launch(self.free_port(), '--enrich')
        self.assertEqual(self.data.read_text(), 'enriched')

    def test_changing_port_cleans_up_old_checkout_server(self):
        old_port, new_port = self.free_port(), self.free_port()
        self.launch(old_port)
        old_pid = self.server_pid(old_port)
        self.launch(new_port)
        new_pid = self.server_pid(new_port)
        self.assert_stopped(old_pid)
        self.assertFalse((self.html / "server_{}.pid".format(old_port)).exists())
        self.assert_http_ready(new_port, "/html_show/tool_workflow_lifecycle.html")
        self.assertEqual(self.fixture_server_pids(), {new_pid})

    def test_stale_pid_file_does_not_kill_unrelated_process(self):
        unrelated_port, selected_port = self.free_port(), self.free_port()
        unrelated = self.start_direct_server(unrelated_port, unrelated=True)
        (self.html / "server_{}.pid".format(selected_port)).write_text(str(unrelated.pid))
        self.launch(selected_port)
        self.assertIsNone(unrelated.poll(), "Unrelated service was terminated")
        self.assert_http_ready(unrelated_port)
        self.assertNotEqual(self.server_pid(selected_port), unrelated.pid)
        self.launch(selected_port, "--stop")
        self.assertIsNone(unrelated.poll(), "Stopping dashboard killed unrelated service")

    def test_generation_failure_preserves_running_service(self):
        port = self.free_port()
        self.launch(port)
        old_pid = self.server_pid(port)
        (self.root / "fail_generation").touch()
        result = self.launch(port, expect_success=False)
        self.assertIn("Intentional generation failure", result.stdout)
        self.assertEqual(self.server_pid(port), old_pid)
        self.assert_http_ready(port, "/html_show/tool_workflow_lifecycle.html")
        self.assertEqual(self.fixture_server_pids(), {old_pid})

    def test_stop_only_terminates_selected_port(self):
        selected_port, other_port = self.free_port(), self.free_port()
        selected = self.start_direct_server(selected_port)
        other = self.start_direct_server(other_port)
        self.launch(selected_port, "--stop")
        self.assert_stopped(selected.pid)
        self.assertIsNone(other.poll(), "--stop terminated the other port")
        self.assert_http_ready(other_port)
        self.assertFalse((self.html / "server_{}.pid".format(selected_port)).exists())
        self.assertEqual(self.server_pid(other_port), other.pid)

    def test_unrelated_listener_on_selected_port_is_not_terminated(self):
        port = self.free_port()
        unrelated = self.start_direct_server(port, unrelated=True)
        self.launch(port, expect_success=False)
        self.assertIsNone(unrelated.poll(), "Occupied port's unrelated service was killed")
        self.assert_http_ready(port)
        self.assertEqual(self.fixture_server_pids(), set())

    def test_unhealthy_new_server_is_cleaned_up(self):
        port = self.free_port()
        (self.root / "unhealthy_server").touch()
        result = self.launch(port, expect_success=False)
        self.assertIn("Dashboard is not accessible", result.stdout)
        self.assertFalse((self.html / "server_{}.pid".format(port)).exists())
        self.assertEqual(self.fixture_server_pids(), set())
        with socket.socket() as sock:
            sock.settimeout(0.5)
            self.assertNotEqual(sock.connect_ex(("127.0.0.1", port)), 0)


if __name__ == "__main__":
    unittest.main()
