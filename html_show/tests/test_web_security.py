"""HTTP regression coverage for the dashboard's public file boundary."""

import base64
from http.client import HTTPConnection
from http.server import HTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.parse import urlencode

from html_show.benchmark_paths import BenchmarkPaths
from html_show.server import SessionViewerHandler, parse_args
from utils.web_security import WebAccessPolicy


class DashboardWebSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="videox-web-security-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.page = self.write("repo/html_show/tool_workflow_security.html", b"<html>dashboard</html>")
        self.write("repo/html_show/brand.css", b"body{}")
        self.write("repo/html_show/server.py", b"private source")
        self.key = self.write("repo/key.yaml", b"api_key: fictitious-regression-only")
        self.trace = self.write("repo/conversations/transcripts/run/trace.jsonl", b'{"type":"test"}\n')
        self.unregistered = self.write("repo/conversations/transcripts/run/other.jsonl", b"unregistered")
        self.media = self.write("repo/output/run/frame.png", b"0123456789")
        self.manifest = self.write("repo/conversations/results.json", json.dumps({
            "sessions": [{"transcript_path": str(self.trace)}],
        }).encode())
        self.start(BenchmarkPaths(self.repo, self.manifest))

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def start(self, paths, token="", static_root=None, dashboard=None):
        class TestHandler(SessionViewerHandler):
            def log_message(self, *args):
                pass
        TestHandler.benchmark_paths = paths
        TestHandler.static_root = static_root or self.repo
        TestHandler.dashboard_file = dashboard
        TestHandler.access_policy = WebAccessPolicy(host="127.0.0.1", token=token)
        self.handler = TestHandler
        self.httpd = HTTPServer(("127.0.0.1", 0), TestHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.thread.join(timeout=2)
            self.httpd = None

    def request(self, path, method="GET", headers=None):
        connection = HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=3)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    @staticmethod
    def api(kind, path, **kwargs):
        return "/api/" + kind + "?" + urlencode(dict(path=str(path), **kwargs))

    def test_registered_trace_and_media_succeed(self):
        for endpoint, file in (("read", self.trace), ("file", self.media)):
            status, headers, body = self.request(self.api(endpoint, file))
            self.assertEqual(status, 200)
            self.assertEqual(body, file.read_bytes())
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
            self.assertNotIn("Access-Control-Allow-Origin", headers)
            self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(self.request("/html_show/tool_workflow_security.html")[0], 200)
        self.assertEqual(self.request("/html_show/brand.css")[0], 200)

    def test_repo_files_never_reachable_via_static_read_or_media(self):
        blocked = [
            "/key.yaml", "/html_show/server.py", "/html_show/", "/.git/config",
            "/%2e%2e/key.yaml", "/html_show/%2e%2e/key.yaml",
            self.api("read", self.key), self.api("file", self.key),
            self.api("read", self.manifest), self.api("read", self.unregistered),
            self.api("file", self.trace),
        ]
        for path in blocked:
            for method in ("GET", "HEAD"):
                with self.subTest(path=path, method=method):
                    status, _headers, body = self.request(path, method, {"Range": "bytes=0-10"})
                    self.assertIn(status, (403, 404))
                    self.assertNotIn(b"fictitious-regression-only", body)
                    if method == "HEAD":
                        self.assertEqual(body, b"")

    def test_removed_endpoints_have_no_implementation_or_routes(self):
        removed = (
            ("/api/projects", "list_projects"), ("/api/sessions/project", "list_sessions"),
            ("/api/log/project/trace.jsonl", "send_session_log"), ("/api/all_sessions", "list_all_sessions"),
            ("/api/conversation_files", "list_conversation_files"),
            ("/api/sessions_from_file?file=/etc/passwd", "list_sessions_from_file"),
        )
        for path, method_name in removed:
            self.assertFalse(hasattr(self.handler, method_name))
            for method in ("GET", "HEAD"):
                self.assertEqual(self.request(path, method)[0], 404)

    def test_head_and_ranges_share_same_boundary(self):
        url = self.api("file", self.media)
        for method in ("GET", "HEAD"):
            status, headers, body = self.request(url, method, {"Range": "bytes=2-5"})
            self.assertEqual(status, 206)
            self.assertEqual(headers["Content-Range"], "bytes 2-5/10")
            self.assertEqual(headers["Content-Length"], "4")
            self.assertEqual(body, b"2345" if method == "GET" else b"")
            self.assertEqual(self.request(url, method, {"Range": "bytes=100-"})[0], 416)
        self.assertEqual(self.request(url, headers={"Range": "bytes=-4"})[2], b"6789")

    def test_symlink_secret_hidden_and_nonpublic_media_are_denied(self):
        external = self.write("outside.png", b"outside secret")
        hidden = self.write("repo/output/.credentials/payload.png", b"hidden secret")
        configuration = self.write("repo/output/configs/payload.png", b"config secret")
        nonpublic = self.write("repo/private-image.png", b"nonpublic")
        alias = self.repo / "output/escaped.png"
        alias.symlink_to(external)
        disguised = self.repo / "output/disguised.png"
        disguised.symlink_to(self.key)
        hidden_alias = self.repo / "output/hidden.png"
        hidden_alias.symlink_to(hidden)
        static_alias = self.repo / "html_show/tool_workflow_leak.html"
        static_alias.symlink_to(self.key)
        for path in (alias, disguised, hidden_alias, hidden, configuration, nonpublic):
            self.assertEqual(self.request(self.api("file", path))[0], 403)
        self.assertEqual(self.request("/html_show/tool_workflow_leak.html")[0], 403)

    def test_default_public_directory_symlink_cannot_publish_another_repo_directory(self):
        private = self.write("repo/internal/screenshot.png", b"private screenshot")
        # Even an in-repository symlink must not broaden the default public roots.
        self.media.unlink()
        self.media.parent.rmdir()
        self.media.parent.parent.rmdir()
        (self.repo / "output").symlink_to(private.parent, target_is_directory=True)
        self.assertEqual(self.request(self.api("file", self.repo / "output/screenshot.png"))[0], 403)

    def test_selected_bundle_and_mapping_keep_trace_and_relative_media(self):
        bundle = self.root / "portable"
        relocated = self.write("portable/conversations/transcripts/run/trace.jsonl", b"relocated trace")
        image = self.write("portable/output/frame.png", b"portable image")
        mapped = self.write("mapped/frame.png", b"mapped image")
        manifest = self.write("portable/conversations/results.json", json.dumps({
            "sessions": [{"transcript_path": "/old/conversations/transcripts/run/trace.jsonl"}],
        }).encode())
        self.handler.benchmark_paths = BenchmarkPaths(self.repo, manifest, bundle, ["/old/assets=" + str(mapped.parent)])
        self.assertEqual(self.request(self.api("read", "/old/conversations/transcripts/run/trace.jsonl"))[2], relocated.read_bytes())
        self.assertEqual(self.request(self.api("file", "../../../output/frame.png", base=str(relocated)))[2], image.read_bytes())
        self.assertEqual(self.request(self.api("file", "/old/assets/frame.png"))[2], mapped.read_bytes())
        self.assertEqual(self.request(self.api("file", self.media))[0], 404)

    def test_mutating_index_revokes_transcript_access(self):
        self.assertEqual(self.request(self.api("read", self.trace))[0], 200)
        self.manifest.write_text('{"sessions":[]}', encoding="utf-8")
        self.assertEqual(self.request(self.api("read", self.trace))[0], 403)

    def test_authentication_covers_health_static_read_file_head_and_range(self):
        token = "fictitious-test-token"
        self.handler.access_policy = WebAccessPolicy(host="0.0.0.0", token=token)
        paths = ("/healthz", "/html_show/tool_workflow_security.html", self.api("read", self.trace), self.api("file", self.media))
        basic = base64.b64encode(("videoxagent:" + token).encode()).decode()
        for path in paths:
            for method in ("GET", "HEAD"):
                status, headers, _body = self.request(path, method, {"Range": "bytes=0-3"})
                self.assertEqual(status, 401)
                self.assertIn("WWW-Authenticate", headers)
                for authorization in ("Bearer " + token, "Basic " + basic):
                    status, _headers, _body = self.request(path, method, {"Authorization": authorization})
                    self.assertEqual(status, 200)

    def test_cross_origin_and_untrusted_host_requests_are_denied(self):
        for path in ("/healthz", "/html_show/tool_workflow_security.html", self.api("read", self.trace), self.api("file", self.media)):
            self.assertEqual(self.request(path, headers={"Host": "attacker.example"})[0], 403)
            self.assertEqual(self.request(path, headers={"Origin": "https://attacker.example"})[0], 403)
            own_origin = "http://127.0.0.1:{}".format(self.httpd.server_port)
            self.assertEqual(self.request(path, headers={"Origin": own_origin})[0], 200)

    def test_external_export_and_explicit_dashboard_do_not_publish_neighbors(self):
        export = self.write("export/dashboard.html", b"export")
        self.write("export/key.yaml", b"private export neighbor")
        self.handler.static_root = export.parent
        self.assertEqual(self.request("/dashboard.html")[2], b"export")
        self.assertEqual(self.request("/key.yaml")[0], 403)
        selected = self.write("export/custom.html", b"custom export")
        self.handler.dashboard_file = selected
        self.assertEqual(self.request("/")[2], b"custom export")
        self.assertEqual(self.request("/custom.html")[2], b"custom export")
        self.assertEqual(self.request("/key.yaml")[0], 403)

    def test_explicit_dashboard_accepts_parent_alias_but_rejects_file_symlink(self):
        selected = self.write("export/index.html", b"selected dashboard")
        parent_alias = self.root / "export-alias"
        parent_alias.symlink_to(selected.parent, target_is_directory=True)
        self.handler.dashboard_file = parent_alias / selected.name
        self.assertEqual(self.request("/index.html")[2], b"selected dashboard")
        self.assertEqual(self.request("/key.yaml")[0], 403)
        alias_file = selected.parent / "alias.html"
        alias_file.symlink_to(self.key)
        self.handler.dashboard_file = parent_alias / alias_file.name
        self.assertEqual(self.request("/alias.html")[0], 403)
        hidden = self.write("export/.hidden/index.html", b"hidden export")
        self.handler.dashboard_file = hidden
        self.assertEqual(self.request("/index.html")[0], 404)

    def test_default_host_is_loopback_and_remote_without_token_fails(self):
        self.assertEqual(parse_args([]).host, "127.0.0.1")
        self.assertEqual(parse_args(["--host", "0.0.0.0"]).host, "0.0.0.0")
        with self.assertRaises(ValueError):
            WebAccessPolicy(host="0.0.0.0", token="")


if __name__ == "__main__":
    unittest.main()
