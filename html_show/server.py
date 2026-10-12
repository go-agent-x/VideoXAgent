#!/usr/bin/env python3
"""Serve selected benchmark traces, media, and generated dashboard assets.

Local access is the default. Remote binds require VIDEOX_WEB_TOKEN and all
requests are checked for authentication, Host and browser Origin.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# Keep direct execution (python html_show/server.py) and package imports equal.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from html_show.benchmark_paths import BenchmarkPaths
from html_show.web_files import DashboardFiles
from utils.web_security import MEDIA_TYPES, SECURITY_HEADERS, WebAccessPolicy


class SessionViewerHandler(BaseHTTPRequestHandler):
    """Serve only explicitly authorized benchmark files; no directory listings."""

    benchmark_paths = None
    static_root = None
    dashboard_file = None
    access_policy = None
    _REPO_ROOT = _REPO_ROOT

    def _file_policy(self):
        paths = self.benchmark_paths or BenchmarkPaths(self._REPO_ROOT)
        return DashboardFiles(paths, self.static_root, self.dashboard_file)

    def _authorize(self):
        policy = self.access_policy or WebAccessPolicy(host="127.0.0.1")
        error = policy.check_request(
            self.headers.get("Host"), self.headers.get("Origin"), self.headers.get("Authorization"),
        )
        if error is None:
            return True
        status, message = error
        self.send_response(status)
        if status == 401:
            self.send_header("WWW-Authenticate", policy.challenge)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        body = message.encode("utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not getattr(self, "_head_only", False):
            self.wfile.write(body)
        return False

    def do_GET(self):
        self._head_only = False
        self._handle_request()

    def do_HEAD(self):
        self._head_only = True
        self._handle_request()

    def _handle_request(self):
        if not self._authorize():
            return
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/healthz":
                self.send_json_response({"status": "ok"})
                return
            if path in ("/api/read", "/api/file"):
                query = parse_qs(parsed.query)
                file_path = query.get("path", [None])[0]
                if not file_path:
                    self.send_error(400, "Missing path query parameter")
                    return
                base = query.get("base", [None])[0]
                if path == "/api/read":
                    self.send_file_content(file_path, base)
                else:
                    self.send_binary_file(file_path, base)
                return
            if path.startswith("/api/"):
                self.send_error(404, "Unknown API endpoint")
                return
            resolved = self._file_policy().static(path)
            if resolved is None:
                self.send_error(404, "Dashboard file not found")
                return
            content_type = {
                ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
            }.get(resolved.suffix.lower(), MEDIA_TYPES.get(resolved.suffix.lower(), "application/octet-stream"))
            self._send_path(resolved, content_type)
        except PermissionError:
            self.send_error(403, "Access denied")
        except (ValueError, OSError, RuntimeError):
            self.send_error(404, "File not found")

    def _resolve_api_path(self, file_path, base=None, binary=False):
        policy = self._file_policy()
        return policy.media(file_path, base) if binary else policy.trace(file_path, base)

    def send_file_content(self, file_path, base=None):
        """Read a JSONL transcript registered by the selected benchmark index."""
        self._send_api_file(file_path, base, binary=False)

    def send_binary_file(self, file_path, base=None):
        """Serve an authorized image, audio, or video, including byte ranges."""
        self._send_api_file(file_path, base, binary=True)

    def _send_api_file(self, file_path, base, binary):
        try:
            resolved = self._resolve_api_path(file_path, base, binary)
            if resolved is None:
                self.send_error(404, "File not found")
                return
            content_type = MEDIA_TYPES[resolved.suffix.lower()] if binary else "text/plain; charset=utf-8"
            self._send_path(resolved, content_type)
        except PermissionError:
            self.send_error(403, "Access denied")
        except (OSError, ValueError, RuntimeError):
            self.send_error(404, "File not found")

    def _send_path(self, path, content_type):
        """GET and HEAD share the same file authorization and Range processing."""
        with path.open("rb") as source:
            stat = os.fstat(source.fileno())
            start, end = 0, stat.st_size - 1
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                if not match or not any(match.groups()):
                    self._range_error(stat.st_size)
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    if last:
                        end = min(int(last), end)
                elif int(last) > 0:
                    start = max(stat.st_size - int(last), 0)
                else:
                    self._range_error(stat.st_size)
                    return
                if start > end or start >= stat.st_size:
                    self._range_error(stat.st_size)
                    return
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(max(end - start + 1, 0)))
            self.send_header("Accept-Ranges", "bytes")
            if range_header:
                self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, stat.st_size))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            if getattr(self, "_head_only", False):
                return
            if not range_header:
                shutil.copyfileobj(source, self.wfile, length=64 * 1024)
                return
            source.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _range_error(self, size):
        self.send_response(416)
        self.send_header("Content-Range", "bytes */{}".format(size))
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_json_response(self, data):
        content = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if not getattr(self, "_head_only", False):
            self.wfile.write(content)

    def end_headers(self):
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        super().end_headers()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Serve benchmark execution traces and media")
    parser.add_argument("port", nargs="?", type=int, default=8000)
    parser.add_argument("root", nargs="?", default=str(Path(__file__).parent),
                        help="Directory containing generated dashboard assets")
    parser.add_argument("--host", default=os.environ.get("VIDEOX_WEB_HOST") or "127.0.0.1",
                        help="Bind address (remote access requires VIDEOX_WEB_TOKEN)")
    parser.add_argument("--dashboard", help="Explicit generated HTML file to serve at / and its basename")
    parser.add_argument("--data-root", default=os.environ.get("VIDEOX_DATA_ROOT") or None,
                        help="Root directory of the downloaded benchmark bundle")
    parser.add_argument("--data", default=os.environ.get("VIDEOX_DATA_FILE") or None,
                        help="Benchmark result JSON authorizing transcript references")
    try:
        mappings = json.loads(os.environ.get("VIDEOX_PATH_MAPS") or "[]")
    except ValueError:
        parser.error("VIDEOX_PATH_MAPS must be a JSON array of OLD=NEW strings")
    if not isinstance(mappings, list) or not all(isinstance(item, str) for item in mappings):
        parser.error("VIDEOX_PATH_MAPS must be a JSON array of OLD=NEW strings")
    parser.add_argument("--path-map", action="append", default=None, metavar="OLD=NEW",
                        help="Map a former absolute path prefix to a local directory; repeatable")
    args = parser.parse_args(argv)
    if args.path_map is None:
        args.path_map = mappings
    return args


def main():
    args = parse_args()
    try:
        SessionViewerHandler.access_policy = WebAccessPolicy(host=args.host)
        SessionViewerHandler.benchmark_paths = BenchmarkPaths(
            _REPO_ROOT,
            data_file=Path(args.data).expanduser().resolve() if args.data else None,
            data_root=Path(args.data_root).expanduser().resolve() if args.data_root else None,
            path_maps=args.path_map,
        )
        SessionViewerHandler.static_root = Path(args.root).expanduser().resolve()
        SessionViewerHandler.dashboard_file = Path(args.dashboard).expanduser().absolute() if args.dashboard else None
        DashboardFiles(SessionViewerHandler.benchmark_paths, SessionViewerHandler.static_root,
                       SessionViewerHandler.dashboard_file)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    server = HTTPServer((args.host, args.port), SessionViewerHandler)
    print(f"Dashboard Server started at http://{args.host}:{args.port}", flush=True)
    print("Benchmark data root: " + str(SessionViewerHandler.benchmark_paths.data_root), flush=True)
    print("API endpoints: /api/read (registered JSONL traces), /api/file (benchmark media)", flush=True)
    print("Press Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
