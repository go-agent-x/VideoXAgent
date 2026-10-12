"""HTTP security regressions using synthetic files and no model execution."""

import base64
import gzip
from html.parser import HTMLParser
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aiohttp
from aiohttp.test_utils import TestClient, TestServer

from demo import agent_server


class _Markup(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.text = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


class DemoRenderingSecurityTests(unittest.TestCase):
    def test_real_javascript_keeps_untrusted_values_out_of_html_attributes(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for browser JavaScript regression checks")
        source = Path(agent_server.__file__).with_name("agent_runner.html")
        script = r'''
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
function element() {
  return {innerHTML: '', textContent: '', value: '', children: [], style: {},
          querySelector: () => null, appendChild(child) { this.children.push(child); }};
}
const payload = '<img src=x onerror="globalThis.attacked=true">';
const tool = 'name" autofocus onfocus="globalThis.attacked=true';
const id = 'chat\');globalThis.attacked=true;//';
const maliciousUrl = 'https://example.invalid/a" onload="globalThis.attacked=true" x=".png';
const context = vm.createContext({
  document: {addEventListener() {}, createElement: element, getElementById(id) {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  }},
  URLSearchParams,
  console,
  fetch: async () => ({ok: true, json: async () => ({tools: [tool]})}),
  payload, tool, id, maliciousUrl,
});
vm.runInContext(script, context);
const result = {
  payload, tool, id, maliciousUrl,
  escaped: vm.runInContext('escapeHtml("\\\"\'&<>")', context),
  relativeUrl: vm.runInContext('pathToImageUrl("output/frame.png")', context),
  image: vm.runInContext('renderImageStrip(extractImagePaths({image: maliciousUrl}))', context),
  cssImage: vm.runInContext('renderImageStrip(["https://example.invalid/\'\\\"()\\\\\\n.png"])', context),
  response: vm.runInContext('formatResponse(payload)', context),
};
vm.runInContext('conversations = [{id, title: payload, pinned: false}]; renderRecents()', context);
result.recents = elements.get('recentsList').innerHTML;
vm.runInContext('appendEvent("complete", {duration: 1, usage: {input_tokens: payload, output_tokens: payload, cache_read_input_tokens: payload}})', context);
result.complete = elements.get('outputContent').children.pop().innerHTML;
vm.runInContext('appendEvent("system-prompt", payload)', context);
result.systemPrompt = elements.get('outputContent').children.pop().innerHTML;
vm.runInContext('appendUserMessage(payload)', context);
result.user = elements.get('outputContent').children.pop().innerHTML;
(async () => {
  await vm.runInContext('loadTools()', context);
  result.toolMarkup = elements.get('toolsContainer').children[0].innerHTML;
  process.stdout.write(JSON.stringify(result));
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
        run = subprocess.run([node, "-e", script, str(source)], text=True, capture_output=True, check=True)
        result = json.loads(run.stdout)
        self.assertIn("&quot;", result["escaped"])
        self.assertIn("&#39;", result["escaped"])
        self.assertEqual(result["relativeUrl"], "/api/file?path=output%2Fframe.png")
        image = _Markup(result["image"])
        images = [attrs for tag, attrs in image.tags if tag == "img"]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["src"], result["maliciousUrl"])
        self.assertEqual(set(images[0]), {"loading", "src", "onerror"})
        self.assertEqual(images[0]["onerror"], "this.style.display='none'")
        for key in ("image", "cssImage"):
            style = next(attrs["style"] for tag, attrs in _Markup(result[key]).tags if "style" in attrs)
            self.assertTrue(style.startswith("--img: url('"))
            self.assertTrue(style.endswith("')"))
            for character in "\"'()\\\n\r<>":
                self.assertNotIn(character, style[len("--img: url('"):-2])
        recents = _Markup(result["recents"])
        recent = next(attrs for tag, attrs in recents.tags if "data-id" in attrs)
        self.assertEqual(recent["data-id"], result["id"])
        self.assertEqual(recent["onclick"], "loadConversation(this.dataset.id)")
        self.assertIn(result["payload"], recents.text)
        tool_markup = _Markup(result["toolMarkup"])
        checkbox = next(attrs for tag, attrs in tool_markup.tags if tag == "input")
        self.assertEqual(set(checkbox), {"type", "id", "value"})
        self.assertEqual(checkbox["value"], result["tool"])
        self.assertEqual(checkbox["id"], "tool_" + result["tool"])
        for key in ("response", "complete", "systemPrompt", "user", "toolMarkup", "recents"):
            markup = _Markup(result[key])
            self.assertNotIn("img", [tag for tag, attrs in markup.tags])
            for tag, attrs in markup.tags:
                self.assertNotIn("onerror", attrs)
                self.assertNotIn("onfocus", attrs)


class DemoWebSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / "repo"
        for name in ("demo", "exp_configs", "data", "output", "uploads"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        (self.root / "demo" / "agent_runner.html").write_text("<!doctype html><title>Demo</title>")
        (self.root / "demo" / "agent_server.py").write_text("synthetic source")
        (self.root / "key.yaml").write_text("TEST_ONLY: synthetic-secret")
        (self.root / "exp_configs" / "selected.yaml").write_text("model: {}\n")
        (self.root / "SYSTEM_PROMPT.md").write_text("Synthetic prompt")
        self.clip = self.root / "data" / "clip.mp4"
        self.clip.write_bytes(b"0123456789")
        self.outside = Path(self.tmp.name).resolve() / "other" / "output"
        self.outside.mkdir(parents=True)
        (self.outside / "clip.mp4").write_bytes(b"outside-media")
        for name, value in {
            "REPO_ROOT": self.root,
            "DEMO_DIR": self.root / "demo",
            "EXP_CONFIGS_DIR": self.root / "exp_configs",
            "UPLOADS_DIR": self.root / "uploads",
            "DEFAULT_EXP_CONFIG": "exp_configs/selected.yaml",
        }.items():
            mock = patch.object(agent_server, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        env = patch.dict(os.environ, {"VIDEOX_WEB_TOKEN": "", "VIDEOX_WEB_HOST": "127.0.0.1", "VIDEOX_MEDIA_ROOTS": ""})
        env.start()
        self.addCleanup(env.stop)
        self.clients = []
        self.client = await self.make_client()

    async def asyncTearDown(self):
        for client in self.clients:
            await client.close()

    async def make_client(self, **kwargs):
        app = agent_server.create_app(**kwargs)
        # These callbacks import the agent/model stack, irrelevant to HTTP security.
        app.on_startup.clear()
        app.on_cleanup.clear()
        client = TestClient(TestServer(app))
        try:
            await client.start_server()
        except BaseException:
            await client.close()
            raise
        self.clients.append(client)
        return client

    async def test_only_the_ui_is_public_static_content(self):
        response = await self.client.get("/agent_runner.html")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        for url in ("/agent_server.py", "/conversation_store.py", "/key.yaml", "/.env", "/demo/", "/../key.yaml"):
            with self.subTest(url=url):
                response = await self.client.get(url)
                self.assertEqual(response.status, 404)
                self.assertNotIn("synthetic-secret", await response.text())

    async def test_media_is_limited_by_type_directory_and_realpath(self):
        (self.root / "output" / "key.yaml").write_text("TEST_ONLY: hidden-secret")
        (self.root / "output" / "page.html").write_text("<script>alert(1)</script>")
        (self.root / "output" / "diagram.svg").write_text("<svg/>")
        (self.root / "output" / ".private").mkdir()
        (self.root / "output" / ".private" / "clip.mp4").write_bytes(b"hidden")
        (self.root / "output" / "escape.mp4").symlink_to(self.outside / "clip.mp4")
        (self.root / "output" / "key.mp4").symlink_to(self.root / "key.yaml")
        forbidden = (
            self.root / "key.yaml",
            self.root / "output" / "key.yaml",
            self.root / "output" / "page.html",
            self.root / "output" / "diagram.svg",
            self.root / "output" / ".private" / "clip.mp4",
            self.root / "output" / "escape.mp4",
            self.root / "output" / "key.mp4",
            self.outside / "clip.mp4",
            self.root / "data",
            self.root / "data" / "missing.mp4",
        )
        for path in forbidden:
            with self.subTest(path=path):
                response = await self.client.get("/api/file", params={"path": str(path)})
                self.assertEqual(response.status, 404)
                self.assertNotIn(str(self.root), await response.text())
        response = await self.client.get("/api/file", params={"path": str(self.clip)})
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), b"0123456789")
        self.assertEqual(response.headers["Content-Type"], "video/mp4")
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)

    async def test_ranges_and_relative_media_paths_still_work(self):
        response = await self.client.get("/api/file", params={"path": "data/clip.mp4"}, headers={"Range": "bytes=2-5"})
        self.assertEqual(response.status, 206)
        self.assertEqual(await response.read(), b"2345")
        self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")

    async def test_compressed_siblings_cannot_bypass_file_allowlists(self):
        paths = [self.clip, self.root / "demo" / "agent_runner.html"]
        urls = [("/api/file", {"path": str(self.clip)}), ("/agent_runner.html", {})]
        for path, (url, params) in zip(paths, urls):
            path.with_suffix(path.suffix + ".gz").write_bytes(gzip.compress(b"unapproved-sibling"))
            response = await self.client.get(url, params=params, headers={"Accept-Encoding": "gzip"})
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.read(), path.read_bytes())
            self.assertNotIn("Content-Encoding", response.headers)

    async def test_explicit_external_media_root_is_supported(self):
        client = await self.make_client(media_roots=[self.outside])
        response = await client.get("/api/file", params={"path": str(self.outside / "clip.mp4")})
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), b"outside-media")

    async def test_default_media_roots_cannot_be_replaced_with_external_symlinks(self):
        # The server has already started, so this also checks post-startup changes.
        self.clip.unlink()
        for name in ("output", "uploads", "data"):
            folder = self.root / name
            folder.rmdir()
            folder.symlink_to(self.outside, target_is_directory=True)
            response = await self.client.get("/api/file", params={"path": str(folder / "clip.mp4")})
            self.assertEqual(response.status, 404)
            response = await self.client.get("/api/file", params={"path": str(self.outside / "clip.mp4")})
            self.assertEqual(response.status, 404)
        body = aiohttp.FormData()
        body.add_field("video", b"must-not-be-written", filename="upload.mp4", content_type="video/mp4")
        response = await self.client.post("/api/upload", data=body)
        self.assertEqual(response.status, 503)
        self.assertEqual(sorted(path.name for path in self.outside.iterdir()), ["clip.mp4"])
        # External locations still work when deliberately selected by the operator.
        client = await self.make_client(media_roots=[self.root / "output"])
        response = await client.get("/api/file", params={"path": str(self.outside / "clip.mp4")})
        self.assertEqual(response.status, 200)

    async def test_non_loopback_bind_requires_token(self):
        with self.assertRaises(ValueError):
            agent_server.create_app(host="0.0.0.0")
        with self.assertRaises(ValueError):
            agent_server.create_app(host="::")

    async def test_remote_auth_covers_static_api_upload_and_websocket(self):
        client = await self.make_client(host="0.0.0.0", token="test-only-password")
        for url in ("/agent_runner.html", "/api/configs", "/api/conversations", "/api/file?path=data/clip.mp4"):
            response = await client.get(url)
            self.assertEqual(response.status, 401)
            self.assertIn("Basic", response.headers["WWW-Authenticate"])
            self.assertNotIn("test-only-password", await response.text())
        response = await client.post("/api/upload", data=b"not multipart")
        self.assertEqual(response.status, 401)
        with self.assertRaises(aiohttp.WSServerHandshakeError) as failed:
            await client.ws_connect("/ws")
        self.assertEqual(failed.exception.status, 401)
        basic = base64.b64encode(b"videoxagent:test-only-password").decode()
        response = await client.get("/agent_runner.html", headers={"Authorization": f"Basic {basic}"})
        self.assertEqual(response.status, 200)
        response = await client.get("/api/configs", headers={"Authorization": "Bearer test-only-password"})
        self.assertEqual(response.status, 200)
        ws = await client.ws_connect("/ws", headers={"Authorization": f"Basic {basic}"})
        await ws.send_json({"action": "ping"})
        self.assertEqual(await ws.receive_json(), {"type": "pong"})
        await ws.close()

    async def test_cross_origin_and_rebinding_requests_are_rejected(self):
        response = await self.client.get("/api/configs", headers={"Origin": "https://attacker.invalid"})
        self.assertEqual(response.status, 403)
        response = await self.client.get("/api/configs", headers={"Host": "attacker.invalid"})
        self.assertEqual(response.status, 403)
        with self.assertRaises(aiohttp.WSServerHandshakeError) as failed:
            await self.client.ws_connect("/ws", origin="https://attacker.invalid")
        self.assertEqual(failed.exception.status, 403)
        origin = str(self.client.make_url("/")).rstrip("/")
        ws = await self.client.ws_connect("/ws", origin=origin)
        await ws.send_json({"action": "ping"})
        self.assertEqual(await ws.receive_json(), {"type": "pong"})
        await ws.close()

    async def test_upload_rejects_active_content_and_accepts_video(self):
        for name in ("payload.html", "payload.svg", "key.yaml"):
            body = aiohttp.FormData()
            body.add_field("video", b"synthetic", filename=name)
            response = await self.client.post("/api/upload", data=body)
            self.assertEqual(response.status, 400)
        self.assertEqual(list((self.root / "uploads").iterdir()), [])
        body = aiohttp.FormData()
        body.add_field("video", b"synthetic-video", filename="../clip.mp4", content_type="video/mp4")
        response = await self.client.post("/api/upload", data=body)
        self.assertEqual(response.status, 200)
        data = await response.json()
        self.assertEqual(Path(data["path"]).parent, self.root / "uploads")
        response = await self.client.get("/api/file", params={"path": data["path"]})
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), b"synthetic-video")

    async def test_config_and_prompt_cannot_be_arbitrary_files(self):
        for path in ("key.yaml", str(self.root / "key.yaml")):
            response = await self.client.get("/api/tools", params={"config": path})
            self.assertEqual(response.status, 400)
            self.assertNotIn("synthetic-secret", await response.text())
        (self.root / "exp_configs" / "escape.yaml").symlink_to(self.root / "key.yaml")
        with self.assertRaises(ValueError):
            agent_server.resolve_experiment_config("exp_configs/escape.yaml")
        self.assertEqual(agent_server.resolve_system_prompt("SYSTEM_PROMPT.md"), self.root / "SYSTEM_PROMPT.md")
        (self.root / "SYSTEM_PROMPT_escape.md").symlink_to(self.root / "key.yaml")
        with self.assertRaises(ValueError):
            agent_server.resolve_system_prompt("SYSTEM_PROMPT_escape.md")
        ws = await self.client.ws_connect("/ws")
        for config, system_prompt in (("key.yaml", "SYSTEM_PROMPT.md"), ("exp_configs/selected.yaml", "key.yaml")):
            await ws.send_json({"action": "start", "prompt": "Synthetic request", "config": config, "system_prompt": system_prompt})
            event = await ws.receive_json()
            self.assertEqual(event["type"], "error")
            self.assertEqual(event["data"]["error"], "Invalid experiment config or system prompt")
        await ws.close()

    async def test_main_defaults_to_loopback(self):
        with patch("sys.argv", ["agent_server.py"]), patch.object(agent_server.web, "run_app") as run:
            agent_server.main()
        self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")
        self.assertEqual(run.call_args.kwargs["port"], 8080)

    async def test_main_passes_explicit_host_and_media_roots(self):
        with patch.dict(os.environ, {"VIDEOX_WEB_TOKEN": "test-only-password", "VIDEOX_MEDIA_ROOTS": str(self.outside)}), patch("sys.argv", ["agent_server.py", "8081", "--host", "0.0.0.0", "--media-root", str(self.root / "custom")]), patch.object(agent_server.web, "run_app") as run:
            agent_server.main()
        self.assertEqual(run.call_args.kwargs["host"], "0.0.0.0")
        self.assertEqual(run.call_args.kwargs["port"], 8081)
        roots = run.call_args.args[0][agent_server.MEDIA_ROOTS_KEY]
        self.assertIn(self.outside, roots)
        self.assertIn(self.root / "custom", roots)


if __name__ == "__main__":
    unittest.main()
