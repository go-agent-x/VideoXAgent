"""Shared browser access rules without opening ports or invoking any models."""

import base64
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.web_security import MEDIA_TYPES, WebAccessPolicy, safe_media_path


class WebAccessPolicyTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop("VIDEOX_WEB_TOKEN", None)
        self.token = "test-only-not-a-real-secret"

    def assert_rejected(self, policy, host, origin=None, authorization=None, status=403):
        failure = policy.check_request(host, origin, authorization)
        self.assertIsNotNone(failure)
        self.assertEqual(failure[0], status, failure)

    def test_loopback_binding_and_host_allowlist(self):
        policy = WebAccessPolicy()
        for host in ("127.0.0.1:8923", "localhost:8923", "[::1]:8923"):
            self.assertIsNone(policy.check_request(host))
        self.assert_rejected(policy, "", status=400)
        for host in ("attacker.example:8923", "192.0.2.1:8923", "localhost.attacker.example"):
            self.assert_rejected(policy, host)

    def test_nonloopback_binding_requires_token(self):
        for host in ("0.0.0.0", "::", "192.0.2.1", "dashboard.example"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                WebAccessPolicy(host=host)
        with patch.dict(os.environ, {"VIDEOX_WEB_TOKEN": self.token}):
            self.assertIsNotNone(WebAccessPolicy(host="0.0.0.0"))

    def test_authentication_is_enforced_even_on_localhost_when_configured(self):
        policy = WebAccessPolicy(token=self.token)
        for auth in (None, "", "Bearer wrong", "Basic !!!", "Digest ignored"):
            self.assert_rejected(policy, "localhost:8923", authorization=auth, status=401)
        basic = base64.b64encode(("videoxagent:" + self.token).encode()).decode()
        for auth in ("Bearer " + self.token, "Basic " + basic):
            self.assertIsNone(policy.check_request("localhost:8923", authorization=auth))

    def test_origin_must_match_request_host(self):
        policy = WebAccessPolicy()
        for origin in ("http://localhost:8923", "https://localhost:8923"):
            self.assertIsNone(policy.check_request("localhost:8923", origin))
        for origin in ("null", "http://attacker.example", "http://localhost:8080",
                       "http://localhost:8923.attacker.example", "http://user@localhost:8923",
                       "file://localhost:8923", "http://localhost:8923/path"):
            with self.subTest(origin=origin):
                self.assert_rejected(policy, "localhost:8923", origin)

    def test_authentication_does_not_bypass_origin_or_host_validation(self):
        policy = WebAccessPolicy(token=self.token)
        self.assert_rejected(policy, "localhost:8923", "https://attacker.example", "Bearer " + self.token)
        self.assert_rejected(policy, "attacker.example:8923", authorization="Bearer " + self.token)

    def test_malformed_authorities_are_rejected(self):
        policy = WebAccessPolicy()
        for host in ("localhost:bad", "localhost:99999", "localhost/secret", "user@localhost",
                     "localhost?query", "localhost#fragment"):
            with self.subTest(host=host):
                self.assert_rejected(policy, host, status=400)


class MediaPathPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="web media policy ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.media = self.root / "public-media"
        self.media.mkdir()

    def write(self, relative, content=b"fixture"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_media_types_exclude_active_content_and_configuration(self):
        self.assertIn(".png", MEDIA_TYPES)
        self.assertIn(".mp4", MEDIA_TYPES)
        self.assertIn(".wav", MEDIA_TYPES)
        for extension in (".svg", ".html", ".js", ".json", ".jsonl", ".yaml", ".pdf", ".txt", ".py"):
            self.assertNotIn(extension, MEDIA_TYPES)

    def test_regular_media_and_relative_base_are_supported(self):
        image = self.write("public-media/frames/frame café #1.png")
        self.assertEqual(safe_media_path(str(image), [self.media]), image)
        self.assertEqual(safe_media_path("frames/frame café #1.png", [self.media], base=self.media), image)

    def test_missing_media_is_distinct_from_denied_path(self):
        with self.assertRaises(FileNotFoundError):
            safe_media_path(str(self.media / "missing.png"), [self.media])
        forbidden = self.write("outside.png")
        for raw in (str(forbidden), str(self.media / "../outside.png"), "\x00.png"):
            with self.subTest(path=raw), self.assertRaises(PermissionError):
                safe_media_path(raw, [self.media])

    def test_hidden_paths_and_non_media_cannot_be_served(self):
        for relative in ("public-media/key.yaml", "public-media/index.html", "public-media/image.svg",
                         "public-media/.hidden/frame.png", "public-media/.private.png"):
            path = self.write(relative)
            with self.subTest(path=path), self.assertRaises(PermissionError):
                safe_media_path(str(path), [self.media])

    def test_symlink_targets_and_lexical_hidden_paths_are_checked(self):
        private = self.write("private/frame.png")
        link = self.media / "linked.png"
        link.symlink_to(private)
        with self.assertRaises(PermissionError):
            safe_media_path(str(link), [self.media])
        public = self.write("public-media/frame.png")
        hidden = self.media / ".hidden-link.png"
        hidden.symlink_to(public)
        with self.assertRaises(PermissionError):
            safe_media_path(str(hidden), [self.media])
        target = self.write("public-media/.private/secret.png")
        (self.media / "public-link.png").symlink_to(target)
        with self.assertRaises(PermissionError):
            safe_media_path(str(self.media / "public-link.png"), [self.media])


if __name__ == "__main__":
    unittest.main()
