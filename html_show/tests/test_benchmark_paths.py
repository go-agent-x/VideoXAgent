"""Portable benchmark bundles and API access, without a listening server."""

import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
from urllib.parse import urlencode

from html_show.benchmark_paths import BenchmarkPaths
from html_show.server import SessionViewerHandler, parse_args
from utils.web_security import WebAccessPolicy


class MemoryHandler(SessionViewerHandler):
    """Exercise the real routes and response body without binding a port."""
    def __init__(self, paths, url="/", headers=None):
        self.benchmark_paths = paths
        self.access_policy = WebAccessPolicy(host="127.0.0.1", token="")
        self.path = url
        self.headers = {"Host": "127.0.0.1"}
        self.headers.update(headers or {})
        self.wfile = io.BytesIO()
        self.status = None
        self.response_headers = {}
        self.error = None

    def send_response(self, code, message=None):
        self.status = code

    def send_header(self, key, value):
        self.response_headers[key] = value

    def end_headers(self):
        pass

    def send_error(self, code, message=None, explain=None):
        self.status = code
        self.error = message


class BenchmarkPathsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="benchmark paths ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "repository"
        self.repo.mkdir()
        self.source = self.root / "original café bundle"
        self.source.mkdir()
        self.trace = self.write(self.source / "conversations/transcripts/run/trace.jsonl", b'{"type":"test"}\n')
        self.image = self.write(self.source / "output/run/frame café %20.png", b"sample image bytes")
        self.local_image = self.write(self.trace.parent / "local frame.png", b"transcript relative image")
        self.data = self.write(self.source / "conversations/results.json", json.dumps({
            "sessions": [{"transcript_path": "transcripts/run/trace.jsonl"}]
        }).encode())

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def copy_bundle(self):
        destination = self.root / "downloaded results café"
        shutil.copytree(str(self.source), str(destination))
        paths = BenchmarkPaths(self.repo, destination / "conversations/results.json", destination)
        return destination, paths

    def test_bundle_can_move_without_rewriting_jsonl(self):
        # Original machine paths remain verbatim in the JSONL download.
        content = json.dumps({"image_path": str(self.image)}, ensure_ascii=False).encode("utf-8")
        self.trace.write_bytes(content)
        destination, paths = self.copy_bundle()
        shutil.rmtree(str(self.source))
        moved_trace = destination / self.trace.relative_to(self.source)
        moved_image = destination / self.image.relative_to(self.source)
        self.assertEqual(paths.resolve(str(self.trace)), moved_trace)
        self.assertEqual(paths.resolve(str(self.image)), moved_image)
        self.assertEqual(paths.resolve("transcripts/run/trace.jsonl"), moved_trace)
        self.assertEqual(paths.resolve("conversations/transcripts/run/trace.jsonl"), moved_trace)
        self.assertEqual(paths.resolve("local frame.png", base=paths.reference(moved_trace)),
                         moved_trace.parent / "local frame.png")
        self.assertEqual(moved_trace.read_bytes(), content)
        self.assertEqual(paths.reference(moved_image), "output/run/frame café %20.png")

    def test_explicit_mapping_overrides_existing_original_and_uses_longest_prefix(self):
        self.write(self.repo / "wrong/output/run/frame café %20.png", b"wrong")
        destination, _ = self.copy_bundle()
        paths = BenchmarkPaths(self.repo, data_root=self.source, path_maps=[
            "{}={}".format(self.source, self.repo / "wrong"),
            "{}={}".format(self.source / "output", destination / "output"),
        ])
        self.assertTrue(self.image.exists())
        self.assertEqual(paths.resolve(str(self.image)), destination / self.image.relative_to(self.source))
        self.assertEqual(paths.reference(destination / self.image.relative_to(self.source)),
                         str(self.image))

    def test_external_mapping_reference_survives_destination_move(self):
        assets_a = self.root / "machine A assets"
        old_trace = self.write(assets_a / "nested/trace.jsonl", b"external trace")
        first = BenchmarkPaths(self.repo, data_root=self.source, path_maps=[
            "/old/workspace={}".format(assets_a),
            "/old/customlogs={}".format(assets_a / "nested"),
        ])
        reference = first.reference(old_trace)
        self.assertEqual(reference, "/old/customlogs/trace.jsonl")
        self.assertEqual(first.resolve(reference), old_trace)
        assets_b = self.root / "machine B assets"
        shutil.move(str(assets_a), str(assets_b))
        second = BenchmarkPaths(self.repo, data_root=self.source, path_maps=[
            "/old/workspace={}".format(assets_b),
            "/old/customlogs={}".format(assets_b / "nested"),
        ])
        moved_trace = assets_b / "nested/trace.jsonl"
        self.assertEqual(second.resolve(reference), moved_trace)
        self.assertEqual(second.reference(moved_trace), reference)
        self.assertEqual(second.resolve(reference).read_bytes(), b"external trace")

    def test_data_root_reference_takes_priority_over_reverse_mapping(self):
        paths = BenchmarkPaths(self.repo, data_root=self.source, path_maps=[
            "/old/media={}".format(self.source / "output"),
        ])
        self.assertEqual(paths.reference(self.image), "output/run/frame café %20.png")

    def test_missing_explicit_mapping_target_does_not_fall_back_to_original(self):
        paths = BenchmarkPaths(self.repo, data_root=self.source, path_maps=[
            "{}={}".format(self.source / "output", self.root / "missing")
        ])
        self.assertTrue(self.image.exists())
        self.assertIsNone(paths.resolve(str(self.image)))
        handler = MemoryHandler(paths)
        handler.send_binary_file(str(self.image))
        self.assertEqual(handler.status, 404)

    def test_relative_mapping_destination_is_relative_to_data_root(self):
        paths = BenchmarkPaths(self.repo, data_root=self.source,
                               path_maps=["/old/workspace=."])
        self.assertEqual(paths.resolve("/old/workspace/output/run/frame café %20.png"), self.image)

    def test_data_root_default_preserves_repo_layout(self):
        data = self.write(self.repo / "conversations/results.json", b"{}")
        paths = BenchmarkPaths(self.repo, data)
        self.assertEqual(paths.data_root, self.repo)
        self.assertFalse(paths.restricted)
        external = BenchmarkPaths(self.repo, self.data)
        self.assertEqual(external.data_root, self.data.parent)
        self.assertTrue(external.restricted)
        self.assertEqual(external.resolve("transcripts/run/trace.jsonl"), self.trace)
        self.assertEqual(external.resolve(str(self.trace)), self.trace)

    def test_data_root_reference_takes_priority_over_manifest_and_trace_relative(self):
        paths = BenchmarkPaths(self.repo, self.data, self.source)
        for prefix in ("conversations", "transcripts", "output", "data"):
            reference = prefix + "/same.png"
            expected = self.write(self.source / reference, b"root")
            self.write(self.data.parent / reference, b"manifest")
            self.write(self.trace.parent / reference, b"trace")
            self.assertEqual(paths.resolve(reference, base=str(self.trace)), expected)

    def test_unanchored_media_uses_current_transcript_before_same_named_files(self):
        paths = BenchmarkPaths(self.repo, self.data, self.source)
        for reference in ("same.png", "images/same.png", "./output/same.png"):
            self.write(self.source / reference, b"wrong root image")
            self.write(self.data.parent / reference, b"wrong manifest image")
            expected = self.write(self.trace.parent / reference, b"correct trace image")
            resolved = paths.resolve(reference, base=str(self.trace))
            self.assertEqual(resolved, expected)
            self.assertEqual(resolved.read_bytes(), b"correct trace image")
        parent_image = self.write(self.trace.parent.parent / "same.png", b"correct parent image")
        self.write(self.source.parent / "same.png", b"outside root image")
        self.assertEqual(paths.resolve("../same.png", base=str(self.trace)), parent_image)

    def test_missing_or_inaccessible_base_does_not_block_valid_root_media(self):
        paths = BenchmarkPaths(self.repo, self.data, self.source)
        expected = self.write(self.source / "same.png", b"root image")
        for base in (None, "missing/trace.jsonl", str(self.root / "private/trace.jsonl"), "\x00"):
            self.assertEqual(paths.resolve("same.png", base=base), expected)

    def test_missing_files_do_not_search_another_experiment_by_basename(self):
        self.write(self.source / "other-experiment/trace.jsonl", b"unrelated")
        paths = BenchmarkPaths(self.repo, self.data, self.source)
        self.assertIsNone(paths.resolve("transcripts/absent/trace.jsonl"))
        self.assertIsNone(paths.resolve("/old/conversations/transcripts/absent/trace.jsonl"))

    def test_parent_traversal_outside_root_is_denied(self):
        self.write(self.root / "secret.txt", b"private")
        paths = BenchmarkPaths(self.repo, data_root=self.source)
        with self.assertRaises(PermissionError):
            paths.resolve("../secret.txt")
        with self.assertRaises(PermissionError):
            paths.resolve(str(self.root / "secret.txt"))

    def test_transcript_relative_parent_paths_inside_bundle_are_allowed(self):
        paths = BenchmarkPaths(self.repo, self.data, self.source)
        self.assertEqual(paths.resolve("../../../output/run/frame café %20.png", base=str(self.trace)), self.image)

    def test_symlink_cannot_escape_data_root(self):
        secret = self.write(self.root / "secret.txt", b"private")
        (self.source / "linked.txt").symlink_to(secret)
        paths = BenchmarkPaths(self.repo, data_root=self.source)
        with self.assertRaises(PermissionError):
            paths.resolve("linked.txt")

    def test_path_map_cannot_escape_via_parent_or_symlink(self):
        mapped = self.root / "mapped"
        mapped.mkdir()
        secret = self.write(self.root / "secret.txt", b"private")
        (mapped / "linked.txt").symlink_to(secret)
        paths = BenchmarkPaths(self.repo, data_root=self.source,
                               path_maps=["/old/media={}".format(mapped)])
        for path in ("/old/media/../secret.txt", "/old/media/linked.txt"):
            with self.assertRaises(PermissionError):
                paths.resolve(path)

    def test_base_does_not_authorize_external_directory(self):
        paths = BenchmarkPaths(self.repo, data_root=self.source)
        external = self.write(self.root / "external/trace.jsonl", b"{}")
        self.write(external.parent / "media.png", b"private")
        with self.assertRaises(PermissionError):
            paths.resolve("media.png", base=str(external))

    def test_invalid_mapping_fails_with_useful_error(self):
        for mapping in ("missing separator", "=target", "/old=", "relative=/new"):
            with self.assertRaises(ValueError):
                BenchmarkPaths(self.repo, path_maps=[mapping])

    def test_api_reads_moved_trace_and_relative_image_with_encoded_filename(self):
        destination, paths = self.copy_bundle()
        shutil.rmtree(str(self.source))
        read = MemoryHandler(paths, "/api/read?" + urlencode({"path": str(self.trace)}))
        read.do_GET()
        self.assertEqual(read.status, 200)
        self.assertEqual(read.wfile.getvalue(), b'{"type":"test"}\n')
        media = MemoryHandler(paths, "/api/file?" + urlencode({"path": str(self.image), "base": str(self.trace)}))
        media.do_GET()
        self.assertEqual(media.status, 200, media.error)
        self.assertEqual(media.wfile.getvalue(), b"sample image bytes")
        self.assertEqual(media.response_headers["Content-Type"], "image/png")
        local = MemoryHandler(paths, "/api/file?" + urlencode({"path": "local frame.png", "base": str(self.trace)}))
        local.do_GET()
        self.assertEqual(local.status, 200, local.error)
        self.assertEqual(local.wfile.getvalue(), b"transcript relative image")

    def test_api_denied_missing_and_bad_query_have_distinct_status(self):
        paths = BenchmarkPaths(self.repo, data_root=self.source)
        for url, status in (("/api/read?path=../private.txt", 403),
                            ("/api/file?path=missing.png", 404), ("/api/read", 400)):
            handler = MemoryHandler(paths, url)
            handler.do_GET()
            self.assertEqual(handler.status, status, handler.error)

    def test_legacy_session_roots_are_never_implicitly_authorized(self):
        legacy = self.write(self.root / "legacy sessions/trace.jsonl", b"legacy")
        for paths in (BenchmarkPaths(self.repo), BenchmarkPaths(self.repo, data_root=self.source)):
            handler = MemoryHandler(paths)
            handler.send_file_content(str(legacy))
            self.assertEqual(handler.status, 403)

    def test_unlisted_transcript_is_denied_inside_selected_data_root(self):
        private = self.write(self.source / "conversations/transcripts/private.jsonl", b"private transcript")
        paths = BenchmarkPaths(self.repo, self.data, self.source)
        handler = MemoryHandler(paths, "/api/read?" + urlencode({"path": str(private)}))
        handler.do_GET()
        self.assertEqual(handler.status, 403)
        self.assertNotIn(b"private transcript", handler.wfile.getvalue())

    def test_media_range_responses_and_invalid_ranges(self):
        paths = BenchmarkPaths(self.repo, data_root=self.source)
        for header, status, content in (("bytes=2-5", 206, b"mple"),
                                        ("bytes=-5", 206, b"bytes"),
                                        ("bytes=999-", 416, b""),
                                        ("bytes=-0", 416, b""),
                                        ("bytes=0-1,4-5", 416, b"")):
            handler = MemoryHandler(paths, headers={"Range": header})
            handler.send_binary_file(str(self.image))
            self.assertEqual(handler.status, status, handler.error)
            self.assertEqual(handler.wfile.getvalue(), content)
            if status == 206:
                self.assertEqual(handler.response_headers["Content-Length"], str(len(content)))
        handler = MemoryHandler(paths, headers={"Range": "bytes=2-5"})
        handler.send_binary_file(str(self.image))
        self.assertEqual(handler.response_headers["Content-Range"], "bytes 2-5/18")

    def test_server_cli_env_and_explicit_override(self):
        env = {"VIDEOX_DATA_ROOT": str(self.source), "VIDEOX_DATA_FILE": str(self.data),
               "VIDEOX_PATH_MAPS": '["/old=/new"]'}
        with mock.patch.dict(os.environ, env):
            args = parse_args(["8923", str(self.repo)])
            self.assertEqual(args.data_root, str(self.source))
            self.assertEqual(args.data, str(self.data))
            self.assertEqual(args.path_map, ["/old=/new"])
            args = parse_args(["8923", str(self.repo), "--path-map", "/a=/b", "--path-map", "/c=/d"])
            self.assertEqual(args.path_map, ["/a=/b", "/c=/d"])


if __name__ == "__main__":
    unittest.main()
