"""Narrow HTTP exposure on top of the benchmark relocation resolver.

BenchmarkPaths understands bundles and former machine paths. This module decides
which of those resolved files are appropriate to expose over HTTP.
"""

import json
from pathlib import Path
from urllib.parse import unquote

from .benchmark_paths import BenchmarkPaths, is_relative_to
from utils.web_security import MEDIA_TYPES, safe_media_path


FRONTEND_ASSETS = frozenset({
    "brand.css", "image_viewer.css", "image_viewer.js", "tailwind.min.js",
    "chart.min.js", "videoxagent-icon.png",
})
_SENSITIVE_PARTS = frozenset({"secrets", "credentials", "config", "configs", "configuration"})


def _public_path(path):
    """Never expose hidden or configuration directories, including symlink targets."""
    return all(
        not (part.startswith(".") and part not in (".", ".."))
        and part.lower() not in _SENSITIVE_PARTS
        for part in Path(path).parts
    )


class DashboardFiles:
    def __init__(self, paths: BenchmarkPaths, static_root=None, dashboard=None):
        self.paths = paths
        self.static_root = Path(static_root or paths.repo_root / "html_show").resolve()
        self.dashboard = None
        if dashboard:
            requested = Path(dashboard).expanduser().absolute()
            if requested.suffix.lower() != ".html" or not _public_path(requested):
                raise ValueError("--dashboard must select a non-hidden HTML file")
            # The administrator selects this exact export. Canonicalize its
            # parent (e.g. macOS /var -> /private/var), but preserve the final
            # component so a symlink cannot disguise a private file as HTML.
            self.dashboard = requested.parent.resolve() / requested.name
        # The normal repo layout must not make every repository file public.
        # Explicit bundle roots and mapping destinations remain supported.
        self.data_roots = (
            [paths.data_root] if paths.restricted else [
                paths.repo_root / name for name in ("conversations", "output", "data")
                if not (paths.repo_root / name).is_symlink()
            ]
        ) + [target for _source, target in paths.path_maps]
        self.data_roots = [root.resolve() for root in self.data_roots]

    def _safe_resolved(self, raw, base=None):
        if not _public_path(str(raw)):
            raise PermissionError("Hidden and configuration paths are not served")
        resolved = self.paths.resolve(raw, base=base)
        if resolved is None:
            return None
        if not _public_path(resolved) or not any(is_relative_to(resolved, root) for root in self.data_roots):
            raise PermissionError("File is outside the public benchmark directories")
        return resolved

    def trace(self, raw, base=None):
        if Path(str(raw)).suffix.lower() != ".jsonl":
            raise PermissionError("Only registered JSONL transcripts may be read")
        resolved = self._safe_resolved(raw, base=base)
        if resolved is None:
            return None
        if resolved.suffix.lower() != ".jsonl":
            raise PermissionError("Only registered JSONL transcripts may be read")
        manifest = self.paths.data_file or self.paths.repo_root / "conversations/all_data_path.json"
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise PermissionError("No readable benchmark index has been selected") from None
        sessions = data if isinstance(data, list) else data.get("sessions", []) if isinstance(data, dict) else []
        if not isinstance(sessions, list):
            sessions = []
        for session in sessions:
            reference = session.get("transcript_path") if isinstance(session, dict) else None
            if not isinstance(reference, str) or Path(reference).suffix.lower() != ".jsonl":
                continue
            try:
                if self._safe_resolved(reference.strip()) == resolved:
                    return resolved
            except (PermissionError, OSError, RuntimeError):
                continue
        raise PermissionError("Transcript is not referenced by the selected benchmark index")

    def media(self, raw, base=None):
        if Path(str(raw)).suffix.lower() not in MEDIA_TYPES:
            raise PermissionError("Only image, video and audio files may be served")
        resolved = self._safe_resolved(raw, base=base)
        if resolved is None:
            return None
        return safe_media_path(resolved, self.data_roots)

    def static(self, request_path):
        decoded = unquote(request_path)
        if "\x00" in decoded or "\\" in decoded:
            raise PermissionError("Invalid static path")
        relative = Path(decoded.lstrip("/"))
        if ".." in relative.parts or not _public_path(relative):
            raise PermissionError("Invalid static path")
        candidate = self.static_root / relative
        # An explicitly selected export is available at / and at its basename.
        if self.dashboard and decoded in ("/", "/" + self.dashboard.name):
            candidate = self.dashboard
            if candidate.is_symlink() or candidate.resolve() != candidate or not _public_path(candidate):
                raise PermissionError("Hidden paths and dashboard symlinks are not served")
            return candidate if candidate.is_file() else None
        if not is_relative_to(candidate.resolve(), self.static_root):
            raise PermissionError("Static path escapes the selected directory")
        html_root = self.paths.repo_root / "html_show"
        if self.static_root in (self.paths.repo_root, html_root):
            expected_parent = html_root
            allowed = candidate.parent == expected_parent and (
                candidate.name in FRONTEND_ASSETS or (
                    candidate.name.startswith("tool_workflow") and candidate.suffix == ".html"
                )
            )
        else:
            # A directory selected for a portable export publishes one page.
            allowed = candidate == self.static_root / "dashboard.html"
        if not allowed:
            raise PermissionError("Static file is not a dashboard asset")
        # Static files are exact trusted assets; do not let a symlink give an
        # allowed public filename to a private file elsewhere in the checkout.
        if candidate.is_symlink() or candidate.resolve() != candidate:
            raise PermissionError("Static symlinks are not served")
        return candidate if candidate.is_file() else None
