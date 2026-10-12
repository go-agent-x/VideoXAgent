"""Resolve benchmark references without rewriting the source JSON or JSONL.

All accepted files must remain inside the selected data root or an explicit
path-map destination after symlinks have been resolved.  No basename searches
are used: directory structure identifies files from an individual run.
"""

from pathlib import Path


def is_relative_to(path, root):
    """Python 3.7-compatible Path.is_relative_to."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class BenchmarkPaths:
    """Path policy shared by HTML generation and the session API.

    A relative mapping destination is relative to ``data_root``. The source
    prefix is absolute, and the longest matching source prefix wins.
    """

    def __init__(self, repo_root, data_file=None, data_root=None, path_maps=None):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.data_file = None
        if data_file:
            source = Path(data_file).expanduser()
            self.data_file = (source if source.is_absolute() else self.repo_root / source).resolve()
        external_data = bool(self.data_file and not is_relative_to(self.data_file, self.repo_root))
        self.restricted = data_root is not None or external_data
        root = data_root if data_root is not None else (
            self.data_file.parent if external_data else self.repo_root
        )
        root = Path(root).expanduser()
        self.data_root = (root if root.is_absolute() else self.repo_root / root).resolve()
        self.path_maps = []
        for item in path_maps or []:
            if not isinstance(item, str) or "=" not in item:
                raise ValueError("Path maps must use OLD=NEW")
            old, new = item.split("=", 1)
            if not old or not new:
                raise ValueError("Path maps must have nonempty OLD and NEW paths")
            old_path = Path(old).expanduser()
            if not old_path.is_absolute() or ".." in old_path.parts:
                raise ValueError("Path-map OLD must be an absolute path without '..'")
            new_path = Path(new).expanduser()
            if not new_path.is_absolute():
                new_path = self.data_root / new_path
            self.path_maps.append((old_path, new_path.resolve()))
        self.path_maps.sort(key=lambda item: len(item[0].parts), reverse=True)
        self.allowed_roots = [self.data_root] + [new for _old, new in self.path_maps]

    def is_allowed(self, path):
        resolved = Path(path).resolve()
        return any(is_relative_to(resolved, root) for root in self.allowed_roots)

    def _existing(self, candidates):
        denied = False
        seen = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError):
                denied = True
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if not self.is_allowed(resolved):
                denied = True
                continue
            if resolved.is_file():
                return resolved
        if denied:
            raise PermissionError("Path is outside the configured benchmark data roots")
        return None

    def _portable_candidates(self, path):
        candidates = []
        parts = path.parts
        for index, part in enumerate(parts):
            if part not in ("conversations", "transcripts", "output", "data"):
                continue
            suffix = Path(*parts[index:])
            candidates.append(self.data_root / suffix)
            # A bundle may itself be its conversations/ or output/ directory.
            if self.data_root.name == part:
                candidates.append(self.data_root.joinpath(*parts[index + 1:]))
        return candidates

    def resolve(self, raw, base=None):
        """Return an existing authorized file, None if absent, or deny access.

        ``base`` is an optional transcript reference for resolving media paths
        relative to that transcript. It never grants access to another root.
        Explicit mappings take precedence even if the original file exists.
        """
        if not raw:
            return None
        raw = str(raw)
        if "\x00" in raw:
            raise PermissionError("Invalid path")
        path = Path(raw).expanduser()
        for old, new in self.path_maps:
            if is_relative_to(path, old):
                mapped = new / path.relative_to(old)
                # A traversal may not escape even into a separately allowed root.
                if not is_relative_to(mapped.resolve(), new):
                    raise PermissionError("Path escapes its configured mapping")
                return self._existing([mapped])
        if path.is_absolute():
            portable = self._portable_candidates(path)
            # Old absolute paths inside the current root remain supported.
            if self.is_allowed(path):
                return self._existing([path] + portable)
            if portable:
                return self._existing(portable)
            raise PermissionError("Absolute path requires --data-root or --path-map")
        candidates = [self.data_root / path]
        if self.data_file:
            candidates.append(self.data_file.parent / path)
        base_error = None
        if base:
            try:
                resolved_base = self.resolve(base)
            except (PermissionError, OSError, RuntimeError) as error:
                # A stale or inaccessible transcript must not prevent an
                # otherwise valid root-relative media reference from loading.
                resolved_base = None
                base_error = error
            if resolved_base:
                base_candidate = resolved_base.parent / path
                root_relative = (path.parts and path.parts[0] in ("conversations", "transcripts", "output", "data")
                                 and not raw.startswith(("./", "../")))
                if root_relative:
                    candidates.append(base_candidate)
                else:
                    candidates.insert(0, base_candidate)
        result = self._existing(candidates)
        if result is None and base_error:
            raise PermissionError("Transcript base is outside the configured benchmark data roots")
        return result

    def reference(self, path):
        """Return a stable root-relative or original mapped API reference.

        External mapped files retain their original prefix so the generated
        HTML can be moved and served with a different mapping destination.
        """
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = self.data_root / path
        resolved = path.resolve()
        if is_relative_to(resolved, self.data_root):
            return resolved.relative_to(self.data_root).as_posix()
        for old, new in sorted(self.path_maps, key=lambda item: len(item[1].parts), reverse=True):
            if is_relative_to(resolved, new):
                return str(old / resolved.relative_to(new))
        return str(resolved)
