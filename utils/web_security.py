"""Dependency-free access controls shared by the local web viewers."""

import base64
import binascii
import hmac
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit


MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff", ".avif": "image/avif",
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
    ".mov": "video/quicktime", ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska", ".mpeg": "video/mpeg", ".mpg": "video/mpeg",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".oga": "audio/ogg", ".flac": "audio/flac",
    ".aac": "audio/aac", ".opus": "audio/ogg",
}
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def is_loopback_host(host):
    name = str(host).strip().lower().rstrip(".")
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name.strip("[]")).is_loopback
    except ValueError:
        return False


def _authority(value):
    if not value or any(character.isspace() for character in value):
        raise ValueError("Invalid host")
    parsed = urlsplit("//" + value)
    if (parsed.username is not None or parsed.password is not None or
            parsed.path or parsed.query or parsed.fragment or not parsed.hostname):
        raise ValueError("Invalid host")
    return parsed.hostname.lower().rstrip("."), parsed.port


class WebAccessPolicy:
    """Allow same-origin browsing; require a token for non-loopback binding.

    Browsers can use HTTP Basic authentication with username ``videoxagent``
    and the token as password. API clients may use a Bearer header instead.
    Tokens are never accepted through URLs or written to logs.
    """

    challenge = 'Basic realm="VideoXAgent", charset="UTF-8"'

    def __init__(self, host="127.0.0.1", token=None):
        self.host = str(host).strip()
        self.token = os.environ.get("VIDEOX_WEB_TOKEN", "") if token is None else token
        if not isinstance(self.token, str):
            raise ValueError("VIDEOX_WEB_TOKEN must be text")
        self.loopback = is_loopback_host(self.host)
        if not self.loopback and not self.token:
            raise ValueError("Non-loopback binding requires VIDEOX_WEB_TOKEN")

    def _authenticated(self, authorization):
        if not isinstance(authorization, str):
            return False
        scheme, separator, credentials = authorization.partition(" ")
        if not separator:
            return False
        candidate = None
        if scheme.lower() == "bearer":
            candidate = credentials
        elif scheme.lower() == "basic":
            try:
                decoded = base64.b64decode(credentials, validate=True).decode("utf-8")
                username, separator, password = decoded.partition(":")
                if separator and username == "videoxagent":
                    candidate = password
            except (ValueError, UnicodeError, binascii.Error):
                return False
        return candidate is not None and hmac.compare_digest(
            candidate.encode("utf-8"), self.token.encode("utf-8")
        )

    def check_request(self, host_header, origin=None, authorization=None):
        try:
            host, port = _authority(host_header)
        except (ValueError, TypeError):
            return 400, "Invalid Host header"
        if self.loopback and not is_loopback_host(host):
            return 403, "Host is not allowed"
        if origin is not None:
            try:
                source = urlsplit(origin)
                if (source.scheme not in ("http", "https") or
                        source.path not in ("", "/") or source.query or source.fragment):
                    raise ValueError("Invalid origin")
                origin_host, origin_port = _authority(source.netloc)
                default_port = 443 if source.scheme == "https" else 80
                request_port = port if port is not None else default_port
                source_port = origin_port if origin_port is not None else default_port
                if (host, request_port) != (origin_host, source_port):
                    return 403, "Cross-origin requests are not allowed"
            except (ValueError, TypeError):
                return 403, "Origin is not allowed"
        if self.token and not self._authenticated(authorization):
            return 401, "Authentication required"
        return None


def is_sensitive_path(path):
    """Reject private path components, preferably relative to an approved root."""
    forbidden = {"key.yaml", "key.yml", "auth.json", "credentials", "credentials.json",
                 "credentials.yaml", "secrets", "secrets.json", "secrets.yaml"}
    return any(part.startswith(".") or part.lower() in forbidden
               for part in Path(path).parts if part not in ("", "/"))


def safe_media_path(raw, roots, *, base=None):
    """Resolve an existing media file inside an explicitly approved directory."""
    if not isinstance(raw, (str, os.PathLike)) or "\x00" in str(raw):
        raise PermissionError("Media path is not allowed")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        if base is None:
            raise PermissionError("A media base directory is required")
        path = Path(base) / path
    try:
        lexical = Path(os.path.abspath(path))
        resolved = path.resolve()
        allowed = False
        for root in roots:
            declared_root = Path(root).expanduser().absolute()
            resolved_root = declared_root.resolve()
            try:
                relative = resolved.relative_to(resolved_root)
            except ValueError:
                continue
            try:
                lexical_relative = lexical.relative_to(declared_root)
            except ValueError:
                # Resolve aliases of the approved root (e.g. macOS /var),
                # while retaining hidden components beneath that root.
                lexical_relative = lexical
                for ancestor in reversed(lexical.parents):
                    if ancestor.resolve() == resolved_root:
                        lexical_relative = lexical.relative_to(ancestor)
                        break
            if not is_sensitive_path(relative) and not is_sensitive_path(lexical_relative):
                allowed = True
                break
    except (OSError, RuntimeError, ValueError) as error:
        raise PermissionError("Media path is not allowed") from error
    if not allowed or resolved.suffix.lower() not in MEDIA_TYPES:
        raise PermissionError("Media path is not allowed")
    if not resolved.is_file():
        raise FileNotFoundError("Media file not found")
    return resolved
