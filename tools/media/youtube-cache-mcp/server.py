#!/usr/bin/env python3
"""YouTube cache MCP server."""
import asyncio
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")

from _base.paths import PROJECT_ROOT

from _base.server_framework import create_mcp_server, make_error_response, make_success_response, run_server

DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "youtube_cache"


def find_cached_video(video_id: str, cache_dir: Path) -> Path | None:
    for ext in [".mp4", ".webm", ".mkv", ".mov", ".avi"]:
        cache_path = cache_dir / f"{video_id}{ext}"
        if cache_path.exists():
            return cache_path
    return None


def check_cache(video_id: str, cache_dir: str | None = None) -> dict:
    resolved_cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cached_path = find_cached_video(video_id, resolved_cache_dir)

    if cached_path:
        return make_success_response(
            exists=True,
            path=str(cached_path),
            video_id=video_id,
        )

    return make_success_response(
        exists=False,
        video_id=video_id,
        cache_dir=str(resolved_cache_dir),
    )


def get_video(video_id: str, cache_dir: str | None = None) -> dict:
    resolved_cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    cached_path = find_cached_video(video_id, resolved_cache_dir)
    if cached_path:
        return make_success_response(
            cached=True,
            path=str(cached_path),
            video_id=video_id,
        )

    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(resolved_cache_dir / f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best/bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--remote-components", "ejs:github",
    ]

    cookies_path = Path(__file__).parent / "www.youtube.com_cookies.txt"
    if cookies_path.exists():
        cmd.extend(["--cookies", str(cookies_path)])

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        cmd.extend(["--proxy", proxy])

    cmd.append(url)

    env = os.environ.copy()
    deno_install = os.path.expanduser("~/.deno")
    env["DENO_INSTALL"] = deno_install
    env["PATH"] = f"{deno_install}/bin:{env.get('PATH', '')}"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return make_error_response("Download timeout (10 minutes)", {"video_id": video_id})
    except FileNotFoundError:
        return make_error_response(
            "yt-dlp not found. From the repository root, run: bash scripts/install_profiles.sh base",
            {"video_id": video_id},
        )
    except Exception as exc:
        return make_error_response(str(exc), {"video_id": video_id})

    if result.returncode != 0:
        return make_error_response(
            result.stderr or "Download failed",
            {"video_id": video_id, "stdout": result.stdout},
        )

    cached_path = find_cached_video(video_id, resolved_cache_dir)
    if cached_path:
        return make_success_response(
            cached=False,
            downloaded=True,
            path=str(cached_path),
            video_id=video_id,
        )

    return make_error_response(
        "Download completed but file not found",
        {"video_id": video_id, "stdout": result.stdout},
    )


async def handle_get_video(arguments: dict) -> dict:
    return get_video(
        video_id=arguments.get("video_id", ""),
        cache_dir=arguments.get("cache_dir"),
    )


async def handle_check_cache(arguments: dict) -> dict:
    return check_cache(
        video_id=arguments.get("video_id", ""),
        cache_dir=arguments.get("cache_dir"),
    )


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="youtube-cache-mcp",
        tool_defs_source="_tool_defs/basic.yaml",
        handlers={
            "get_video_from_youtube_id": handle_get_video,
            "check_youtube_cache": handle_check_cache,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
