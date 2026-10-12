#!/usr/bin/env python3
"""
WebSocket server for agent execution with real-time streaming.
Provides both REST API and WebSocket endpoints.

Usage:
    python agent_server.py [port] [--host HOST] [--media-root PATH]
    # Default port: 8080
    # Then visit: http://localhost:8080/agent_runner.html
"""

import argparse
import asyncio
import json
import os
import sys
import logging
from pathlib import Path
import re
import time
import uuid
import yaml

# aiohttp for async WebSocket support
try:
    from aiohttp import web
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. From the repository root, run: bash scripts/install_profiles.sh base")
    sys.exit(1)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add demo directory for agent_runner import
sys.path.insert(0, str(Path(__file__).parent))

from utils.web_security import MEDIA_TYPES, SECURITY_HEADERS, WebAccessPolicy, safe_media_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
REPO_ROOT = Path(__file__).parent.parent.resolve()
EXP_CONFIGS_DIR = REPO_ROOT / "exp_configs"
DEFAULT_EXP_CONFIG = os.environ.get(
    "AGENT_EXP_CONFIG", "exp_configs/validation/videomme_v2_gemini.yaml"
)
DEMO_DIR = Path(__file__).parent
UPLOADS_DIR = REPO_ROOT / "uploads"
MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024

# Running agents tracker
running_agents = {}
ACCESS_POLICY_KEY = web.AppKey("access_policy", WebAccessPolicy)
MEDIA_ROOTS_KEY = web.AppKey("media_roots", list)


def is_safe_default_media_root(path):
    """Default folders must stay in the repository; external roots are opt-in."""
    try:
        return not path.is_symlink() and path.resolve().is_relative_to(REPO_ROOT.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def default_media_roots():
    # Check on every request as a directory can become a symlink after startup.
    return [root for root in (REPO_ROOT / "output", UPLOADS_DIR, REPO_ROOT / "data")
            if is_safe_default_media_root(root)]


class ExactFileResponse(web.FileResponse):
    """Serve the validated path, without negotiating unchecked .gz/.br siblings."""

    async def prepare(self, request):
        headers = request.headers.copy()
        headers.popall("Accept-Encoding", None)
        return await super().prepare(request.clone(headers=headers))


def resolve_experiment_config(config_path):
    """Only allow repository experiment configs or the startup-selected config."""
    if not isinstance(config_path, str) or not config_path:
        raise ValueError("Invalid experiment config")
    candidate = REPO_ROOT / config_path
    resolved = candidate.resolve()
    config_root = EXP_CONFIGS_DIR.resolve()
    default = (REPO_ROOT / DEFAULT_EXP_CONFIG).resolve()
    if candidate.suffix.lower() not in {".yaml", ".yml"} or not resolved.is_file():
        raise ValueError("Invalid experiment config")
    if resolved != default:
        try:
            relative = resolved.relative_to(config_root)
            lexical = candidate.absolute().relative_to(EXP_CONFIGS_DIR.absolute())
        except ValueError as exc:
            raise ValueError("Invalid experiment config") from exc
        if any(part.startswith(".") for part in (*relative.parts, *lexical.parts)):
            raise ValueError("Invalid experiment config")
    return resolved


def resolve_system_prompt(prompt_path):
    """Constrain file-based prompts; arbitrary custom prompts use text input."""
    if not isinstance(prompt_path, str) or not prompt_path:
        raise ValueError("Invalid system prompt")
    candidate = REPO_ROOT / prompt_path
    resolved = candidate.resolve()
    root = REPO_ROOT.resolve()
    try:
        relative = resolved.relative_to(root)
        lexical = candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise ValueError("Invalid system prompt") from exc
    if any(part.startswith(".") for part in (*relative.parts, *lexical.parts)):
        raise ValueError("Invalid system prompt")
    if not resolved.is_file() or resolved.suffix.lower() != ".md":
        raise ValueError("Invalid system prompt")
    if not (
        len(relative.parts) == 1 and relative.name.startswith("SYSTEM_PROMPT")
        or relative.parts[0] == "prompts"
    ):
        raise ValueError("Invalid system prompt")
    return resolved


@web.middleware
async def access_middleware(request, handler):
    """Protect REST, static files and WebSocket handshakes alike."""
    policy = request.app[ACCESS_POLICY_KEY]
    failure = policy.check_request(
        request.headers.get("Host", ""),
        request.headers.get("Origin"),
        request.headers.get("Authorization"),
    )
    if failure:
        status, error = failure
        headers = {"WWW-Authenticate": policy.challenge} if status == 401 else {}
        return web.json_response({"error": error}, status=status, headers=headers)
    return await handler(request)


async def security_headers(request, response):
    response.headers.update(SECURITY_HEADERS)
    response.headers["X-Frame-Options"] = "DENY"


async def index_handler(request):
    """Redirect to agent_runner.html"""
    raise web.HTTPFound('/agent_runner.html')


async def list_configs_handler(request):
    """List available experiment config files."""
    configs = []

    # Scan exp_configs directory recursively
    if EXP_CONFIGS_DIR.exists():
        for yaml_file in EXP_CONFIGS_DIR.rglob("*.yaml"):
            try:
                resolve_experiment_config(str(yaml_file))
                rel_path = yaml_file.relative_to(REPO_ROOT)
            except (OSError, ValueError):
                continue
            configs.append({
                "path": str(rel_path),
                "name": yaml_file.stem,
                "mtime": yaml_file.stat().st_mtime
            })

    # Include explicitly selected configs outside exp_configs as well.
    default_path = REPO_ROOT / DEFAULT_EXP_CONFIG
    if default_path.is_file() and not any(c["path"] == DEFAULT_EXP_CONFIG for c in configs):
        configs.append({
            "path": DEFAULT_EXP_CONFIG,
            "name": default_path.stem,
            "mtime": default_path.stat().st_mtime,
        })

    # Sort by path
    configs.sort(key=lambda x: x["path"])

    return web.json_response({"configs": configs, "default_config": DEFAULT_EXP_CONFIG})


async def list_tools_handler(request):
    """List MCP servers from the selected experiment's configuration."""
    config_path = request.query.get("config") or DEFAULT_EXP_CONFIG
    try:
        with resolve_experiment_config(config_path).open(encoding="utf-8") as f:
            exp_config = yaml.safe_load(f)
        if not isinstance(exp_config, dict):
            raise ValueError("Experiment config must be a mapping")
        mcp_file = exp_config.get("mcp_json_file", ".mcp.profiles.json")
        if not isinstance(mcp_file, str) or not mcp_file.strip():
            raise ValueError("mcp_json_file must be a non-empty string")
        mcp_config_path = REPO_ROOT / mcp_file
        with open(mcp_config_path, "r", encoding="utf-8") as f:
            mcp_config = json.load(f)
        tools = list(mcp_config.get("mcpServers", {}))
    except (OSError, ValueError, yaml.YAMLError, AttributeError, TypeError) as e:
        logger.warning("Failed to list tools for %s: %s", config_path, e)
        return web.json_response({"error": "Unable to load tools from experiment config"}, status=400)
    return web.json_response({"tools": sorted(tools)})


def make_safe_upload_filename(filename: str) -> str:
    """Create a filesystem-safe upload filename with a timestamp prefix."""
    basename = Path(filename).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    if not safe_name:
        safe_name = "video"
    return f"{int(time.time() * 1000)}_{uuid.uuid4().hex}_{safe_name}"


async def upload_handler(request):
    """Handle video file upload from browser."""
    reader = await request.multipart()
    field = await reader.next()

    if field is None or field.name != "video":
        return web.json_response({"error": "Expected multipart field named 'video'"}, status=400)

    if not field.filename:
        return web.json_response({"error": "No upload filename provided"}, status=400)

    suffix = Path(field.filename).suffix.lower()
    if suffix not in MEDIA_TYPES or not MEDIA_TYPES[suffix].startswith("video/"):
        return web.json_response({"error": "Unsupported video file type"}, status=400)

    if not is_safe_default_media_root(UPLOADS_DIR):
        return web.json_response({"error": "Upload directory is unavailable"}, status=503)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_filename = make_safe_upload_filename(field.filename)
    filepath = UPLOADS_DIR / safe_filename
    size = 0

    try:
        with open(filepath, "xb") as output_file:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    filepath.unlink(missing_ok=True)
                    return web.json_response({"error": "File too large (max 2GB)"}, status=413)
                output_file.write(chunk)
    except Exception as e:
        filepath.unlink(missing_ok=True)
        logger.error(f"Failed to save uploaded file: {e}")
        return web.json_response({"error": "Failed to save uploaded file"}, status=500)

    resolved = filepath.resolve()
    logger.info(f"Uploaded video saved: {resolved} ({size} bytes)")
    return web.json_response({
        "path": str(resolved),
        "filename": field.filename,
        "saved_filename": safe_filename,
        "size": size,
    })


async def safe_send_json(ws, data: dict) -> bool:
    """Safely send JSON to websocket, returns False if failed."""
    try:
        if ws.closed:
            return False
        await ws.send_json(data)
        return True
    except Exception as e:
        logger.debug(f"Failed to send websocket message: {e}")
        return False


async def run_agent_turn(ws, agent_id: str, runner, prompt: str, specified_tools: list):
    """Run a single conversation turn for an existing runner.

    The runner is created once per conversation and reused for every turn, so its
    checkpointer keeps the full history and follow-up prompts build on prior turns.
    After the turn completes the full transcript is persisted so the conversation
    can be resumed (context replayed) even after a server restart.
    """
    try:
        async for event in runner.run_agent(prompt, specified_tools):
            if ws.closed:
                logger.info(f"WebSocket closed, stopping agent {agent_id}")
                break
            if not await safe_send_json(ws, event):
                break

        persist_agent_transcript(agent_id, runner)

    except asyncio.CancelledError:
        # Task was cancelled by stop request
        logger.info(f"Agent {agent_id} was cancelled")
        await safe_send_json(ws, {"type": "stopped", "data": {"agent_id": agent_id, "reason": "User requested stop"}})
        raise  # Re-raise to properly handle cancellation

    except Exception as e:
        import traceback
        logger.error(f"Agent {agent_id} error: {e}")
        await safe_send_json(ws, {
            "type": "error",
            "data": {"error": str(e), "traceback": traceback.format_exc()}
        })

    finally:
        # Turn finished: mark idle so a follow-up turn can be started.
        if agent_id in running_agents:
            running_agents[agent_id]["status"] = "idle"


def persist_agent_transcript(agent_id: str, runner):
    """Save the runner's message history + build config for resume-after-restart."""
    try:
        import conversation_store
        from agent_runner import serialize_messages
        meta = running_agents.get(agent_id, {}).get("meta", {})
        conversation_store.save_agent_transcript(agent_id, {
            "config_path": meta.get("config_path"),
            "system_prompt_path": meta.get("system_prompt_path"),
            "system_prompt_text": meta.get("system_prompt_text"),
            "messages": serialize_messages(getattr(runner, "last_messages", [])),
        })
    except Exception as e:
        logger.warning(f"Failed to persist agent transcript {agent_id}: {e}")


def rebuild_runner_from_store(agent_id: str):
    """Rebuild a seeded runner from a persisted transcript (after a restart).

    Returns (runner, meta) or (None, None) if no transcript is stored.
    """
    try:
        import conversation_store
        from agent_runner import AgentRunner
        rec = conversation_store.load_agent_transcript(agent_id)
        if not rec:
            return None, None
        meta = {
            "config_path": rec.get("config_path") or DEFAULT_EXP_CONFIG,
            "system_prompt_path": rec.get("system_prompt_path") or "SYSTEM_PROMPT.md",
            "system_prompt_text": rec.get("system_prompt_text"),
        }
        meta["config_path"] = str(resolve_experiment_config(meta["config_path"]))
        if not meta["system_prompt_text"]:
            meta["system_prompt_path"] = str(resolve_system_prompt(meta["system_prompt_path"]))
        runner = AgentRunner(
            config_path=meta["config_path"],
            system_prompt_path=meta["system_prompt_path"],
            system_prompt_text=meta["system_prompt_text"],
        )
        runner.seed_messages = rec.get("messages", [])
        return runner, meta
    except Exception as e:
        logger.warning(f"Failed to rebuild runner {agent_id}: {e}")
        return None, None


async def websocket_handler(request):
    """WebSocket handler for agent execution with streaming."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    agent_id = None
    agent_task = None

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    action = data.get("action")

                    if action == "start":
                        # Cancel any existing task first
                        if agent_task and not agent_task.done():
                            agent_task.cancel()
                            try:
                                await asyncio.wait_for(agent_task, timeout=2.0)
                            except (asyncio.CancelledError, asyncio.TimeoutError):
                                pass

                        # Start a fresh conversation
                        prompt = data.get("prompt", "")
                        config_path = data.get("config") or DEFAULT_EXP_CONFIG
                        system_prompt_path = data.get("system_prompt", "SYSTEM_PROMPT.md")
                        system_prompt_text = data.get("system_prompt_text", None)  # Direct text input
                        specified_tools = data.get("tools", [])

                        if not prompt:
                            await ws.send_json({"type": "error", "data": {"error": "Prompt is required"}})
                            continue

                        try:
                            config_path = str(resolve_experiment_config(config_path))
                            if not system_prompt_text:
                                system_prompt_path = str(resolve_system_prompt(system_prompt_path))
                        except (OSError, ValueError):
                            await ws.send_json({"type": "error", "data": {"error": "Invalid experiment config or system prompt"}})
                            continue

                        # Create a runner that persists for the whole conversation,
                        # so follow-up turns reuse its history via the checkpointer.
                        from agent_runner import AgentRunner
                        runner = AgentRunner(
                            config_path=config_path,
                            system_prompt_path=system_prompt_path,
                            system_prompt_text=system_prompt_text,
                        )

                        # Generate agent ID
                        agent_id = str(uuid.uuid4())[:8]
                        running_agents[agent_id] = {
                            "status": "running", "ws": ws, "task": None, "runner": runner,
                            "meta": {
                                "config_path": config_path,
                                "system_prompt_path": system_prompt_path,
                                "system_prompt_text": system_prompt_text,
                            },
                        }

                        await ws.send_json({
                            "type": "started",
                            "data": {"agent_id": agent_id, "config": config_path}
                        })

                        # Create agent task (runs in background, doesn't block message receiving)
                        agent_task = asyncio.create_task(
                            run_agent_turn(ws, agent_id, runner, prompt, specified_tools)
                        )
                        running_agents[agent_id]["task"] = agent_task

                        # Don't await here - let the task run in background
                        # This allows us to receive stop messages while agent is running

                    elif action == "continue":
                        # Follow-up turn in an existing conversation (steering / hints).
                        prompt = data.get("prompt", "")
                        cont_id = data.get("agent_id") or agent_id
                        specified_tools = data.get("tools", [])

                        if not prompt:
                            await ws.send_json({"type": "error", "data": {"error": "Prompt is required"}})
                            continue

                        if not cont_id:
                            await ws.send_json({"type": "error", "data": {"error": "No active conversation. Start one first."}})
                            continue

                        entry = running_agents.get(cont_id)

                        # After a server restart the in-memory runner is gone; rebuild
                        # it from the persisted transcript so context is replayed.
                        if entry is None or entry.get("runner") is None:
                            runner, meta = rebuild_runner_from_store(cont_id)
                            if runner is None:
                                await ws.send_json({"type": "error", "data": {"error": "This conversation can no longer be continued (no saved transcript)."}})
                                continue
                            entry = {"status": "running", "ws": ws, "task": None, "runner": runner, "meta": meta}
                            running_agents[cont_id] = entry
                        else:
                            prev_task = entry.get("task")
                            if prev_task and not prev_task.done():
                                await ws.send_json({"type": "error", "data": {"error": "Agent is still running. Please wait or stop it first."}})
                                continue

                        runner = entry.get("runner")
                        agent_id = cont_id
                        entry["ws"] = ws
                        entry["status"] = "running"

                        await ws.send_json({
                            "type": "turn_started",
                            "data": {"agent_id": cont_id}
                        })

                        agent_task = asyncio.create_task(
                            run_agent_turn(ws, cont_id, runner, prompt, specified_tools)
                        )
                        entry["task"] = agent_task

                    elif action == "stop":
                        # Stop running agent by cancelling its task
                        stop_id = data.get("agent_id")
                        if stop_id in running_agents:
                            task = running_agents[stop_id].get("task")
                            if task and not task.done():
                                logger.info(f"Cancelling agent {stop_id}...")
                                task.cancel()
                                # Don't wait here - let the task handle its own cleanup
                            running_agents[stop_id]["status"] = "stopped"
                        else:
                            await ws.send_json({"type": "error", "data": {"error": f"Agent {stop_id} not found"}})

                    elif action == "ping":
                        await ws.send_json({"type": "pong"})

                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "data": {"error": "Invalid JSON"}})

            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")

    finally:
        # Cancel any running task when websocket closes
        if agent_task and not agent_task.done():
            agent_task.cancel()
            try:
                await asyncio.wait_for(agent_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if agent_id and agent_id in running_agents:
            del running_agents[agent_id]

    return ws


async def file_handler(request):
    """Serve only inert media within explicitly configured media directories."""
    file_path = request.query.get("path", "")
    if not file_path:
        raise web.HTTPBadRequest(text="Missing 'path' parameter")

    try:
        roots = default_media_roots() + request.app[MEDIA_ROOTS_KEY]
        resolved = safe_media_path(file_path, roots, base=REPO_ROOT)
    except (OSError, ValueError):
        raise web.HTTPNotFound(text="Media not found")
    return ExactFileResponse(resolved, headers={"Content-Type": MEDIA_TYPES[resolved.suffix.lower()]})


async def static_handler(request):
    """Publish the UI document only, never server sources or directory listings."""
    filename = request.match_info.get('filename', '')
    if filename != "agent_runner.html":
        raise web.HTTPNotFound()
    filepath = DEMO_DIR / filename
    if filepath.is_symlink() or not filepath.is_file():
        raise web.HTTPNotFound()
    return ExactFileResponse(filepath, headers={"Content-Type": "text/html; charset=utf-8"})


async def list_conversations_handler(request):
    """Return saved demo conversations (metadata only)."""
    import conversation_store
    return web.json_response({"conversations": conversation_store.list_conversations()})


async def get_conversation_handler(request):
    """Return a single saved conversation, including its rendered transcript."""
    import conversation_store
    conv_id = request.match_info.get("conv_id", "")
    try:
        record = conversation_store.get_conversation(conv_id)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    if record is None:
        return web.json_response({"error": "Conversation not found"}, status=404)
    return web.json_response(record)


async def put_conversation_handler(request):
    """Create or update a saved conversation."""
    import conversation_store
    conv_id = request.match_info.get("conv_id", "")
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    try:
        record = conversation_store.save_conversation(conv_id, payload or {})
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response(record)


async def delete_conversation_handler(request):
    """Delete a saved conversation."""
    import conversation_store
    conv_id = request.match_info.get("conv_id", "")
    try:
        existed = conversation_store.delete_conversation(conv_id)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response({"deleted": existed})


def create_app(host="127.0.0.1", token=None, media_roots=None):
    """Create the aiohttp application."""
    policy = WebAccessPolicy(host, token)
    app = web.Application(client_max_size=MAX_UPLOAD_SIZE + 1024 * 1024, middlewares=[access_middleware])
    app[ACCESS_POLICY_KEY] = policy
    app[MEDIA_ROOTS_KEY] = [Path(root).expanduser().resolve() for root in (media_roots or [])]
    app.on_response_prepare.append(security_headers)

    # API routes
    app.router.add_get('/api/configs', list_configs_handler)
    app.router.add_get('/api/tools', list_tools_handler)
    app.router.add_get('/api/file', file_handler)
    app.router.add_post('/api/upload', upload_handler)
    app.router.add_get('/api/conversations', list_conversations_handler)
    app.router.add_get('/api/conversations/{conv_id}', get_conversation_handler)
    app.router.add_put('/api/conversations/{conv_id}', put_conversation_handler)
    app.router.add_delete('/api/conversations/{conv_id}', delete_conversation_handler)
    app.router.add_get('/ws', websocket_handler)

    # Static files and index
    app.router.add_get('/', index_handler)
    app.router.add_get('/{filename:.*}', static_handler)

    # Register startup/cleanup handlers for MCP tools cache
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


async def on_startup(app):
    """Start MCP tools preloading in background (non-blocking)."""
    logger.info("Server started. MCP tools will be loaded on first request or in background...")
    # Start preloading in background task (non-blocking)
    asyncio.create_task(preload_mcp_tools_background())


async def preload_mcp_tools_background():
    """Background task to preload MCP tools."""
    try:
        logger.info("Background: Preloading MCP tools (this may take 10-20 seconds)...")
        from agent_runner import get_mcp_tools
        tools, _, _ = await get_mcp_tools()
        logger.info(f"Background: MCP tools preloaded: {len(tools)} tools ready")
    except Exception as e:
        logger.warning(f"Background: Failed to preload MCP tools: {e}")
        logger.warning("Tools will be loaded on first request instead")


async def on_cleanup(app):
    """Cleanup MCP sessions on server shutdown."""
    logger.info("Cleaning up MCP sessions...")
    try:
        from agent_runner import cleanup_mcp_cache
        await cleanup_mcp_cache()
        logger.info("MCP sessions cleaned up")
    except Exception as e:
        logger.warning(f"MCP cleanup warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run the VideoXAgent Demo UI")
    parser.add_argument("port", type=int, nargs="?", default=8080)
    parser.add_argument("--host", default=os.environ.get("VIDEOX_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--media-root", action="append", default=[], metavar="PATH", help="Additional trusted media directory (repeatable)")
    args = parser.parse_args()
    env_roots = [root for root in os.environ.get("VIDEOX_MEDIA_ROOTS", "").split(os.pathsep) if root]
    try:
        app = create_app(args.host, os.environ.get("VIDEOX_WEB_TOKEN"), args.media_root + env_roots)
    except ValueError as exc:
        parser.error(str(exc))
    host_label = f"[{args.host}]" if ":" in args.host else args.host
    server_url = f"http://{host_label}:{args.port}"

    print(f"=" * 60)
    print(f"Agent Runner Server")
    print(f"=" * 60)
    print(f"Server URL: {server_url}")
    print(f"Agent Runner UI: {server_url}/agent_runner.html")
    if os.environ.get("VIDEOX_WEB_TOKEN"):
        print("Authentication enabled (browser username: videoxagent).")
    print(f"")
    print(f"API endpoints:")
    print(f"  GET  /api/configs  - List available config files")
    print(f"  GET  /api/tools    - List available MCP tools")
    print(f"  GET  /api/file     - Serve local image/video files")
    print(f"  POST /api/upload   - Upload browser video files (max 2GB)")
    print(f"  WS   /ws           - WebSocket for agent execution")
    print(f"")
    print(f"Press Ctrl+C to stop")
    print(f"=" * 60)

    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
