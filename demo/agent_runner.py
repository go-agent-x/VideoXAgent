#!/usr/bin/env python3
"""
Agent runner module for web-based execution.
Provides async streaming of agent events via queue.

Optimizations:
- MCP tools are cached globally to avoid reloading on each request
- Model and config are loaded once per runner instance
"""

import asyncio
import json
import sys
import os
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from demo.event_serialization import make_tool_start_event

logger = logging.getLogger(__name__)


def _extract_text(content) -> str:
    """Extract plain text from a message's content (str or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def serialize_messages(messages) -> list:
    """Reduce LangChain messages to a compact, replayable [{role, content}] list.

    Only human/assistant *text* is kept (tool calls, tool results and thinking are
    dropped) so the transcript can be safely replayed into a fresh agent thread
    without dangling tool-call references.
    """
    out = []
    for m in messages or []:
        if isinstance(m, dict):
            mtype = m.get("type") or m.get("role")
            content = m.get("content")
        else:
            mtype = getattr(m, "type", None)
            content = getattr(m, "content", None)
        text = _extract_text(content)
        if not text.strip():
            continue
        if mtype in ("human", "user"):
            out.append({"role": "user", "content": text})
        elif mtype in ("ai", "assistant"):
            out.append({"role": "assistant", "content": text})
    return out


# ============ Global Cache for MCP Tools ============
# Loading MCP tools starts ~30 subprocess servers, which takes 10-20 seconds.
# Cache them globally so subsequent requests reuse the same tools.
_mcp_cache = {
    "tools": None,
    "session_manager": None,
    "server_info": None,
    "initialized": False,
    "lock": None,  # Will be set to asyncio.Lock()
    "vlm_model_name": None,  # Track which model was used to initialize
    "config_key": None,
}


async def get_mcp_tools(vlm_model_name: Optional[str] = None, vlm_api_url: Optional[str] = None, mcp_json_file: str = ".mcp.profiles.json"):
    """Get cached MCP tools, loading them if necessary.

    Reload MCP sessions when the model, API URL, or config file changes.
    """
    global _mcp_cache

    # Initialize lock if needed (must be done in async context)
    if _mcp_cache["lock"] is None:
        _mcp_cache["lock"] = asyncio.Lock()

    async with _mcp_cache["lock"]:
        config_key = (vlm_model_name, vlm_api_url, mcp_json_file)
        if _mcp_cache["initialized"] and _mcp_cache["config_key"] != config_key:
            logger.info("MCP configuration changed, reloading MCP tools...")
            if _mcp_cache["session_manager"]:
                try:
                    await _mcp_cache["session_manager"].cleanup()
                except Exception as e:
                    logger.warning(f"MCP cleanup during reload: {e}")
            _mcp_cache["initialized"] = False
            _mcp_cache["tools"] = None
            _mcp_cache["session_manager"] = None

        if _mcp_cache["initialized"]:
            return _mcp_cache["tools"], _mcp_cache["session_manager"], _mcp_cache["server_info"]

        import yaml
        from deepagents_cli.mcp_tools import load_mcp_config, _load_tools_from_config

        # Load API keys
        key_config_path = Path(__file__).parent.parent / "key.yaml"
        with open(key_config_path, "r", encoding="utf-8") as f:
            key_config = yaml.safe_load(f)

        if key_config.get("VLM_API_KEY"):
            os.environ["VLM_API_KEY"] = key_config["VLM_API_KEY"]

        # Load MCP config
        mcp_config = load_mcp_config(str(Path(__file__).parent.parent / mcp_json_file))
        inject_env = {
            k: v for k, v in [
                ("VLM_API_KEY", key_config.get("VLM_API_KEY")),
                ("DINOX_API_KEY", key_config.get("DINOX_API_KEY")),
                ("VLM_MODEL_NAME", vlm_model_name),
                ("VLM_API_URL", vlm_api_url),
            ] if v
        }
        for srv in mcp_config["mcpServers"].values():
            env = srv.setdefault("env", {})
            env['http_proxy'] = ''
            env['https_proxy'] = ''
            for k, v in inject_env.items():
                if k in ("VLM_MODEL_NAME", "VLM_API_URL") or k not in env:
                    env[k] = v

        logger.info("Loading MCP tools (this may take 10-20 seconds on first request)...")
        mcp_tools, session_manager, server_info = await _load_tools_from_config(mcp_config)

        # Strip server name prefix from tool names
        seen = {}
        for t in mcp_tools:
            parts = t.name.split("__")
            short = parts[-1] if len(parts) >= 2 else t.name
            if short in seen:
                short = t.name.replace("__", "_")
            seen[short] = True
            t.name = short

        _mcp_cache["tools"] = mcp_tools
        _mcp_cache["session_manager"] = session_manager
        _mcp_cache["server_info"] = server_info
        _mcp_cache["initialized"] = True
        _mcp_cache["vlm_model_name"] = vlm_model_name
        _mcp_cache["config_key"] = config_key

        logger.info(f"MCP tools loaded: {len(mcp_tools)} tools available")
        return mcp_tools, session_manager, server_info


async def cleanup_mcp_cache():
    """Cleanup cached MCP sessions (call on server shutdown)."""
    global _mcp_cache
    if _mcp_cache["session_manager"]:
        try:
            await _mcp_cache["session_manager"].cleanup()
        except Exception as e:
            logger.warning(f"MCP cleanup warning: {e}")
        _mcp_cache["initialized"] = False
        _mcp_cache["tools"] = None
        _mcp_cache["session_manager"] = None


class AgentRunner:
    """Runs video-agent and yields events for streaming."""

    def __init__(self, config_path: str, system_prompt_path: str = "SYSTEM_PROMPT.md", system_prompt_text: Optional[str] = None):
        self.config_path = config_path
        self.system_prompt_path = system_prompt_path
        self.system_prompt_text = system_prompt_text
        self._thinking_started = False
        # Multi-turn conversation state. Populated once by _initialize() and then
        # reused for every follow-up turn so the checkpointer keeps full history.
        self.agent = None
        self.config = None
        self.thread_id = None
        self.recursion_limit_val = 125
        self._initialized = False
        # Resume support: prior messages replayed on the first turn (used when a
        # runner is rebuilt after a server restart), and the latest full message
        # list captured after each turn (for persistence).
        self.seed_messages = None
        self._seed_consumed = False
        self.last_messages = []

    async def _initialize(self) -> AsyncGenerator[dict, None]:
        """One-time setup shared across all turns of a conversation.

        Loads the model, system prompt and MCP tools, then builds the agent with a
        persistent checkpointer + thread_id. Reusing the same thread_id for every
        turn lets LangGraph's checkpointer retain the full conversation history,
        which is what enables multi-turn dialogue (follow-ups, hints, steering).
        """
        import yaml
        from langchain_anthropic import ChatAnthropic
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Import from parent directory
        from utils.chat_openai_with_reasoning import ChatOpenAIWithReasoning
        from deepagents.backends import FilesystemBackend
        from create_deep_agent import create_deep_agent
        from langgraph.checkpoint.memory import MemorySaver

        # Load API keys
        key_config_path = Path(__file__).parent.parent / "key.yaml"
        with open(key_config_path, "r", encoding="utf-8") as f:
            key_config = yaml.safe_load(f)

        if key_config.get("LLM_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = key_config["LLM_API_KEY"]
            os.environ["OPENAI_API_KEY"] = key_config["LLM_API_KEY"]
            os.environ["GOOGLE_API_KEY"] = key_config["LLM_API_KEY"]

        if not os.environ.get("GOOGLE_API_KEY") and key_config.get("VLM_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = key_config["VLM_API_KEY"]

        # Load model config
        model_config_path = Path(__file__).parent.parent / self.config_path
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)

        # Create LLM
        model_cfg = dict(model_config["model"])
        provider = model_cfg.pop("provider")

        if provider == "ChatOpenAI":
            # Match the Google branch's key.yaml fallback. Passing the key
            # explicitly also avoids stale process-level OPENAI_API_KEY values.
            configured_key = key_config.get("LLM_API_KEY") or key_config.get("VLM_API_KEY")
            if configured_key:
                model_cfg.setdefault("api_key", configured_key)
            model = ChatOpenAIWithReasoning(**model_cfg)
        elif provider == "ChatAnthropic":
            model = ChatAnthropic(**model_cfg)
        elif provider == "ChatGoogleGenerativeAI":
            model = ChatGoogleGenerativeAI(**model_cfg)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        yield {"type": "status", "data": f"Loaded model: {model_cfg.get('model', 'unknown')}"}

        # Load system prompt
        if self.system_prompt_text:
            # Use directly provided text
            system_prompt = self.system_prompt_text
            yield {"type": "status", "data": f"Using custom system prompt ({len(system_prompt)} chars)"}
            preview = system_prompt[:2000] + "..." if len(system_prompt) > 2000 else system_prompt
            yield {"type": "system_prompt", "data": preview}
        else:
            # Load from file
            prompt_path = Path(__file__).parent.parent / self.system_prompt_path
            if prompt_path.exists():
                system_prompt = prompt_path.read_text(encoding="utf-8")
                yield {"type": "status", "data": f"Loaded system prompt: {self.system_prompt_path} ({len(system_prompt)} chars)"}
                preview = system_prompt[:2000] + "..." if len(system_prompt) > 2000 else system_prompt
                yield {"type": "system_prompt", "data": preview}
            else:
                system_prompt = None
                yield {"type": "status", "data": f"System prompt not found: {self.system_prompt_path}"}

        # Get cached MCP tools (fast if already loaded)
        yield {"type": "status", "data": "Loading MCP tools..."}
        vlm_model_name = model_config.get("vlm_model_name")
        vlm_api_url = model_config.get("VLM_API_URL")
        mcp_json_file = model_config.get("mcp_json_file", ".mcp.profiles.json")
        mcp_tools, session_manager, server_info = await get_mcp_tools(vlm_model_name=vlm_model_name, vlm_api_url=vlm_api_url, mcp_json_file=mcp_json_file)

        tool_names = [t.name for t in mcp_tools]
        yield {"type": "tools_loaded", "data": tool_names}

        # Get disabled builtin tools from config
        disabled_builtin_tools = model_config.get("disabled_builtin_tools", [])

        # Read recursion_limit from config (default 125)
        recursion_limit_val = model_config.get("recursion_limit", 125)
        # Whether to enable budget warning mechanism (default True)
        _enable_budget_warning = model_config.get("enable_budget_warning", True)

        # Create agent with checkpointer
        checkpointer = MemorySaver()
        thread_id = str(uuid.uuid4())

        agent = create_deep_agent(
            model=model,
            tools=mcp_tools,
            system_prompt=system_prompt,
            subagents=[],
            backend=FilesystemBackend(virtual_mode=False),
            disabled_builtin_tools=disabled_builtin_tools,
            recursion_limit=recursion_limit_val,
            enable_budget_warning=_enable_budget_warning,
            checkpointer=checkpointer,
        )

        self.agent = agent
        self.thread_id = thread_id
        self.recursion_limit_val = recursion_limit_val
        self.config = {
            "recursion_limit": recursion_limit_val,
            "configurable": {"thread_id": thread_id},
        }
        self._initialized = True

        yield {"type": "status", "data": "Agent created, starting execution..."}

    async def run_agent(
        self,
        prompt: str,
        specified_tools: Optional[list[str]] = None
    ) -> AsyncGenerator[dict, None]:
        """Run one conversation turn and stream events.

        The first call triggers one-time initialization. Subsequent calls reuse the
        same agent and thread_id, so previous turns stay in context and the user can
        intervene, add hints, or steer the agent across turns.

        Args:
            prompt: The user prompt for this turn
            specified_tools: Optional list of tools to require (prompt injection)

        Yields:
            dict: Event objects with type and data
        """
        if not self._initialized:
            async for ev in self._initialize():
                yield ev

        agent = self.agent
        config = self.config

        # Prepare input for this turn
        if specified_tools:
            tool_instruction = (
                "\n\n[IMPORTANT] You MUST use the following tools to complete the task: "
                + ", ".join(specified_tools)
                + ". Prioritize these tools over others."
            )
            prompt = prompt + tool_instruction

        # On the first turn of a rebuilt runner, replay the prior transcript so the
        # model regains full context (enables continuing a chat after a restart).
        turn_messages = []
        if self.seed_messages and not self._seed_consumed:
            turn_messages.extend(self.seed_messages)
            self._seed_consumed = True
        turn_messages.append({"role": "user", "content": prompt})
        input_data = {"messages": turn_messages}

        # Accumulated usage
        accumulated_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

        result = None
        start_time = datetime.now()

        try:
            async for event in agent.astream_events(input_data, config, version="v2"):
                kind = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                # Stream token output
                if kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", "")
                        additional_kwargs = getattr(chunk, 'additional_kwargs', {}) or {}
                        reasoning_content = additional_kwargs.get('reasoning_content', '')

                        if reasoning_content:
                            yield {"type": "thinking", "data": reasoning_content}

                        if isinstance(content, str) and content:
                            yield {"type": "token", "data": content}
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict):
                                    item_type = item.get("type", "")
                                    if item_type in ("thinking", "reasoning", "reasoning_content", "thought"):
                                        thinking_text = (
                                            item.get("thinking") or
                                            item.get("reasoning_content") or
                                            item.get("reasoning") or
                                            item.get("thought") or
                                            item.get("content") or
                                            item.get("text") or
                                            ""
                                        )
                                        if thinking_text:
                                            yield {"type": "thinking", "data": thinking_text}
                                    elif item.get("type") == "text":
                                        text = item.get("text", "")
                                        if text:
                                            yield {"type": "token", "data": text}
                                    elif item.get("type") == "tool_use":
                                        tool_name = item.get("name", "unknown")
                                        yield {"type": "tool_call_start", "data": {"name": tool_name}}

                # Tool execution events
                elif kind == "on_tool_start":
                    yield make_tool_start_event(name, data.get("input", {}))

                elif kind == "on_tool_end":
                    output = data.get("output", "")
                    output_str = str(output)
                    output_display = output_str[:800] + "..." if len(output_str) > 800 else output_str
                    yield {
                        "type": "tool_end",
                        "data": {"name": name, "output": output_display, "output_raw": output_str}
                    }

                elif kind == "on_tool_error":
                    error = data.get("error", "")
                    yield {
                        "type": "tool_error",
                        "data": {"name": name, "error": str(error)}
                    }

                # Capture token usage
                if kind == "on_chat_model_end":
                    output = data.get("output")
                    if output:
                        usage_meta = getattr(output, 'usage_metadata', None) or {}
                        if usage_meta:
                            accumulated_usage["input_tokens"] += usage_meta.get('input_tokens', 0)
                            accumulated_usage["output_tokens"] += usage_meta.get('output_tokens', 0)
                            input_details = usage_meta.get('input_token_details', {}) or {}
                            accumulated_usage["cache_read_input_tokens"] += input_details.get('cache_read', 0)
                            accumulated_usage["cache_creation_input_tokens"] += input_details.get('cache_creation', 0)

                # Capture final result
                if kind == "on_chain_end" and name == "LangGraph":
                    result = data.get("output", {"messages": []})

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Remember the full message list so the server can persist it for resume.
            if result and isinstance(result, dict):
                self.last_messages = result.get("messages", []) or []

            yield {
                "type": "complete",
                "data": {
                    "duration": duration,
                    "usage": accumulated_usage,
                    "status": "success"
                }
            }

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            yield {
                "type": "error",
                "data": {"error": str(e), "traceback": error_detail}
            }

        # Note: Do NOT cleanup MCP sessions here - they are cached globally


async def test_runner():
    """Test the agent runner."""
    runner = AgentRunner(
        config_path="exp_configs/validation/videomme_v2_gemini.yaml",
        system_prompt_path="SYSTEM_PROMPT.md"
    )

    prompt = "What is 2+2? Just answer the number."

    async for event in runner.run_agent(prompt):
        print(f"[{event['type']}] {event.get('data', '')}")


if __name__ == "__main__":
    asyncio.run(test_runner())
