#!/usr/bin/env python3
"""
Persistent batch worker for video-agent.

Loads MCP tools ONCE at startup and processes multiple prompts sequentially,
avoiding the overhead of restarting MCP servers for each sample.

Compared to the per-sample subprocess approach (python video-agent.py per prompt),
this saves ~2min of MCP loading time per sample after the first one.

Usage:
    python batch_worker.py -c config.yaml -sys SYSTEM_PROMPT.md --task-file /tmp/worker_0_tasks.jsonl [--gpu-id 0]

Task file format (one JSON per line):
    {"idx": 1, "prompt": "Given a youtube video..."}
    {"idx": 3, "prompt": "What is the speaker..."}

Worker writes results to the same conversations/ directory as video-agent.py
(multi-process safe via file locking). Exit code 0 = all tasks completed,
non-zero = at least one fatal error during setup.
"""

import asyncio
import json
import sys
import os
import uuid
import fcntl
import signal
import atexit
import logging
import argparse
import ast
import random
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.chat_openai_with_reasoning import ChatOpenAIWithReasoning
from deepagents_cli.mcp_tools import load_mcp_config, _load_tools_from_config
from deepagents.backends import FilesystemBackend
from create_deep_agent import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import yaml

# API error types for retry
try:
    from anthropic import RateLimitError as AnthropicRateLimitError
    from anthropic import APIConnectionError as AnthropicAPIConnectionError
    from anthropic import APIStatusError as AnthropicAPIStatusError
    from anthropic import InternalServerError as AnthropicInternalServerError
except ImportError:
    AnthropicRateLimitError = Exception
    AnthropicAPIConnectionError = Exception
    AnthropicAPIStatusError = Exception
    AnthropicInternalServerError = Exception

try:
    from openai import RateLimitError as OpenAIRateLimitError
    from openai import APIConnectionError as OpenAIAPIConnectionError
    from openai import APITimeoutError as OpenAIAPITimeoutError
    from openai import InternalServerError as OpenAIInternalServerError
except ImportError:
    OpenAIRateLimitError = Exception
    OpenAIAPIConnectionError = Exception
    OpenAIAPITimeoutError = Exception
    OpenAIInternalServerError = Exception

try:
    from google.api_core.exceptions import ResourceExhausted as GoogleRateLimitError
    from google.api_core.exceptions import ServiceUnavailable as GoogleServiceUnavailable
    from google.api_core.exceptions import DeadlineExceeded as GoogleDeadlineExceeded
except ImportError:
    GoogleRateLimitError = Exception
    GoogleServiceUnavailable = Exception
    GoogleDeadlineExceeded = Exception

try:
    from langgraph.errors import GraphRecursionError
except ImportError:
    GraphRecursionError = type('GraphRecursionError', (RecursionError,), {})


class ConsecutiveToolFailureError(RuntimeError):
    """Raised when MCP tools fail consecutively, indicating infrastructure issues."""
    pass

RETRYABLE_EXCEPTIONS = (
    AnthropicRateLimitError,
    AnthropicAPIConnectionError,
    AnthropicInternalServerError,
    OpenAIRateLimitError,
    OpenAIAPIConnectionError,
    OpenAIAPITimeoutError,
    OpenAIInternalServerError,
    GoogleRateLimitError,
    GoogleServiceUnavailable,
    GoogleDeadlineExceeded,
)

NON_RETRYABLE_EXCEPTIONS = (
    GraphRecursionError,
    RecursionError,
    ConsecutiveToolFailureError,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ============ Global MCP session management ============
_session_manager = None
_cleanup_done = False


def _cleanup_mcp_sessions():
    """Cleanup MCP sessions on exit."""
    global _session_manager, _cleanup_done
    if _cleanup_done or _session_manager is None:
        return
    _cleanup_done = True
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_session_manager.cleanup())
        print("[Worker] MCP sessions cleaned up")
    except Exception as e:
        print(f"[Worker] MCP cleanup warning: {e}")


def _signal_handler(signum, frame):
    sig_name = signal.Signals(signum).name
    print(f"\n[Worker] Received signal {sig_name}, exiting...")
    _cleanup_mcp_sessions()
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
atexit.register(_cleanup_mcp_sessions)


# ============ Result saving (same logic as video-agent.py) ============

def _message_to_entry(msg, session_id, timestamp):
    """Convert a LangChain message to Claude Code session JSONL entry."""
    msg_type = msg.__class__.__name__

    if msg_type == "HumanMessage":
        return {
            "type": "user",
            "message": {"role": "user", "content": msg.content},
            "timestamp": timestamp,
            "sessionId": session_id
        }

    elif msg_type == "AIMessage":
        if isinstance(msg.content, list):
            content = list(msg.content)
        elif msg.content:
            content = [{"type": "text", "text": msg.content}]
        else:
            content = []

        # Normalize bare strings to text blocks
        normalized_content = []
        for item in content:
            if isinstance(item, str):
                if item:
                    normalized_content.append({"type": "text", "text": item})
            else:
                normalized_content.append(item)
        content = normalized_content

        # Fix tool_use items with partial_json
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                if not item.get("input") and item.get("partial_json"):
                    try:
                        item["input"] = json.loads(item["partial_json"])
                    except json.JSONDecodeError:
                        pass
                item.pop("partial_json", None)
                item.pop("caller", None)
                item.pop("index", None)

        # Merge tool_calls not already in content
        existing_ids = {c.get("id") for c in content if isinstance(c, dict) and c.get("type") == "tool_use"}
        for tc in getattr(msg, 'tool_calls', None) or []:
            if tc.get("id") not in existing_ids:
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": tc["name"],
                    "input": tc.get("args", {})
                })

        # Extract model name and usage
        meta = getattr(msg, 'response_metadata', None) or {}
        model = meta.get('model', meta.get('model_id', meta.get('model_name', '')))

        usage_meta = getattr(msg, 'usage_metadata', None) or {}
        if usage_meta:
            input_details = usage_meta.get('input_token_details', {}) or {}
            usage = {
                "input_tokens": usage_meta.get('input_tokens', 0),
                "output_tokens": usage_meta.get('output_tokens', 0),
                "cache_read_input_tokens": input_details.get('cache_read', 0),
                "cache_creation_input_tokens": input_details.get('cache_creation', 0),
            }
        else:
            raw_usage = meta.get('usage', {}) or meta.get('token_usage', {}) or {}
            usage = {
                "input_tokens": raw_usage.get('input_tokens', 0) or raw_usage.get('prompt_tokens', 0),
                "output_tokens": raw_usage.get('output_tokens', 0) or raw_usage.get('completion_tokens', 0),
                "cache_read_input_tokens": raw_usage.get('cache_read_input_tokens', 0),
                "cache_creation_input_tokens": raw_usage.get('cache_creation_input_tokens', 0),
            }

        return {
            "type": "assistant",
            "message": {
                "id": getattr(msg, 'id', '') or "",
                "role": "assistant",
                "content": content,
                "model": model,
                "usage": usage
            },
            "timestamp": timestamp,
            "sessionId": session_id
        }

    elif msg_type == "ToolMessage":
        content_val = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False, default=str)
        return {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": getattr(msg, 'tool_call_id', ''),
                    "content": content_val
                }]
            },
            "timestamp": timestamp,
            "sessionId": session_id
        }

    return None


def _extract_vlm_usage_from_tool_content(content):
    if not content:
        return None
    parsed = None
    if isinstance(content, dict):
        parsed = content
    elif isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(content)
            except (ValueError, SyntaxError):
                return None
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not text:
                continue
            nested_usage = _extract_vlm_usage_from_tool_content(text)
            if nested_usage:
                return nested_usage
    if not isinstance(parsed, dict):
        return None
    usage = parsed.get("vlm_usage")
    if not isinstance(usage, dict):
        return None
    return usage


def _accumulate_vlm_usage_from_entry(entry, totals):
    if not entry or entry.get("type") != "user":
        return
    blocks = entry.get("message", {}).get("content", [])
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        usage = _extract_vlm_usage_from_tool_content(block.get("content"))
        if not usage:
            continue
        totals["total_input_tokens"] += usage.get("input_tokens", 0) or usage.get("total_input_tokens", 0) or 0
        totals["total_output_tokens"] += usage.get("output_tokens", 0) or usage.get("total_output_tokens", 0) or 0
        totals["tool_call_count"] += usage["calls"] if "calls" in usage else 1


def save_result_as_conversation(result, prompt, start_time=None, end_time=None,
                                exp_name=None, accumulated_usage=None,
                                completed="success", model_name="unknown"):
    """Save agent result in Claude Code compatible format (same as video-agent.py)."""
    session_id = str(uuid.uuid4())
    now = datetime.now()
    start_time = start_time or now
    end_time = end_time or now
    messages = result.get("messages", [])

    conversations_dir = Path(os.environ.get("CONVERSATIONS_DIR", "conversations"))
    transcripts_dir = conversations_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # Compute token usage
    if accumulated_usage:
        total_input_tokens = accumulated_usage.get("input_tokens", 0)
        total_output_tokens = accumulated_usage.get("output_tokens", 0)
        total_cache_read_tokens = accumulated_usage.get("cache_read_input_tokens", 0)
        total_cache_write_tokens = accumulated_usage.get("cache_creation_input_tokens", 0)
    else:
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        total_cache_write_tokens = 0

    vlm_usage_totals = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "tool_call_count": 0,
    }

    # Compute per-message time interval
    total_secs = (end_time - start_time).total_seconds()
    msg_count = len(messages)
    interval = total_secs / max(msg_count - 1, 1) if msg_count > 0 else 0

    transcript_path = transcripts_dir / f"{session_id}.jsonl"
    with open(transcript_path, 'w', encoding='utf-8') as f:
        for i, msg in enumerate(messages):
            timestamp = (start_time + timedelta(seconds=interval * i)).isoformat()
            entry = _message_to_entry(msg, session_id, timestamp)
            if entry:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')
                _accumulate_vlm_usage_from_entry(entry, vlm_usage_totals)
                if not accumulated_usage and entry.get("type") == "assistant":
                    usage = entry["message"].get("usage", {})
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
                    total_cache_read_tokens += usage.get("cache_read_input_tokens", 0)
                    total_cache_write_tokens += usage.get("cache_creation_input_tokens", 0)

    # Append session info (multi-process safe with file locking)
    session_info = {
        "session_id": session_id,
        "transcript_path": str(transcript_path.resolve()),
        "init_user_query": prompt,
        "timestamp": start_time.isoformat(),
        "completed": completed,
        "usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cache_read_tokens": total_cache_read_tokens,
            "total_cache_write_tokens": total_cache_write_tokens,
        },
        "vlm_usage": {
            "total_input_tokens": vlm_usage_totals["total_input_tokens"],
            "total_output_tokens": vlm_usage_totals["total_output_tokens"],
            "total_tokens": vlm_usage_totals["total_input_tokens"] + vlm_usage_totals["total_output_tokens"],
            "tool_call_count": vlm_usage_totals["tool_call_count"],
        }
    }

    if exp_name:
        json_path = conversations_dir / f"all_data_path_{exp_name}.json"
    else:
        json_path = conversations_dir / "all_data_path.json"
    lock_path = conversations_dir / ".video_agent.lock"

    with open(lock_path, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            data = {"sessions": []}
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
            data.setdefault("sessions", []).append(session_info)
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(data, jf, indent=2, ensure_ascii=False)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    return session_id


# ============ Core: run a single prompt ============

async def run_single_prompt(prompt: str, model, mcp_tools, system_prompt: str | None,
                            model_config: dict, stream: bool = False) -> dict:
    """Run a single prompt through the agent. Returns {"status": "ok"} or {"status": "error", "error": "..."}.

    This reuses the same MCP tools (already loaded) but creates a fresh agent
    and checkpointer for each prompt.
    """
    _disabled_builtin_tools = model_config.get("disabled_builtin_tools", [])

    # Retry config from model config
    retry_config = model_config.get("retry", {})
    api_max_retries = retry_config.get("max_retries", 8)
    api_initial_wait = retry_config.get("initial_wait", 30.0)
    api_max_wait = retry_config.get("max_wait", 180.0)

    # Consecutive tool failure detection — terminate sample when MCP tools
    # (especially VLM) fail repeatedly, indicating an infrastructure issue.
    MAX_CONSECUTIVE_TOOL_FAILURES = model_config.get("max_consecutive_tool_failures", 5)
    consecutive_tool_failures = 0

    # Read recursion_limit from config (default 125)
    recursion_limit_val = model_config.get("recursion_limit", 125)
    # Whether to enable budget warning mechanism (default True)
    _enable_budget_warning = model_config.get("enable_budget_warning", True)

    # Fresh checkpointer for each prompt
    checkpointer = MemorySaver()
    thread_id = str(uuid.uuid4())

    agent = create_deep_agent(
        model=model,
        tools=mcp_tools,
        system_prompt=system_prompt,
        subagents=[],
        backend=FilesystemBackend(virtual_mode=False),
        disabled_builtin_tools=_disabled_builtin_tools,
        recursion_limit=recursion_limit_val,
        enable_budget_warning=_enable_budget_warning,
        checkpointer=checkpointer,
    )

    start_time = datetime.now()

    # Accumulated token usage
    accumulated_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }

    result = None
    execution_error = None
    is_recursion_error = False
    recursion_count = 0

    # Incremental message collection (same as video-agent.py)
    _msg_by_id = {}
    _tool_results_by_id = {}
    _pending_tool_calls = {}
    _human_msg_key = "__human__"

    # API retry loop
    api_retry_count = 0
    is_first_attempt = True
    last_api_error = None
    config = {
        "recursion_limit": recursion_limit_val,
        "configurable": {"thread_id": thread_id},
    }

    while api_retry_count <= api_max_retries:
        try:
            if is_first_attempt:
                input_data = {"messages": [{"role": "user", "content": prompt}]}
            else:
                # Use None (not Command(resume=None)) to resume from checkpoint.
                # Command(resume=None) triggers a LangGraph 1.1.4 bug (resume_is_map UnboundLocalError).
                # None achieves the same "resume from checkpoint" semantics without the buggy code path.
                input_data = None

            async for event in agent.astream_events(input_data, config, version="v2"):
                # Incremental message collection
                if event.get("event") == "on_chain_end":
                    chain_output = event.get("data", {}).get("output")
                    if isinstance(chain_output, dict) and "messages" in chain_output:
                        raw_msgs = chain_output["messages"]
                        if hasattr(raw_msgs, 'value'):
                            raw_msgs = raw_msgs.value
                        if not isinstance(raw_msgs, list):
                            continue
                        for msg in raw_msgs:
                            msg_type = msg.__class__.__name__
                            if msg_type == "HumanMessage":
                                key = getattr(msg, 'id', None) or _human_msg_key
                            elif msg_type == "AIMessage":
                                key = getattr(msg, 'id', None) or None
                            elif msg_type == "ToolMessage":
                                key = getattr(msg, 'tool_call_id', None) or getattr(msg, 'id', None) or None
                            else:
                                key = getattr(msg, 'id', None) or None
                            if key:
                                _msg_by_id[key] = msg

                if event.get("event") == "on_chat_model_end":
                    output = event.get("data", {}).get("output")
                    if output and isinstance(output, AIMessage):
                        msg_id = getattr(output, 'id', None)
                        if msg_id:
                            _msg_by_id[msg_id] = output
                        for tc in getattr(output, 'tool_calls', None) or []:
                            tc_id = tc.get("id")
                            if tc_id:
                                _pending_tool_calls[tc_id] = {"name": tc["name"], "args": tc.get("args", {})}

                if event.get("event") == "on_tool_end":
                    tool_name = event.get("name", "")
                    tool_output = event.get("data", {}).get("output", "")
                    matched_tc_id = None
                    for tc_id, tc_info in _pending_tool_calls.items():
                        if tc_info["name"] == tool_name and tc_id not in _tool_results_by_id:
                            matched_tc_id = tc_id
                            break
                    if matched_tc_id:
                        tool_msg = ToolMessage(
                            content=str(tool_output) if not isinstance(tool_output, str) else tool_output,
                            tool_call_id=matched_tc_id,
                            name=tool_name,
                        )
                        _tool_results_by_id[matched_tc_id] = tool_msg
                        _msg_by_id[matched_tc_id] = tool_msg

                    # Consecutive tool failure detection
                    output_str = str(tool_output) if not isinstance(tool_output, str) else tool_output
                    if '"success": false' in output_str or '"success":false' in output_str:
                        consecutive_tool_failures += 1
                        if consecutive_tool_failures >= MAX_CONSECUTIVE_TOOL_FAILURES:
                            raise ConsecutiveToolFailureError(
                                f"{consecutive_tool_failures} consecutive tool failures (> {MAX_CONSECUTIVE_TOOL_FAILURES}) "
                                f"(last tool: {tool_name}). Likely infrastructure issue."
                            )
                    else:
                        consecutive_tool_failures = 0

                    # Recursion tracking
                    recursion_count += 1
                    if recursion_count >= recursion_limit_val:
                        break

                # Token usage collection
                if event.get("event") == "on_chat_model_end":
                    output = event.get("data", {}).get("output")
                    if output:
                        usage_meta = getattr(output, 'usage_metadata', None) or {}
                        if usage_meta:
                            accumulated_usage["input_tokens"] += usage_meta.get('input_tokens', 0)
                            accumulated_usage["output_tokens"] += usage_meta.get('output_tokens', 0)
                            input_details = usage_meta.get('input_token_details', {}) or {}
                            accumulated_usage["cache_read_input_tokens"] += input_details.get('cache_read', 0)
                            accumulated_usage["cache_creation_input_tokens"] += input_details.get('cache_creation', 0)
                        else:
                            meta = getattr(output, 'response_metadata', None) or {}
                            raw_usage = meta.get('usage', {}) or meta.get('token_usage', {}) or {}
                            accumulated_usage["input_tokens"] += raw_usage.get('input_tokens', 0) or raw_usage.get('prompt_tokens', 0)
                            accumulated_usage["output_tokens"] += raw_usage.get('output_tokens', 0) or raw_usage.get('completion_tokens', 0)
                            accumulated_usage["cache_read_input_tokens"] += raw_usage.get('cache_read_input_tokens', 0)
                            accumulated_usage["cache_creation_input_tokens"] += raw_usage.get('cache_creation_input_tokens', 0)

                # Final result capture
                if event.get("event") == "on_chain_end" and event.get("name") == "LangGraph":
                    result = event.get("data", {}).get("output", {"messages": []})

            break  # Normal completion

        except RETRYABLE_EXCEPTIONS as e:
            if isinstance(e, NON_RETRYABLE_EXCEPTIONS) or isinstance(e.__cause__, NON_RETRYABLE_EXCEPTIONS):
                execution_error = f"{type(e).__name__}: {e}"
                if isinstance(e, GraphRecursionError) or isinstance(e.__cause__, GraphRecursionError):
                    is_recursion_error = True
                break

            last_api_error = f"{type(e).__name__}: {e}"
            api_retry_count += 1
            if api_retry_count > api_max_retries:
                last_tool_info = f" | last_tool: {list(_pending_tool_calls.values())[-1]['name']}" if _pending_tool_calls else ""
                execution_error = (
                    f"API retry exhausted ({api_max_retries}x). "
                    f"Last error: {last_api_error}{last_tool_info} | "
                    f"msgs: {len(_msg_by_id)}, tools: {len(_tool_results_by_id)}"
                )
                break

            wait_time = min(api_initial_wait * (2 ** (api_retry_count - 1)), api_max_wait)
            jitter = random.uniform(0, min(wait_time * 0.2, 10))
            actual_wait = wait_time + jitter
            last_tool_name = list(_pending_tool_calls.values())[-1]['name'] if _pending_tool_calls else 'N/A'
            print(
                f"  [Retry {api_retry_count}/{api_max_retries}] {type(e).__name__}: {e}"
                f" | last_tool: {last_tool_name} | waiting {actual_wait:.1f}s...",
                file=sys.stderr
            )
            await asyncio.sleep(actual_wait)
            is_first_attempt = False

        except NON_RETRYABLE_EXCEPTIONS as e:
            execution_error = f"{type(e).__name__}: {e}"
            if isinstance(e, GraphRecursionError):
                is_recursion_error = True
            break

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"  [ERROR] {type(e).__name__}: {e}", file=sys.stderr)
            print(f"  [TRACEBACK]\n{tb_str}", file=sys.stderr)

            # Collect context info
            last_tool_info = f" | last_tool: {list(_pending_tool_calls.values())[-1]['name']}" if _pending_tool_calls else ""
            context_info = f"msgs: {len(_msg_by_id)}, tools: {len(_tool_results_by_id)}{last_tool_info}"

            if last_api_error and not is_first_attempt:
                execution_error = (
                    f"API retry failed. Original: {last_api_error} | "
                    f"Recovery: {type(e).__name__}: {e} | {context_info}"
                )
            else:
                execution_error = f"{type(e).__name__}: {e} | {context_info}"
            break

    # Build final result
    if result is None:
        # Try to get ordered state from checkpointer (preserves correct message order)
        try:
            checkpoint_config = {"configurable": {"thread_id": thread_id}}
            state = agent.get_state(checkpoint_config)
            if state and state.values and state.values.get("messages"):
                result = {"messages": list(state.values["messages"])}
        except Exception:
            pass

    if result is None:
        if _msg_by_id:
            ordered = []
            if _human_msg_key in _msg_by_id:
                ordered.append(_msg_by_id[_human_msg_key])
            elif HumanMessage(content=prompt) not in ordered:
                ordered.append(HumanMessage(content=prompt))
            for key in sorted(k for k in _msg_by_id if k != _human_msg_key):
                ordered.append(_msg_by_id[key])
            result = {"messages": ordered}
        else:
            result = {"messages": [HumanMessage(content=prompt)]}

    if execution_error:
        error_msg = AIMessage(content=f"[EXECUTION ERROR] {execution_error}")
        result["messages"].append(error_msg)

    end_time = datetime.now()

    # Determine completion status
    if is_recursion_error:
        completed_status = "exceed_recursion_count"
    elif execution_error:
        completed_status = "execute_error"
    elif recursion_count >= recursion_limit_val:
        completed_status = "exceed_recursion_count"
    else:
        completed_status = "success"

    # Save result
    exp_name = model_config.get("exp_name")
    model_name = model_config.get("model", {}).get("model", "unknown") if isinstance(model_config.get("model"), dict) else "unknown"
    save_result_as_conversation(
        result, prompt, start_time, end_time,
        exp_name=exp_name,
        accumulated_usage=accumulated_usage,
        completed=completed_status,
        model_name=model_name,
    )

    if completed_status == "success":
        return {"status": "ok"}
    else:
        return {"status": "error", "error": execution_error or completed_status}


# ============ Dynamic task queue ============

def fetch_next_task(queue_file: str) -> dict | None:
    """Atomically fetch the next task from a shared queue file.

    Uses fcntl.flock on a companion .lock file to ensure only one worker
    advances the byte-offset cursor at a time. Returns None when the queue
    is exhausted.
    """
    cursor_file = queue_file.replace('.jsonl', '.cursor')
    lock_file = queue_file.replace('.jsonl', '.lock')

    with open(lock_file, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(cursor_file, 'r') as cf:
                offset = int(cf.read().strip())
            with open(queue_file, 'rb') as qf:
                qf.seek(offset)
                raw_line = qf.readline()
                if not raw_line.strip():
                    return None
                new_offset = offset + len(raw_line)
            with open(cursor_file, 'w') as cf:
                cf.write(str(new_offset))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    return json.loads(raw_line.decode('utf-8'))


# ============ Worker main ============

async def worker_main(config_path: str, system_prompt_path: str, task_file: str = None,
                      queue_file: str = None, worker_id: int = 0, total_tasks: int = 0,
                      stream: bool = False):
    """Main worker function: load MCP once, process all tasks."""
    global _session_manager

    _key_config_path = Path(__file__).resolve().parent / "key.yaml"

    # Load API keys
    with open(_key_config_path, "r", encoding="utf-8") as f:
        key_config = yaml.safe_load(f)

    if key_config.get("LLM_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = key_config["LLM_API_KEY"]
        os.environ["OPENAI_API_KEY"] = key_config["LLM_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = key_config["LLM_API_KEY"]

    if not os.environ.get("GOOGLE_API_KEY") and key_config.get("VLM_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = key_config["VLM_API_KEY"]

    if key_config.get("VLM_API_KEY"):
        os.environ["VLM_API_KEY"] = key_config["VLM_API_KEY"]

    # ---- Load model config EARLY (before MCP) to get vlm_model_name ----
    model_config_path = Path(config_path)
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    for key, value in (model_config.get("env") or {}).items():
        if value is not None:
            os.environ[str(key)] = str(value)

    # ---- Load MCP tools ONCE ----
    print("[Worker] Loading MCP tools (one-time setup)...")
    mcp_json_file = model_config.get("mcp_json_file", ".mcp.json")
    if not os.path.exists(mcp_json_file):
        raise FileNotFoundError(f"MCP configuration file not found: {mcp_json_file}")
    mcp_config = load_mcp_config(mcp_json_file)
    print(f"[Worker-{worker_id}] MCP config loaded from: {mcp_json_file}")
    _inject_env = {
        k: v for k, v in [
            ("VLM_API_KEY", key_config.get("VLM_API_KEY")),
            ("DINOX_API_KEY", key_config.get("DINOX_API_KEY")),
            ("VLM_MODEL_NAME", model_config.get("vlm_model_name")),
            ("VLM_API_URL", model_config.get("VLM_API_URL")),
            ("VLM_ENABLE_THINKING", model_config.get("VLM_ENABLE_THINKING")),
            ("VLM_MAX_TOKENS", model_config.get("VLM_MAX_TOKENS")),
            ("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES")),
        ] if v
    }
    if model_config.get("vlm_model_name"):
        print(f"[Worker-{worker_id}] VLM_MODEL_NAME will be injected to MCP: {model_config['vlm_model_name']}")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"[Worker-{worker_id}] CUDA_VISIBLE_DEVICES will be injected to MCP: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    for srv in mcp_config["mcpServers"].values():
        env = srv.setdefault("env", {})
        env['http_proxy'] = ''
        env['https_proxy'] = ''
        for k, v in _inject_env.items():
            if k not in env:
                env[k] = v

    mcp_tools, session_manager, server_info = await _load_tools_from_config(mcp_config)
    _session_manager = session_manager

    # Strip server name prefix from tool names
    # Support double-underscore prefixes in langchain-mcp-adapters v0.1.x (server__toolname)
    # Support single-underscore prefixes in langchain-mcp-adapters v0.2.x (server_toolname)
    # Keep the full name if the shortened name conflicts
    seen = {}
    for t in mcp_tools:
        name = t.name
        if "__" in name:
            short = name.split("__")[-1]    # Support v0.1.x double-underscore prefixes
        else:
            pos = name.find("_")
            if pos > 0 and "-" in name[:pos]:
                short = name[pos + 1:]     # Support v0.2.x single-underscore prefixes
            else:
                short = name
        if short in seen:
            short = name        # Keep the full name if the shortened name conflicts
        seen[short] = True
        t.name = short

    print(f"[Worker-{worker_id}] MCP tools loaded: {len(mcp_tools)} tools ready")

    # ---- Setup LLM model ----
    model_cfg = dict(model_config["model"])
    provider = model_cfg.pop("provider")

    if provider == "ChatOpenAI":
        model = ChatOpenAIWithReasoning(**model_cfg)
    elif provider == "ChatAnthropic":
        model = ChatAnthropic(**model_cfg)
    elif provider == "ChatGoogleGenerativeAI":
        model = ChatGoogleGenerativeAI(**model_cfg)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    print(f"[Worker] Model: {model_cfg.get('model', 'unknown')}")

    # ---- Load system prompt ONCE ----
    prompt_path = Path(system_prompt_path)
    system_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else None

    # ---- Read and process tasks ----
    if task_file:
        # Legacy mode: static pre-assigned task list
        with open(task_file, 'r', encoding='utf-8') as f:
            tasks = [json.loads(line.strip()) for line in f if line.strip()]
        total = len(tasks)
        print(f"[Worker-{worker_id}] Starting {total} tasks (static mode, MCP already loaded), total samples across all workers: {total_tasks}")
    elif queue_file:
        # Dynamic mode: pull tasks from shared queue on demand
        tasks = None
        total = total_tasks
        print(f"[Worker-{worker_id}] Dynamic queue mode (MCP already loaded), total samples across all workers: {total_tasks}")
    else:
        print(f"[Worker-{worker_id}] ERROR: No task-file or queue-file specified")
        return 0, 0

    success_count = 0
    fail_count = 0
    worker_start = datetime.now()

    if tasks is not None:
        # Static mode: iterate over pre-assigned tasks
        for i, task in enumerate(tasks):
            idx = task["idx"]
            prompt = task["prompt"]

            print(f"\n{'='*10} [Worker-{worker_id}] : [progress={i+1}/{total};  idx={idx}/{total_tasks}] {'='*10}")
            print(f"  Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            print(f"  [Worker-{worker_id}] Agent is working...")

            task_start = datetime.now()
            result = await run_single_prompt(
                prompt=prompt,
                model=model,
                mcp_tools=mcp_tools,
                system_prompt=system_prompt,
                model_config=model_config,
                stream=stream,
            )
            task_end = datetime.now()
            duration = (task_end - task_start).total_seconds()

            now_str = task_end.strftime("%m/%d %H:%M:%S")
            if result["status"] == "ok":
                success_count += 1
                print(f" ✅ [{now_str}][Worker-{worker_id}] ====> [progress={i+1}/{total};  idx={idx}/{total_tasks}] Agent Done ({duration:.1f}s)")
            else:
                fail_count += 1
                print(f" ❌ [{now_str}][Worker-{worker_id}] ====> [progress={i+1}/{total};  idx={idx}/{total_tasks}] Agent Failed ({duration:.1f}s): {result.get('error', 'unknown')}")

            elapsed = (datetime.now() - worker_start).total_seconds()
            avg_per_case = elapsed / (i + 1)
            remaining = avg_per_case * (total - i - 1)
            print(f"  ⏱  [Worker-{worker_id}] Elapsed: {elapsed/60:.1f}min | Avg: {avg_per_case:.1f}s/case | ETA: {remaining/60:.1f}min")
    else:
        # Dynamic mode: pull tasks from shared queue until exhausted
        i = 0
        while True:
            task = fetch_next_task(queue_file)
            if task is None:
                break
            idx = task["idx"]
            prompt = task["prompt"]

            print(f"\n{'='*10} [Worker-{worker_id}] : [completed={i};  idx={idx}/{total_tasks}] {'='*10}")
            print(f"  Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            print(f"  [Worker-{worker_id}] Agent is working...")

            task_start = datetime.now()
            result = await run_single_prompt(
                prompt=prompt,
                model=model,
                mcp_tools=mcp_tools,
                system_prompt=system_prompt,
                model_config=model_config,
                stream=stream,
            )
            task_end = datetime.now()
            duration = (task_end - task_start).total_seconds()

            now_str = task_end.strftime("%m/%d %H:%M:%S")
            if result["status"] == "ok":
                success_count += 1
                print(f" ✅ [{now_str}][Worker-{worker_id}] ====> [completed={i+1};  idx={idx}/{total_tasks}] Agent Done ({duration:.1f}s)")
            else:
                fail_count += 1
                print(f" ❌ [{now_str}][Worker-{worker_id}] ====> [completed={i+1};  idx={idx}/{total_tasks}] Agent Failed ({duration:.1f}s): {result.get('error', 'unknown')}")

            elapsed = (datetime.now() - worker_start).total_seconds()
            avg_per_case = elapsed / (i + 1)
            print(f"  ⏱  [Worker-{worker_id}] Elapsed: {elapsed/60:.1f}min | Avg: {avg_per_case:.1f}s/case")
            i += 1

    print(f"\n[Worker-{worker_id}] All tasks completed: {success_count} success, {fail_count} failed")

    # Cleanup MCP sessions
    try:
        await session_manager.cleanup()
        _session_manager = None
    except Exception as e:
        print(f"[Worker-{worker_id}] MCP cleanup warning: {e}")

    return success_count, fail_count


def parse_args():
    parser = argparse.ArgumentParser(description='Persistent batch worker for video-agent')
    parser.add_argument('-c', '--config', type=str, required=True, help='Experiment config YAML path')
    parser.add_argument('-sys', '--system-prompt', type=str, required=True, help='System prompt MD path')
    task_source = parser.add_mutually_exclusive_group(required=True)
    task_source.add_argument('--task-file', type=str, help='Task file path (JSONL) - static pre-assigned tasks')
    task_source.add_argument('--queue-file', type=str, help='Shared queue file path (JSONL) - dynamic task fetching')
    parser.add_argument('--worker-id', type=int, default=0, help='Worker index (for logging)')
    parser.add_argument('--total-tasks', type=int, default=0, help='Total number of tasks across all workers (for logging)')
    parser.add_argument('--stream', action='store_true', help='Print streaming output')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success, fail = asyncio.run(worker_main(
        config_path=args.config,
        system_prompt_path=args.system_prompt,
        task_file=args.task_file,
        queue_file=args.queue_file,
        worker_id=args.worker_id,
        total_tasks=args.total_tasks,
        stream=args.stream,
    ))
    sys.exit(1 if fail > 0 else 0)
