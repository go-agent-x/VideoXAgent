"""Custom ChatOpenAI that extracts reasoning_content from API responses.

LangChain's ChatOpenAI does not extract the `reasoning_content` field from
third-party providers like Doubao, nor the `reasoning` field from OpenAI's
reasoning models (o1/o3/o4-mini). This module provides a subclass that
extracts reasoning content and includes it in the message content as a
thinking block.

Supported providers:
    - OpenAI reasoning models (o1, o3, o4-mini, etc.): uses `reasoning` field
    - Doubao and other third-party providers: uses `reasoning_content` field

Usage:
    from utils.chat_openai_with_reasoning import ChatOpenAIWithReasoning

    # For Doubao
    model = ChatOpenAIWithReasoning(
        model_name="doubao-seed-2-0-pro-260215",
        base_url: "http://example-llm-ip:8000/v1"
        temperature=0.0,
    )

    # For OpenAI reasoning models
    model = ChatOpenAIWithReasoning(
        model_name="o3",
        temperature=1.0,  # OpenAI reasoning models require temperature=1
    )
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _flatten_message_content_to_string(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Flatten list-of-dicts/list-of-strings content to plain strings for OpenAI-compatible API.

    Many OpenAI-compatible APIs (DeepSeek, Kimi, etc.) only accept content as string
    for system/assistant messages. This function flattens list content to plain strings
    for ALL message types that have list content.
    """
    from langchain_core.messages import AIMessage  # noqa: PLC0415
    flattened = []
    for msg in messages:
        if isinstance(msg.content, list):
            text_parts = []
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            new_content = "\n".join(text_parts) if text_parts else ""
            # For AIMessage with tool_calls, content should be empty string or the text
            if isinstance(msg, AIMessage) and not new_content:
                new_content = ""
            flattened.append(msg.model_copy(update={"content": new_content}))
        else:
            flattened.append(msg)
    return flattened


def _extract_reasoning_content(delta: Any) -> str | None:
    """Extract reasoning content from delta, supporting multiple providers.

    Args:
        delta: The delta object from a streaming chunk.

    Returns:
        The reasoning content string, or None if not present.

    Supported fields:
        - `reasoning`: OpenAI reasoning models (o1, o3, o4-mini)
        - `reasoning_content`: Doubao and other third-party providers
    """
    # OpenAI reasoning models use `reasoning` field
    reasoning = getattr(delta, 'reasoning', None)
    if reasoning:
        return reasoning

    # Doubao and other providers use `reasoning_content` field
    reasoning_content = getattr(delta, 'reasoning_content', None)
    if reasoning_content:
        return reasoning_content

    return None


def _extract_reasoning_tokens(usage: Any) -> int:
    """Extract reasoning tokens from usage object.

    Args:
        usage: The usage object from API response.

    Returns:
        The number of reasoning tokens, or 0 if not available.
    """
    if not usage:
        return 0

    # OpenAI includes reasoning_tokens in completion_tokens_details
    completion_details = getattr(usage, 'completion_tokens_details', None)
    if completion_details:
        reasoning_tokens = getattr(completion_details, 'reasoning_tokens', 0)
        if reasoning_tokens:
            return reasoning_tokens

    return 0


class ChatOpenAIWithReasoning(ChatOpenAI):
    """ChatOpenAI subclass that extracts reasoning content from API responses.

    This is useful for:
    - OpenAI reasoning models (o1, o3, o4-mini) that return thinking in `reasoning` field
    - Doubao and other providers that return thinking in `reasoning_content` field

    The reasoning content is converted to a thinking block format:
    {"type": "thinking", "thinking": "..."}

    Attributes:
        stream_usage: Whether to include usage info in streaming mode (default True).
    """

    # Enable stream usage by default so we get token counts in streaming mode
    stream_usage: bool = True

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Override _stream to inject reasoning_content into chunks."""
        messages = _flatten_message_content_to_string(messages)
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        # Inject stream_options to get usage info in streaming mode
        if self.stream_usage:
            payload["stream_options"] = {"include_usage": True}

        with self.client.create(**payload) as response:
            for chunk in response:
                if not chunk.choices:
                    # Handle usage-only chunk (no choices but has usage)
                    usage = getattr(chunk, 'usage', None)
                    if usage:
                        reasoning_tokens = _extract_reasoning_tokens(usage)
                        usage_dict = {
                            "input_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                            "output_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                            "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                        }
                        response_usage = {
                            "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                            "completion_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                            "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                        }
                        if reasoning_tokens:
                            usage_dict["reasoning_tokens"] = reasoning_tokens
                            response_usage["reasoning_tokens"] = reasoning_tokens
                        usage_chunk = AIMessageChunk(
                            content="",
                            usage_metadata=usage_dict,
                            response_metadata={"usage": response_usage},
                        )
                        yield ChatGenerationChunk(message=usage_chunk)
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Check for reasoning content (OpenAI `reasoning` or Doubao `reasoning_content`)
                reasoning_content = _extract_reasoning_content(delta)
                if reasoning_content:
                    # Create a thinking block
                    thinking_chunk = AIMessageChunk(
                        content=[{
                            "type": "thinking",
                            "thinking": reasoning_content,
                            "index": 0,
                        }]
                    )
                    yield ChatGenerationChunk(message=thinking_chunk)
                    if run_manager:
                        run_manager.on_llm_new_token(reasoning_content)

                # Handle regular content
                content = getattr(delta, 'content', None) or ""
                tool_calls = getattr(delta, 'tool_calls', None)

                if content or tool_calls or choice.finish_reason:
                    # Build the message chunk
                    tool_call_chunks = []

                    if tool_calls:
                        tool_call_chunks = [
                            {
                                "name": tc.function.name if tc.function else None,
                                "args": tc.function.arguments if tc.function else None,
                                "id": tc.id,
                                "index": tc.index,
                            }
                            for tc in tool_calls
                        ]

                    # Extract usage from the final chunk (when finish_reason is set)
                    usage = getattr(chunk, 'usage', None)
                    usage_metadata = None
                    response_metadata = {}
                    if usage:
                        reasoning_tokens = _extract_reasoning_tokens(usage)
                        usage_metadata = {
                            "input_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                            "output_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                            "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                        }
                        if reasoning_tokens:
                            usage_metadata["reasoning_tokens"] = reasoning_tokens
                        response_metadata["usage"] = {
                            "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                            "completion_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                            "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                        }
                        if reasoning_tokens:
                            response_metadata["usage"]["reasoning_tokens"] = reasoning_tokens

                    msg_chunk = AIMessageChunk(
                        content=content,
                        tool_call_chunks=tool_call_chunks,
                        **({"usage_metadata": usage_metadata} if usage_metadata else {}),
                        **({"response_metadata": response_metadata} if response_metadata else {}),
                    )

                    generation_info = {}
                    if choice.finish_reason:
                        generation_info["finish_reason"] = choice.finish_reason

                    yield ChatGenerationChunk(
                        message=msg_chunk,
                        generation_info=generation_info or None,
                    )

                    if run_manager and content:
                        run_manager.on_llm_new_token(content)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Override _astream to inject reasoning_content into chunks."""
        messages = _flatten_message_content_to_string(messages)
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        # Inject stream_options to get usage info in streaming mode
        if self.stream_usage:
            payload["stream_options"] = {"include_usage": True}

        # # DEBUG: dump the payload to find the unsupported format
        # import json as _json
        # _debug_payload = _json.dumps(payload, ensure_ascii=False, default=str)
        # logger.error(f"[DEBUG_PAYLOAD] Sending payload (first 5000 chars):\n{_debug_payload[:5000]}")
        # # Also dump each message's role and content type
        # for i, m in enumerate(payload.get("messages", [])):
        #     role = m.get("role", "?")
        #     content = m.get("content")
        #     content_type = type(content).__name__
        #     content_preview = str(content)[:200] if content else "None"
        #     logger.error(f"[DEBUG_MSG {i}] role={role} content_type={content_type} content={content_preview}")

        response = await self.async_client.create(**payload)

        async for chunk in response:
            if not chunk.choices:
                # Handle usage-only chunk (no choices but has usage)
                usage = getattr(chunk, 'usage', None)
                if usage:
                    reasoning_tokens = _extract_reasoning_tokens(usage)
                    usage_dict = {
                        "input_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                        "output_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                        "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                    }
                    response_usage = {
                        "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                        "completion_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                        "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                    }
                    if reasoning_tokens:
                        usage_dict["reasoning_tokens"] = reasoning_tokens
                        response_usage["reasoning_tokens"] = reasoning_tokens
                    usage_chunk = AIMessageChunk(
                        content="",
                        usage_metadata=usage_dict,
                        response_metadata={"usage": response_usage},
                    )
                    yield ChatGenerationChunk(message=usage_chunk)
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Check for reasoning content (OpenAI `reasoning` or Doubao `reasoning_content`)
            reasoning_content = _extract_reasoning_content(delta)
            if reasoning_content:
                # Create a thinking block
                thinking_chunk = AIMessageChunk(
                    content=[{
                        "type": "thinking",
                        "thinking": reasoning_content,
                        "index": 0,
                    }]
                )
                yield ChatGenerationChunk(message=thinking_chunk)
                if run_manager:
                    await run_manager.on_llm_new_token(reasoning_content)

            # Handle regular content
            content = getattr(delta, 'content', None) or ""
            tool_calls = getattr(delta, 'tool_calls', None)

            if content or tool_calls or choice.finish_reason:
                # Build the message chunk
                tool_call_chunks = []

                if tool_calls:
                    tool_call_chunks = [
                        {
                            "name": tc.function.name if tc.function else None,
                            "args": tc.function.arguments if tc.function else None,
                            "id": tc.id,
                            "index": tc.index,
                        }
                        for tc in tool_calls
                    ]

                # Extract usage from the final chunk (when finish_reason is set)
                usage = getattr(chunk, 'usage', None)
                usage_metadata = None
                response_metadata = {}
                if usage:
                    reasoning_tokens = _extract_reasoning_tokens(usage)
                    usage_metadata = {
                        "input_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                        "output_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                        "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                    }
                    if reasoning_tokens:
                        usage_metadata["reasoning_tokens"] = reasoning_tokens
                    response_metadata["usage"] = {
                        "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                        "completion_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                        "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                    }
                    if reasoning_tokens:
                        response_metadata["usage"]["reasoning_tokens"] = reasoning_tokens

                msg_chunk = AIMessageChunk(
                    content=content,
                    tool_call_chunks=tool_call_chunks,
                    **({"usage_metadata": usage_metadata} if usage_metadata else {}),
                    **({"response_metadata": response_metadata} if response_metadata else {}),
                )

                generation_info = {}
                if choice.finish_reason:
                    generation_info["finish_reason"] = choice.finish_reason

                yield ChatGenerationChunk(
                    message=msg_chunk,
                    generation_info=generation_info or None,
                )

                if run_manager and content:
                    await run_manager.on_llm_new_token(content)
