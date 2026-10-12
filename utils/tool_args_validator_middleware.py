"""Middleware for validating tool call arguments before execution.

When LLM generates tool calls with missing required parameters, this middleware
intercepts the error and returns a friendly message instead of crashing,
allowing the LLM to retry with correct parameters.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

logger = logging.getLogger(__name__)


class ToolArgsValidatorMiddleware(AgentMiddleware):
    """Middleware that validates tool call arguments before execution.

    When the LLM generates a tool call with missing required parameters,
    this middleware catches the validation error and returns a helpful
    error message, allowing the LLM to retry instead of crashing.

    Example:
        >>> middleware = ToolArgsValidatorMiddleware()
        >>> # Add to your middleware stack in create_deep_agent
    """

    def __init__(
        self,
        tools: list[BaseTool | dict | Callable] | None = None,
        tool_schemas: dict[str, dict] | None = None,
    ):
        """Initialize the middleware.

        Args:
            tools: Optional list of tools to extract schemas from.
                   Can be BaseTool, dict, or Callable.
            tool_schemas: Optional dict mapping tool names to their JSON schemas.
                         If not provided, will try to extract from tools at runtime.
        """
        self._tool_schemas: dict[str, dict] = tool_schemas or {}
        if tools:
            for tool in tools:
                self._extract_schema_from_tool(tool)

    def _extract_schema_from_tool(self, tool: BaseTool | dict | Callable | Any) -> None:
        """Extract schema from a tool and store it.

        Args:
            tool: A tool object (BaseTool, dict, or Callable).
        """
        try:
            # Handle dict type tools
            if isinstance(tool, dict):
                tool_name = tool.get('name')
                if tool_name:
                    # Check if schema is provided in the dict
                    if 'args_schema' in tool:
                        args_schema = tool['args_schema']
                        if hasattr(args_schema, 'model_json_schema'):
                            self._tool_schemas[tool_name] = args_schema.model_json_schema()
                        elif hasattr(args_schema, 'schema'):
                            self._tool_schemas[tool_name] = args_schema.schema()
                    elif 'parameters' in tool:
                        # OpenAI function format
                        self._tool_schemas[tool_name] = tool['parameters']
                return

            # Handle BaseTool or similar objects
            tool_name = getattr(tool, 'name', None)
            if not tool_name:
                return

            if hasattr(tool, 'args_schema') and tool.args_schema:
                if hasattr(tool.args_schema, 'model_json_schema'):
                    self._tool_schemas[tool_name] = tool.args_schema.model_json_schema()
                elif hasattr(tool.args_schema, 'schema'):
                    self._tool_schemas[tool_name] = tool.args_schema.schema()
            elif hasattr(tool, 'get_input_schema'):
                try:
                    schema_obj = tool.get_input_schema()
                    if hasattr(schema_obj, 'model_json_schema'):
                        self._tool_schemas[tool_name] = schema_obj.model_json_schema()
                    elif hasattr(schema_obj, 'schema'):
                        self._tool_schemas[tool_name] = schema_obj.schema()
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Failed to extract schema from tool: {e}")

    def _get_required_params(self, tool_name: str, runtime_tools: list[BaseTool] | None = None) -> list[str]:
        """Get required parameters for a tool.

        Args:
            tool_name: Name of the tool.
            runtime_tools: Optional list of tools from runtime to check.

        Returns:
            List of required parameter names.
        """
        # First check cached schemas
        if tool_name in self._tool_schemas:
            schema = self._tool_schemas[tool_name]
            return schema.get('required', [])

        # Try to get from runtime tools
        if runtime_tools:
            for tool in runtime_tools:
                # Handle dict type tools
                if isinstance(tool, dict):
                    if tool.get('name') == tool_name:
                        self._extract_schema_from_tool(tool)
                        if tool_name in self._tool_schemas:
                            return self._tool_schemas[tool_name].get('required', [])
                    continue

                # Handle BaseTool objects
                if getattr(tool, 'name', None) == tool_name:
                    self._extract_schema_from_tool(tool)
                    if tool_name in self._tool_schemas:
                        return self._tool_schemas[tool_name].get('required', [])

        return []

    def _validate_args(
        self,
        tool_name: str,
        args: dict[str, Any],
        runtime_tools: list[BaseTool] | None = None,
    ) -> tuple[bool, str | None]:
        """Validate tool call arguments.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments passed to the tool.
            runtime_tools: Optional list of tools from runtime.

        Returns:
            Tuple of (is_valid, error_message).
        """
        required_params = self._get_required_params(tool_name, runtime_tools)

        if not required_params:
            # No schema available, skip validation
            return True, None

        missing_params = [p for p in required_params if p not in args or args[p] is None]

        if missing_params:
            error_msg = (
                f"Missing required parameters for tool '{tool_name}': {missing_params}. "
                f"Required parameters are: {required_params}. "
                f"Received arguments: {json.dumps(args, ensure_ascii=False)}. "
                f"Please retry with all required parameters."
            )
            return False, error_msg

        # Check for empty args (likely a generation error)
        if not args and required_params:
            error_msg = (
                f"Empty arguments received for tool '{tool_name}'. "
                f"This appears to be a generation error. "
                f"Required parameters are: {required_params}. "
                f"Please regenerate the tool call with all required parameters."
            )
            return False, error_msg

        return True, None

    def _create_error_response(
        self,
        request: ToolCallRequest,
        error_msg: str,
    ) -> ToolMessage:
        """Create a ToolMessage with the validation error.

        Args:
            request: The original tool call request.
            error_msg: The error message to include.

        Returns:
            ToolMessage with the error.
        """
        tool_call_id = request.tool_call.get("id", "unknown")
        tool_name = request.tool_call.get("name", "unknown")

        return ToolMessage(
            content=json.dumps({
                "success": False,
                "error": error_msg,
                "tool_name": tool_name,
                "hint": "Please check the required parameters and retry the tool call."
            }, ensure_ascii=False),
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Validate tool arguments before calling the handler.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the request.

        Returns:
            ToolMessage with result or error, or Command.
        """
        tool_name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {})

        # Get runtime tools if available
        runtime_tools = None
        if hasattr(request, 'runtime') and hasattr(request.runtime, 'tools'):
            runtime_tools = request.runtime.tools

        # Validate arguments (pre-check)
        is_valid, error_msg = self._validate_args(tool_name, args, runtime_tools)

        if not is_valid:
            logger.warning(f"Tool call validation failed (pre-check): {error_msg}")
            return self._create_error_response(request, error_msg)

        # Call handler and catch any validation errors
        try:
            return handler(request)
        except Exception as e:
            error_str = str(e)
            # Check if it's a validation error (missing required parameters)
            if "required property" in error_str.lower() or "validation error" in error_str.lower():
                friendly_error = (
                    f"Tool '{tool_name}' failed due to missing or invalid parameters. "
                    f"Original error: {error_str}. "
                    f"Received arguments: {json.dumps(args, ensure_ascii=False)}. "
                    f"Please retry with all required parameters."
                )
                logger.warning(f"Tool call validation failed (caught exception): {friendly_error}")
                return self._create_error_response(request, friendly_error)
            # Re-raise other exceptions
            raise

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async) Validate tool arguments before calling the handler.

        Args:
            request: The tool call request being processed.
            handler: The async handler function to call with the request.

        Returns:
            ToolMessage with result or error, or Command.
        """
        tool_name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {})

        # Get runtime tools if available
        runtime_tools = None
        if hasattr(request, 'runtime') and hasattr(request.runtime, 'tools'):
            runtime_tools = request.runtime.tools

        # Validate arguments (pre-check)
        is_valid, error_msg = self._validate_args(tool_name, args, runtime_tools)

        if not is_valid:
            logger.warning(f"Tool call validation failed (pre-check): {error_msg}")
            return self._create_error_response(request, error_msg)

        # Call handler and catch any validation errors
        try:
            return await handler(request)
        except Exception as e:
            error_str = str(e)
            # Check if it's a validation error (missing required parameters)
            if "required property" in error_str.lower() or "validation error" in error_str.lower():
                friendly_error = (
                    f"Tool '{tool_name}' failed due to missing or invalid parameters. "
                    f"Original error: {error_str}. "
                    f"Received arguments: {json.dumps(args, ensure_ascii=False)}. "
                    f"Please retry with all required parameters."
                )
                logger.warning(f"Tool call validation failed (caught exception): {friendly_error}")
                return self._create_error_response(request, friendly_error)
            # Re-raise other exceptions
            raise
