"""MCP Server Framework - Reusable server base.

This module provides a framework for creating MCP servers with decoupled
tool definitions and implementations.
"""

import asyncio
import inspect
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional

import anyio

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent

from _base.tool_registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mcp_server(
    server_name: str,
    tool_defs_source: str,
    handlers: Dict[str, Callable],
    is_file: bool = False,
) -> Server:
    """Create a configured MCP Server.
    
    Args:
        server_name: Server identifier.
        tool_defs_source: Path to YAML file or directory with tool definitions.
        handlers: Dict mapping tool_name to handler function.
        is_file: If True, treat tool_defs_source as a file path; otherwise as directory.
    
    Returns:
        Configured Server instance.
        
    Raises:
        FileNotFoundError: If tool definitions not found.
        ValueError: If handler registered for non-existent tool.
    """
    # Each server gets its own isolated registry so that multiple servers
    # created in the same process do not overwrite each other's definitions.
    registry = ToolRegistry()

    # Load tool definitions
    defs_path = Path(tool_defs_source)
    if not defs_path.is_absolute():
        # Make relative to project root
        project_root = Path(__file__).parent.parent
        defs_path = project_root / defs_path

    if is_file:
        registry.load_tool_defs_from_file(defs_path)
    else:
        registry.load_tool_defs(defs_path)

    # Register handlers
    for tool_name, handler in handlers.items():
        registry.register_handler(tool_name, handler)

    # Only keep tool defs that have registered handlers
    registry.filter_tool_defs(set(handlers.keys()))

    # Create server
    app = Server(server_name)

    @app.list_tools()
    async def list_tools():
        """List available tools."""
        return registry.build_mcp_tools()

    @app.call_tool()
    async def call_tool(name: str, arguments: dict):
        """Handle tool call."""
        logger.info(f"Tool call: [[{name}]] with args: {arguments}")

        # Validate arguments
        valid, error = registry.validate_arguments(name, arguments)
        if not valid:
            logger.error(f"Validation failed for {name}: {error}")
            return [TextContent(
                type="text",
                text=json.dumps({"success": False, "error": error}, ensure_ascii=False)
            )]

        # Route to handler
        try:
            handler = registry.route_tool_call(name)
            if inspect.iscoroutinefunction(handler):
                result = await handler(arguments)
            else:
                result = handler(arguments)
            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]
        except Exception as e:
            logger.error(f"Handler error for {name}: {e}")
            logger.error(traceback.format_exc())
            return [TextContent(
                type="text",
                text=json.dumps(
                    {"success": False, "error": f"Internal error: {str(e)}"},
                    ensure_ascii=False
                )
            )]

    return app


async def run_server(app: Server):
    """Run the MCP server with stdio transport.
    
    Args:
        app: Configured Server instance.
    
    Handles ``anyio.ClosedResourceError`` that occurs when the client
    disconnects (stdin EOF) while a long-running handler (e.g. a VLM HTTP
    call) is still in progress.  In the MCP library (v1.x), the session's
    ``_receive_loop`` closes **both** the read and write memory-object
    streams when stdin ends.  Any handler task that later calls
    ``message.respond()`` → ``_write_stream.send()`` will therefore raise
    ``ClosedResourceError``.  Catching it here prevents the server from
    crashing with an unhandled exception and allows a clean shutdown.
    """
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    except anyio.ClosedResourceError:
        logger.info("Client disconnected; shutting down gracefully.")
    except BaseExceptionGroup as eg:
        # MCP library wraps handler errors in nested ExceptionGroup(s).
        # If every leaf exception is ClosedResourceError, treat as clean disconnect.
        def _all_closed(exc):
            if isinstance(exc, BaseExceptionGroup):
                return all(_all_closed(e) for e in exc.exceptions)
            return isinstance(exc, anyio.ClosedResourceError)
        if _all_closed(eg):
            logger.info("Client disconnected; shutting down gracefully.")
        else:
            raise


def make_success_response(
    data: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """Create a standardized success response.
    
    Args:
        data: Optional data dict to merge.
        **kwargs: Additional fields to include.
        
    Returns:
        Response dict with success=True.
    """
    result = {"success": True}
    if data:
        result.update(data)
    result.update(kwargs)
    return result


def make_error_response(
    error: str,
    details: Optional[Dict] = None,
) -> Dict:
    """Create a standardized error response.
    
    Args:
        error: Error message.
        details: Optional details dict.
        
    Returns:
        Response dict with success=False.
    """
    result = {"success": False, "error": error}
    if details:
        result.update(details)
    return result
