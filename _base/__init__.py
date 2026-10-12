"""_base module - Shared base utilities for MCP servers."""

from .tool_registry import (
    load_tool_defs,
    build_mcp_tools,
    register_handler,
    get_tool_def,
    route_tool_call,
    validate_arguments,
    list_all_tools,
    clear_registry,
)

from .server_framework import create_mcp_server, run_server

from .config import (
    get_config,
    pick_api_key,
    get_env,
    normalize_vlm_url,
    get_output_dir,
    CONFIG_SCHEMAS,
)

from .subtitle_utils import (
    DATASET_SUBTITLE_CONFIG,
    register_dataset,
    detect_dataset_from_video_path,
    parse_timestamp,
    format_seconds_to_timestamp,
    strip_html_tags,
    clean_subtitle_text,
    parse_srt_file,
    parse_json_subtitle_file,
    parse_jsonl_subtitle_file,
    parse_subtitle_file,
    get_video_id_from_path,
    find_subtitle_file,
    filter_subtitles_by_time_range,
    get_subtitles_for_segments,
    format_subtitles_as_text,
    group_subtitles_by_scenes,
)

__all__ = [
    "load_tool_defs",
    "build_mcp_tools",
    "register_handler",
    "get_tool_def",
    "route_tool_call",
    "validate_arguments",
    "list_all_tools",
    "clear_registry",
    "create_mcp_server",
    "run_server",
    "get_config",
    "pick_api_key",
    "get_env",
    "normalize_vlm_url",
    "get_output_dir",
    "CONFIG_SCHEMAS",
    "DATASET_SUBTITLE_CONFIG",
    "register_dataset",
    "detect_dataset_from_video_path",
    "parse_timestamp",
    "format_seconds_to_timestamp",
    "strip_html_tags",
    "clean_subtitle_text",
    "parse_srt_file",
    "parse_json_subtitle_file",
    "parse_jsonl_subtitle_file",
    "parse_subtitle_file",
    "get_video_id_from_path",
    "find_subtitle_file",
    "filter_subtitles_by_time_range",
    "get_subtitles_for_segments",
    "format_subtitles_as_text",
    "group_subtitles_by_scenes",
]
