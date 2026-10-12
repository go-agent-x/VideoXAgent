#!/usr/bin/env python3
"""Subtitle MCP Server — Read and query SRT subtitle files for videos."""
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent))
# Support direct execution from any working directory after service relocation.
for _ancestor in Path(__file__).resolve().parents:
    if (_ancestor / "_base" / "paths.py").is_file():
        sys.path.insert(0, str(_ancestor))
        break
else:
    raise RuntimeError("Cannot locate the VideoXAgent repository root")

from _base.paths import PROJECT_ROOT

from _base.server_framework import create_mcp_server, run_server, make_success_response, make_error_response
from _base.subtitle_utils import (
    DATASET_SUBTITLE_CONFIG,
    detect_dataset_from_video_path,
    parse_timestamp,
    format_seconds_to_timestamp,
    clean_subtitle_text,
    strip_html_tags,
    parse_srt_file,
    parse_json_subtitle_file,
    parse_jsonl_subtitle_file,
    parse_subtitle_file,
    get_video_id_from_path,
    find_subtitle_file,
    get_subtitles_for_segments,
    format_subtitles_as_text,
)

_PROJECT_ROOT = str(PROJECT_ROOT)


def get_subtitle_content(
    video_path: str,
    segments: Optional[list[dict]] = None,
    include_timestamps: bool = False,
    workspace_root: str = ""
) -> dict[str, Any]:
    """
    Main function to get subtitle content for a video.
    
    Args:
        video_path: Path to video file
        segments: Optional list of time segments [{"start": float, "end": float}, ...]
        include_timestamps: Whether to include timestamps in the output text
        workspace_root: Root directory of the workspace
        
    Returns:
        Dict with subtitle content and metadata
    """
    # Find subtitle file
    subtitle_path = find_subtitle_file(video_path, workspace_root)
    
    if not subtitle_path:
        video_id = get_video_id_from_path(video_path)
        ws = workspace_root or os.environ.get("WORKSPACE_ROOT", _PROJECT_ROOT)
        dataset = detect_dataset_from_video_path(video_path)

        # Build expected paths based on dataset or show all possibilities
        expected_paths = []
        if dataset and dataset in DATASET_SUBTITLE_CONFIG:
            config = DATASET_SUBTITLE_CONFIG[dataset]
            filename = config["naming_pattern"].format(video_id=video_id)
            expected_paths.append(os.path.join(ws, config["subtitle_dir"], filename))
        else:
            for name, config in DATASET_SUBTITLE_CONFIG.items():
                filename = config["naming_pattern"].format(video_id=video_id)
                expected_paths.append(os.path.join(ws, config["subtitle_dir"], filename))

        return {
            "error": f"Subtitle file not found for video: {video_path}",
            "detected_dataset": dataset,
            "expected_paths": expected_paths,
            "video_id": video_id
        }
    
    # Parse subtitle file
    try:
        all_subtitles = parse_subtitle_file(subtitle_path)
    except Exception as e:
        return {
            "error": f"Failed to parse subtitle file: {e}",
            "subtitle_path": subtitle_path
        }
    
    if not all_subtitles:
        return {
            "error": "No subtitles found in file",
            "subtitle_path": subtitle_path
        }
    
    # Filter by segments if provided
    if segments and len(segments) > 0:
        # Convert segments to proper format
        parsed_segments = []
        for seg in segments:
            start = seg.get('start', 0)
            end = seg.get('end', float('inf'))
            
            # Handle string timestamps
            if isinstance(start, str):
                start = parse_timestamp(start)
            if isinstance(end, str):
                end = parse_timestamp(end)
                
            parsed_segments.append({'start': start, 'end': end})
        
        filtered_subtitles = get_subtitles_for_segments(all_subtitles, parsed_segments)
    else:
        filtered_subtitles = all_subtitles
    
    # Format output
    text_content = format_subtitles_as_text(filtered_subtitles, include_timestamps)
    
    return {
        "video_path": video_path,
        "video_id": get_video_id_from_path(video_path),
        # "subtitle_path": subtitle_path,
        "total_subtitles": len(all_subtitles),
        "matched_subtitles": len(filtered_subtitles),
        # "segments_requested": segments,
        "text": text_content
    }


# Initialize MCP server


async def handle_get_video_subtitle(arguments: dict) -> dict:
    workspace_root = os.environ.get("WORKSPACE_ROOT", "")
    return get_subtitle_content(
        video_path=arguments.get("video_path", ""),
        segments=arguments.get("segments"),
        include_timestamps=arguments.get("include_timestamps", False),
        workspace_root=workspace_root,
    )


async def handle_get_multiple_videos_subtitle(arguments: dict) -> dict:
    workspace_root = os.environ.get("WORKSPACE_ROOT", "")
    videos = arguments.get("videos", [])
    include_timestamps = arguments.get("include_timestamps", False)

    results = {}
    for video_spec in videos:
        video_path = video_spec.get("video_path", "")
        segments = video_spec.get("segments")
        result = get_subtitle_content(
            video_path=video_path,
            segments=segments,
            include_timestamps=include_timestamps,
            workspace_root=workspace_root,
        )
        results[video_path] = result

    return {
        "total_videos": len(videos),
        "results": results,
    }


async def handle_check_subtitle_exists(arguments: dict) -> dict:
    workspace_root = os.environ.get("WORKSPACE_ROOT", "")
    video_path = arguments.get("video_path", "")
    video_id = get_video_id_from_path(video_path)
    subtitle_path = find_subtitle_file(video_path, workspace_root)

    ws = workspace_root or os.environ.get("WORKSPACE_ROOT", _PROJECT_ROOT)
    dataset = detect_dataset_from_video_path(video_path)

    # Build expected path based on detected dataset
    expected_path = None
    if not subtitle_path:
        if dataset and dataset in DATASET_SUBTITLE_CONFIG:
            config = DATASET_SUBTITLE_CONFIG[dataset]
            filename = config["naming_pattern"].format(video_id=video_id)
            expected_path = os.path.join(ws, config["subtitle_dir"], filename)
        else:
            # Default to first config
            config = list(DATASET_SUBTITLE_CONFIG.values())[0]
            filename = config["naming_pattern"].format(video_id=video_id)
            expected_path = os.path.join(ws, config["subtitle_dir"], filename)

    return {
        "subtitle_exists": subtitle_path is not None,
        "subtitle_path": subtitle_path or expected_path,
        "detected_dataset": dataset,
    }


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="subtitle-mcp",
        tool_defs_source="_tool_defs/basic.yaml",
        handlers={
            "get_video_subtitle": handle_get_video_subtitle,
            "get_multiple_videos_subtitle": handle_get_multiple_videos_subtitle,
            "check_subtitle_exists": handle_check_subtitle_exists,
        },
        is_file=True,
    )
    asyncio.run(run_server(app))
