#!/usr/bin/env python3
"""Thumbnail Navigator MCP Server — Generate thumbnail grids for video scene navigation."""

import asyncio
import sys
import os
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

from _base.server_framework import create_mcp_server, run_server

# ── Business logic ──

import cv2
from thumbnail_utils import (
    extract_thumbnails,
    create_grid,
    calculate_timestamps,
    get_video_info,
)

OUTPUT_DIR = PROJECT_ROOT / "output"


def generate_thumbnail_grid(
    video_path: str,
    start_time: float = 0,
    end_time: float = None,
    interval: float = None,
    max_thumbnails: int = 20,
    thumbnail_width: int = 320,
    columns: int = 5,
    show_timestamp: bool = True,
    output_name: str = "",
) -> dict:
    """
    Generate a thumbnail grid from a video for scene navigation.
    
    Args:
        video_path: Absolute path to the video file
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (default: video end)
        interval: Sampling interval in seconds (auto-calculated if not specified)
        max_thumbnails: Maximum number of thumbnails when interval is auto
        thumbnail_width: Width of each thumbnail in pixels
        columns: Number of columns in the grid
        show_timestamp: Whether to overlay timestamp on each thumbnail
        output_name: Output file name (auto-generated if empty)
        
    Returns:
        Dict with success status and output information
    """
    # Validate video path
    if not video_path:
        return {"success": False, "error": "video_path is required"}
    
    if not os.path.exists(video_path):
        return {"success": False, "error": f"Video not found: {video_path}"}
    
    try:
        # Get video info
        video_info = get_video_info(video_path)
        duration = video_info["duration"]
        
        if duration <= 0:
            return {"success": False, "error": "Invalid video duration"}
        
        # Set default end_time
        if end_time is None or end_time <= 0:
            end_time = duration
        
        # Validate time range
        if start_time < 0:
            start_time = 0
        if start_time >= duration:
            return {"success": False, "error": f"start_time ({start_time}s) exceeds video duration ({duration:.2f}s)"}
        if end_time > duration:
            end_time = duration
        if start_time >= end_time:
            return {"success": False, "error": f"Invalid time range: start_time ({start_time}s) >= end_time ({end_time}s)"}
        
        # Calculate timestamps
        timestamps, interval_used = calculate_timestamps(
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            max_thumbnails=max_thumbnails,
        )
        
        if not timestamps:
            return {"success": False, "error": "No timestamps generated"}
        
        # Extract thumbnails
        thumbnails = extract_thumbnails(
            video_path=video_path,
            timestamps=timestamps,
            width=thumbnail_width,
        )
        
        if not thumbnails:
            return {"success": False, "error": "No frames could be extracted from the video"}
        
        # Create grid
        grid = create_grid(
            thumbnails=thumbnails,
            columns=columns,
            show_timestamp=show_timestamp,
        )
        
        # Generate output path
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if not output_name:
            video_name = Path(video_path).stem
            time_range_str = f"{int(start_time)}s-{int(end_time)}s"
            output_name = f"{video_name}_thumbnail_grid_{time_range_str}.jpg"
        
        if not any(output_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            output_name = output_name + '.jpg'
        
        output_path = (OUTPUT_DIR / output_name).resolve()
        
        # Save grid
        success = cv2.imwrite(str(output_path), grid)
        if not success:
            return {"success": False, "error": f"Failed to save image to {output_path}"}
        
        # Calculate grid dimensions
        thumb_h = thumbnails[0][1].shape[0]
        rows = (len(thumbnails) + columns - 1) // columns
        
        # Build summary
        summary = (
            f"Generated {len(thumbnails)} thumbnails from {start_time:.1f}s to {end_time:.1f}s "
            f"(interval: {interval_used:.1f}s). Grid: {columns}x{rows}."
        )
        
        return {
            "success": True,
            "output_path": str(output_path),
            "image_context": "thumbnail_grid",
            "thumbnail_count": len(thumbnails),
            "interval_used": round(interval_used, 2),
            "time_range": {
                "start": round(start_time, 2),
                "end": round(end_time, 2),
            },
            "timestamps": [round(t, 2) for t in [ts for ts, _ in thumbnails]],
            "grid_size": {
                "columns": columns,
                "rows": rows,
                "thumbnail_width": thumbnail_width,
                "thumbnail_height": thumb_h,
            },
            "video_duration": round(duration, 2),
            "summary": summary,
        }
        
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Failed to generate thumbnail grid: {str(e)}"}


async def handle_generate_thumbnail_grid(arguments: dict) -> dict:
    """Async handler for generate_thumbnail_grid tool."""
    return generate_thumbnail_grid(
        video_path=arguments.get("video_path", ""),
        start_time=float(arguments.get("start_time", 0)),
        end_time=arguments.get("end_time"),  # None if not provided
        interval=arguments.get("interval"),  # None if not provided
        max_thumbnails=int(arguments.get("max_thumbnails", 20)),
        thumbnail_width=int(arguments.get("thumbnail_width", 320)),
        columns=int(arguments.get("columns", 5)),
        show_timestamp=arguments.get("show_timestamp", True),
        output_name=arguments.get("output_name", ""),
    )


if __name__ == "__main__":
    app = create_mcp_server(
        server_name="thumbnail-navigator-mcp",
        tool_defs_source="_tool_defs/basic.yaml",
        handlers={"generate_thumbnail_grid": handle_generate_thumbnail_grid},
        is_file=True,
    )
    asyncio.run(run_server(app))
