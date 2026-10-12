#!/usr/bin/env python3
"""
VideoMME Benchmark Evaluation Script for Gemini Models

Supports both frames mode (extract frames) and video mode (native video input).
Quota reporting is disabled by default. Configure --quota-url, YAML quota_url,
or VLM_QUOTA_URL (in that order); an explicit empty value disables it.

Usage:
    # Frames mode (default)
    python eval_mllms/eval_gemini.py \
        --data data/videomme/test/short_duration_data.jsonl \
        --model gemini-2.0-flash \
        --input-mode frames

    # Video mode (native Gemini video support)
    python eval_mllms/eval_gemini.py \
        --data data/videomme/test/short_duration_data.jsonl \
        --model gemini-2.0-flash \
        --input-mode video
"""

import argparse
import atexit
import base64
import io
import json
import os
import sys
import termios
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx
import yaml

# Import shared utilities from eval_seed
sys.path.insert(0, str(Path(__file__).parent))
from eval_seed import (
    DEFAULT_API_URL,
    DEFAULT_DATA,
    build_prompt,
    extract_frames,
    extract_frames_in_range,
    parse_timestamps,
    extract_answer,
    read_srt_subtitle,
    load_completed_ids_from_output,
    _save_output_file,
    _save_terminal_settings,
    _ThreadLocalStdout,
    reencode_video,
    get_video_format,
    video_to_base64,
    normalize_api_url,
    infer_subtitle_dir,
    VIDECUT_TOOL_DEFINITION,
    VIDECUT_TOOL_PROMPT,
    _query_remain_quota,
    _start_quota_tracking,
)

_result_lock = threading.Lock()


# ── Gemini API Call ───────────────────────────────────────────────────────────


# Gemini-format VideoCut tool definition
GEMINI_VIDECUT_TOOL = {
    "functionDeclarations": [{
        "name": "VideoCut",
        "description": VIDECUT_TOOL_DEFINITION["function"]["description"],
        "parameters": VIDECUT_TOOL_DEFINITION["function"]["parameters"],
    }]
}


def _make_gemini_request(
    url: str,
    headers: dict,
    payload: dict,
    timeout: float,
    max_retries: int = 5,
) -> tuple[dict | None, str | None]:
    """
    Make Gemini API request with retry logic.

    Returns:
        (data, error): data is the JSON response, error is None on success.
    """
    payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    print(f"[INFO] Request payload size: {payload_bytes / 1024 / 1024:.2f} MB")

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload, headers=headers)

                if resp.status_code != 200:
                    print(f"[DEBUG] Response status code: {resp.status_code} (attempt {attempt + 1}/{max_retries})")

                if resp.status_code == 429 or resp.status_code >= 500:
                    wait_time = min(2 ** attempt, 60)
                    if attempt < max_retries - 1:
                        label = "Rate limited (429)" if resp.status_code == 429 else f"Server error ({resp.status_code})"
                        print(f"[WARN] {label}. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return None, f"HTTP {resp.status_code} after {max_retries} retries"

                elif resp.status_code != 200:
                    try:
                        error_data = resp.json()
                        error_msg = error_data.get("error", {}).get("message", str(error_data))
                    except Exception:
                        error_msg = resp.text[:200]
                    return None, f"HTTP {resp.status_code}: {error_msg}"

                return resp.json(), None

        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"[WARN] Timeout. Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
                continue
            return None, "timeout"

        except Exception as e:
            return None, str(e)[:200]

    return None, f"Failed after {max_retries} retries"


def _build_gemini_parts(
    prompt_text: str,
    frames_b64: list[tuple[float, str]] | None = None,
    video_path: str | None = None,
) -> list[dict]:
    """Build Gemini content parts from prompt + optional frames or video."""
    parts = [{"text": prompt_text}]

    if frames_b64:
        for _, img_b64 in frames_b64:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_b64,
                }
            })
    elif video_path:
        video_b64 = video_to_base64(video_path)
        video_format = get_video_format(video_path)
        mime_type = f"video/{video_format}"
        parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": video_b64,
            }
        })

    return parts


def _extract_gemini_text(data: dict) -> str:
    """Extract text content from Gemini API response."""
    if "candidates" in data and len(data["candidates"]) > 0:
        candidate = data["candidates"][0]
        if "content" in candidate and "parts" in candidate["content"]:
            text_parts = [
                p["text"] for p in candidate["content"]["parts"]
                if "text" in p
            ]
            return "\n".join(text_parts).strip()
    return ""


def _extract_gemini_function_calls(data: dict) -> list[dict]:
    """Extract functionCall parts from Gemini API response."""
    calls = []
    if "candidates" in data and len(data["candidates"]) > 0:
        candidate = data["candidates"][0]
        if "content" in candidate and "parts" in candidate["content"]:
            for part in candidate["content"]["parts"]:
                if "functionCall" in part:
                    calls.append(part["functionCall"])
    return calls


def _extract_gemini_usage(data: dict) -> tuple[int, int]:
    """Extract token usage from Gemini API response. Returns (input_tokens, output_tokens)."""
    if "usageMetadata" in data:
        usage = data["usageMetadata"]
        return usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)
    return 0, 0


def call_gemini_vlm(
    prompt_text: str,
    api_key: str,
    model: str,
    frames_b64: list[tuple[float, str]] | None = None,
    video_path: str | None = None,
    base_url: str = DEFAULT_API_URL,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: float = 600.0,
    thinking_level: str | None = None,
    # Tool options (aligned with call_vlm)
    use_videcut_tool: bool = False,
    fc_loop: bool = False,
    max_videcut_calls: int = 3,
) -> dict:
    """
    Call Gemini API with either images (frames) or video (native format),
    with optional VideoCut function-call loop.

    Returns dict with keys: content, rounds, input_tokens, output_tokens, error, tool_calls.
    """
    if fc_loop and video_path is None:
        raise ValueError("video_path is required for fc_loop mode (needed for VideoCut)")

    base_url = base_url.rstrip("/")
    url = f"{base_url}/v1beta/models/{model}:generateContent"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    result = {
        "content": "",
        "rounds": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None,
    }
    if fc_loop:
        result["tool_calls"] = []

    # Build initial parts
    parts = _build_gemini_parts(prompt_text, frames_b64, video_path)

    # ─── Single-call mode (no loop) ──────────────────────────────────────────
    if not fc_loop:
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "mediaResolution": "MEDIA_RESOLUTION_MEDIUM",
                **({"thinking_level": thinking_level} if thinking_level else {}),
            },
        }
        if use_videcut_tool:
            payload["tools"] = [GEMINI_VIDECUT_TOOL]

        data, error = _make_gemini_request(url, headers, payload, timeout)

        if error:
            result["error"] = error
            return result

        in_tok, out_tok = _extract_gemini_usage(data)
        result["input_tokens"] = in_tok
        result["output_tokens"] = out_tok

        text = _extract_gemini_text(data)
        if text:
            round_info = {"round": 1, "answer": text}
            result["rounds"].append(round_info)
            result["content"] = text
        else:
            result["error"] = "No text in response"

        return result

    # ─── Function-call loop mode ─────────────────────────────────────────────
    contents = [{"role": "user", "parts": parts}]
    videcut_count = 0
    round_num = 0

    while videcut_count < max_videcut_calls:
        round_num += 1

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "mediaResolution": "MEDIA_RESOLUTION_MEDIUM",
                **({"thinking_level": thinking_level} if thinking_level else {}),
            },
            "tools": [GEMINI_VIDECUT_TOOL],
        }

        print(f"[API Call] Round {round_num}, contents count: {len(contents)}")
        data, error = _make_gemini_request(url, headers, payload, timeout)

        if error:
            result["error"] = error
            break

        in_tok, out_tok = _extract_gemini_usage(data)
        result["input_tokens"] += in_tok
        result["output_tokens"] += out_tok

        if "candidates" not in data or len(data["candidates"]) == 0:
            result["error"] = "No candidates in response"
            break

        candidate = data["candidates"][0]
        finish_reason = candidate.get("finishReason", "")
        resp_parts = candidate.get("content", {}).get("parts", [])

        # Extract text and function calls from this round
        text = "\n".join(p["text"] for p in resp_parts if "text" in p).strip()
        fc_calls = [p["functionCall"] for p in resp_parts if "functionCall" in p]

        print(f"[API Response] finishReason: {finish_reason}, has_text: {bool(text)}, function_calls: {len(fc_calls)}")

        round_info = {"round": round_num}
        if text:
            round_info["answer"] = text

        # No function calls → done
        if not fc_calls:
            print("[Done] No more tool calls, exiting loop")
            result["rounds"].append(round_info)
            break

        # Append model's response to conversation
        contents.append({"role": "model", "parts": resp_parts})

        # Process each function call
        tool_calls_info = []
        function_response_parts = []

        for fc in fc_calls:
            tool_name = fc["name"]
            fc_args = fc.get("args", {})

            if tool_name == "VideoCut":
                videcut_count += 1

                try:
                    start_sec, end_sec = parse_timestamps(str(fc_args.get("timestamps", "")))
                except ValueError:
                    function_response_parts.append({
                        "functionResponse": {
                            "name": "VideoCut",
                            "response": {"content": f"Invalid timestamps format: {fc_args.get('timestamps')}. Use format 'start - end' (e.g., '10 - 20')."},
                        }
                    })
                    continue

                fps = int(fc_args.get("fps", 3))
                print(f"[VideoCut #{videcut_count}] Extracting frames: {start_sec}-{end_sec}s @ {fps}fps")

                try:
                    extra_frames = extract_frames_in_range(video_path, start_sec, end_sec, fps)
                    tool_call_info = {
                        "name": "VideoCut",
                        "params": {"start": start_sec, "end": end_sec, "fps": fps},
                        "frames_extracted": len(extra_frames),
                    }
                    tool_calls_info.append(tool_call_info)
                    result["tool_calls"].append(tool_call_info)
                    print(f"[VideoCut #{videcut_count}] Extracted {len(extra_frames)} frames")

                    # Build function response with images as inlineData
                    response_parts = [
                        {"text": f"[VideoCut Result] Extracted {len(extra_frames)} frames from {start_sec}-{end_sec}s. Use <think></think> tags to analyze these frames. If still insufficient, you can call VideoCut again with a different time range or higher fps. Otherwise, provide your final answer."}
                    ]
                    for _, img_b64 in extra_frames:
                        response_parts.append({
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": img_b64,
                            }
                        })

                    function_response_parts.append({
                        "functionResponse": {
                            "name": "VideoCut",
                            "response": {"content": response_parts},
                        }
                    })

                except Exception as e:
                    function_response_parts.append({
                        "functionResponse": {
                            "name": "VideoCut",
                            "response": {"content": f"Error extracting frames: {e}"},
                        }
                    })
            else:
                function_response_parts.append({
                    "functionResponse": {
                        "name": tool_name,
                        "response": {"content": f"Unknown tool: {tool_name}"},
                    }
                })

        if tool_calls_info:
            round_info["tool_calls"] = tool_calls_info
        result["rounds"].append(round_info)

        # Append function responses to conversation
        if function_response_parts:
            contents.append({"role": "user", "parts": function_response_parts})

    # Build content for extract_answer (use last round's answer)
    for r in reversed(result["rounds"]):
        if r.get("answer"):
            result["content"] = r["answer"]
            break

    return result


# ── Per-item Evaluation ───────────────────────────────────────────────────────

def evaluate_item(
    item: dict,
    api_url: str,
    api_key: str,
    model: str,
    input_mode: str,
    fps: float,
    max_frames: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    item_index: int,
    total_items: int,
    with_subtitle: bool = False,
    subtitle_dir: str = "",
    with_videcut: bool = False,
    max_long_edge: int = 480,
    sampling_strategy: str = "uniform",
    thinking_level: str | None = None,
) -> dict:
    """Evaluate a single VideoMME item with Gemini. Returns result dict."""
    qid = item["question_id"]
    video_path = item["video_path"]
    gt_answer = item["answer"]

    print(f"\n{'='*70}")
    print(f"[{item_index}/{total_items}] QID: {qid}")
    print(f"Video: {video_path}")
    print(f"GT Answer: {gt_answer}")
    print(f"Input Mode: {input_mode}")
    if with_subtitle:
        print(f"With Subtitle: Yes")
    if with_videcut:
        print(f"With VideoCut: Yes")
    print(f"{'='*70}")

    result = {
        "question_id": qid,
        "video_id": item.get("video_id", ""),
        "videoID": item.get("videoID", ""),
        "duration": item.get("duration", ""),
        "domain": item.get("domain", ""),
        "sub_category": item.get("sub_category", ""),
        "task_type": item.get("task_type", ""),
        "question": item["question"],
        "options": item["options"],
        "gt_answer": gt_answer,
        "model_answer": None,
        "is_correct": False,
        "model_output": "",
        "input_mode": input_mode,
        "num_frames": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None,
        "with_subtitle": with_subtitle,
        "with_videcut": with_videcut,
    }

    if not os.path.isfile(video_path):
        result["error"] = f"Video not found: {video_path}"
        print(f"[ERROR] {result['error']}")
        return result

    # Load subtitle if requested
    subtitle_text = ""
    if with_subtitle:
        if subtitle_dir and os.path.isdir(subtitle_dir):
            video_stem = Path(video_path).stem
            subtitle_path = os.path.join(subtitle_dir, f"{video_stem}.srt")
        else:
            subtitle_path = str(Path(video_path).with_suffix(".srt"))

        if os.path.isfile(subtitle_path):
            subtitle_text = read_srt_subtitle(subtitle_path)
            print(f"[OK] Loaded subtitle: {subtitle_path}")
            print(f"[INFO] Subtitle length: {len(subtitle_text)} chars")
        else:
            print(f"[WARN] Subtitle not found: {subtitle_path}")

    video_path_for_api = video_path  # default, overridden in video mode

    if input_mode == "frames":
        # Frame extraction mode
        print(f"[INFO] Extracting frames (fps={fps}, strategy={sampling_strategy})...")
        try:
            frames_b64, actual_fps, video_duration = extract_frames(
                video_path, fps=fps, max_long_edge=max_long_edge,
                max_frames=max_frames, sampling_strategy=sampling_strategy
            )
            result["num_frames"] = len(frames_b64)
            result["actual_fps"] = actual_fps
            result["video_duration"] = video_duration
            print(f"[OK] Extracted {result['num_frames']} frames (actual fps={actual_fps:.2f})")
        except Exception as e:
            result["error"] = f"Frame extraction failed: {e}"
            print(f"[ERROR] {result['error']}")
            return result

        if not frames_b64:
            result["error"] = "No frames extracted"
            print(f"[ERROR] {result['error']}")
            return result

        prompt_text = build_prompt(
            item["question"],
            item["options"],
            subtitle_text,
            with_videcut=with_videcut,
            input_type="frames",
            video_duration=video_duration,
            current_fps=actual_fps,
            num_frames=len(frames_b64),
        )
        mode_info = f"frames (requested fps={fps}, actual fps={actual_fps:.2f}), {len(frames_b64)} frames"
        call_info = f"{len(frames_b64)} frames"

    else:
        # Video mode: re-encode with target fps and resolution
        try:
            reencoded_path, original_size_mb, new_size_mb = reencode_video(
                video_path, fps=fps, max_long_edge=max_long_edge,
            )
        except Exception as e:
            result["error"] = f"Video re-encode failed: {e}"
            print(f"[ERROR] {result['error']}")
            return result

        prompt_text = build_prompt(
            item["question"],
            item["options"],
            subtitle_text,
            with_videcut=with_videcut,
            input_type="video",
        )
        # Use re-encoded video for API call, keep original path for VideoCut
        video_path_for_api = reencoded_path
        result["num_frames"] = -1  # Mark as video mode
        mode_info = f"video (original: {original_size_mb:.2f} MB → re-encoded: {new_size_mb:.2f} MB, fps={fps}, max_edge={max_long_edge})"
        call_info = f"video (original: {original_size_mb:.2f} MB → re-encoded: {new_size_mb:.2f} MB)"

    # Log prompt (compact)
    sub_info = f", subtitle={len(subtitle_text)}chars" if subtitle_text else ""
    videcut_info = ", videcut=on" if with_videcut else ""
    thinking_info = f", thinking={thinking_level}" if thinking_level else ""
    print(f"\n[PROMPT] mode={mode_info}{sub_info}{videcut_info} | max_tokens={max_tokens}, temp={temperature}, timeout={timeout}s{thinking_info}")
    prompt_preview = prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text
    print(f"[PROMPT TEXT] {prompt_preview}")
    print(f"[INFO] Calling Gemini API with {call_info}...")

    frames_arg = frames_b64 if input_mode == "frames" else None
    # For video mode: use re-encoded video for API content, original video for VideoCut
    api_video_path = video_path_for_api if input_mode == "video" else video_path
    if with_videcut:
        api_result = call_gemini_vlm(
            prompt_text=prompt_text,
            api_key=api_key,
            model=model,
            frames_b64=frames_arg,
            video_path=api_video_path,
            base_url=api_url,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            thinking_level=thinking_level,
            fc_loop=True,
        )
    else:
        api_result = call_gemini_vlm(
            prompt_text=prompt_text,
            api_key=api_key,
            model=model,
            frames_b64=frames_arg,
            video_path=api_video_path,
            base_url=api_url,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            thinking_level=thinking_level,
        )

    # Clean up re-encoded temp file
    if input_mode == "video":
        try:
            os.unlink(video_path_for_api)
        except OSError:
            pass

    # Store structured output (aligned with eval_seed)
    result["prompt"] = prompt_text
    result["model_output"] = {
        "rounds": api_result.get("rounds", []),
        "tool_calls": api_result.get("tool_calls", []),
    }
    result["input_tokens"] = api_result["input_tokens"]
    result["output_tokens"] = api_result["output_tokens"]
    result["error"] = api_result["error"]

    if result["error"]:
        print(f"[ERROR] API call failed: {result['error']}")
        return result

    # Print structured output
    print(f"\n[MODEL OUTPUT]")
    print("-" * 50)
    for r in api_result.get("rounds", []):
        print(f"[Round {r['round']}]")
        if r.get("reasoning"):
            reasoning_preview = r["reasoning"][:500] + "..." if len(r.get("reasoning", "")) > 500 else r.get("reasoning", "")
            print(f"  Reasoning: {reasoning_preview}")
        if r.get("tool_calls"):
            for tc in r["tool_calls"]:
                print(f"  Tool Call: {tc['name']} -> {tc['params']}, extracted {tc['frames_extracted']} frames")
        if r.get("answer"):
            print(f"  Answer: {r['answer']}")
    print("-" * 50)

    # Extract answer from content
    if api_result["content"]:
        model_answer = extract_answer(api_result["content"])
        result["model_answer"] = model_answer
        if model_answer:
            result["is_correct"] = model_answer == gt_answer.upper()

    print(f"\n[RESULT]")
    print(f"  Model Answer: {result['model_answer']}")
    print(f"  GT Answer:    {gt_answer}")
    print(f"  Match:        {'✓ YES' if result['is_correct'] else '✗ NO'}")
    print(f"  Tokens:       {result['input_tokens']} in / {result['output_tokens']} out")

    return result


# ── Main Evaluation Loop ──────────────────────────────────────────────────────

def _build_summary(args, total, correct, errors, total_input_tokens, total_output_tokens, elapsed):
    """Build summary dict from evaluation stats."""
    valid = total - errors
    acc = correct / valid if valid > 0 else 0  # Store as decimal (0.8922), not percentage
    api_url = getattr(args, "api_url", DEFAULT_API_URL) or DEFAULT_API_URL
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "timestamp": timestamp,
        "data_file": args.data,
        "model": args.model,
        "input_mode": args.input_mode,
        "metrics": {
            "total": total,
            "correct": correct,
            "valid": valid,
            "accuracy": round(acc, 4),
            "errors": errors,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "elapsed_seconds": round(elapsed, 1),
        },
        "config": {
            "api_url": api_url,
            "features": {
                "subtitle": {
                    "enabled": args.with_subtitle,
                    "directory": args.subtitle_dir if args.subtitle_dir else "auto-detect (same dir as video)",
                },
                "videcut": {
                    "enabled": args.with_videcut,
                },
            },
            "sampling": {
                "fps": args.fps if args.input_mode == "frames" else None,
                "strategy": args.sampling_strategy if args.input_mode == "frames" else None,
                "max_frames": args.max_frames if args.input_mode == "frames" else None,
            },
            "generation": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "thinking_level": args.thinking_level,
                "timeout": args.timeout,
            },
        },
    }


def _recalculate_summary(results: list[dict], old_summary: dict | None = None) -> dict:
    """Recalculate summary from results list."""
    total = len(results)
    correct = 0
    errors = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for r in results:
        if r.get("error") is not None:
            errors += 1
        elif r.get("is_correct"):
            correct += 1
        total_input_tokens += r.get("input_tokens", 0)
        total_output_tokens += r.get("output_tokens", 0)

    valid = total - errors
    acc = correct / valid if valid > 0 else 0  # Store as decimal (0.8922), not percentage

    # Build new summary, preserving config from old summary
    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "metrics": {
            "total": total,
            "correct": correct,
            "valid": valid,
            "accuracy": round(acc, 4),
            "errors": errors,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
    }

    # Copy other fields from old summary
    if old_summary:
        for key in ["data_file", "model", "input_mode", "config"]:
            if key in old_summary:
                summary[key] = old_summary[key]
        # Preserve elapsed_seconds if exists
        if "metrics" in old_summary and "elapsed_seconds" in old_summary["metrics"]:
            summary["metrics"]["elapsed_seconds"] = old_summary["metrics"]["elapsed_seconds"]

    return summary


def _print_summary(args, summary):
    """Print evaluation summary to console."""
    m = summary["metrics"]
    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE")
    print(f"=" * 70)
    print(f"Data:      {summary['data_file']}")
    print(f"Total:     {m['total']}")
    print(f"Correct:   {m['correct']}/{m['valid']} = {m['accuracy']*100:.2f}%")
    print(f"Errors:    {m['errors']}")
    print(f"Tokens:    {m['total_input_tokens']:,} in / {m['total_output_tokens']:,} out")
    elapsed = m["elapsed_seconds"]
    total = m["total"]
    print(f"Time:      {elapsed:.1f}s ({total/elapsed:.2f} it/s)" if elapsed > 0 else f"Time:      {elapsed:.1f}s")
    print(f"\n[Configuration]")
    print(f"Input Mode:       {args.input_mode}")
    if args.input_mode == "frames":
        print(f"  FPS:            {args.fps}")
        print(f"  Sampling:       {args.sampling_strategy}")
    print(f"Subtitle:         {args.with_subtitle} {f'(dir: {args.subtitle_dir})' if args.subtitle_dir else '(auto-detect)'}")
    print(f"VideoCut:         {args.with_videcut}")


def run_evaluation(args, output_file: Path | None = None, shared_results: list[dict] | None = None):
    """Run evaluation on a single data file.

    Args:
        args: Parsed CLI arguments.
        output_file: If provided, use this path for output instead of auto-generating.
        shared_results: If provided, append results to this list (for multi-file mode).

    Returns:
        (summary_dict, results_list) tuple.
    """
    api_url = getattr(args, "api_url", DEFAULT_API_URL) or DEFAULT_API_URL
    api_key = args.api_key or os.environ.get("VLM_API_KEY", "")

    sample_rate = getattr(args, "sample_rate", 1) or 1
    workers = getattr(args, "workers", 1) or 1
    skip_completed = getattr(args, "skip_completed", False)
    qids_file = getattr(args, "qids_file", "") or ""

    # Load qids filter if provided
    qids_filter: set[str] | None = None
    if qids_file:
        qids_filter_path = Path(qids_file)
        if qids_filter_path.exists():
            with open(qids_filter_path, "r", encoding="utf-8") as f:
                qids_filter = set(line.strip() for line in f if line.strip())
            print(f"[INFO] Loaded {len(qids_filter)} question IDs from {qids_file}")
        else:
            print(f"[WARN] QIDs file not found: {qids_file}, ignoring filter")

    # Load data
    items = []
    if qids_filter:
        # Load all data, filter by qids_filter
        with open(args.data, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if item.get("question_id") in qids_filter:
                        items.append(item)
        print(f"[INFO] Loaded {len(items)} items matching qids_filter from {args.data}")
    else:
        total_line_nums = sample_rate * args.limit if args.limit else 0
        with open(args.data, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if total_line_nums and i >= total_line_nums:
                    break
                if i % sample_rate == 0:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        if args.limit:
            items = items[: args.limit]

    print(f"{'='*70}")
    print(f"Gemini Evaluation")
    print(f"{'='*70}")
    print(f"Data: {args.data}")
    if qids_filter:
        print(f"QIDs Filter: {qids_file} ({len(qids_filter)} IDs)")
    print(f"Items: {len(items)}")
    if sample_rate > 1:
        print(f"Sample Rate: every {sample_rate}th item")
    if workers > 1:
        print(f"Workers: {workers}")
    print(f"API URL: {api_url}")
    print(f"Model: {args.model}")
    print(f"Input Mode: {args.input_mode}")
    if args.input_mode == "frames":
        print(f"FPS: {args.fps}")
        print(f"Sampling Strategy: {args.sampling_strategy}")
    if args.with_subtitle:
        subtitle_info = f"Subtitle Dir: {args.subtitle_dir}" if args.subtitle_dir else "Subtitle Dir: auto-detect"
        print(f"With Subtitle: Yes ({subtitle_info})")
    if args.with_videcut:
        print(f"With VideoCut: Yes")
    print()

    # Output file - with configuration tags
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_stem = Path(args.data).stem
        out_dir = Path(args.output_dir) / args.model
        out_dir.mkdir(parents=True, exist_ok=True)

        config_parts = [args.input_mode]
        if args.input_mode == "frames":
            config_parts.append(args.sampling_strategy)
        if args.with_subtitle:
            config_parts.append("sub")
        if args.with_videcut:
            config_parts.append("videcut")

        config_str = "_".join(config_parts)
        output_file = out_dir / f"{data_stem}_{args.model}_{config_str}_{timestamp}.json"

    # Load completed question_ids and results for resume
    completed_ids: set[str] = set()
    all_results: list[dict] = shared_results if shared_results is not None else []
    # Track initial count for multi-file mode (to calculate per-file stats correctly)
    initial_results_count = len(all_results)

    # Extract completed_ids from shared_results (for multi-file resume)
    if shared_results:
        for rec in shared_results:
            completed_ids.add(rec["question_id"])
        if completed_ids:
            print(f"[INFO] {len(completed_ids)} items already completed from previous results")

    if skip_completed:
        loaded_ids = load_completed_ids_from_output(args.output_dir, args.model)
        if loaded_ids:
            completed_ids.update(loaded_ids)
            print(f"Skip-completed: found {len(loaded_ids)} completed items in {args.output_dir}/{args.model}/")

    if args.resume_file:
        resume_path = Path(args.resume_file)
        if resume_path.exists():
            with open(resume_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_records = data.get("results", [])
                old_summary = data.get("summary", None)
                total_count = len(all_records)
                success_count = 0
                failed_count = 0
                skipped_dup = 0
                skipped_not_in_data = 0
                # Build set of valid question_ids from current items for filtering
                item_qids = {it["question_id"] for it in items} if shared_results is None else None
                seen_qids = set(completed_ids)  # already-seen qids (from shared_results or skip_completed)
                for rec in all_records:
                    has_error = rec.get("error") is not None
                    has_answer = rec.get("model_answer") is not None
                    qid = rec.get("question_id", "")
                    if not has_error and has_answer:
                        # Dedup: skip if already in all_results
                        if qid in seen_qids:
                            skipped_dup += 1
                            continue
                        # Filter: skip if not in current data (single-file mode only)
                        if item_qids is not None and qid not in item_qids:
                            skipped_not_in_data += 1
                            continue
                        seen_qids.add(qid)
                        completed_ids.add(qid)
                        all_results.append(rec)
                        success_count += 1
                    else:
                        print(f"  [DEBUG] Removing QID={qid}: error={has_error}, answer={has_answer}")
                        failed_count += 1
            output_file = resume_path
            print(f"Resuming from: {resume_path}")
            print(f"  - Total {total_count} cases")
            print(f"  - Keeping {success_count} successful (no error + has answer)")
            print(f"  - Removing {failed_count} failed/incomplete")
            if skipped_dup > 0:
                print(f"  - Skipped {skipped_dup} duplicate question_ids")
            if skipped_not_in_data > 0:
                print(f"  - Skipped {skipped_not_in_data} results not in current data")
            print(f"  - all_results now has {len(all_results)} items")

            if failed_count > 0 or skipped_dup > 0 or skipped_not_in_data > 0:
                new_summary = _recalculate_summary(all_results, old_summary)
                _save_output_file(output_file, all_results, new_summary)
                print(f"  - Cleaned file saved to: {output_file}")
                metrics = new_summary.get("metrics", new_summary)
                print(f"  - New accuracy: {metrics['accuracy']*100:.2f}% ({metrics['correct']}/{metrics['valid']})")
        else:
            print(f"Resume file not found: {resume_path}, starting fresh → {output_file}")
    else:
        print(f"Output: {output_file}")

    # Filter out completed items
    pending = [it for it in items if it["question_id"] not in completed_ids]
    print(f"Pending: {len(pending)} items\n")

    if not pending:
        print("All items already completed!")
        if all_results:
            summary = _recalculate_summary(all_results)
            summary.setdefault("data_file", args.data)
            summary.setdefault("model", args.model)
            summary.setdefault("input_mode", args.input_mode)
            summary["metrics"].setdefault("elapsed_seconds", 0.0)
            _print_summary(args, summary)
            return summary, all_results
        return _build_summary(args, 0, 0, 0, 0, 0, 0.0), all_results

    # Stats — base counts from resumed results for cumulative tracking
    resumed_count = len(all_results)
    resumed_correct = sum(1 for r in all_results if r.get("is_correct"))
    resumed_errors = sum(1 for r in all_results if r.get("error") is not None)

    total = len(pending)
    done = 0
    correct = 0
    errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    # Common kwargs for evaluate_item
    eval_kwargs = dict(
        api_url=api_url, api_key=api_key, model=args.model,
        input_mode=args.input_mode,
        fps=args.fps,
        max_frames=getattr(args, "max_frames", 1024),
        sampling_strategy=getattr(args, "sampling_strategy", "uniform"),
        max_tokens=args.max_tokens, temperature=args.temperature,
        timeout=args.timeout,
        with_subtitle=args.with_subtitle,
        subtitle_dir=args.subtitle_dir,
        with_videcut=args.with_videcut,
        max_long_edge=getattr(args, "max_long_edge", 480),
        thinking_level=getattr(args, "thinking_level", None),
    )

    def _process_result(result):
        nonlocal done, correct, errors, total_input_tokens, total_output_tokens
        with _result_lock:
            done += 1
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            if result["error"]:
                errors += 1
            elif result["is_correct"]:
                correct += 1
            all_results.append(result)

            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            valid = done - errors
            acc = correct / valid * 100 if valid > 0 else 0
            if resumed_count > 0:
                cum_correct = resumed_correct + correct
                cum_total = resumed_count + done
                cum_errors = resumed_errors + errors
                cum_valid = cum_total - cum_errors
                cum_acc = cum_correct / cum_valid * 100 if cum_valid > 0 else 0
                print(f"\n[SUMMARY] Run: {acc:.1f}% ({correct}/{valid}) | All: {cum_acc:.1f}% ({cum_correct}/{cum_valid}) | Errors: {errors} | Speed: {rate:.2f} it/s")
            else:
                print(f"\n[SUMMARY] Acc: {acc:.1f}% ({correct}/{valid}) | Errors: {errors} | Speed: {rate:.2f} it/s")

            if done % 5 == 0:
                _save_output_file(output_file, all_results, None)

            if done % 20 == 0:
                qb = getattr(args, "quota_before", None)
                if qb is not None:
                    api_key = args.api_key or os.environ.get("VLM_API_KEY", "")
                    qn = _query_remain_quota(api_key, getattr(args, "quota_url", ""))
                    if qn is not None:
                        cost = qb - qn
                        avg = cost / done
                        print(f"[QUOTA] used={cost:.4f}, avg/case={avg:.4f}, remain={qn:.4f}")

    if workers > 1:
        print(f"[INFO] Using {workers} parallel workers")
        _tl_stdout = _ThreadLocalStdout(sys.stdout)
        sys.stdout = _tl_stdout

        def _worker_fn(item, idx):
            _tl_stdout.start_buffering()
            try:
                return evaluate_item(
                    item, **eval_kwargs,
                    item_index=idx, total_items=total,
                )
            finally:
                _tl_stdout.flush_buffer()

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for idx, item in enumerate(pending, 1):
                    future = executor.submit(_worker_fn, item, idx)
                    futures[future] = idx

                for future in as_completed(futures):
                    result = future.result()
                    _process_result(result)
        finally:
            sys.stdout = _tl_stdout._real
    else:
        for idx, item in enumerate(pending, 1):
            result = evaluate_item(
                item, **eval_kwargs,
                item_index=idx, total_items=total,
            )
            _process_result(result)

    # Final summary - recalculate from results
    elapsed = time.time() - start_time

    # In multi-file mode, only count results from THIS file (not cumulative)
    if shared_results is not None:
        current_file_results = all_results[initial_results_count:]
    else:
        current_file_results = all_results

    final_total = len(current_file_results)
    final_correct = sum(1 for r in current_file_results if r.get("is_correct"))
    final_errors = sum(1 for r in current_file_results if r.get("error") is not None)
    final_input_tokens = sum(r.get("input_tokens", 0) for r in current_file_results)
    final_output_tokens = sum(r.get("output_tokens", 0) for r in current_file_results)

    summary = _build_summary(args, final_total, final_correct, final_errors, final_input_tokens, final_output_tokens, elapsed)

    # Inject quota info into summary
    quota_before = getattr(args, "quota_before", None)
    if quota_before is not None:
        api_key = args.api_key or os.environ.get("VLM_API_KEY", "")
        quota_after = _query_remain_quota(api_key, getattr(args, "quota_url", ""))
        args.quota_after = quota_after
        cost = round(quota_before - quota_after, 4) if quota_after is not None else None
        summary["quota"] = {"before": round(quota_before, 4), "after": round(quota_after, 4) if quota_after is not None else None, "cost": cost}

    _print_summary(args, summary)
    if done < final_total:
        print(f"\n[This Run] Processed {done} items (total: {final_total})")

    # Write output file (skip in multi-file mode, handled by caller)
    if shared_results is None:
        _save_output_file(output_file, all_results, summary)
        print(f"\n[Output File]")
        print(f"  {output_file}")
    else:
        # Multi-file mode: periodic save for crash recovery
        _save_output_file(output_file, all_results, None)

    return summary, all_results


def _build_per_file_summaries(all_results: list[dict], data_files: list[str]) -> list[dict]:
    """Build per-file summaries by matching results to data files via question_id."""
    # Load question_ids for each data file
    file_qids = {}
    for data_file in data_files:
        qids = set()
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        qids.add(item.get("question_id", ""))
        except Exception:
            pass
        file_qids[data_file] = qids

    # Build per-file summaries
    summaries = []
    matched_qids = set()
    for data_file in data_files:
        qids = file_qids[data_file]
        file_results = [r for r in all_results if r.get("question_id") in qids]
        matched_qids.update(r.get("question_id") for r in file_results)

        total = len(file_results)
        correct = sum(1 for r in file_results if r.get("is_correct"))
        errors = sum(1 for r in file_results if r.get("error") is not None)
        valid = total - errors
        acc = correct / valid if valid > 0 else 0
        tokens_in = sum(r.get("input_tokens", 0) for r in file_results)
        tokens_out = sum(r.get("output_tokens", 0) for r in file_results)

        summaries.append({
            "data_file": data_file,
            "metrics": {
                "total": total,
                "correct": correct,
                "valid": valid,
                "accuracy": round(acc, 4),
                "errors": errors,
                "total_input_tokens": tokens_in,
                "total_output_tokens": tokens_out,
            },
        })

    # Handle results not matching any data file
    unmatched = [r for r in all_results if r.get("question_id") not in matched_qids]
    if unmatched:
        total = len(unmatched)
        correct = sum(1 for r in unmatched if r.get("is_correct"))
        errors = sum(1 for r in unmatched if r.get("error") is not None)
        valid = total - errors
        acc = correct / valid if valid > 0 else 0
        summaries.append({
            "data_file": "(unmatched)",
            "metrics": {
                "total": total, "correct": correct, "valid": valid,
                "accuracy": round(acc, 4), "errors": errors,
                "total_input_tokens": sum(r.get("input_tokens", 0) for r in unmatched),
                "total_output_tokens": sum(r.get("output_tokens", 0) for r in unmatched),
            },
        })

    return summaries


def run_multi_file_evaluation(args, data_files: list[str]):
    """Run evaluation across multiple data files, saving all results to a single output file."""
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    all_summaries = []
    completed_ids: set[str] = set()

    # Handle resume: use existing file instead of creating new one
    if args.resume_file:
        resume_path = Path(args.resume_file)
        if resume_path.exists():
            # Build set of valid qids from all data files for filtering
            all_valid_qids = set()
            qids_file = getattr(args, "qids_file", "") or ""
            qids_filter: set[str] | None = None
            if qids_file and Path(qids_file).exists():
                with open(qids_file, "r", encoding="utf-8") as f:
                    qids_filter = set(line.strip() for line in f if line.strip())
            for data_file in data_files:
                try:
                    with open(data_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                item = json.loads(line)
                                qid = item.get("question_id", "")
                                if qids_filter is None or qid in qids_filter:
                                    all_valid_qids.add(qid)
                except Exception:
                    pass

            with open(resume_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_records = data.get("results", [])
                old_summary = data.get("summary", None)
                success_count = 0
                failed_count = 0
                skipped_dup = 0
                skipped_not_in_data = 0
                seen_qids: set[str] = set()
                for rec in all_records:
                    has_error = rec.get("error") is not None
                    has_answer = rec.get("model_answer") is not None
                    qid = rec.get("question_id", "")
                    if not has_error and has_answer:
                        if qid in seen_qids:
                            skipped_dup += 1
                            continue
                        if all_valid_qids and qid not in all_valid_qids:
                            skipped_not_in_data += 1
                            continue
                        seen_qids.add(qid)
                        completed_ids.add(qid)
                        all_results.append(rec)
                        success_count += 1
                    else:
                        failed_count += 1
            output_file = resume_path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"[MULTI] Resuming from: {resume_path}")
            print(f"  - Keeping {success_count} successful, removing {failed_count} failed")
            if skipped_dup > 0:
                print(f"  - Skipped {skipped_dup} duplicate question_ids")
            if skipped_not_in_data > 0:
                print(f"  - Skipped {skipped_not_in_data} results not in current data files")

            if failed_count > 0 or skipped_dup > 0 or skipped_not_in_data > 0:
                new_summary = _recalculate_summary(all_results, old_summary)
                _save_output_file(output_file, all_results, new_summary)
                print(f"  - Cleaned file saved, new accuracy: {new_summary['metrics']['accuracy']*100:.2f}%")
        else:
            print(f"[WARN] Resume file not found: {resume_path}, creating new file")
            args.resume_file = ""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_stem = Path(args.config).stem if args.config else "multi"
            config_parts = [args.input_mode]
            if args.input_mode == "frames":
                config_parts.append(args.sampling_strategy)
            if args.with_subtitle:
                config_parts.append("sub")
            if args.with_videcut:
                config_parts.append("videcut")
            config_str = "_".join(config_parts)
            output_file = out_dir / f"{config_stem}_{args.model}_{config_str}_{timestamp}.json"
    else:
        # No resume - create new output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_stem = Path(args.config).stem if args.config else "multi"
        config_parts = [args.input_mode]
        if args.input_mode == "frames":
            config_parts.append(args.sampling_strategy)
        if args.with_subtitle:
            config_parts.append("sub")
        if args.with_videcut:
            config_parts.append("videcut")
        config_str = "_".join(config_parts)
        output_file = out_dir / f"{config_stem}_{args.model}_{config_str}_{timestamp}.json"

    # Clear resume_file in args so run_evaluation doesn't process it again
    original_resume_file = args.resume_file
    args.resume_file = ""

    print(f"[CONFIG] Output file: {output_file}")

    for i, data_file in enumerate(data_files, 1):
        print(f"\n{'#'*70}")
        print(f"# File {i}/{len(data_files)}: {data_file}")
        print(f"{'#'*70}\n")

        args.data = data_file
        summary, _ = run_evaluation(args, output_file=output_file, shared_results=all_results)
        if summary:
            all_summaries.append(summary)

    # Build combined summary — recalculate from all_results to include resumed results
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get("is_correct"))
    errors_total = sum(1 for r in all_results if r.get("error") is not None)
    valid = total - errors_total
    tokens_in = sum(r.get("input_tokens", 0) for r in all_results)
    tokens_out = sum(r.get("output_tokens", 0) for r in all_results)
    elapsed = sum(s["metrics"].get("elapsed_seconds", 0) for s in all_summaries)
    acc = correct / valid if valid > 0 else 0  # Store as decimal

    # Per-file summaries: rebuild from all_results to include resumed results
    per_file_summaries = _build_per_file_summaries(all_results, data_files)

    combined_summary = {
        "timestamp": timestamp,
        "model": args.model,
        "input_mode": args.input_mode,
        "data_files": data_files,
        "per_file_summaries": per_file_summaries,
        "metrics": {
            "total": total,
            "correct": correct,
            "valid": valid,
            "accuracy": round(acc, 4),
            "errors": errors_total,
            "total_input_tokens": tokens_in,
            "total_output_tokens": tokens_out,
            "elapsed_seconds": round(elapsed, 1),
        },
    }

    # Write final combined output
    _save_output_file(output_file, all_results, combined_summary)

    print(f"\n{'='*70}")
    print(f"COMBINED RESULTS ({len(data_files)} files)")
    print(f"{'='*70}")
    for s in per_file_summaries:
        m = s["metrics"]
        print(f"  {s['data_file']}: {m['accuracy']*100:.2f}% ({m['correct']}/{m['valid']})")
    print(f"  {'─'*50}")
    print(f"  Overall: {acc:.4f} ({correct}/{valid}) | Errors: {errors_total}")
    print(f"  Tokens:  {tokens_in:,} in / {tokens_out:,} out")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"\n[Output File]")
    print(f"  {output_file}")


def main():
    _save_terminal_settings()

    parser = argparse.ArgumentParser(
        description="VideoMME Benchmark Evaluation — Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file (iterates over data_files, uses max_samples/sample_rate/workers)
  python eval_mllms/eval_gemini.py -c exp_configs/videomme_exp.yaml

  # Run with config + export data first
  python eval_mllms/eval_gemini.py -c exp_configs/videomme_exp.yaml --export

  # Single data file (backward compatible)
  python eval_mllms/eval_gemini.py --data data/videomme/test/short_duration_data.jsonl --limit 3

  # Filter by specific question IDs (ignores limit/sample_rate)
  python eval_mllms/eval_gemini.py --data data.jsonl --qids-file data/sampled_qids/sampled_150.txt

  # Parallel workers + skip completed
  python eval_mllms/eval_gemini.py --data data.jsonl -w 4 --skip-completed

  # Disable subtitles
  python eval_mllms/eval_gemini.py --limit 3 --no-subtitle

  # Test with VideoCut tool
  python eval_mllms/eval_gemini.py --input-mode video --limit 3 --with-videcut

  # Resume from previous run
  python eval_mllms/eval_gemini.py --resume-file eval_mllm/model/xxx.json
        """,
    )

    # Config
    parser.add_argument("--config", "-c", default="", help="Path to experiment config YAML file")
    parser.add_argument("--export", action="store_true", help="Run export_commands from config before evaluation")

    # Data
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to VideoMME JSONL data file")
    parser.add_argument("--qids-file", default="", help="Path to file containing question IDs to evaluate (one per line, e.g., '602-1')")
    parser.add_argument("--limit", type=int, default=3, help="Limit number of items (default: 3, 0=all)")
    parser.add_argument("--sample-rate", "-r", type=int, default=1, help="Sample rate: evaluate every N-th item (default: 1)")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Number of parallel workers (default: 8)")
    parser.add_argument("--skip-completed", action="store_true", help="Auto-skip completed items by scanning existing output files")

    # Input mode
    parser.add_argument(
        "--input-mode",
        choices=["video", "frames"],
        default="video",
        help="Input mode: 'frames' (extract frames) or 'video' (native video, default: video)"
    )

    # API
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help=f"Gemini API base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""),
                        help="API key (or set VLM_API_KEY env var)")
    parser.add_argument(
        "--quota-url", default=None,
        help="Optional quota endpoint (CLI > YAML quota_url > VLM_QUOTA_URL). "
             "Disabled when unset; pass an empty string to disable. Uses the evaluator API key.",
    )
    parser.add_argument("--model", default="gemini-3.1-pro-preview",
                        help="Gemini model name")

    # Frame extraction (only used when input-mode=frames)
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample")
    parser.add_argument("--max-frames", type=int, default=1024, help="Maximum number of frames to extract (default: 1024)")
    parser.add_argument("--max-long-edge", type=int, default=480, help="Resize frames so long edge <= this value (default: 480)")
    parser.add_argument("--sampling-strategy", choices=["adaptive", "uniform"], default="uniform",
                        help="Frame sampling strategy: 'adaptive' reduces fps for long videos, 'uniform' uses 1fps by default (default: uniform)")

    # Generation
    parser.add_argument("--max-tokens", type=int, default=8196, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--thinking-level", type=str, default='high', choices=["off", "low", "medium", "high"],
                        help="Gemini thinking level (default: None, not set)")
    parser.add_argument("--timeout", type=float, default=600.0, help="API timeout in seconds")

    # Output
    parser.add_argument("--output-dir", default="eval_mllm", help="Output directory")

    # Subtitle (enabled by default)
    parser.add_argument(
        "--no-subtitle",
        action="store_true",
        help="Disable subtitle inclusion (subtitles are included by default)"
    )
    parser.add_argument(
        "--subtitle-dir",
        default="",
        help="Directory containing subtitle files (empty = auto-detect from data path)"
    )

    # VideoCut tool (optional)
    parser.add_argument(
        "--with-videcut",
        action="store_true",
        help="Enable VideoCut tool for high-density frame sampling (Gemini function calling)"
    )

    # Resume
    parser.add_argument("--resume-file", default="",
                        help="Resume from this result JSON file (re-run failed cases)")

    args = parser.parse_args()

    # Subtitle is enabled by default, --no-subtitle disables it
    args.with_subtitle = not args.no_subtitle

    # Load config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {args.config}")
            return
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        print(f"[CONFIG] Loaded: {args.config}")

        # Map config fields to args (config overrides defaults, CLI overrides config)
        if "max_samples" in config:
            args.limit = config["max_samples"]
        if "sample_rate" in config:
            args.sample_rate = config["sample_rate"]
        if "workers" in config:
            args.workers = config["workers"]
        if "skip_completed" in config:
            args.skip_completed = config["skip_completed"]
        if "qids_file" in config and not args.qids_file:
            args.qids_file = config["qids_file"]
        if "model" in config:
            args.model = config["model"]
        if "input_mode" in config:
            args.input_mode = config["input_mode"]
        if "with_subtitle" in config:
            args.with_subtitle = config["with_subtitle"]
        if "subtitle_dir" in config:
            args.subtitle_dir = config["subtitle_dir"]

        try:
            quota_before = _start_quota_tracking(args, config)
        except ValueError as error:
            parser.error(str(error))

        # Auto-detect subtitle_dir from data_files when not explicitly set
        if not args.subtitle_dir:
            data_files_for_detect = config.get("data_files", [])
            detect_path = data_files_for_detect[0] if data_files_for_detect else args.data
            inferred = infer_subtitle_dir(detect_path)
            if inferred is None:
                args.with_subtitle = False
                print(f"[CONFIG] Auto-detected: no subtitles for this dataset")
            elif inferred:
                args.subtitle_dir = inferred
                print(f"[CONFIG] Auto-detected subtitle_dir: {inferred}")

        # Run export commands if --export flag is set
        import subprocess
        if args.export and config.get("export_commands"):
            for cmd in config["export_commands"]:
                print(f"[EXPORT] Running: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
            print()

        # Multi-file or single-file evaluation
        data_files = config.get("data_files", [])
        if data_files:
            run_multi_file_evaluation(args, data_files)
        else:
            run_evaluation(args)
    else:
        try:
            quota_before = _start_quota_tracking(args)
        except ValueError as error:
            parser.error(str(error))
        # Auto-detect subtitle_dir from data path when not explicitly set
        if not args.subtitle_dir:
            inferred = infer_subtitle_dir(args.data)
            if inferred is None:
                args.with_subtitle = False
                print(f"[INFO] Auto-detected: no subtitles for this dataset")
            elif inferred:
                args.subtitle_dir = inferred
                print(f"[INFO] Auto-detected subtitle_dir: {inferred}")
        run_evaluation(args)

    # Print quota summary
    quota_after = getattr(args, "quota_after", None)
    if quota_before is not None and quota_after is not None:
        print(f"[QUOTA] after evaluation: {quota_after:.4f}, total cost: {quota_before - quota_after:.4f}")
    elif quota_after is not None:
        print(f"[QUOTA] after evaluation: {quota_after:.4f}")


if __name__ == "__main__":
    main()
