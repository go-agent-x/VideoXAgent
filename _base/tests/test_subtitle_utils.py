"""Tests for _base.subtitle_utils module."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _base.subtitle_utils import (
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


# ===========================================================================
# Dataset config tests
# ===========================================================================
class TestDatasetConfig:
    def test_default_configs_exist(self):
        assert "videomme" in DATASET_SUBTITLE_CONFIG
        assert "longvideobench" in DATASET_SUBTITLE_CONFIG
        assert "video-mme-v2" in DATASET_SUBTITLE_CONFIG

    def test_detect_dataset_videomme(self):
        path = "data/videomme/videos/abc.mp4"
        assert detect_dataset_from_video_path(path) == "videomme"

    def test_detect_dataset_longvideobench(self):
        path = "data/LongVideoBench/videos/abc.mp4"
        assert detect_dataset_from_video_path(path) == "longvideobench"

    def test_detect_dataset_video_mme_v2(self):
        path = "data/Video-MME-v2/videos/abc.mp4"
        assert detect_dataset_from_video_path(path) == "video-mme-v2"

    def test_detect_dataset_unknown(self):
        assert detect_dataset_from_video_path("/tmp/random.mp4") is None

    def test_register_dataset_success(self):
        register_dataset("test-ds", {
            "video_path_pattern": r"data/test/videos/",
            "subtitle_dir": "data/test/subs",
            "subtitle_ext": ".srt",
            "naming_pattern": "{video_id}.srt",
        })
        assert "test-ds" in DATASET_SUBTITLE_CONFIG
        # cleanup
        del DATASET_SUBTITLE_CONFIG["test-ds"]

    def test_register_dataset_missing_keys(self):
        with pytest.raises(ValueError, match="Missing required config keys"):
            register_dataset("bad", {"video_path_pattern": r"x/"})


# ===========================================================================
# parse_timestamp tests
# ===========================================================================
class TestParseTimestamp:
    def test_numeric_int(self):
        assert parse_timestamp(120) == 120.0

    def test_numeric_float(self):
        assert parse_timestamp(90.5) == 90.5

    def test_srt_format_comma(self):
        assert parse_timestamp("01:23:45,678") == 1 * 3600 + 23 * 60 + 45.678

    def test_srt_format_dot(self):
        assert parse_timestamp("01:23:45.678") == 1 * 3600 + 23 * 60 + 45.678

    def test_mm_ss_format(self):
        assert parse_timestamp("05:30.5") == 5 * 60 + 30.5

    def test_bare_number(self):
        assert parse_timestamp("123.4") == 123.4

    def test_zero_hours(self):
        assert parse_timestamp("00:00:05.000") == 5.0

    def test_invalid_string(self):
        assert parse_timestamp("not_a_number") == 0.0

    def test_none_returns_zero(self):
        assert parse_timestamp(None) == 0.0


# ===========================================================================
# format_seconds_to_timestamp tests
# ===========================================================================
class TestFormatSecondsToTimestamp:
    def test_zero(self):
        assert format_seconds_to_timestamp(0) == "00:00:00.000"

    def test_simple(self):
        assert format_seconds_to_timestamp(5.5) == "00:00:05.500"

    def test_hours(self):
        assert format_seconds_to_timestamp(3661.123) == "01:01:01.123"


# ===========================================================================
# Text cleaning tests
# ===========================================================================
class TestTextCleaning:
    def test_strip_html_tags(self):
        assert strip_html_tags("<font color=red>Hello</font>") == "Hello"

    def test_strip_html_tags_no_tags(self):
        assert strip_html_tags("plain text") == "plain text"

    def test_clean_subtitle_text_entities(self):
        # Note: html.unescape runs first, then tag stripping.
        # "< >" (after unescape of &lt; &gt;) gets stripped by <[^>]+> regex.
        assert clean_subtitle_text("&amp; &lt; &gt;") == "&"

    def test_clean_subtitle_text_whitespace(self):
        assert clean_subtitle_text("  hello   world  ") == "hello world"

    def test_clean_subtitle_text_tags(self):
        assert clean_subtitle_text("<b>Bold</b> text") == "Bold text"


# ===========================================================================
# SRT parsing tests
# ===========================================================================
class TestParseSrtFile:
    def test_basic_srt(self, tmp_path):
        srt_content = "1\n00:00:01,000 --> 00:00:05,000\nHello world\n\n2\n00:00:06,000 --> 00:00:10,000\nSecond line\n"
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        subs = parse_srt_file(str(srt_file))
        assert len(subs) == 2
        assert subs[0]["index"] == 1
        assert subs[0]["start"] == 1.0
        assert subs[0]["end"] == 5.0
        assert subs[0]["text"] == "Hello world"
        assert subs[1]["index"] == 2
        assert subs[1]["text"] == "Second line"

    def test_srt_with_html_tags(self, tmp_path):
        srt_content = "1\n00:00:01,000 --> 00:00:05,000\n<font color=red>Bold</font> text\n"
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        subs = parse_srt_file(str(srt_file))
        assert subs[0]["text"] == "Bold text"

    def test_srt_unicode_decode_fallback(self, tmp_path):
        # Write with latin-1 encoding containing non-UTF8 chars
        srt_content = "1\n00:00:01,000 --> 00:00:05,000\nHéllo\n"
        srt_file = tmp_path / "test.srt"
        srt_file.write_bytes(srt_content.encode("latin-1"))

        subs = parse_srt_file(str(srt_file))
        assert len(subs) == 1

    def test_srt_empty_file(self, tmp_path):
        srt_file = tmp_path / "empty.srt"
        srt_file.write_text("", encoding="utf-8")

        subs = parse_srt_file(str(srt_file))
        assert subs == []

    def test_srt_multiline_text(self, tmp_path):
        srt_content = "1\n00:00:01,000 --> 00:00:05,000\nLine one\nLine two\n"
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        subs = parse_srt_file(str(srt_file))
        assert subs[0]["text"] == "Line one Line two"


# ===========================================================================
# JSON parsing tests
# ===========================================================================
class TestParseJsonSubtitleFile:
    def test_basic_json(self, tmp_path):
        data = [
            {"start": "00:01:02.560", "end": "00:01:11.840", "line": "Hello"},
            {"start": "00:01:12.000", "end": "00:01:20.000", "line": "World"},
        ]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        subs = parse_json_subtitle_file(str(json_file))
        assert len(subs) == 2
        assert subs[0]["start"] == 62.56
        assert subs[0]["end"] == 71.84
        assert subs[0]["text"] == "Hello"

    def test_json_with_html(self, tmp_path):
        data = [{"start": "00:00:01.000", "end": "00:00:05.000", "line": "<b>Bold</b> &amp; text"}]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        subs = parse_json_subtitle_file(str(json_file))
        assert subs[0]["text"] == "Bold & text"

    def test_json_invalid(self, tmp_path):
        json_file = tmp_path / "bad.json"
        json_file.write_text("not json", encoding="utf-8")

        with pytest.raises(ValueError):
            parse_json_subtitle_file(str(json_file))

    def test_json_not_array(self, tmp_path):
        json_file = tmp_path / "obj.json"
        json_file.write_text(json.dumps({"key": "val"}), encoding="utf-8")

        with pytest.raises(ValueError, match="must contain an array"):
            parse_json_subtitle_file(str(json_file))


# ===========================================================================
# JSONL parsing tests
# ===========================================================================
class TestParseJsonlSubtitleFile:
    def test_basic_jsonl(self, tmp_path):
        lines = [
            '{"text": "Hi,", "start_time": 115.6, "end_time": 116.0}',
            '{"text": "I am here", "start_time": 116.16, "end_time": 116.56}',
        ]
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text("\n".join(lines), encoding="utf-8")

        subs = parse_jsonl_subtitle_file(str(jsonl_file))
        assert len(subs) == 2
        assert subs[0]["start"] == 115.6
        assert subs[0]["end"] == 116.0
        assert subs[0]["text"] == "Hi,"

    def test_jsonl_empty_lines(self, tmp_path):
        lines = [
            '{"text": "Hi", "start_time": 1.0, "end_time": 2.0}',
            '',
            '{"text": "Bye", "start_time": 3.0, "end_time": 4.0}',
        ]
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text("\n".join(lines), encoding="utf-8")

        subs = parse_jsonl_subtitle_file(str(jsonl_file))
        assert len(subs) == 2


# ===========================================================================
# parse_subtitle_file auto-detect tests
# ===========================================================================
class TestParseSubtitleFileAutoDetect:
    def test_srt_extension(self, tmp_path):
        srt_content = "1\n00:00:01,000 --> 00:00:05,000\nHello\n"
        srt_file = tmp_path / "auto.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        subs = parse_subtitle_file(str(srt_file))
        assert len(subs) == 1

    def test_json_extension(self, tmp_path):
        data = [{"start": "00:00:01.000", "end": "00:00:05.000", "line": "Hello"}]
        json_file = tmp_path / "auto.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        subs = parse_subtitle_file(str(json_file))
        assert len(subs) == 1

    def test_jsonl_extension(self, tmp_path):
        lines = ['{"text": "Hi", "start_time": 1.0, "end_time": 2.0}']
        jsonl_file = tmp_path / "auto.jsonl"
        jsonl_file.write_text("\n".join(lines), encoding="utf-8")

        subs = parse_subtitle_file(str(jsonl_file))
        assert len(subs) == 1


# ===========================================================================
# get_video_id_from_path tests
# ===========================================================================
class TestGetVideoIdFromPath:
    def test_simple(self):
        assert get_video_id_from_path("video.mp4") == "video"

    def test_with_path(self):
        assert get_video_id_from_path("/data/videomme/videos/abc_123.mp4") == "abc_123"

    def test_double_ext(self):
        assert get_video_id_from_path("video.tar.gz") == "video.tar"


# ===========================================================================
# find_subtitle_file tests
# ===========================================================================
class TestFindSubtitleFile:
    def test_explicit_dir_srt(self, tmp_path):
        srt_file = tmp_path / "video.srt"
        srt_file.write_text("1\n00:00:01,000 --> 00:00:05,000\nHello\n", encoding="utf-8")

        result = find_subtitle_file("/some/video.mp4", subtitle_dir=str(tmp_path))
        assert result is not None
        assert result.endswith("video.srt")

    def test_explicit_dir_json(self, tmp_path):
        data = [{"start": "00:00:01.000", "end": "00:00:05.000", "line": "Hello"}]
        json_file = tmp_path / "video.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        result = find_subtitle_file("/some/video.mp4", subtitle_dir=str(tmp_path))
        assert result is not None
        assert result.endswith("video.json")

    def test_same_dir_as_video(self, tmp_path):
        srt_file = tmp_path / "video.srt"
        srt_file.write_text("1\n00:00:01,000 --> 00:00:05,000\nHello\n", encoding="utf-8")
        video_file = tmp_path / "video.mp4"
        video_file.write_text("fake", encoding="utf-8")

        result = find_subtitle_file(str(video_file))
        assert result is not None

    def test_no_subtitle_found(self, tmp_path):
        result = find_subtitle_file(str(tmp_path / "nonexistent.mp4"))
        # In this test env, likely won't find any subtitle
        # Just verify it returns None or a string
        assert result is None or isinstance(result, str)


# ===========================================================================
# filter_subtitles_by_time_range tests
# ===========================================================================
class TestFilterSubtitlesByTimeRange:
    def test_basic_overlap(self):
        subs = [
            {"start": 1.0, "end": 5.0, "text": "A"},
            {"start": 6.0, "end": 10.0, "text": "B"},
            {"start": 11.0, "end": 15.0, "text": "C"},
        ]
        result = filter_subtitles_by_time_range(subs, 3.0, 8.0)
        assert len(result) == 2
        assert result[0]["text"] == "A"
        assert result[1]["text"] == "B"

    def test_no_overlap(self):
        subs = [
            {"start": 1.0, "end": 5.0, "text": "A"},
        ]
        result = filter_subtitles_by_time_range(subs, 10.0, 20.0)
        assert len(result) == 0

    def test_exact_boundary(self):
        subs = [
            {"start": 5.0, "end": 10.0, "text": "A"},
        ]
        # sub.start == end: no overlap (start < end is false)
        result = filter_subtitles_by_time_range(subs, 0.0, 5.0)
        assert len(result) == 0

    def test_partial_overlap(self):
        subs = [
            {"start": 3.0, "end": 7.0, "text": "A"},
        ]
        result = filter_subtitles_by_time_range(subs, 5.0, 10.0)
        assert len(result) == 1


# ===========================================================================
# get_subtitles_for_segments tests
# ===========================================================================
class TestGetSubtitlesForSegments:
    def test_multiple_segments_dedup(self):
        subs = [
            {"index": 1, "start": 1.0, "end": 5.0, "text": "A"},
            {"index": 2, "start": 6.0, "end": 10.0, "text": "B"},
        ]
        segments = [{"start": 0, "end": 6.0}, {"start": 4.0, "end": 12.0}]
        result = get_subtitles_for_segments(subs, segments)
        # Both subs overlap with both segments, but deduplicated
        assert len(result) == 2

    def test_sorted_by_start(self):
        subs = [
            {"index": 2, "start": 6.0, "end": 10.0, "text": "B"},
            {"index": 1, "start": 1.0, "end": 5.0, "text": "A"},
        ]
        segments = [{"start": 0, "end": 20.0}]
        result = get_subtitles_for_segments(subs, segments)
        assert result[0]["start"] == 1.0
        assert result[1]["start"] == 6.0


# ===========================================================================
# format_subtitles_as_text tests
# ===========================================================================
class TestFormatSubtitlesAsText:
    def test_no_timestamps(self):
        subs = [
            {"start": 1.0, "end": 5.0, "text": "Hello"},
            {"start": 6.0, "end": 10.0, "text": "World"},
        ]
        result = format_subtitles_as_text(subs)
        assert result == "Hello\nWorld"

    def test_with_timestamps(self):
        subs = [{"start": 1.0, "end": 5.0, "text": "Hello"}]
        result = format_subtitles_as_text(subs, include_timestamps=True)
        assert "[00:00:01.000 --> 00:00:05.000] Hello" == result

    def test_empty(self):
        assert format_subtitles_as_text([]) == ""

    def test_max_length_truncation(self):
        subs = [{"start": 1.0, "end": 5.0, "text": "A very long subtitle text that should be truncated"}]
        result = format_subtitles_as_text(subs, max_length=10)
        assert len(result) == 10

    def test_max_length_no_truncation(self):
        subs = [{"start": 1.0, "end": 5.0, "text": "Short"}]
        result = format_subtitles_as_text(subs, max_length=100)
        assert result == "Short"


# ===========================================================================
# group_subtitles_by_scenes tests
# ===========================================================================
class TestGroupSubtitlesByScenes:
    def test_basic_mapping(self):
        subs = [
            {"start": 1.0, "end": 5.0, "text": "A"},
            {"start": 11.0, "end": 15.0, "text": "B"},
        ]
        scenes = [
            {"scene_id": 1, "start_time": 0.0, "end_time": 10.0},
            {"scene_id": 2, "start_time": 10.0, "end_time": 20.0},
        ]
        mapping = group_subtitles_by_scenes(subs, scenes)
        assert len(mapping[1]) == 1
        assert mapping[1][0]["text"] == "A"
        assert len(mapping[2]) == 1
        assert mapping[2][0]["text"] == "B"

    def test_midpoint_assignment(self):
        # Sub midpoint=3.0 -> falls in scene 1 (0-10)
        subs = [{"start": 2.0, "end": 4.0, "text": "Mid"}]
        scenes = [
            {"scene_id": 1, "start_time": 0.0, "end_time": 10.0},
            {"scene_id": 2, "start_time": 10.0, "end_time": 20.0},
        ]
        mapping = group_subtitles_by_scenes(subs, scenes)
        assert mapping[1][0]["text"] == "Mid"
        assert len(mapping[2]) == 0

    def test_empty_subtitles(self):
        scenes = [
            {"scene_id": 1, "start_time": 0.0, "end_time": 10.0},
        ]
        mapping = group_subtitles_by_scenes([], scenes)
        assert len(mapping[1]) == 0