"""Tests for _base.config module."""
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure _base is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _base.config import (
    CONFIG_SCHEMAS,
    get_config,
    get_env,
    get_output_dir,
    normalize_vlm_url,
    pick_api_key,
)


# ---------------------------------------------------------------------------
# Helper: clean env vars used by schemas to avoid cross-test pollution
# ---------------------------------------------------------------------------

def _all_schema_env_vars():
    """Collect all env var names referenced in CONFIG_SCHEMAS."""
    env_vars = set()
    for schema in CONFIG_SCHEMAS.values():
        for _, (env_var, _, _) in schema.items():
            env_vars.add(env_var)
    return env_vars


@pytest.fixture(autouse=True)
def clean_env():
    """Isolate schema env vars and restore the full environment after each test."""
    all_vars = _all_schema_env_vars()
    all_vars.update(
        var for var in os.environ
        if var.startswith("TOOL_") and var.partition("__")[2] in all_vars
    )
    # Also clean common test vars
    all_vars.update([
        "TOOL_MY_TOOL__VLM_API_KEY", "TOOL_MY_TOOL__VLM_MODEL_NAME",
        "TOOL_MY_TOOL__VLM_API_URL", "TOOL_MY_TOOL__VLM_MODEL_FALLBACKS",
        "TOOL_MY_TOOL__FER_FUSION_WEIGHT",
        "ACOUSTID_API_KEY", "MY_INT_VAR", "MY_FLOAT_VAR", "MY_BOOL_VAR",
        "OUTPUT_DIR", "CUSTOM_OUTPUT", "TOTALLY_UNKNOWN_KEY", "MISSING_VAR",
    ])
    # Tests write directly to os.environ, including previously unseen tool names.
    # Snapshot the whole mapping so those additions cannot leak into later tests.
    with mock.patch.dict(os.environ):
        for var in all_vars:
            os.environ.pop(var, None)
        yield


# ===================================================================
# get_config
# ===================================================================

class TestGetConfig:
    """Tests for get_config()."""

    def test_l3_defaults(self):
        """With no env vars set, returns schema defaults."""
        cfg = get_config("vlm")
        assert cfg["api_url"] == ""
        assert cfg["api_key"] == ""
        assert cfg["model"] == "gpt-4.1"
        assert isinstance(cfg["fallbacks"], str)

    def test_l3_defaults_grounding_dino(self):
        cfg = get_config("grounding_dino")
        assert cfg["device"] == "cuda"
        assert cfg["checkpoint"] == "models/GroundingDINO/groundingdino_swinb_cogcoor.pth"
        assert cfg["bert_path"] == "models/bert-base-uncased"

    def test_l2_global_override(self):
        """Global env var overrides schema default."""
        os.environ["VLM_API_KEY"] = "global-key-123"
        os.environ["VLM_MODEL_NAME"] = "gemini-pro"
        cfg = get_config("vlm")
        assert cfg["api_key"] == "global-key-123"
        assert cfg["model"] == "gemini-pro"

    def test_l1_tool_override(self):
        """Tool-level env var overrides global."""
        os.environ["VLM_API_KEY"] = "global-key"
        os.environ["TOOL_MY_TOOL__VLM_API_KEY"] = "tool-key"
        cfg = get_config("vlm", tool_name="my_tool")
        assert cfg["api_key"] == "tool-key"

    def test_l1_over_l2_over_l3(self):
        """Full priority chain: L1 > L2 > L3."""
        # L3 only
        cfg = get_config("vlm")
        assert cfg["model"] == "gpt-4.1"

        # L2 overrides L3
        os.environ["VLM_MODEL_NAME"] = "l2-model"
        cfg = get_config("vlm")
        assert cfg["model"] == "l2-model"

        # L1 overrides L2
        os.environ["TOOL_MY_TOOL__VLM_MODEL_NAME"] = "l1-model"
        cfg = get_config("vlm", tool_name="my_tool")
        assert cfg["model"] == "l1-model"

        # Without tool_name, still gets L2
        cfg = get_config("vlm")
        assert cfg["model"] == "l2-model"

    def test_type_cast_float(self):
        """Float type cast works for emotion schema."""
        os.environ["FER_FUSION_WEIGHT"] = "0.8"
        cfg = get_config("emotion")
        assert cfg["fusion_weight"] == 0.8
        assert isinstance(cfg["fusion_weight"], float)

    def test_type_cast_invalid_fallback(self):
        """Invalid value falls back to schema default."""
        os.environ["FER_FUSION_WEIGHT"] = "not-a-number"
        cfg = get_config("emotion")
        assert cfg["fusion_weight"] == "0.6"  # falls back to default string "0.6"

    def test_tool_override_with_type_cast(self):
        """Tool-level override also gets type cast."""
        os.environ["TOOL_MY_TOOL__FER_FUSION_WEIGHT"] = "0.9"
        cfg = get_config("emotion", tool_name="my_tool")
        assert cfg["fusion_weight"] == 0.9

    def test_unknown_schema_raises(self):
        """Unknown schema name raises KeyError."""
        with pytest.raises(KeyError):
            get_config("nonexistent_schema")

    def test_empty_string_env_var_is_set(self):
        """Empty string env var counts as 'set' (L2), not fallback to L3."""
        os.environ["VLM_API_KEY"] = ""
        cfg = get_config("vlm")
        assert cfg["api_key"] == ""

    def test_all_schemas_loadable(self):
        """Every registered schema can be loaded without error."""
        for schema_name in CONFIG_SCHEMAS:
            cfg = get_config(schema_name)
            assert isinstance(cfg, dict)
            assert len(cfg) > 0


# ===================================================================
# pick_api_key
# ===================================================================

class TestPickApiKey:
    """Tests for pick_api_key()."""

    def test_single_key(self):
        os.environ["VLM_API_KEY"] = "key-abc"
        assert pick_api_key("VLM_API_KEY") == "key-abc"

    def test_multi_key_rotation(self):
        os.environ["VLM_API_KEY"] = "key1,key2,key3"
        results = {pick_api_key("VLM_API_KEY") for _ in range(50)}
        assert results == {"key1", "key2", "key3"}

    def test_multi_key_with_spaces(self):
        os.environ["VLM_API_KEY"] = " key1 , key2 , key3 "
        results = {pick_api_key("VLM_API_KEY") for _ in range(50)}
        assert results == {"key1", "key2", "key3"}

    def test_tool_override(self):
        os.environ["VLM_API_KEY"] = "global-vlm"
        os.environ["TOOL_INSPECT__VLM_API_KEY"] = "tool-vlm"
        assert pick_api_key("VLM_API_KEY", tool_name="inspect") == "tool-vlm"

    def test_tool_override_falls_back_to_global(self):
        os.environ["VLM_API_KEY"] = "global-vlm"
        assert pick_api_key("VLM_API_KEY", tool_name="inspect") == "global-vlm"

    def test_no_env_returns_schema_default(self):
        # With no env var set, pick_api_key falls back to schema default
        assert pick_api_key("VLM_API_KEY") == ""

    def test_unregistered_var_returns_empty(self):
        # An env var not in any schema and not set returns empty
        assert pick_api_key("TOTALLY_UNKNOWN_KEY") == ""

    def test_comma_only_returns_empty(self):
        os.environ["VLM_API_KEY"] = ",,,  ,"
        assert pick_api_key("VLM_API_KEY") == ""


# ===================================================================
# get_env
# ===================================================================

class TestGetEnv:
    """Tests for get_env()."""

    def test_basic_read(self):
        os.environ["ACOUSTID_API_KEY"] = "my-acoustid-key"
        assert get_env("ACOUSTID_API_KEY") == "my-acoustid-key"

    def test_default_when_missing(self):
        assert get_env("ACOUSTID_API_KEY", "fallback") == "fallback"

    def test_cast_int(self):
        os.environ["MY_INT_VAR"] = "42"
        assert get_env("MY_INT_VAR", "0", cast=int) == 42

    def test_cast_float(self):
        os.environ["MY_FLOAT_VAR"] = "3.14"
        assert get_env("MY_FLOAT_VAR", "0.0", cast=float) == 3.14

    def test_cast_bool_true(self):
        for val in ("1", "true", "True", "YES", "yes"):
            os.environ["MY_BOOL_VAR"] = val
            assert get_env("MY_BOOL_VAR", "", cast=bool) is True

    def test_cast_bool_false(self):
        for val in ("0", "false", "no", "anything"):
            os.environ["MY_BOOL_VAR"] = val
            assert get_env("MY_BOOL_VAR", "", cast=bool) is False

    def test_invalid_cast_returns_default(self):
        os.environ["MY_INT_VAR"] = "not-a-number"
        assert get_env("MY_INT_VAR", "0", cast=int) == "0"

    def test_default_returned_as_is(self):
        """Default is returned without cast when env var is missing."""
        result = get_env("MISSING_VAR", 42, cast=int)
        assert result == 42


# ===================================================================
# normalize_vlm_url
# ===================================================================

class TestNormalizeVlmUrl:
    """Tests for normalize_vlm_url()."""

    def test_existing_v1_prefix_is_not_duplicated(self):
        for url in ("http://example.com/v1", "http://example.com/v1/", "example.com/v1"):
            result = normalize_vlm_url(url)
            assert result == "http://example.com/v1/chat/completions"
            assert normalize_vlm_url(result) == result

    def test_empty_string(self):
        assert normalize_vlm_url("") == ""

    def test_already_complete(self):
        url = "http://example.com/v1/chat/completions"
        assert normalize_vlm_url(url) == url

    def test_adds_http_prefix(self):
        assert normalize_vlm_url("example.com") == \
            "http://example.com/v1/chat/completions"

    def test_preserves_https(self):
        assert normalize_vlm_url("https://example.com") == \
            "https://example.com/v1/chat/completions"

    def test_adds_suffix_only(self):
        assert normalize_vlm_url("http://example.com") == \
            "http://example.com/v1/chat/completions"

    def test_strips_trailing_slash(self):
        assert normalize_vlm_url("http://example.com/") == \
            "http://example.com/v1/chat/completions"

    def test_adds_suffix_to_api_host(self):
        assert normalize_vlm_url("https://api.example.com") == \
            "https://api.example.com/v1/chat/completions"

    def test_does_not_double_suffix(self):
        url = "http://example.com/v1/chat/completions"
        assert normalize_vlm_url(url) == url

    def test_suffix_as_substring_not_at_end(self):
        """URL containing /v1/chat/completions as substring but not at end."""
        url = "http://example.com/v1/chat/completions-extra/path"
        assert normalize_vlm_url(url) == \
            "http://example.com/v1/chat/completions-extra/path/v1/chat/completions"

    def test_trailing_slash_on_complete_url(self):
        """Complete URL with trailing slash should be normalized."""
        url = "http://example.com/v1/chat/completions/"
        assert normalize_vlm_url(url) == \
            "http://example.com/v1/chat/completions"


# ===================================================================
# get_output_dir
# ===================================================================

class TestGetOutputDir:
    """Tests for get_output_dir()."""

    def test_default_dir(self, tmp_path):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OUTPUT_DIR", None)
            d = get_output_dir(default=str(tmp_path / "test_output"))
            assert d.exists()
            assert d == tmp_path / "test_output"

    def test_env_var_override(self, tmp_path):
        custom = str(tmp_path / "custom_output")
        os.environ["CUSTOM_OUTPUT"] = custom
        d = get_output_dir(env_var="CUSTOM_OUTPUT")
        assert d.exists()
        assert str(d) == custom

    def test_creates_nested_dirs(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c")
        d = get_output_dir(default=nested)
        assert d.exists()
