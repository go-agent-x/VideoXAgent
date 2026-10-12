"""Tests for tool_registry module."""

import pytest
from pathlib import Path
import tempfile
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from _base.tool_registry import (
    load_tool_defs,
    load_tool_defs_from_file,
    build_mcp_tools,
    register_handler,
    get_tool_def,
    route_tool_call,
    validate_arguments,
    list_all_tools,
    clear_registry,
    get_tool_backend,
    get_tool_media_type,
    _check_type,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def sample_yaml_content():
    """Sample tool definitions for testing."""
    return {
        "detect": {
            "description": "Detect objects in an image",
            "backend": "grounding_dino",
            "media_type": "image",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to image"},
                    "categories": {"type": "string", "description": "Categories to detect"},
                    "threshold": {"type": "number", "default": 0.5},
                },
                "required": ["image_path", "categories"],
            },
        },
        "count": {
            "description": "Count objects in an image",
            "backend": "grounding_dino",
            "media_type": "image",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["image_path", "category"],
            },
        },
    }


class TestLoadToolDefs:
    """Tests for load_tool_defs function."""
    
    def test_load_from_directory(self, sample_yaml_content, tmp_path):
        """Test loading tool definitions from directory."""
        # Create test YAML file
        yaml_file = tmp_path / "test_tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_content, f)
        
        # Load definitions
        defs = load_tool_defs(tmp_path)
        
        assert "detect" in defs
        assert "count" in defs
        assert defs["detect"]["backend"] == "grounding_dino"
        assert defs["detect"]["media_type"] == "image"
    
    def test_load_multiple_files(self, tmp_path):
        """Test loading from multiple YAML files."""
        # Create first file
        file1 = tmp_path / "tools1.yaml"
        with open(file1, 'w') as f:
            yaml.dump({"tool1": {"description": "Tool 1", "backend": "test"}}, f)
        
        # Create second file
        file2 = tmp_path / "tools2.yaml"
        with open(file2, 'w') as f:
            yaml.dump({"tool2": {"description": "Tool 2", "backend": "test"}}, f)
        
        defs = load_tool_defs(tmp_path)
        
        assert "tool1" in defs
        assert "tool2" in defs
    
    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory raises error."""
        with pytest.raises(FileNotFoundError):
            load_tool_defs(Path("/nonexistent/path"))


class TestLoadToolDefsFromFile:
    """Tests for load_tool_defs_from_file function."""
    
    def test_load_single_file(self, sample_yaml_content, tmp_path):
        """Test loading from single file."""
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_content, f)
        
        defs = load_tool_defs_from_file(yaml_file)
        
        assert "detect" in defs
        assert "count" in defs
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_tool_defs_from_file(Path("/nonexistent/file.yaml"))


class TestBuildMcpTools:
    """Tests for build_mcp_tools function."""
    
    def test_build_tools(self, sample_yaml_content, tmp_path):
        """Test building MCP Tool objects."""
        # Setup
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_content, f)
        load_tool_defs_from_file(yaml_file)
        
        # Build tools
        tools = build_mcp_tools()
        
        assert len(tools) == 2
        
        # Check tool names
        tool_names = [t.name for t in tools]
        assert "detect" in tool_names
        assert "count" in tool_names
    
    def test_build_tools_empty_registry(self):
        """Test building with empty registry."""
        tools = build_mcp_tools()
        assert tools == []


class TestRegisterAndRouteHandler:
    """Tests for handler registration and routing."""
    
    async def sample_handler(self, arguments):
        return {"success": True, "result": "test"}
    
    def test_register_and_route(self):
        """Test registering and routing handler."""
        # Setup - add a tool definition first
        from _base import tool_registry
        tool_registry._default_registry._tool_defs["test_tool"] = {
            "description": "Test tool",
            "input_schema": {"type": "object"},
        }
        
        # Register handler
        register_handler("test_tool", self.sample_handler)
        
        # Route to handler
        handler = route_tool_call("test_tool")
        assert handler == self.sample_handler
    
    def test_route_unknown_tool(self):
        """Test routing unknown tool raises error."""
        with pytest.raises(ValueError, match="No handler registered"):
            route_tool_call("unknown_tool")


class TestGetToolDef:
    """Tests for get_tool_def function."""
    
    def test_get_existing_tool(self, sample_yaml_content, tmp_path):
        """Test getting existing tool definition."""
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_content, f)
        load_tool_defs_from_file(yaml_file)
        
        tdef = get_tool_def("detect")
        assert tdef is not None
        assert tdef["backend"] == "grounding_dino"
    
    def test_get_nonexistent_tool(self):
        """Test getting non-existent tool returns None."""
        tdef = get_tool_def("nonexistent")
        assert tdef is None


class TestValidateArguments:
    """Tests for validate_arguments function."""
    
    def test_valid_arguments(self, sample_yaml_content, tmp_path):
        """Test validating valid arguments."""
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_content, f)
        load_tool_defs_from_file(yaml_file)
        
        args = {
            "image_path": "/path/to/image.jpg",
            "categories": "person.car",
        }
        
        valid, error = validate_arguments("detect", args)
        assert valid is True
        assert error is None
    
    def test_missing_required_field(self, sample_yaml_content, tmp_path):
        """Test validating with missing required field."""
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_content, f)
        load_tool_defs_from_file(yaml_file)
        
        args = {"image_path": "/path/to/image.jpg"}  # Missing categories
        
        valid, error = validate_arguments("detect", args)
        assert valid is False
        assert "categories" in error
    
    def test_unknown_tool(self):
        """Test validating arguments for unknown tool."""
        valid, error = validate_arguments("unknown", {})
        assert valid is False
        assert "Unknown tool" in error


class TestListAllTools:
    """Tests for list_all_tools function."""
    
    def test_list_all(self, sample_yaml_content, tmp_path):
        """Test listing all tools."""
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_content, f)
        load_tool_defs_from_file(yaml_file)
        
        tools = list_all_tools()
        assert "detect" in tools
        assert "count" in tools


class TestClearRegistry:
    """Tests for clear_registry function."""
    
    def test_clear_registry(self, sample_yaml_content, tmp_path):
        """Test clearing registry."""
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_content, f)
        load_tool_defs_from_file(yaml_file)
        register_handler("detect", lambda x: x)
        
        clear_registry()
        
        assert list_all_tools() == {}
        with pytest.raises(ValueError):
            route_tool_call("detect")


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_get_tool_backend(self, sample_yaml_content, tmp_path):
        """Test getting tool backend."""
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_content, f)
        load_tool_defs_from_file(yaml_file)
        
        backend = get_tool_backend("detect")
        assert backend == "grounding_dino"
    
    def test_get_tool_media_type(self, sample_yaml_content, tmp_path):
        """Test getting tool media type."""
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_content, f)
        load_tool_defs_from_file(yaml_file)
        
        media_type = get_tool_media_type("detect")
        assert media_type == "image"


class TestCheckType:
    """Tests for _check_type, especially bool-vs-integer confusion."""

    def test_bool_rejected_for_integer(self):
        assert _check_type(True, "integer") is False
        assert _check_type(False, "integer") is False

    def test_bool_rejected_for_number(self):
        assert _check_type(True, "number") is False
        assert _check_type(False, "number") is False

    def test_bool_accepted_for_boolean(self):
        assert _check_type(True, "boolean") is True
        assert _check_type(False, "boolean") is True

    def test_int_accepted_for_integer(self):
        assert _check_type(0, "integer") is True
        assert _check_type(42, "integer") is True
        assert _check_type(-1, "integer") is True

    def test_int_accepted_for_number(self):
        assert _check_type(42, "number") is True

    def test_float_accepted_for_number(self):
        assert _check_type(3.14, "number") is True

    def test_float_rejected_for_integer(self):
        assert _check_type(3.14, "integer") is False

    def test_int_rejected_for_boolean(self):
        assert _check_type(0, "boolean") is False
        assert _check_type(1, "boolean") is False

    def test_string_type(self):
        assert _check_type("hello", "string") is True
        assert _check_type(123, "string") is False

    def test_array_type(self):
        assert _check_type([1, 2], "array") is True
        assert _check_type("not a list", "array") is False

    def test_object_type(self):
        assert _check_type({"key": "val"}, "object") is True
        assert _check_type([1], "object") is False

    def test_null_type(self):
        assert _check_type(None, "null") is True
        assert _check_type(0, "null") is False

    def test_unknown_type_returns_true(self):
        assert _check_type("anything", "custom") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
