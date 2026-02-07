"""Tests for template resolution in YAML commands."""

import pytest

from tensortruth.extensions.errors import TemplateResolutionError
from tensortruth.extensions.yaml_command import (
    _build_args_context,
    _resolve_params,
    resolve_template,
)


class TestResolveTemplate:
    """Tests for resolve_template()."""

    def test_simple_variable(self):
        ctx = {"name": "hello"}
        assert resolve_template("{{name}}", ctx) == "hello"

    def test_no_template(self):
        assert resolve_template("plain text", {}) == "plain text"

    def test_mixed_text_and_template(self):
        ctx = {"x": "world"}
        assert resolve_template("hello {{x}}!", ctx) == "hello world!"

    def test_multiple_variables(self):
        ctx = {"a": "1", "b": "2"}
        assert resolve_template("{{a}} and {{b}}", ctx) == "1 and 2"

    def test_dot_path(self):
        ctx = {"result": {"id": "abc123"}}
        assert resolve_template("{{result.id}}", ctx) == "abc123"

    def test_nested_dot_path(self):
        ctx = {"a": {"b": {"c": "deep"}}}
        assert resolve_template("{{a.b.c}}", ctx) == "deep"

    def test_direct_key_with_dot(self):
        """Keys like args.0 are stored directly in context."""
        ctx = {"args.0": "pytorch", "args.rest": "batch norm"}
        assert resolve_template("{{args.0}}", ctx) == "pytorch"
        assert resolve_template("{{args.rest}}", ctx) == "batch norm"

    def test_non_string_value_json_serialised(self):
        ctx = {"count": 42}
        assert resolve_template("{{count}}", ctx) == "42"

    def test_list_value_json_serialised(self):
        ctx = {"items": [1, 2, 3]}
        assert resolve_template("{{items}}", ctx) == "[1, 2, 3]"

    def test_missing_variable_raises(self):
        with pytest.raises(TemplateResolutionError):
            resolve_template("{{missing}}", {})

    def test_missing_nested_raises(self):
        ctx = {"a": {"b": "value"}}
        with pytest.raises(TemplateResolutionError):
            resolve_template("{{a.nonexistent}}", ctx)

    def test_whitespace_in_braces(self):
        ctx = {"x": "val"}
        assert resolve_template("{{ x }}", ctx) == "val"


class TestBuildArgsContext:
    """Tests for _build_args_context()."""

    def test_full_args(self):
        ctx = _build_args_context("pytorch batch normalization")
        assert ctx["args"] == "pytorch batch normalization"
        assert ctx["args.0"] == "pytorch"
        assert ctx["args.1"] == "batch"
        assert ctx["args.2"] == "normalization"
        assert ctx["args.rest"] == "batch normalization"

    def test_single_arg(self):
        ctx = _build_args_context("pytorch")
        assert ctx["args"] == "pytorch"
        assert ctx["args.0"] == "pytorch"
        assert ctx["args.rest"] == ""

    def test_empty_args(self):
        ctx = _build_args_context("")
        assert ctx["args"] == ""
        assert ctx["args.rest"] == ""

    def test_no_args_0_when_empty(self):
        ctx = _build_args_context("")
        assert "args.0" not in ctx


class TestResolveParams:
    """Tests for _resolve_params()."""

    def test_string_values_resolved(self):
        ctx = {"args": "test"}
        result = _resolve_params({"q": "{{args}}"}, ctx)
        assert result == {"q": "test"}

    def test_non_string_values_passed_through(self):
        result = _resolve_params({"count": 5, "flag": True}, {})
        assert result == {"count": 5, "flag": True}

    def test_nested_dict_resolved(self):
        ctx = {"x": "val"}
        result = _resolve_params({"outer": {"inner": "{{x}}"}}, ctx)
        assert result == {"outer": {"inner": "val"}}

    def test_list_values_resolved(self):
        ctx = {"x": "val"}
        result = _resolve_params({"items": ["{{x}}", "literal"]}, ctx)
        assert result == {"items": ["val", "literal"]}
