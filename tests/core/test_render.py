# tests/test_render.py
"""Comprehensive unit‑test suite for `_render.py`.

This file unifies the earlier baseline tests with the additional
edge/error‑case tests that were identified as missing.  It should bring the
behavioural coverage of `_render.py` close to 100 %.

To run:
    pytest -q tests/test_render.py
"""

from __future__ import annotations

import ast
import json
from typing import Any, Iterable
import typing
import pytest


# Dynamic import so the tests work regardless of project layout.
from dachi.core import (
    Templatable, model_to_text,
    render, is_renderable,
    render_multi, Renderable,
    TemplateField
)

from dachi.core import _render
from dachi.core import Renderable
from pydantic import BaseModel




try:
    from pydantic import BaseModel  # type: ignore
except ImportError:  # pragma: no cover – pydantic is a hard requirement for library
    raise RuntimeError("pydantic must be installed to run the _render test‑suite")

# --------------------------------------------------------------------------- #
# Helpers & reusable fixtures
# --------------------------------------------------------------------------- #
class DummyRenderable(Renderable):
    """Simple concrete implementation of `Renderable`."""

    def render(self) -> str:  # type: ignore[override]
        return "dummy"


class DemoModel(BaseModel):
    a: int
    b: str = "foo"


@pytest.fixture
def pf() -> TemplateField:
    """A canonical `TemplateField` instance reused across tests."""

    return TemplateField(
        type_="string",
        description="desc",
        default="x",
        is_required=False,
    )


#                               TemplateField
class TestTemplateField:
    def test_to_dict_regular(self, pf: TemplateField) -> None:
        assert pf.to_dict() == {
            "type": "string",
            "description": "desc",
            "default": "x",
            "is_required": False,
        }

    def test_render_matches_to_dict(self, pf: TemplateField) -> None:
        assert ast.literal_eval(pf.render()) == pf.to_dict()

    def test_missing_default_and_required(self) -> None:
        field = TemplateField("int", "d", default=None, is_required=True)
        d = field.to_dict()
        assert d["default"] is None and d["is_required"] is True


#  model_to_text
class TestModelToText:
    def test_without_escape_returns_json(self) -> None:
        m = DemoModel(a=1, b="bar")
        assert json.loads(model_to_text(m)) == m.model_dump()

    # def test_with_escape_calls_escape_curly_braces(self, monkeypatch) -> None:
    #     m = DemoModel(a=2)
    #     called: dict[str, Any] = {}

    #     def fake_escape(arg):  # noqa: ANN001  – runtime replacement
    #         called["arg"] = arg
    #         return "<<ESCAPED>>"

    #     monkeypatch.setattr(render_mod, "escape_curly_braces", fake_escape)
    #     assert model_to_text(m, escape=True) == "<<ESCAPED>>"
    #     assert called["arg"] == m.model_dump()

    def test_invalid_input_raises(self):
        with pytest.raises(AttributeError):
            model_to_text(42)  # type: ignore[arg-type]



#  render
class TestRender:
    #  TemplateField
    def test_templatefield_default(self, pf: TemplateField) -> None:
        assert render(pf) == pf.render()

    def test_templatefield_custom_renderer_called_once(self, pf: TemplateField) -> None:
        calls: list[TemplateField] = []

        def custom(tf: TemplateField) -> str:  # noqa: ANN001
            calls.append(tf)
            return "CUSTOM"

        out = render(pf, template_render=custom)
        assert out == "CUSTOM" and len(calls) == 1


    def test_dummy_renderable(self) -> None:
        assert render(DummyRenderable()) == "dummy"

    @pytest.mark.parametrize("escape", [False, True])
    def test_pydantic_model(self, escape: bool) -> None:
        m = DemoModel(a=5)
        assert render(m, escape_braces=escape) == model_to_text(m, escape)

    @pytest.mark.parametrize(
        "value, expected",
        [
            (42, "42"),
            ("hi", "hi"),
            (True, "True"),
            (None, "None"),
        ],
    )
    def test_primitives(self, value: Any, expected: str) -> None:  # noqa: ANN401
        assert render(value) == expected

    def test_dict_escaped(self) -> None:
        d = {"x": 1, "y": "z"}
        assert render(d, escape_braces=True) == '{{"x": 1, "y": "z"}}'

    def test_dict_no_escape(self) -> None:
        d = {"x": 1, "y": "z"}
        assert render(d, escape_braces=False) == '{"x": 1, "y": "z"}'

    def test_empty_dict_variants(self):
        assert render({}) == "{}"
        assert render({}, escape_braces=True) == "{{}}"

    def test_dict_with_unrenderable_value(self):
        result = render({"bad": object()})
        # Should include str(object) but not crash.  We can't know object's repr;
        # key presence suffices.
        assert result.startswith('{') and 'bad' in result

    def test_nested_dict_brace_escape(self):
        nested = {"wrapper": {"inner": 1}}
        assert render(nested, escape_braces=True) == '{{"wrapper": {"inner": 1}}}'

    # list
    def test_list_mixed(self) -> None:
        lst = [1, "a", DummyRenderable()]
        assert render(lst) == '[1, "a", dummy]'

    def test_empty_list(self):
        assert render([]) == "[]"

    def test_list_nested_containers(self):
        data = [[1, 2], {"k": "v"}]
        assert render(data) == '[[1, 2], {"k": "v"}]'

    # fallback 
    def test_fallback_str(self) -> None:
        class Silly:
            def __str__(self) -> str:  # noqa: D401
                return "silly"

        assert render(Silly()) == "silly"



#  is_renderable
class TestIsRenderable:
    @pytest.mark.parametrize(
        "obj",
        [
            DummyRenderable(),
            3.14,
            "hi",
            None,
            {"k": 1},
            [1, 2, 3],
            DemoModel(a=9),
        ],
    )
    def test_positive_cases(self, obj: Any):  # noqa: ANN401
        assert is_renderable(obj) is True

    def test_negative_case(self) -> None:
        class NotRenderable:  # noqa: D401 — test helper
            pass

        assert is_renderable(NotRenderable()) is False



# render_multi
class TestRenderMulti:
    def test_basic(self) -> None:
        assert render_multi([1, "x"]) == ["1", "x"]

    def test_iterable(self) -> None:
        gen: Iterable[Any] = (x for x in (DummyRenderable(), 7))
        assert render_multi(gen) == ["dummy", "7"]

    def test_empty_iterable(self):
        assert render_multi([]) == []



class NestedModel(BaseModel):
    nested_field: int


class DummyModel(BaseModel):
    required_field: str
    optional_field: typing.Optional[int]
    nested_model: NestedModel


# class TestModelTemplate:
    
#     def test_model_template_with_simple_model(self):
#         class SimpleModel(BaseModel):
#             field: str

#         expected = {
#             "field": {
#                 "is_required": True,
#                 "type": str
#             }
#         }
#         assert model_template(SimpleModel) == expected

#     def test_model_template_with_optional_field(self):
#         class OptionalModel(BaseModel):
#             optional_field: typing.Optional[int]

#         expected = {
#             "optional_field": {
#                 "is_required": True,
#                 "type": typing.Optional[int]
#             }
#         }
#         result = model_template(OptionalModel)
#         assert result == expected

#     def test_model_template_with_optional_field(self):
#         class OptionalModel(BaseModel):
#             optional_field: typing.Optional[int] = 2

#         expected = {
#             "optional_field": {
#                 "is_required": False,
#                 "type": typing.Optional[int]
#             }
#         }
#         result = model_template(OptionalModel)
#         assert result == expected

#     def test_model_template_with_nested_model(self):
#         expected = {
#             "required_field": {
#                 "is_required": True,
#                 "type": str
#             },
#             "optional_field": {
#                 "is_required": True,
#                 "type": typing.Optional[int]
#             },
#             "nested_model": {
#                 "nested_field": {
#                     "is_required": True,
#                     "type": int
#                 }
#             }
#         }
#         assert model_template(DummyModel) == expected

#     def test_model_template_with_empty_model(self):
#         class EmptyModel(BaseModel):
#             pass

#         expected = {}
#         assert model_template(EmptyModel) == expected

class TestModelToText:
    def test_model_to_text_with_simple_model(self):
        class SimpleModel(BaseModel):
            field: str

        model = SimpleModel(field="value")
        expected = '{"field":"value"}'
        assert _render.model_to_text(model) == expected

    def test_model_to_text_with_optional_field(self):
        class OptionalModel(BaseModel):
            optional_field: typing.Optional[int] = None

        model = OptionalModel()
        expected = '{"optional_field":null}'
        assert _render.model_to_text(model) == expected

    # def test_model_to_text_with_nested_model(self):
    #     nested_model = NestedModel(nested_field=42)
    #     model = DummyModel(
    #         required_field="required",
    #         optional_field=123,
    #         nested_model=nested_model
    #     )
    #     print(_render.model_to_text(model))
    #     expected = '{"required_field":"required","optional_field":123, "nested_model":{"nested_field":42}}'
    #     assert _render.model_to_text(model) == expected

    def test_model_to_text_with_escape(self):
        class EscapedModel(BaseModel):
            field: str

        model = EscapedModel(field="value")
        expected = '{{"field": "value"}}'  # Assuming escape_curly_braces does not alter this
        assert _render.model_to_text(model, escape=True) == expected

    def test_model_to_text_with_empty_model(self):
        class EmptyModel(BaseModel):
            pass

        model = EmptyModel()
        expected = '{}'
        assert _render.model_to_text(model) == expected


class TestRender:

    def test_render_with_primitive(self):
        result = _render.render(42)
        assert result == "42"

    def test_render_with_string(self):
        result = _render.render("hello")
        assert result == "hello"

    def test_render_with_dict(self):
        input_data = {"key": "value", "number": 42}
        expected = '{"key": "value", "number": 42}'
        result = _render.render(input_data, escape_braces=False)
        assert result == expected

    def test_render_with_dict_escaped(self):
        input_data = {"key": "value", "number": 42}
        expected = '{{"key": "value", "number": 42}}'
        result = _render.render(input_data, escape_braces=True)
        assert result == expected

    def test_render_with_list(self):
        input_data = [1, "two", 3.0]
        expected = '[1, "two", 3.0]'
        result = _render.render(input_data)
        assert result == expected

    def test_render_with_nested_dict(self):
        input_data = {"outer": {"inner": "value"}}
        expected = '{"outer": {"inner": "value"}}'
        result = _render.render(input_data, escape_braces=False)
        assert result == expected

    def test_render_with_nested_dict_escaped(self):
        input_data = {"outer": {"inner": "value"}}
        expected = '{{"outer": {{"inner": "value"}}}}'
        result = _render.render(input_data, escape_braces=True)
        assert result == expected

    # def test_render_with_pydantic_model(self):
    #     model = DummyModel(
    #         required_field="required",
    #         optional_field=123,
    #         nested_model=NestedModel(nested_field=42)
    #     )
    #     expected = '{"required_field":"required","optional_field":123, "nested_model": {"nested_field":42}}'
    #     result = _render.render(model, escape_braces=False)
    #     assert result == expected

    def test_render_with_template_field(self):
        class MockTemplateField(Renderable):
            def render(self):
                return "rendered_template"

        template_field = MockTemplateField()
        result = _render.render(template_field)
        assert result == "rendered_template"

    def test_render_with_unknown_type(self):
        class UnknownType:
            def __str__(self):
                return "unknown"

        unknown = UnknownType()
        result = _render.render(unknown)
        assert result == "unknown"


