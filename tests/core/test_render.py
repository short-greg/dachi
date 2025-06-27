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


# Helpers & reusable fixtures
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



from dachi.core import style_formatter, bullet
import pytest
from dachi.core import generate_numbered_list, numbered


class TestGenerateNumberedList:

    def test_arabic_numbering(self):
        """Tests arabic numbering."""
        result = generate_numbered_list(5, numbering_type='arabic')
        expected = ['1', '2', '3', '4', '5']
        assert result == expected

    def test_roman_numbering(self):
        """Tests roman numeral numbering."""
        result = generate_numbered_list(5, numbering_type='roman')
        expected = ['i', 'ii', 'iii', 'iv', 'v']
        assert result == expected

    def test_alphabet_numbering(self):
        """Tests alphabetic numbering."""
        result = generate_numbered_list(5, numbering_type='alphabet')
        expected = ['A', 'B', 'C', 'D', 'E']
        assert result == expected

    def test_invalid_numbering_type(self):
        """Tests invalid numbering type."""
        with pytest.raises(ValueError):
            generate_numbered_list(5, numbering_type='invalid')

    def test_alphabet_numbering_exceeds_limit(self):
        """Tests alphabetic numbering exceeding the limit."""
        with pytest.raises(ValueError, match="Alphabetic numbering can only handle up to 26 items"):
            generate_numbered_list(27, numbering_type='alphabet')

    def test_zero_items(self):
        """Tests generating a list with zero items."""
        result = generate_numbered_list(0, numbering_type='arabic')
        assert result == []

    def test_negative_items(self):
        """Tests generating a list with negative items."""
        with pytest.raises(ValueError):
            generate_numbered_list(-5, numbering_type='arabic')

    def test_default_numbering_type(self):
        """Tests default numbering type (arabic)."""
        result = generate_numbered_list(3)
        expected = ['1', '2', '3']
        assert result == expected


class TestNumbered:

    def test_arabic_numbering(self):
        """Tests arabic numbering with default indent."""
        result = numbered(["Apple", "Banana", "Cherry"], numbering="arabic")
        expected = "1. Apple\n2. Banana\n3. Cherry"
        assert result == expected

    def test_roman_numbering(self):
        """Tests roman numeral numbering."""
        result = numbered(["Apple", "Banana", "Cherry"], numbering="roman")
        expected = "i. Apple\nii. Banana\niii. Cherry"
        assert result == expected

    def test_alphabet_numbering(self):
        """Tests alphabetic numbering."""
        result = numbered(["Apple", "Banana", "Cherry"], numbering="alphabet")
        expected = "A. Apple\nB. Banana\nC. Cherry"
        assert result == expected

    def test_with_indent(self):
        """Tests numbering with indentation."""
        result = numbered(["Apple", "Banana", "Cherry"], indent=4, numbering="arabic")
        expected = "    1. Apple\n    2. Banana\n    3. Cherry"
        assert result == expected

    def test_empty_list(self):
        """Tests numbering with an empty list."""
        result = numbered([], numbering="arabic")
        assert result == ""

    def test_invalid_numbering_type(self):
        """Tests invalid numbering type."""
        with pytest.raises(ValueError):
            numbered(["Apple", "Banana"], numbering="invalid")

    def test_single_item(self):
        """Tests numbering with a single item."""
        result = numbered(["Apple"], numbering="arabic")
        expected = "1. Apple"
        assert result == expected

    def test_negative_indent(self):
        """Tests numbering with a negative indent."""
        result = numbered(["Apple", "Banana"], indent=-2, numbering="arabic")
        expected = "1. Apple\n2. Banana"
        assert result == expected  # Negative indent should be treated as no indent

    def test_large_list(self):
        """Tests numbering with a large list."""
        with pytest.raises(ValueError):
            items = [f"Item {i}" for i in range(1, 28)]
            numbered(items, numbering="alphabet")
            # expected = "\n".join([f"{chr(64 + i)}. Item {i}" for i in range(1, 27)])
            # assert result == expected

class TestBullet:

    def test_basic_bullet_list(self):
        """Tests basic bullet list generation."""
        result = bullet(["Apple", "Banana", "Cherry"])
        expected = "- Apple\n- Banana\n- Cherry"
        assert result == expected

    def test_custom_bullet_character(self):
        """Tests bullet list with a custom bullet character."""
        result = bullet(["Apple", "Banana", "Cherry"], bullets="*")
        expected = "* Apple\n* Banana\n* Cherry"
        assert result == expected

    def test_with_indent(self):
        """Tests bullet list with indentation."""
        result = bullet(["Apple", "Banana", "Cherry"], indent=4)
        expected = "    - Apple\n    - Banana\n    - Cherry"
        assert result == expected

    def test_empty_list(self):
        """Tests bullet list with an empty list."""
        result = bullet([])
        assert result == ""

    def test_single_item(self):
        """Tests bullet list with a single item."""
        result = bullet(["Apple"])
        expected = "- Apple"
        assert result == expected

    def test_negative_indent(self):
        """Tests bullet list with a negative indent."""
        result = bullet(["Apple", "Banana"], indent=-2)
        expected = "- Apple\n- Banana"  # Negative indent should be treated as no indent
        assert result == expected

    def test_numeric_items(self):
        """Tests bullet list with numeric items."""
        result = bullet([1, 2, 3])
        expected = "- 1\n- 2\n- 3"
        assert result == expected

    def test_mixed_type_items(self):
        """Tests bullet list with mixed type items."""
        result = bullet(["Apple", 42, 3.14])
        expected = "- Apple\n- 42\n- 3.14"
        assert result == expected

    def test_custom_bullet_and_indent(self):
        """Tests bullet list with custom bullet and indentation."""
        result = bullet(["Apple", "Banana"], bullets=">", indent=2)
        expected = "  > Apple\n  > Banana"
        assert result == expected


class TestStyleFormat:

    def test_basic_string_formatting(self):
        """Tests standard Python formatting without special styling."""
        result = style_formatter("Hello, {name}!", name="Alice")
        assert result == "Hello, Alice!"

    def test_bullet_list_formatting(self):
        """Tests whether lists are correctly formatted as bullet points."""
        result = style_formatter("Shopping List:\n\n{x:bullet}", x=["Milk", "Eggs", "Bread"])
        expected = "Shopping List:\n\n- Milk\n- Eggs\n- Bread"
        print(result, expected)
        assert result == expected

    def test_bold_formatting(self):
        """Tests bold formatting."""
        result = style_formatter("Total: {total:bold}", total=100)
        expected = "Total: **100**"
        assert result == expected

    def test_italic_formatting(self):
        """Tests italic formatting."""
        result = style_formatter("Style: {text_:italic}", text_="Fancy")
        expected = "Style: *Fancy*"
        assert result == expected

    def test_mixed_formatting(self):
        """Tests multiple styles in one string."""
        result = style_formatter(
            "Receipt:\nItems:\n{x:bullet}\nTotal: {total:bold}",
            x=["Apple", "Banana"],
            total=12.50
        )
        expected = "Receipt:\nItems:\n- Apple\n- Banana\nTotal: **12.5**"
        assert result == expected

    def test_no_style_fallback(self):
        """Tests that standard formatting still works if no styles are applied."""
        result = style_formatter("Hello, {name}", name="World")
        assert result == "Hello, World"

    def test_empty_list_bullet(self):
        """Tests bullet formatting with an empty list."""
        result = style_formatter("Items:\n{x:bullet}", x=[])
        expected = "Items:\n"
        assert result == expected

    def test_numeric_formatting(self):
        """Tests if numbers are formatted properly."""
        result = style_formatter("The value is: {value}", value=42)
        assert result == "The value is: 42"

    def test_style_with_empty_string(self):
        """Tests that styling works even when the string is empty."""
        result = style_formatter("{x:bold}", x="")
        expected = "****"  # Bold empty string should still be two asterisks
        assert result == expected

    def test_positional_arguments(self):
        """Tests positional arguments instead of named arguments."""
        result = style_formatter("Values:\n{0:bullet}", ["A", "B", "C"])
        expected = "Values:\n- A\n- B\n- C"
        assert result == expected

    def test_function_call_style(self):
        """Tests positional arguments instead of named arguments."""
        result = style_formatter("Values:\n{0:bullet('-', 1)}", ["A", "B", "C"])
        expected = "Values:\n - A\n - B\n - C"
        assert result == expected
