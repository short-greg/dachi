from __future__ import annotations

import ast
import json
from typing import Any, Iterable
import typing
import pytest

from dachi.core._render import model_to_text, struct_template
from pydantic import BaseModel
import pytest


# Dynamic import so the tests work regardless of project layout.
from dachi.core import (
    model_to_text,
    render, is_renderable,
    render_multi, Renderable,
    TemplateField
)
from dachi.utils.store._store import is_undefined, UNDEFINED, is_primitive, primitives
from dachi.utils._internal import is_nested_model

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



class SimpleStruct(BaseModel):

    x: str


class NestedStruct(BaseModel):

    simple: SimpleStruct


class TestIsUndefined(object):

    def test_is_undefined(self):

        assert is_undefined(
            UNDEFINED
        )

    def test_not_is_undefined(self):

        assert not is_undefined(
            1
        )


class TestIsNestedModel:

    def test_is_nested_model_returns_true_for_nested(self):

        assert is_nested_model(NestedStruct) is True

    def test_is_nested_model_returns_false_for_not_nested(self):

        assert is_nested_model(SimpleStruct) is False


class TestStruct(object):

    def test_simple_struct_gets_string(self):

        struct = SimpleStruct(x="2")
        assert struct.x == '2'
    
    def test_template_gives_correct_template(self):

        struct = SimpleStruct(x="2")
        template = struct_template(struct)
        print(template)
        assert template['x'].is_required is True
        assert template['x'].type_ == type('text')

    def test_template_gives_correct_template_with_nested(self):

        struct = NestedStruct(simple=SimpleStruct(x="2"))
        template = struct_template(struct)
        assert template['simple']['x'].is_required is True
        assert template['simple']['x'].type_ == type('text')

    def test_to_text_converts_to_text(self):
        struct = SimpleStruct(x="2")
        text = model_to_text(struct)
        assert "2" in text

    def test_to_text_doubles_the_braces(self):
        struct = SimpleStruct(x="2")
        text = model_to_text(struct, True)
        assert "{{" in text
        assert "}}" in text

    def test_to_text_works_for_nested(self):
        struct = NestedStruct(simple=SimpleStruct(x="2"))
        text = model_to_text(struct, True)
        assert text.count('{{') == 2
        assert text.count("}}") == 2

    def test_to_dict_converts_to_a_dict(self):
        struct = SimpleStruct(x="2")
        d = struct.model_dump()
        assert d['x'] == "2"
