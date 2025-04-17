from dachi.msg import _render
from dachi.base import Renderable
import typing
import pytest
from pydantic import BaseModel
from dachi.msg import model_template


class NestedModel(BaseModel):
    nested_field: int


class DummyModel(BaseModel):
    required_field: str
    optional_field: typing.Optional[int]
    nested_model: NestedModel


class TestModelTemplate:

    def test_model_template_with_simple_model(self):
        class SimpleModel(BaseModel):
            field: str

        expected = {
            "field": {
                "is_required": True,
                "type": str
            }
        }
        assert model_template(SimpleModel) == expected

    def test_model_template_with_optional_field(self):
        class OptionalModel(BaseModel):
            optional_field: typing.Optional[int]

        expected = {
            "optional_field": {
                "is_required": True,
                "type": typing.Optional[int]
            }
        }
        result = model_template(OptionalModel)
        assert result == expected

    def test_model_template_with_optional_field(self):
        class OptionalModel(BaseModel):
            optional_field: typing.Optional[int] = 2

        expected = {
            "optional_field": {
                "is_required": False,
                "type": typing.Optional[int]
            }
        }
        result = model_template(OptionalModel)
        assert result == expected

    def test_model_template_with_nested_model(self):
        expected = {
            "required_field": {
                "is_required": True,
                "type": str
            },
            "optional_field": {
                "is_required": True,
                "type": typing.Optional[int]
            },
            "nested_model": {
                "nested_field": {
                    "is_required": True,
                    "type": int
                }
            }
        }
        assert model_template(DummyModel) == expected

    def test_model_template_with_empty_model(self):
        class EmptyModel(BaseModel):
            pass

        expected = {}
        assert model_template(EmptyModel) == expected

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

