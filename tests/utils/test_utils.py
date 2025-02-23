from dachi import utils
from dachi.proc import model_to_text, struct_template
from pydantic import BaseModel
import pytest


class SimpleStruct(BaseModel):

    x: str


class NestedStruct(BaseModel):

    simple: SimpleStruct


class TestIsUndefined(object):

    def test_is_undefined(self):

        assert utils.is_undefined(
            utils.UNDEFINED
        )

    def test_not_is_undefined(self):

        assert not utils.is_undefined(
            1
        )


class TestIsNestedModel:

    def test_is_nested_model_returns_true_for_nested(self):

        assert utils.is_nested_model(NestedStruct) is True

    def test_is_nested_model_returns_false_for_not_nested(self):

        assert utils.is_nested_model(SimpleStruct) is False


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

    # def test_render_works_for_nested(self):
    #     struct = NestedStruct(simple=SimpleStruct(x="2"))
    #     text = utils.render(struct)
    #     assert text.count('{{') == 2
    #     assert text.count("}}") == 2


class TestStrFormatter(object):

    def test_formatter_formats_positional_variables(self):

        assert utils.str_formatter(
            '{} {}', 1, 2
        ) == '1 2'

    def test_formatter_formats_positional_variables(self):

        assert utils.str_formatter(
            '{0} {1}', 1, 2
        ) == '1 2'

    def test_formatter_formats_named_variables(self):

        assert utils.str_formatter(
            '{x} {y}', x=1, y=2
        ) == '1 2'

    def test_formatter_raises_error_if_positional_and_named_variables(self):

        with pytest.raises(ValueError):
            utils.str_formatter(
                '{0} {y}', 1, y=2
            )

    def test_get_variables_gets_all_pos_variables(self):

        assert utils.get_str_variables(
            '{0} {1}'
        ) == [0, 1]

    def test_get_variables_gets_all_named_variables(self):

        assert utils.get_str_variables(
            '{x} {y}'
        ) == ['x', 'y']


class TestGetMember(object):

    def test_get_member_gets_immediate_child(self):

        class X:
            y = 2

        x = X()

        assert utils.get_member(
            x, 'y'
        ) == 2

    def test_get_member_gets_sub_child(self):

        class X:
            y = 2

            def __getattr__(self, key):

                o = X()
                object.__setattr__(self, key, o)
                return o

        x = X()

        assert utils.get_member(
            x, 'z.y'
        ) == 2

