from dachi.store import _struct
import pydantic
import typing

# TODO Move

# class TestRole:

#     def test_role_sets_name(self):

#         assistant = _struct.Role(name='assistant')
#         assert assistant.name == 'assistant'

#     def test_role_has_default_descr(self):

#         assistant = _struct.Role(name='assistant')
#         assert assistant.descr == ''


# class TestText:

#     def test_text_has_text_field(self):

#         text = _struct.Text(text='assistant')
#         assert text.text == 'assistant'



class SimpleStruct(_struct.Struct):

    x: _struct.Str


class NestedStruct(_struct.Struct):

    simple: SimpleStruct


class TestStr(object):

    def test_str_formats_the_string(self):

        str_ = _struct.Str(text='Here is the text {field}', vars=['field'])
        result = str_(
            field=1
        )
        assert result.text == 'Here is the text 1'

    def test_str_formats_the_string_with_str(self):

        str_ = _struct.Str(text='Here is the text {field}', vars=['field'])
        str2 = _struct.Str(text='1', vars=[])
        result = str_(
            field=str2
        )
        assert result.text == 'Here is the text 1'

    def test_str_does_not_format_if_var_not_available(self):

        str_ = _struct.Str(text='Here is the text {field}', vars=['field'])
        result = str_(
            field2="1"
        )
        assert result.text == 'Here is the text {field}'

    def test_can_format_str_after_fails_once(self):

        str_ = _struct.Str(text='Here is the text {field}', vars=['field'])
        result = str_(
            field2="1"
        )
        result = result(
            field="1"
        )
        assert result.text == 'Here is the text 1'

    def test_vars_remaining_after_format(self):

        str_ = _struct.Str(text='Here is the text {field}', vars=['field'])
        result = str_(
            field2="1"
        )
        assert result.vars == ['field']


class TestStruct(object):

    def test_simple_struct_converts_to_Str(self):

        struct = SimpleStruct(x="2")
        assert struct.x.text == '2'

    def test_simple_struct_pudates_text_on_forward(self):

        struct = SimpleStruct(x=_struct.Str(text='{adj} job', vars=['adj']))
        struct = struct(adj='great')
        assert struct.x.text == 'great job'

    def test_nested_struct_pudates_text_on_forward(self):
        simple = SimpleStruct(x=_struct.Str(text='{adj} job', vars=['adj']))
        struct = NestedStruct(simple=simple)
        struct = struct(adj='great')
        assert struct.simple.x.text == 'great job'

    def test_template_gives_correct_template(self):

        struct = SimpleStruct(x="2")
        template = struct.template()
        assert template['x']['required'] is True
        assert template['x']['type'] == type('text')
