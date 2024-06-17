# from dachi.instruct.depracate import _core as _instruct
from dachi._core import _struct


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

        struct = SimpleStruct(name='x', x="2")
        assert struct.x.text == '2'

    # def test_simple_struct_pudates_text_on_forward(self):

    #     struct = SimpleStruct(name='x', x=_struct.Str(text='{adj} job', vars=['adj']))
    #     struct = struct.update(adj='great')
    #     assert struct.x.text == 'great job'

    # def test_nested_struct_pudates_text_on_forward(self):
    #     simple = SimpleStruct(name='x', x=_struct.Str(text='{adj} job', vars=['adj']))
    #     struct = NestedStruct(name='y', simple=simple)
    #     struct = struct.update(adj='great')
    #     assert struct.simple.x.text == 'great job'

    def test_template_gives_correct_template(self):

        struct = SimpleStruct(name='x', x="2")
        template = struct.template()
        assert template['x']['text']['is_required'] is True
        assert template['x']['text']['type'] == type('text')

    def test_template_gives_correct_template_with_nested(self):

        struct = NestedStruct(name='x', simple=SimpleStruct(name='x', x="2"))
        template = struct.template()
        assert template['simple']['x']['text']['is_required'] is True
        assert template['simple']['x']['text']['type'] == type('text')
