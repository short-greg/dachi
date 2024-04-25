from dachi.instruct import _core as _instruct
from dachi._core import _struct


class SimpleStruct(_instruct.Instruct):

    x: _instruct.Str


class NestedStruct(_instruct.Instruct):

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

    def test_simple_struct_pudates_text_on_forward(self):

        struct = SimpleStruct(name='x', x=_instruct.Str(text='{adj} job', vars=['adj']))
        struct = struct.update(adj='great')
        assert struct.x.text == 'great job'

    def test_nested_struct_pudates_text_on_forward(self):
        simple = SimpleStruct(name='x', x=_instruct.Str(text='{adj} job', vars=['adj']))
        struct = NestedStruct(name='y', simple=simple)
        struct = struct.update(adj='great')
        assert struct.simple.x.text == 'great job'

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


class TestMessage(object):

    def test_message_role_is_a_string(self):

        message = _struct.Message(role='assistant', text='hi, how are you')
        assert message.role.text == 'assistant'
        assert message.text.text == 'hi, how are you'


class TestDoc(object):

    def test_doc_text_is_a_string(self):

        doc = _struct.Doc(name='document name', text='hi, how are you')
        assert doc.name.text == 'document name'
        assert doc.text.text == 'hi, how are you'


class TestChat(object):

    def test_chat_adds_several_messages_correctly(self):

        message = _struct.Message(role='assistant', text='hi, how are you')
        message2 = _struct.Message(role='user', text="i'm fine and you?")
        chat = _struct.Chat(
            messages=[message, message2]
        )
        assert chat.messages[0].text.text == 'hi, how are you'
        assert chat.messages[1].text.text == "i'm fine and you?"
