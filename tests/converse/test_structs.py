# from dachi.instruct.depracate import _core as _instruct
from dachi._core import _struct
from dachi.adapt import Message, Doc, MessageList

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
