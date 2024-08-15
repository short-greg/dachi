from dachi._core import _structs_doc as _struct



# class TestDoc(object):

#     def test_doc_text_is_a_string(self):

#         doc = _struct.Doc(
#             name='document name', 
#             text='hi, how are you'
#         )
#         assert doc.name == 'document name'
#         assert doc.text == 'hi, how are you'


# class TestMessageList(object):

#     def test_chat_adds_several_messages_correctly(self):

#         message = _struct.Message(
#             role='assistant', 
#             content={'text': 'hi, how are you'}
#         )
#         message2 = _struct.Message(
#             role='user', content={'text': "i'm fine and you?"}
#         )
#         chat = _struct.MessageList(
#             structs=[message, message2]
#         )
#         assert chat.messages[0].content['text'] == 'hi, how are you'
#         assert chat.messages[1].content['text'] == "i'm fine and you?"
