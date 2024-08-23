from dachi._core import _structs, str_formatter
from pydantic import Field
from test_core import SimpleStruct


class Role(_structs.Description):

    duty: str = Field(description='The duty of the role')

    def render(self) -> str:

        return f"""
        # Role {self.name}

        {self.duty}
        """
    
    def update(self, **kwargs) -> _structs.Description:
        return Role(name=self.name, duty=str_formatter(self.duty, **kwargs))


class TestRendering:

    def test_text_for_description_is_correct(self):
        
        role = Role(name='Assistant', duty='You are a helpful assistant')
        text = role.render()

        assert text == f"""
        # Role Assistant

        {role.duty}
        """

    def test_text_for_description_is_correct_after_updating(self):
        
        role = Role(name='Assistant', duty='You are a helpful {role}')
        
        role = role.update(role='Sales Assistant')
        text = role.render()
        assert 'Sales Assistant' in text


class TestRef:

    def test_ref_does_not_output_text(self):

        role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
        ref = _structs.Ref(desc=role)
        assert 'Helpful Assistant' in ref.desc.render()

    def test_name_returns_name_of_reference(self):

        role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
        ref = _structs.Ref(desc=role)
        # ref = ref.update(role='Helpful Assistant')
        assert ref.name == 'Assistant'

    def test_text_is_empty_string(self):

        role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
        ref = _structs.Ref(desc=role)
        # ref = ref.update(role='Helpful Assistant')
        assert ref.render() == role.name


class TestStructList:

    def test_struct_list_retrieves_item(self):

        l = _structs.StructList[SimpleStruct](
            [SimpleStruct(x='2'), SimpleStruct(x='3')]
        )
        assert l[0].x == '2'
        assert l[1].x == '3'

    def test_struct_sets_the_item(self):

        l = _structs.StructList[SimpleStruct](
            [SimpleStruct(x='4'), SimpleStruct(x='5')]
        )
        l[1] = SimpleStruct(x='8')
        assert l[1].x == '8'
    
    def test_struct_sets_the_item_with_none(self):

        l = _structs.StructList[SimpleStruct](
            [SimpleStruct(x='4'), SimpleStruct(x='5')]
        )
        l[None] = SimpleStruct(x='8')
        assert l[2].x == '8'
    


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
