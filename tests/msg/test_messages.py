from dachi.msg import _messages as M


class TextMessage(M.Msg):

    def __init__(self, role: str, content: str):

        return super().__init__(
            role=role, content=content
        )


class TestMessage(object):

    def test_message_sets_data(self):

        message = M.Msg(role='assistant', question='How?')
        assert message.role == 'assistant'
        assert message.question == 'How?'

    def test_message_role_is_a_string(self):

        message = M.Msg(role='assistant', content='hi, how are you')
        assert message.role == 'assistant'
        assert message.content == 'hi, how are you'

    def test_message_sets_data(self):

        message = M.Msg(role='assistant', question='How?')
        assert message.role == 'assistant'
        assert message.question == 'How?'

    # def test_render_renders_the_message_with_colon(self):

    #     message = M.Msg(role='assistant', content='hi, how are you')
    #     rendered = message.render()
    #     assert rendered == 'assistant: hi, how are you'

class TestListDialog:

    def test_pop_removes_and_returns_message(self):
        message1 = TextMessage('assistant', 'help')
        message2 = TextMessage('system', 'help the user')
        dialog = M.ListDialog(messages=[message1, message2])
        popped_message = dialog.pop(0)
        assert popped_message is message1
        assert len(dialog) == 1
        assert dialog[0] is message2

    def test_remove_deletes_message(self):
        message1 = TextMessage('assistant', 'help')
        message2 = TextMessage('system', 'help the user')
        dialog = M.ListDialog(messages=[message1, message2])
        dialog.remove(message1)
        assert len(dialog) == 1
        assert dialog[0] is message2

    def test_extend_with_list_of_messages(self):
        message1 = TextMessage('assistant', 'help')
        message2 = TextMessage('system', 'help the user')
        message3 = TextMessage('user', 'thank you')
        dialog = M.ListDialog(messages=[message1, message2])
        dialog.extend([message3])
        assert len(dialog) == 3
        assert dialog[2] is message3

    def test_extend_with_another_dialog(self):
        message1 = TextMessage('assistant', 'help')
        message2 = TextMessage('system', 'help the user')
        message3 = TextMessage('user', 'thank you')
        dialog1 = M.ListDialog(messages=[message1])
        dialog2 = M.ListDialog(messages=[message2, message3])
        dialog1.extend(dialog2)
        assert len(dialog1) == 3
        assert dialog1[1] is message2
        assert dialog1[2] is message3

    def test_render_creates_correct_string(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        dialog = M.ListDialog(messages=[message1, message2])
        rendered = dialog.render()
        assert rendered == 'assistant: Hello\nuser: Hi'

    # def test_to_input_converts_to_api_format(self):
    #     message1 = M.Msg(role='assistant', content='Hello')
    #     message2 = M.Msg(role='user', content='Hi')
    #     dialog = M.ListDialog(messages=[message1, message2])
    #     inputs = dialog.to_input()
    #     assert isinstance(inputs, list)
    #     assert inputs[0] == message1.to_input()
    #     assert inputs[1] == message2.to_input()

    # def test_clone_creates_independent_copy(self):
    #     message1 = TextMessage('assistant', 'help')
    #     message2 = TextMessage('system', 'help the user')
    #     dialog = M.ListDialog(messages=[message1, message2])
    #     cloned_dialog = dialog.clone()
    #     assert cloned_dialog is not dialog
    #     assert cloned_dialog[0] is message1
    #     assert cloned_dialog[1] is message2
    #     cloned_dialog.append(TextMessage('user', 'thank you'))
    #     assert len(dialog) == 2
    #     assert len(cloned_dialog) == 3

    # def test_dialog_creates_message_list(self):

    #     message = TextMessage('assistant', 'help')
    #     message2 = TextMessage('system', 'help the user')
    #     dialog = M.ListDialog(
    #         messages=[message, message2]
    #     )
    #     assert dialog[0] is message
    #     assert dialog[1] is message2

    # def test_dialog_replaces_the_message(self):

    #     message = TextMessage('assistant', 'help')
    #     message2 = TextMessage('system', 'help the user')
    #     dialog = M.ListDialog(
    #         messages=[message, message2]
    #     )
    #     dialog.replace(0, M.Msg(role="system", content="Stop!"))
    #     assert dialog[1] is message2
    #     assert dialog[0].content == 'Stop!'

    # def test_dialog_inserts_into_correct_position(self):

    #     message = TextMessage('assistant', 'help')
    #     message2 = TextMessage('system', 'help the user')
    #     dialog = M.ListDialog(
    #         messages=[message, message2]
    #     )
    #     dialog.insert(
    #         0, M.Msg(role="system", content="Stop!")
    #     )
    #     assert len(dialog) == 3
    #     assert dialog[2] is message2
    #     assert dialog[0].content == 'Stop!'

    # def test_aslist_converts_to_a_list(self):

    #     message = TextMessage('assistant', 'help')
    #     message2 = TextMessage('system', 'help the user')
    #     dialog = M.ListDialog(
    #         messages=[message, message2]
    #     )
    #     dialog.insert(
    #         0, M.Msg(role="system", content="Stop!")
    #     )
    #     dialog.insert(
    #         0, M.Msg(role='system', content='Stop!')
    #     )
    #     assert isinstance(dialog.aslist(), list)

    # def test_append_returns_the_dialog(self):

    #     message = TextMessage('assistant', 'help')
    #     message2 = TextMessage('system', 'help the user')
    #     dialog = M.ListDialog(
    #         messages=[message, message2]
    #     )
    #     dialog2 = dialog.append(
    #         M.Msg(role="system", content="Stop!")
    #     )
    #     assert dialog is dialog2

    # def test_clone_creates_a_new_dialog(self):

    #     message = TextMessage('assistant', 'help')
    #     message2 = TextMessage('system', 'help the user')
    #     dialog = M.ListDialog(
    #         messages=[message, message2]
    #     )
    #     dialog2 = dialog.clone()
    #     assert dialog is not dialog2
    #     assert dialog[0] is dialog2[0]
