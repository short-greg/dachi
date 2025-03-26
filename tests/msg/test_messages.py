from dachi.msg import _messages as M


class TextMessage(M.Msg):

    def __init__(self, role: str, content: str):

        return super().__init__(
            role=role, content=content
        )


class TestListDialog(object):

    def test_dialog_creates_message_list(self):

        message = TextMessage('assistant', 'help')
        message2 = TextMessage('system', 'help the user')
        dialog = M.ListDialog(
            messages=[message, message2]
        )
        assert dialog[0] is message
        assert dialog[1] is message2

    def test_dialog_replaces_the_message(self):

        message = TextMessage('assistant', 'help')
        message2 = TextMessage('system', 'help the user')
        dialog = M.ListDialog(
            messages=[message, message2]
        )
        dialog.replace(0, M.Msg(role="system", content="Stop!"))
        assert dialog[1] is message2
        assert dialog[0].content == 'Stop!'

    def test_dialog_inserts_into_correct_position(self):

        message = TextMessage('assistant', 'help')
        message2 = TextMessage('system', 'help the user')
        dialog = M.ListDialog(
            messages=[message, message2]
        )
        dialog.insert(
            0, M.Msg(role="system", content="Stop!")
        )
        assert len(dialog) == 3
        assert dialog[2] is message2
        assert dialog[0].content == 'Stop!'

    def test_aslist_converts_to_a_list(self):

        message = TextMessage('assistant', 'help')
        message2 = TextMessage('system', 'help the user')
        dialog = M.ListDialog(
            messages=[message, message2]
        )
        dialog.insert(
            0, M.Msg(role="system", content="Stop!")
        )
        dialog.insert(
            0, M.Msg(role='system', content='Stop!')
        )
        assert isinstance(dialog.aslist(), list)

    def test_append_returns_the_dialog(self):

        message = TextMessage('assistant', 'help')
        message2 = TextMessage('system', 'help the user')
        dialog = M.ListDialog(
            messages=[message, message2]
        )
        dialog2 = dialog.append(
            M.Msg(role="system", content="Stop!")
        )
        assert dialog is dialog2

    def test_clone_creates_a_new_dialog(self):

        message = TextMessage('assistant', 'help')
        message2 = TextMessage('system', 'help the user')
        dialog = M.ListDialog(
            messages=[message, message2]
        )
        dialog2 = dialog.clone()
        assert dialog is not dialog2
        assert dialog[0] is dialog2[0]


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

