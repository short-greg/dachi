from dachi.msg import _messages as M
from dachi import proc as P


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
        msg_renderer = P.FieldRenderer()
        dialog = M.ListDialog(
            messages=[message1, message2], 
        )
        rendered = msg_renderer(dialog)
        assert rendered == 'assistant: Hello\nuser: Hi'

    def test_to_input_converts_to_api_format(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        dialog = M.ListDialog(messages=[message1, message2])
        inputs = dialog.to_input()
        assert isinstance(inputs, list)
        assert inputs[0] == message1.to_input()
        assert inputs[1] == message2.to_input()

    def test_clone_creates_independent_copy(self):
        message1 = TextMessage('assistant', 'help')
        message2 = TextMessage('system', 'help the user')
        dialog = M.ListDialog(messages=[message1, message2])
        cloned_dialog = dialog.clone()
        assert cloned_dialog is not dialog
        assert cloned_dialog[0] is message1
        assert cloned_dialog[1] is message2
        cloned_dialog.append(TextMessage('user', 'thank you'))
        assert len(dialog) == 2
        assert len(cloned_dialog) == 3

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


class TestDialogTurn:

    def test_root_returns_root_node(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        root_turn = M.DialogTurn(message=message1)
        child_turn = root_turn.append(message2)
        assert child_turn.root() is root_turn

    def test_messages_returns_all_messages(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        root_turn = M.DialogTurn(message=message1)
        root_turn.append(message2)
        messages = root_turn.list_messages()
        assert len(messages) == 1
        assert messages[0] == root_turn.message

    def test_iter_yields_messages_in_order(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        root_turn = M.DialogTurn(message=message1)
        child_turn = root_turn.append(message2)

        messages = list(child_turn)
        assert len(messages) == 2
        assert messages[0] == message1
        assert messages[1] == message2

    def test_child_returns_correct_child(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        root_turn = M.DialogTurn(message=message1)
        child_turn = root_turn.append(message2)
        assert root_turn.child(0) is child_turn

    def test_find_locates_message(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        root_turn = M.DialogTurn(message=message1)
        child_turn = root_turn.append(message2)
        found_turn = root_turn.find(message2)
        assert found_turn is child_turn

    def test_pop_removes_and_returns_message(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        root_turn = M.DialogTurn(message=message1)
        child_turn = root_turn.append(message2)
        dialog, popped_message = child_turn.pop(1, get_msg=True)
        assert popped_message == message2
        assert len(dialog.list_messages()) == 1

    def test_remove_deletes_message(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        root_turn = M.DialogTurn(message=message1)
        root_turn.append(message2)
        root_turn.remove(message2)
        assert len(root_turn.list_messages()) == 1
        assert root_turn.list_messages()[0] == message1

    def test_extend_adds_messages(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        message3 = M.Msg(role='system', content='Goodbye')
        root_turn = M.DialogTurn(message=message1)
        child_turn = root_turn.append(message2)
        child_turn = child_turn.extend([message3])
        assert len(child_turn.list_messages()) == 3
        assert child_turn.list_messages()[2] == message3

    def test_clone_creates_independent_copy(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        root_turn = M.DialogTurn(message=message1)
        child_turn = root_turn.append(message2)
        cloned_turn = child_turn.clone()
        assert cloned_turn is not root_turn
        assert cloned_turn.list_messages() == root_turn.list_messages()

    def test_append_adds_message(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        root_turn = M.DialogTurn(message=message1)
        appended_turn = root_turn.append(message2)
        assert len(appended_turn.list_messages()) == 2
        assert appended_turn._message == message2

    def test_insert_inserts_message_at_position(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        message3 = M.Msg(role='system', content='Goodbye')
        root_turn = M.DialogTurn(message=message1)
        child_turn = root_turn.append(message2)
        inserted_turn = child_turn.insert(1, message3)
        assert len(child_turn.list_messages()) == 3
        assert child_turn.list_turns()[1] is inserted_turn

    def test_replace_replaces_message_at_index(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        message3 = M.Msg(role='system', content='Goodbye')
        root_turn = M.DialogTurn(message=message1)
        child_turn = root_turn.append(message2)
        replaced_turn = child_turn.replace(message3, 1)
        assert replaced_turn.list_messages()[1] == message3

    def test_replace_replaces_message_at_parent_index(self):
        message1 = M.Msg(role='assistant', content='Hello')
        message2 = M.Msg(role='user', content='Hi')
        message3 = M.Msg(role='system', content='Goodbye')
        root_turn = M.DialogTurn(message=message1)
        child_turn = root_turn.append(message2)
        child_turn.replace(message3, 0)
        assert child_turn.list_messages()[1] == message2


class TestExcludeRole:

    def test_exclude_single_role(self):
        messages = [
            M.Msg(role="user", content="hi"),
            M.Msg(role="system", content="hello"),
            M.Msg(role="assistant", content="how can I help?")
        ]
        filtered = M.exclude_role(messages, "system")
        assert len(filtered) == 2
        assert all(msg.role != "system" for msg in filtered)

    def test_exclude_multiple_roles(self):
        messages = [
            M.Msg(role="user", content="hi"),
            M.Msg(role="system", content="hello"),
            M.Msg(role="assistant", content="how can I help?")
        ]
        filtered = M.exclude_role(messages, "system", "assistant")
        assert len(filtered) == 1
        assert filtered[0].role == "user"

    def test_exclude_no_roles(self):
        messages = [
            M.Msg(role="user", content="hi"),
            M.Msg(role="system", content="hello"),
            M.Msg(role="assistant", content="how can I help?")
        ]
        filtered = M.exclude_role(messages)
        assert len(filtered) == 3
        assert filtered == messages

    def test_exclude_all_roles(self):
        messages = [
            M.Msg(role="user", content="hi"),
            M.Msg(role="system", content="hello"),
            M.Msg(role="assistant", content="how can I help?")
        ]
        filtered = M.exclude_role(messages, "user", "system", "assistant")
        assert len(filtered) == 0

    def test_exclude_with_empty_messages(self):
        messages = []
        filtered = M.exclude_role(messages, "user")
        assert len(filtered) == 0

    def test_exclude_with_nonexistent_role(self):
        messages = [
            M.Msg(role="user", content="hi"),
            M.Msg(role="system", content="hello")
        ]
        filtered = M.exclude_role(messages, "assistant")
        assert len(filtered) == 2
        assert filtered == messages

class TestIncludeRole:

    def test_include_single_role(self):
        messages = [
            M.Msg(role="user", content="hi"),
            M.Msg(role="system", content="hello"),
            M.Msg(role="assistant", content="how can I help?")
        ]
        filtered = M.include_role(messages, "system")
        assert len(filtered) == 1
        assert filtered[0].role == "system"

    def test_include_multiple_roles(self):
        messages = [
            M.Msg(role="user", content="hi"),
            M.Msg(role="system", content="hello"),
            M.Msg(role="assistant", content="how can I help?")
        ]
        filtered = M.include_role(messages, "system", "assistant")
        assert len(filtered) == 2
        assert all(msg.role in {"system", "assistant"} for msg in filtered)

    def test_include_no_roles(self):
        messages = [
            M.Msg(role="user", content="hi"),
            M.Msg(role="system", content="hello"),
            M.Msg(role="assistant", content="how can I help?")
        ]
        filtered = M.include_role(messages)
        assert len(filtered) == 0

    def test_include_all_roles(self):
        messages = [
            M.Msg(role="user", content="hi"),
            M.Msg(role="system", content="hello"),
            M.Msg(role="assistant", content="how can I help?")
        ]
        filtered = M.include_role(messages, "user", "system", "assistant")
        assert len(filtered) == 3
        assert filtered == messages

    def test_include_with_empty_messages(self):
        messages = []
        filtered = M.include_role(messages, "user")
        assert len(filtered) == 0

    def test_include_with_nonexistent_role(self):
        messages = [
            M.Msg(role="user", content="hi"),
            M.Msg(role="system", content="hello")
        ]
        filtered = M.include_role(messages, "assistant")
        assert len(filtered) == 0

