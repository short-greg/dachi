# from dachi.core import _msg as M
# from dachi import proc as P


# class TextMessage(M.Msg):

#     def __init__(self, role: str, content: str):

#         return super().__init__(
#             role=role, content=content
#         )


# class TestMessage(object):

#     def test_message_sets_data(self):

#         message = M.Msg(role='assistant', question='How?')
#         assert message.role == 'assistant'
#         assert message.question == 'How?'

#     def test_message_role_is_a_string(self):

#         message = M.Msg(role='assistant', content='hi, how are you')
#         assert message.role == 'assistant'
#         assert message.content == 'hi, how are you'

#     def test_message_sets_data(self):

#         message = M.Msg(role='assistant', question='How?')
#         assert message.role == 'assistant'
#         assert message.question == 'How?'

#     # def test_render_renders_the_message_with_colon(self):

#     #     message = M.Msg(role='assistant', content='hi, how are you')
#     #     rendered = message.render()
#     #     assert rendered == 'assistant: hi, how are you'

# class TestListDialog:

#     def test_pop_removes_and_returns_message(self):
#         message1 = TextMessage('assistant', 'help')
#         message2 = TextMessage('system', 'help the user')
#         dialog = M.ListDialog(messages=[message1, message2])
#         popped_message = dialog.pop(0)
#         assert popped_message is message1
#         assert len(dialog) == 1
#         assert dialog[0] is message2

#     def test_remove_deletes_message(self):
#         message1 = TextMessage('assistant', 'help')
#         message2 = TextMessage('system', 'help the user')
#         dialog = M.ListDialog(messages=[message1, message2])
#         dialog.remove(message1)
#         assert len(dialog) == 1
#         assert dialog[0] is message2

#     def test_extend_with_list_of_messages(self):
#         message1 = TextMessage('assistant', 'help')
#         message2 = TextMessage('system', 'help the user')
#         message3 = TextMessage('user', 'thank you')
#         dialog = M.ListDialog(messages=[message1, message2])
#         dialog.extend([message3])
#         assert len(dialog) == 3
#         assert dialog[2] is message3

#     def test_extend_with_another_dialog(self):
#         message1 = TextMessage('assistant', 'help')
#         message2 = TextMessage('system', 'help the user')
#         message3 = TextMessage('user', 'thank you')
#         dialog1 = M.ListDialog(messages=[message1])
#         dialog2 = M.ListDialog(messages=[message2, message3])
#         dialog1.extend(dialog2)
#         assert len(dialog1) == 3
#         assert dialog1[1] is message2
#         assert dialog1[2] is message3

#     def test_render_creates_correct_string(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         msg_renderer = M.FieldRenderer()
#         dialog = M.ListDialog(
#             messages=[message1, message2], 
#         )
#         rendered = msg_renderer(dialog)
#         assert rendered == 'assistant: Hello\nuser: Hi'

#     def test_to_input_converts_to_api_format(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         dialog = M.ListDialog(messages=[message1, message2])
#         inputs = dialog.to_input()
#         assert isinstance(inputs, list)
#         assert inputs[0] == message1.to_input()
#         assert inputs[1] == message2.to_input()

#     def test_clone_creates_independent_copy(self):
#         message1 = TextMessage('assistant', 'help')
#         message2 = TextMessage('system', 'help the user')
#         dialog = M.ListDialog(messages=[message1, message2])
#         cloned_dialog = dialog.clone()
#         assert cloned_dialog is not dialog
#         assert cloned_dialog[0] is message1
#         assert cloned_dialog[1] is message2
#         cloned_dialog.append(TextMessage('user', 'thank you'))
#         assert len(dialog) == 2
#         assert len(cloned_dialog) == 3

#     def test_dialog_creates_message_list(self):

#         message = TextMessage('assistant', 'help')
#         message2 = TextMessage('system', 'help the user')
#         dialog = M.ListDialog(
#             messages=[message, message2]
#         )
#         assert dialog[0] is message
#         assert dialog[1] is message2

#     def test_dialog_replaces_the_message(self):

#         message = TextMessage('assistant', 'help')
#         message2 = TextMessage('system', 'help the user')
#         dialog = M.ListDialog(
#             messages=[message, message2]
#         )
#         dialog.replace(0, M.Msg(role="system", content="Stop!"))
#         assert dialog[1] is message2
#         assert dialog[0].content == 'Stop!'

#     def test_dialog_inserts_into_correct_position(self):

#         message = TextMessage('assistant', 'help')
#         message2 = TextMessage('system', 'help the user')
#         dialog = M.ListDialog(
#             messages=[message, message2]
#         )
#         dialog.insert(
#             0, M.Msg(role="system", content="Stop!")
#         )
#         assert len(dialog) == 3
#         assert dialog[2] is message2
#         assert dialog[0].content == 'Stop!'

#     def test_aslist_converts_to_a_list(self):

#         message = TextMessage('assistant', 'help')
#         message2 = TextMessage('system', 'help the user')
#         dialog = M.ListDialog(
#             messages=[message, message2]
#         )
#         dialog.insert(
#             0, M.Msg(role="system", content="Stop!")
#         )
#         dialog.insert(
#             0, M.Msg(role='system', content='Stop!')
#         )
#         assert isinstance(dialog.aslist(), list)

#     def test_append_returns_the_dialog(self):

#         message = TextMessage('assistant', 'help')
#         message2 = TextMessage('system', 'help the user')
#         dialog = M.ListDialog(
#             messages=[message, message2]
#         )
#         dialog2 = dialog.append(
#             M.Msg(role="system", content="Stop!")
#         )
#         assert dialog is dialog2

#     def test_clone_creates_a_new_dialog(self):

#         message = TextMessage('assistant', 'help')
#         message2 = TextMessage('system', 'help the user')
#         dialog = M.ListDialog(
#             messages=[message, message2]
#         )
#         dialog2 = dialog.clone()
#         assert dialog is not dialog2
#         assert dialog[0] is dialog2[0]


# class TestDialogTurn:

#     def test_root_returns_root_node(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         child_turn = root_turn.append(message2)
#         assert child_turn.root() is root_turn

#     def test_depth_returns_2(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         leaf_turn = root_turn.append(message2)
#         assert leaf_turn.depth() == 2

#     def test_depth_returns_1_if_root(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         root_turn.append(message2)

#         assert root_turn.depth() == 1

#     def test_child_returns_correct_child(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         child_turn = root_turn.append(message2)
#         assert root_turn.child(0) is child_turn

#     def test_find_locates_message(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         child_turn = root_turn.append(message2)
#         found_turn = root_turn.find(child_turn)
#         assert found_turn is child_turn

#     def test_prune_removes_and_returns_message(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         message3 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         child_turn = root_turn.append(message2)
#         grandchild_turn = child_turn.append(message3)
#         pruned = root_turn.prune(0)
#         assert pruned.message == message2
#         assert grandchild_turn.depth() == 2

#     def test_prune_results_in_one_child(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         message3 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         root_turn.append(message2)
#         root_turn.append(message3)
#         root_turn.prune(0)
#         assert root_turn.n_children == 1

#     def test_leaf_returns_self_if_no_children(self):
#         message = M.Msg(role='assistant', content='Hello')
#         turn = M.DialogTurn(message=message)
#         assert turn.leaf() is turn

#     def test_leaf_returns_leftmost_leaf(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         message3 = M.Msg(role='system', content='Goodbye')
#         root_turn = M.DialogTurn(message=message1)
#         child_turn = root_turn.append(message2)
#         child_turn = child_turn.append(message3)
#         assert root_turn.leaf() is child_turn

#     def test_ancestors_returns_correct_order(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         message3 = M.Msg(role='system', content='Goodbye')
#         root_turn = M.DialogTurn(message=message1)
#         child_turn = root_turn.append(message2)
#         grandchild_turn = child_turn.append(message3)
#         ancestors = list(grandchild_turn.ancestors)
#         assert ancestors == [child_turn, root_turn]

#     def test_prepend_creates_new_parent(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='system', content='Goodbye')
#         turn = M.DialogTurn(message=message1)
#         new_parent = turn.prepend(message2)
#         assert new_parent.message == message2
#         assert new_parent.child(0) is turn
#         assert turn.parent is new_parent

#     def test_append_creates_new_child(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         turn = M.DialogTurn(message=message1)
#         child_turn = turn.append(message2)
#         assert child_turn.message == message2
#         assert child_turn.parent is turn
#         assert turn.child(0) is child_turn

#     def test_ascend_raises_error_if_too_high(self):
#         message = M.Msg(role='assistant', content='Hello')
#         turn = M.DialogTurn(message=message)
#         try:
#             turn.ancestor(1)
#         except ValueError as e:
#             assert str(e) == "Cannot ascend 1.Only 0 parents."

#     def test_ascend_returns_correct_ancestor(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         child_turn = root_turn.append(message2)
#         assert child_turn.ancestor(1) is root_turn

#     def test_find_val_locates_message(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         child_turn = root_turn.append(message2)
#         found_turn = root_turn.find_val(message2)
#         assert found_turn is child_turn

#     def test_find_val_returns_none_if_not_found(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         child_turn = M.DialogTurn(message=message2)
#         child_turn = root_turn.append(child_turn)
#         assert child_turn.find_val(message1) is None

#     def test_prune_removes_correct_child(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         child_turn = root_turn.append(message2)
#         pruned = root_turn.prune(0)
#         assert pruned is child_turn
#         assert pruned.parent is None
#         assert root_turn.n_children == 0

#     def test_prune_raises_error_for_invalid_index(self):
#         message = M.Msg(role='assistant', content='Hello')
#         root_turn = M.DialogTurn(message=message)
#         try:
#             root_turn.prune(0)
#         except IndexError as e:
#             assert str(e) == "Index out of range for pruning."

#     def test_sibling_returns_correct_sibling(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         message3 = M.Msg(role='system', content='Goodbye')
#         root_turn = M.DialogTurn(message=message1)
#         child1 = root_turn.append(message2)
#         child2 = root_turn.append(message3)
#         assert child1.sibling(1) is child2

#     def test_sibling_raises_error_if_no_parent(self):
#         message = M.Msg(role='assistant', content='Hello')
#         turn = M.DialogTurn(message=message)
#         try:
#             turn.sibling(1)
#         except RuntimeError as e:
#             assert str(e) == "There is no parent so must be 1."

#     def test_n_children_returns_correct_count(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         root_turn = M.DialogTurn(message=message1)
#         root_turn.append(message2)
#         assert root_turn.n_children == 1

#     def test_n_siblings_returns_correct_count(self):
#         message1 = M.Msg(role='assistant', content='Hello')
#         message2 = M.Msg(role='user', content='Hi')
#         message3 = M.Msg(role='system', content='Goodbye')
#         root_turn = M.DialogTurn(message=message1)
#         root_turn.append(message2)
#         root_turn.append(message3)
#         child_turn = root_turn.child(0)
#         assert child_turn.n_siblings == 2

#     def test_sibling_returns_correct_sibling(self):
#         message1 = M.Msg(
#             role='assistant', 
#             content='Hello'
#         )
#         message2 = M.Msg(role='user', content='Hi')
#         message3 = M.Msg(role='user', content='Hi2')
#         root_turn = M.DialogTurn(
#             message=message1
#         )
#         child1 = root_turn.append(message2)
#         child2 = root_turn.append(message3)
#         assert child2.sibling(0) is child1


# class TestTreeDialog:

#     def test_initialization_creates_empty_dialog(self):
#         dialog = M.TreeDialog()
#         assert len(dialog) == 0
#         assert dialog._leaf is None
#         assert dialog._root is None

#     def test_append_adds_message_to_dialog(self):
#         dialog =M.TreeDialog()
#         message = M.Msg(role="assistant", content="Hello")
#         dialog.append(message)
#         assert len(dialog) == 1
#         assert dialog._leaf.message == message

#     def test_extend_with_list_of_messages(self):
#         dialog = M.TreeDialog()
#         messages = [
#             M.Msg(role="assistant", content="Hello"),
#             M.Msg(role="user", content="Hi")
#         ]
#         dialog.extend(messages)
#         assert len(dialog) == 2
#         # print(type(dialog[0]), type(messages[0]))
#         assert dialog[0] is messages[0]
#         assert dialog[1] is messages[1]

#     def test_extend_with_another_dialog(self):
#         dialog1 =M.TreeDialog()
#         dialog2 =M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog1.append(message1)
#         dialog2.append(message2)
#         dialog1.extend(dialog2)
#         assert len(dialog1) == 2
#         assert dialog1[1] == message2

#     def test_replace_replaces_message_at_index(self):
#         dialog =M.TreeDialog()
#         message1 = M.Msg(
#             role="assistant", 
#             content="Hello"
#         )
#         message2 = M.Msg(
#             role="user", content="Hi"
#         )
#         dialog.append(message1)
#         dialog.replace(0, message2)
#         assert dialog[0] == message2

#     def test_insert_inserts_message_at_index(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.append(message1)
#         dialog.insert(0, message2)
#         assert len(dialog) == 2
#         assert dialog[0] == message2

#     # def test_pop_removes_message_at_index(self):
#     #     dialog = M.TreeDialog()
#     #     message1 = M.Msg(role="assistant", content="Hello")
#     #     message2 = M.Msg(role="user", content="Hi")
#     #     dialog.extend([message1, message2])
#     #     dialog.pop(0)
#     #     assert len(dialog) == 1
#     #     assert dialog[0] == message2

#     def test_clone_creates_independent_copy(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.extend([message1, message2])
#         cloned_dialog = dialog.clone()
#         assert cloned_dialog is not dialog
#         assert len(cloned_dialog) == len(dialog)
#         assert cloned_dialog[0] == dialog[0]

#     def test_clone_creates_independent_copy_with_none(self):
#         dialog = M.TreeDialog()
#         cloned_dialog = dialog.clone()
#         assert cloned_dialog is not dialog
#         assert len(cloned_dialog) == 0

#     def test_clone_creates_independent_copy_with_complex(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         message3 = M.Msg(role="user", content="Yo")
#         message4 = M.Msg(role="user", content="Mo")
#         dialog.extend([message1, message2])
#         dialog.ancestor(1)
#         dialog.extend([message3, message4])
#         cloned_dialog = dialog.clone()
#         assert cloned_dialog is not dialog
#         assert len(cloned_dialog) == len(dialog)
#         assert len(cloned_dialog) == 3
#         assert cloned_dialog[2] == dialog[2]

#     def test_remove_deletes_message(self):
#         pass
#         # dialog = M.TreeDialog()
#         # message1 = M.Msg(role="assistant", content="Hello")
#         # message2 = M.Msg(role="user", content="Hi")
#         # dialog.extend([message1, message2])
#         # dialog.remove(message1)
#         # assert len(dialog) == 1
#         # assert dialog[0] == message2

#     # def test_list_messages_returns_all_messages(self):
#     #     dialog = M.TreeDialog()
#     #     message1 = M.Msg(role="assistant", content="Hello")
#     #     message2 = M.Msg(role="user", content="Hi")
#     #     dialog.extend([message1, message2])
#     #     messages = dialog.list_messages()
#     #     assert len(messages) == 2
#     #     assert messages[0] == message1
#     #     assert messages[1] == message2

#     def test_indices_property_returns_correct_indices(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.extend([message1, message2])
#         assert dialog.indices == [0, 0]

#     def test_indices_property_returns_correct_indices_with2(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.extend([message1, message2])
#         dialog.ancestor(1)
#         dialog.append(M.Msg(role="user", content="Hi"))
#         assert dialog.indices == [0, 1]

#     def test_counts_property_returns_correct_counts(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.extend([message1, message2])
#         assert dialog.counts == [1, 1]

#     def test_counts_property_returns_correct_counts_with2(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.extend([message1, message2])
#         dialog.ancestor(1)
#         dialog.append(M.Msg(role="user", content="Hi"))
#         assert dialog.counts == [1, 2]

#     def test_ascend_moves_leaf_up(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.extend([message1, message2])
#         dialog.ancestor(1)
#         assert dialog._leaf.message == message1

#     def test_sibling_moves_to_correct_sibling(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.extend([message1, message2])
#         dialog.ancestor(1)
#         msg3 = M.Msg(role="user", content="Hi")
#         dialog.append(msg3)
#         dialog.sibling(1)
#         assert dialog._leaf.message is msg3

#     def test_len_returns_correct_count(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.extend([message1, message2])
#         assert len(dialog) == 2

#     def test_iter_yields_all_messages(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.extend([message1, message2])
#         messages = list(iter(dialog))
#         assert len(messages) == 2
#         assert messages[0] == message1
#         assert messages[1] == message2

#     def test_getitem_retrieves_correct_message(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(role="assistant", content="Hello")
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.extend([message1, message2])
#         assert dialog[0] == message1
#         assert dialog[1] == message2

#     def test_setitem_updates_message_at_index(self):
#         dialog = M.TreeDialog()
#         message1 = M.Msg(
#             role="assistant", 
#             content="Hello"
#         )
#         message2 = M.Msg(role="user", content="Hi")
#         dialog.append(message1)
#         dialog[0] = message2
#         assert dialog[0] == message2


# class TestExcludeRole:

#     def test_exclude_single_role(self):
#         messages = [
#             M.Msg(role="user", content="hi"),
#             M.Msg(role="system", content="hello"),
#             M.Msg(role="assistant", content="how can I help?")
#         ]
#         filtered = M.exclude_role(messages, "system")
#         assert len(filtered) == 2
#         assert all(msg.role != "system" for msg in filtered)

#     def test_exclude_multiple_roles(self):
#         messages = [
#             M.Msg(role="user", content="hi"),
#             M.Msg(role="system", content="hello"),
#             M.Msg(role="assistant", content="how can I help?")
#         ]
#         filtered = M.exclude_role(messages, "system", "assistant")
#         assert len(filtered) == 1
#         assert filtered[0].role == "user"

#     def test_exclude_no_roles(self):
#         messages = [
#             M.Msg(role="user", content="hi"),
#             M.Msg(role="system", content="hello"),
#             M.Msg(role="assistant", content="how can I help?")
#         ]
#         filtered = M.exclude_role(messages)
#         assert len(filtered) == 3
#         assert filtered == messages

#     def test_exclude_all_roles(self):
#         messages = [
#             M.Msg(role="user", content="hi"),
#             M.Msg(role="system", content="hello"),
#             M.Msg(role="assistant", content="how can I help?")
#         ]
#         filtered = M.exclude_role(messages, "user", "system", "assistant")
#         assert len(filtered) == 0

#     def test_exclude_with_empty_messages(self):
#         messages = []
#         filtered = M.exclude_role(messages, "user")
#         assert len(filtered) == 0

#     def test_exclude_with_nonexistent_role(self):
#         messages = [
#             M.Msg(role="user", content="hi"),
#             M.Msg(role="system", content="hello")
#         ]
#         filtered = M.exclude_role(messages, "assistant")
#         assert len(filtered) == 2
#         assert filtered == messages

# class TestIncludeRole:

#     def test_include_single_role(self):
#         messages = [
#             M.Msg(role="user", content="hi"),
#             M.Msg(role="system", content="hello"),
#             M.Msg(role="assistant", content="how can I help?")
#         ]
#         filtered = M.include_role(messages, "system")
#         assert len(filtered) == 1
#         assert filtered[0].role == "system"

#     def test_include_multiple_roles(self):
#         messages = [
#             M.Msg(role="user", content="hi"),
#             M.Msg(role="system", content="hello"),
#             M.Msg(role="assistant", content="how can I help?")
#         ]
#         filtered = M.include_role(messages, "system", "assistant")
#         assert len(filtered) == 2
#         assert all(msg.role in {"system", "assistant"} for msg in filtered)

#     def test_include_no_roles(self):
#         messages = [
#             M.Msg(role="user", content="hi"),
#             M.Msg(role="system", content="hello"),
#             M.Msg(role="assistant", content="how can I help?")
#         ]
#         filtered = M.include_role(messages)
#         assert len(filtered) == 0

#     def test_include_all_roles(self):
#         messages = [
#             M.Msg(role="user", content="hi"),
#             M.Msg(role="system", content="hello"),
#             M.Msg(role="assistant", content="how can I help?")
#         ]
#         filtered = M.include_role(messages, "user", "system", "assistant")
#         assert len(filtered) == 3
#         assert filtered == messages

#     def test_include_with_empty_messages(self):
#         messages = []
#         filtered = M.include_role(messages, "user")
#         assert len(filtered) == 0

#     def test_include_with_nonexistent_role(self):
#         messages = [
#             M.Msg(role="user", content="hi"),
#             M.Msg(role="system", content="hello")
#         ]
#         filtered = M.include_role(messages, "assistant")
#         assert len(filtered) == 0

