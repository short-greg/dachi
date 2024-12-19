from dachi._core import _messages as M
from dachi._core._core import Cue, NullRead
import numpy as np

class TestDialog(object):

    def test_dialog_creates_message_list(self):

        message = M.TextMessage('assistant', 'help')
        message2 = M.TextMessage('system', 'help the user')
        dialog = M.Dialog(
            messages=[message, message2]
        )
        assert dialog[0] is message
        assert dialog[1] is message2

    def test_dialog_replaces_the_message(self):

        message = M.TextMessage('assistant', 'help')
        message2 = M.TextMessage('system', 'help the user')
        dialog = M.Dialog(
            messages=[message, message2]
        )
        dialog.system('Stop!', _ind=0, _replace=True)
        assert dialog[1] is message2
        assert dialog[0].text == 'Stop!'

    def test_dialog_inserts_into_correct_position(self):

        message = M.TextMessage('assistant', 'help')
        message2 = M.TextMessage('system', 'help the user')
        dialog = M.Dialog(
            messages=[message, message2]
        )
        dialog.system('Stop!', _ind=0, _replace=False)
        assert len(dialog) == 3
        assert dialog[2] is message2
        assert dialog[0].text == 'Stop!'

    def test_aslist_converts_to_a_list(self):

        message = M.TextMessage('assistant', 'help')
        message2 = M.TextMessage('system', 'help the user')
        dialog = M.Dialog(
            messages=[message, message2]
        )
        dialog.system('Stop!', _ind=0, _replace=False)
        assert isinstance(dialog.aslist(), list)


# # TODO: Add tests - Test all functionality here

class TestMessage(object):

    def test_message_sets_data(self):

        message = M.Message(role='assistant', question='How?')
        assert message.role == 'assistant'
        assert message.question == 'How?'

    def test_message_role_is_a_string(self):

        message = M.TextMessage(role='assistant', content='hi, how are you')
        assert message.role == 'assistant'
        assert message.text == 'hi, how are you'

    def test_render_renders_the_message_with_colon(self):

        message = M.TextMessage(role='assistant', content='hi, how are you')
        rendered = message.render()
        assert rendered == 'assistant: hi, how are you'


class TestMessage(object):

    def test_message_sets_data(self):

        message = M.Message(role='assistant', question='How?')
        assert message.role == 'assistant'
        assert message.question == 'How?'

class TestTextMessage(object):

    def test_message_role_is_a_string(self):

        message = M.TextMessage(role='assistant', content='hi, how are you')
        assert message.role == 'assistant'
        assert message.text == 'hi, how are you'

    def test_render_renders_the_message_with_colon(self):

        message = M.TextMessage(role='assistant', content='hi, how are you')
        rendered = message.render()
        assert rendered == 'assistant: hi, how are you'

    def test_delta_returns_the_latest_value(self):

        message = M.TextMessage(
            role='assistant', 
            content='hi, how are you',
            delta=M.Delta(content='u')
        )
        assert message.delta.content == 'u'


class TestEmbeddingMessage(object):

    def test_embedding_message_gets_embedding(self):

        message = M.EmbeddingMessage(
            role='assistant', source='hi, how are you', embedding=np.random.random((5,))
        )
        assert isinstance(message.embedding, np.ndarray)

    def test_embedding_message_gets_source(self):

        message = M.EmbeddingMessage(
            role='assistant', source='hi, how are you', embedding=np.random.random((5,))
        )
        assert message.source == 'hi, how are you'

    def test_render_renders_the_source(self):

        message = M.EmbeddingMessage(
            role='assistant', source='hi, how are you', embedding=np.random.random((5,))
        )
        assert message.source == 'hi, how are you'


class TestCueMessage(object):

    def test_cue_text_gets_the_cue_text(self):

        message = M.CueMessage(
            role='assistant', cue=Cue('Help the user')
        )
        assert message.cue.text == 'Help the user'

    def test_cue_text_gets_the_cue_text(self):

        message = M.CueMessage(
            role='assistant', cue=Cue('Help the user', out=NullRead())
        )
        assert isinstance(message.reader, NullRead)


class TestObjMessage(object):

    def test_obj_returns_the_object(self):

        message = M.ObjMessage(
            role='assistant', obj={'x': 'y'}, source= "{'x': 'y'}"
        )
        assert message.obj['x'] == 'y'

    def test_source_returns_the_message_source(self):

        message = M.ObjMessage(
            role='assistant', obj={'x': 'y'}, source="{'x': 'y'}"
        )
        assert message.source == "{'x': 'y'}"

    def test_text_returns_the_source(self):

        message = M.ObjMessage(
            role='assistant', obj={'x': 'y'}, source="{'x': 'y'}"
        )
        assert message.text == "{'x': 'y'}"


class TestFunctionMessage(object):

    def test_returns_name_of_function(self):

        message = M.FunctionMessage(
            role='assistant', name='exec', response={'y'}
        )
        assert message.name == 'exec'

    def test_returns_response_of_function(self):

        message = M.FunctionMessage(
            role='assistant', name='exec', response={'x': 'y'}
        )
        assert message.response['x'] == 'y'

    def test_render_renders_the_output(self):

        message = M.FunctionMessage(
            role='assistant', name='exec', response={'x': 'y'}
        )
        print(message.render())
        assert message.render() == "assistant: [exec] => {'x': 'y'}"


class TestToolOptionMessage(object):

    def test_returns_name_of_function(self):

        message = M.ToolOptionMessage(
            [M.FunctionTool('f', [M.ToolParam(name='x', type_='string')])]
        )
        assert message.tools[0].name == 'f'

    def test_returns_correct_param_for_function(self):

        message = M.ToolOptionMessage(
            [M.FunctionTool('f', [M.ToolParam(name='x', type_='string')])]
        )
        assert message.tools[0].params[0].name == 'x'

    def test_render_includes_all_function_names(self):

        message = M.ToolOptionMessage(
            [M.FunctionTool('f', [M.ToolParam(name='x', type_='string')]), M.FunctionTool('f2', [M.ToolParam(name='y', type_='string')])]
        )
        assert 'f' in message.render()
        assert 'f2' in message.render()

    def test_render_includes_all_function_names(self):

        message = M.ToolOptionMessage(
            [M.FunctionTool('f', [M.ToolParam(name='x', type_='string')]), M.FunctionTool('f2', [M.ToolParam(name='y', type_='string')])]
        )
        assert 'f' in message.text
        assert 'f2' in message.text
