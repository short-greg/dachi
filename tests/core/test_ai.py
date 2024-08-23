from dachi._core import _core
# from dachi._core import _instruct as core
from dachi._core import Struct, str_formatter
import pytest
from pydantic import Field

import asyncio
from typing import Any, Iterator, Tuple
import pytest
from dachi._core import _process as p
from dachi._core._core import Module
import typing


class DummyAIModel(_core.AIModel):
    """APIAdapter allows one to adapt various WebAPI or other
    API for a consistent interface
    """

    target = 'Great!'

    def forward(self, prompt: _core.AIPrompt, **kwarg_override) -> _core.AIResponse:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        message = prompt.aslist()[0]
        result = self.convert(message)
        return _core.AIResponse(
            _core.TextMessage('assistant', self.target), result, self.target
        )

    def convert(self, message: _core.Message) -> typing.Dict:
        """Convert a message to the format needed for the model

        Args:
            messages (Message): The messages to convert

        Returns:
            typing.List[typing.Dict]: The format to pass to the "model"
        """
        return {'text': message['text']}


class TestDialog(object):

    def test_dialog_creates_message_list(self):

        message = _core.TextMessage('assistant', 'help')
        message2 = _core.TextMessage('system', 'help the user')
        dialog = _core.Dialog(
            messages=[message, message2]
        )
        assert dialog[0] is message
        assert dialog[1] is message2

    def test_dialog_replaces_the_message(self):

        message = _core.TextMessage('assistant', 'help')
        message2 = _core.TextMessage('system', 'help the user')
        dialog = _core.Dialog(
            messages=[message, message2]
        )
        dialog.system('Stop!', _ind=0, _replace=True)
        assert dialog[1] is message2
        assert dialog[0].text == 'Stop!'

    def test_dialog_inserts_into_correct_position(self):

        message = _core.TextMessage('assistant', 'help')
        message2 = _core.TextMessage('system', 'help the user')
        dialog = _core.Dialog(
            messages=[message, message2]
        )
        dialog.system('Stop!', _ind=0, _replace=False)
        assert len(dialog) == 3
        assert dialog[2] is message2
        assert dialog[0].text == 'Stop!'

    def test_aslist_converts_to_a_list(self):

        message = _core.TextMessage('assistant', 'help')
        message2 = _core.TextMessage('system', 'help the user')
        dialog = _core.Dialog(
            messages=[message, message2]
        )
        dialog.system('Stop!', _ind=0, _replace=False)
        assert isinstance(dialog.aslist(), list)

    def test_prompt_returns_a_message(self):

        message = _core.TextMessage('assistant', 'help')
        message2 = _core.TextMessage('system', 'help the user')
        dialog = _core.Dialog(
            messages=[message, message2]
        )
        result = dialog.prompt(DummyAIModel())
        assert result.val == DummyAIModel.target

    def test_stream_prompt_returns_a_message(self):

        message = _core.TextMessage('assistant', 'help')
        message2 = _core.TextMessage('system', 'help the user')
        dialog = _core.Dialog(
            messages=[message, message2]
        )
        for d, dx in dialog.stream_prompt(DummyAIModel()):
            pass
        assert dx.val == DummyAIModel.target



class TestMessage(object):

    def test_message_sets_data(self):

        message = _core.Message(source='assistant', data={'question': 'How?'})
        assert message.source == 'assistant'
        assert message.data['question'] == 'How?'

    def test_message_clones_correctly(self):

        message = _core.Message(source='assistant', data={'question': 'How?'})
        message2 = message.clone()
        assert message['question'] == message2['question']

    def test_message_role_is_a_string(self):

        message = _core.TextMessage(source='assistant', text='hi, how are you')
        assert message.source == 'assistant'
        assert message.data['text'] == 'hi, how are you'

    def test_message_returns_the_reader(self):

        message = _core.TextMessage(source='assistant', text='hi, how are you')
        
        assert isinstance(message.reader(), _core.NullRead)

    def test_clone_copies_the_message(self):

        message = _core.TextMessage(source='assistant', text='hi, how are you')
        message2 = message.clone()
        assert message2['text'] == message['text']

    def test_render_renders_the_message_with_colon(self):

        message = _core.TextMessage(source='assistant', text='hi, how are you')
        rendered = message.render()
        assert rendered == 'assistant: hi, how are you'

    def test_aslist_returns_self_in_a_list(self):
        message = _core.TextMessage(source='assistant', text='hi, how are you')
        assert message.aslist()[0] is message


