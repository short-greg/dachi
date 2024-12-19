from dachi._core import _core
# from dachi._core import _instruct as core

import asyncio
from typing import Any, Iterator, Tuple
import pytest
from dachi._core import _process as p
from dachi._core._core import Module
from dachi.ai import _ai
import typing


class DummyAIModel(_ai.AIModel):
    """APIAdapter allows one to adapt various WebAPI or other
    API for a consistent interface
    """

    def __init__(self, target='Great!'):
        super().__init__()
        self.target = target

    def forward(self, prompt: _ai.AIPrompt, **kwarg_override) -> _ai.AIResponse:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        message = prompt.aslist()[0]
        result = self.convert(message)
        return _ai.AIResponse(
            _ai.TextMessage('assistant', self.target), result, self.target
        )
    
    def stream(self, prompt: _ai.AIPrompt, **kwarg_override) -> Iterator[Tuple[_ai.AIResponse]]:
        message = prompt.aslist()[0]
        result = self.convert(message)

        cur_out = ''
        for c in self.target:
            cur_out += c
            yield _ai.AIResponse(
                _ai.TextMessage('assistant', cur_out), result, cur_out
            ), _ai.AIResponse(
                _ai.TextMessage('assistant', c), result, c
            )

    def convert(self, message: _ai.Message) -> typing.Dict:
        """Convert a message to the format needed for the model

        Args:
            messages (Message): The messages to convert

        Returns:
            typing.List[typing.Dict]: The format to pass to the "model"
        """
        return {'text': message['text']}



# TODO: Test stream text, TextMessage, and AIResponse


# TODO: Add tests - Test all functionality here

