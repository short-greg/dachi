from dachi._core import _core
from dachi._core._messages import Msg

from typing import Iterator
from dachi.ai import _ai
import typing


class DummyAIModel(_ai.LLM):
    """APIAdapter allows one to adapt various WebAPI or other
    API for a consistent interface
    """

    def __init__(self, target='Great!'):
        super().__init__()
        self.target = target

    def system(self, *args, type_ = 'data', delta = None, meta = None, **kwargs):
        return 

    def forward(self, prompt: _ai.LLM_PROMPT, **kwarg_override) -> _ai.LLM_RESPONSE:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        return _ai.Msg(
            role='assistant', content=self.target
        ), self.target
    
    def stream(self, prompt: _ai.LLM_PROMPT, **kwarg_override) -> Iterator[_ai.LLM_RESPONSE]:

        cur_out = ''
        for c in self.target:
            cur_out += c
            yield _ai.Msg(
                'assistant', cur_out, delta={'content': c}
            ), c
