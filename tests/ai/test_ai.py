from dachi._core import _core
from dachi._core import Msg

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
        return _core.Msg(
            role='assistant', content=self.target
        ), self.target
    
    def stream(self, prompt: _ai.LLM_PROMPT, **kwarg_override) -> Iterator[_ai.LLM_RESPONSE]:

        cur_out = ''
        for c in self.target:
            cur_out += c
            yield _core.Msg(
                'assistant', cur_out, delta={'content': c}
            ), c


def forward(msg: str) -> typing.Dict:
    """Use to test forward
    """
    return {'content': f'Hi! {msg}'}


async def aforward(msg: str) -> typing.Dict:
    """Use to test aforward
    """

    return {'content': f'Hi! {msg}'}


def stream(msg: str) -> typing.Iterator[typing.Dict]:
    """Use to test stream
    """

    response = f'Hi! {msg}'
    for c in response:
        yield {'content': c}


async def astream(msg: str) -> typing.AsyncIterator[typing.Dict]:
    """Use to test astream
    """

    response = f'Hi! {msg}'
    for c in response:
        yield {'content': c}


class TextResp(_ai.ResponseProc):

    def __init__(self):
        super().__init__(True)

    def __call__(self, response, msg: Msg) -> typing.Any:
        msg['content'] = response['content']
        return msg['content']

    def delta(self, response, msg: Msg, delta: typing.Dict) -> typing.Any: 
        
        if 'content' not in delta:
            delta['content'] = '' 
        delta['content'] += response['content']
        msg['content'] = delta['content']
        return msg['content']


class DeltaResp(_ai.ResponseProc):

    def __init__(self):
        super().__init__(True)

    def __call__(self, response, msg: Msg) -> typing.Any:
        msg['delta_content'] = ''
        return ''

    def delta(self, response, msg: Msg, delta: typing.Dict) -> typing.Any: 
        
        msg['delta_content'] = response['content']
        return response['content']


class TestLLM:

    def test_llm_executes_forward(self):
        res = _ai.llm_forward(forward, 'Jack')
        print(res['meta']['response'])
        assert res['meta']['response']['content'] == 'Hi! Jack'

    def test_llm_executes_forward_with_processor(self):
        res, content = _ai.llm_forward(forward, 'Jack', _resp_proc=[TextResp()])
        assert content == 'Hi! Jack'
        assert res['meta']['response']['content'] == 'Hi! Jack'

    def test_llm_executes_stream_with_processor(self):
        responses = []
        contents = []
        for r, content in _ai.llm_stream(stream, 'Jack', _resp_proc=[TextResp()]):
            responses.append(r)
            contents.append(content)
        assert contents[0] == 'H'
        assert contents[-1] == 'Hi! Jack'
        assert responses[-1]['content'] == 'Hi! Jack'

    def test_llm_executes_stream_with_two_processors(self):
        responses = []
        contents = []
        deltas = []
        for r, (content, delta) in _ai.llm_stream(stream, 'Jack', _resp_proc=[TextResp(), DeltaResp()]):
            responses.append(r)
            contents.append(content)
            deltas.append(delta)
        assert contents[0] == 'H'
        assert contents[-1] == 'Hi! Jack'
        assert delta[-1] == 'k'
        assert responses[-1]['content'] == 'Hi! Jack'
