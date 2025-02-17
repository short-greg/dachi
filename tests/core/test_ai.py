from dachi._core import Msg

from typing import Iterator
from dachi._core import _ai
import typing


class DummyAIModel(
    _ai.LLM, _ai.StreamLLM,
    _ai.AsyncLLM, _ai.AsyncStreamLLM
):
    """APIAdapter allows one to adapt various WebAPI or other
    API for a consistent interface
    """

    def __init__(self, target='Great!'):
        super().__init__()
        self.target = target

    def forward(
        self, 
        prompt: _ai.LLM_PROMPT, 
        **kwarg_override
    ) -> _ai.LLM_RESPONSE:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        return Msg(
            role='assistant', content=self.target
        ), self.target
    
    def stream(
        self, prompt: _ai.LLM_PROMPT, 
        **kwarg_override
    ) -> Iterator[_ai.LLM_RESPONSE]:

        cur_out = ''
        for c in self.target:
            cur_out += c
            
            msg = Msg(
                role='assistant', content=cur_out, delta={'content': c}
            )
            print('Yielding ', c)
            yield msg, c
        
    async def aforward(self, dialog, **kwarg_overrides):
        return self.forward(dialog, **kwarg_overrides)

    async def astream(self, dialog, **kwarg_overrides):
        
        for msg, c in self.stream(dialog, **kwarg_overrides):
            yield msg, c


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


class TextResp(_ai.RespProc):

    def __init__(self):
        super().__init__(True)

    def __call__(self, response, msg: Msg) -> typing.Any:
        msg['content'] = response['content']
        return msg['content']

    def delta(self, response, msg: Msg, delta_store: typing.Dict) -> typing.Any: 
        
        if 'content' not in delta_store:
            delta_store['content'] = '' 
        if response is not None and response is not _ai.END_TOK:
            delta_store['content'] += response['content']
        msg['content'] = delta_store['content']
        return msg['content']


class DeltaResp(_ai.RespProc):

    def __init__(self):
        super().__init__(True)

    def __call__(self, response, msg: Msg) -> typing.Any:
        msg['delta_content'] = ''
        return ''

    def delta(self, response, msg: Msg, delta_store: typing.Dict) -> typing.Any: 
        if response is _ai.END_TOK:
            msg['delta_content'] = None
            return None
        
        msg['delta_content'] = response['content']
        return response['content']


class TestLLM:

    def test_llm_executes_forward(self):
        res = _ai.llm_forward(forward, 'Jack')
        print(res['meta']['response'])
        assert res['meta']['response']['content'] == 'Hi! Jack'

    def test_llm_executes_forward_with_processor(self):
        res, content = _ai.llm_forward(forward, 'Jack', _resp_proc=TextResp())
        assert content == 'Hi! Jack'
        assert res['meta']['response']['content'] == 'Hi! Jack'

    def test_llm_executes_stream_with_processor(self):
        responses = []
        contents = []
        for r, content in _ai.llm_stream(
            stream, 'Jack', _resp_proc=TextResp()
        ):
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
        assert delta is None
        assert responses[-1]['content'] == 'Hi! Jack'
