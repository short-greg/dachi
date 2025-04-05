from dachi.msg._messages import Msg, StreamMsg
from dachi.asst._msg import MsgProc

from typing import Iterator
from dachi.asst import _ai
import typing
from dachi import utils


class DummyAIModel(
    _ai.LLM
):
    """APIAdapter allows one to adapt various WebAPI or other
    API for a consistent interface
    """

    def __init__(self, target='Great!', proc: typing.List[MsgProc]=None):
        super().__init__()
        self.proc = proc or []
        self.target = target

    def forward(
        self, 
        prompt: _ai.LLM_PROMPT, 
        **kwarg_override
    ):
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        msg = Msg(
            role='assistant', content=self.target,
            meta={'content': self.target}
        )
        
        for p in self.proc:
            p(msg)
        return msg
        # , self.target
    
    def stream(
        self, prompt: _ai.LLM_PROMPT, 
        **kwarg_override
    ) -> Iterator:

        cur_out = ''
        
        for i, c in enumerate(self.target):
            cur_out += c
            is_last = i == len(cur_out) - 1
            
            msg = StreamMsg(
                role='assistant', content=cur_out, meta={'content': c}, is_last=is_last
            )
            for p in self.proc:
                p(msg)
                print(msg.m)
            yield msg
        
    async def aforward(self, dialog, **kwarg_overrides):
        return self.forward(dialog, **kwarg_overrides)

    async def astream(self, dialog, **kwarg_overrides):
        
        for msg in self.stream(dialog, **kwarg_overrides):

            yield msg


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


class TextResp(_ai.RespConv):

    def __init__(self):
        super().__init__('content')
    
    def post(self, msg, result, delta_store, streamed = False, is_last = True):
        msg[self.name] = delta_store.get('content', '')

    def delta(self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=True) -> typing.Any: 
        if resp is not _ai.END_TOK:
            utils.add(delta_store, 'content', resp['content'])
            return resp['content']
        return ''


class DeltaResp(_ai.RespConv):

    def __init__(self, name: str):
        super().__init__(name, from_='response')

    # def __call__(self, response, msg: Msg) -> typing.Any:
    #     msg['delta_content'] = ''
    #     return ''

    # def delta(self, response, msg: Msg, delta_store: typing.Dict) -> typing.Any: 
    #     if response is _ai.END_TOK:
    #         msg['delta_content'] = None
    #         return None
        
    #     msg['delta_content'] = response['content']
    #     return response['content']

    def post(self, msg, result, delta_store, streamed = False, is_last = True):
        msg[self.name] = delta_store.get('content', '')

    def delta(self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=True) -> typing.Any: 
        if resp is not _ai.END_TOK:
            return resp['content']
        return ''



class TestLLM:

    def test_llm_executes_forward(self):
        res = _ai.llm_forward(forward, 'Jack')
        print(res['meta']['response'])
        assert res['meta']['response']['content'] == 'Hi! Jack'

    def test_llm_executes_forward_with_processor(self):
        res = _ai.llm_forward(forward, 'Jack', _proc=TextResp())
        assert res['content'] == 'Hi! Jack'
        assert res['meta']['content'] == 'Hi! Jack'

    def test_llm_executes_stream_with_processor(self):
        responses = []
        contents = []
        for r in _ai.llm_stream(
            stream, 'Jack', _proc=TextResp()
        ):
            responses.append(r.m['response'])
            contents.append(r['content'])
        assert contents[0] == 'H'
        print(r)
        assert contents[-1] == 'Hi! Jack'

    def test_llm_executes_stream_with_two_processors(self):
        responses = []
        contents = []
        deltas = []
        for r in _ai.llm_stream(stream, 'Jack', _proc=[TextResp(), DeltaResp('delta')]):
            print('R: ', type(r))
            responses.append(r)
            contents.append(r['content'])
            deltas.append(r.m['content'])
        assert contents[0] == 'H'
        assert contents[-1] == 'Hi! Jack'
        assert deltas[-1] is ''
        assert responses[-1]['content'] == 'Hi! Jack'
