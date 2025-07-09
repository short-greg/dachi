from dachi.core._msg import Msg, Resp
from dachi.proc._resp import RespProc
from dachi.proc import Process
from dachi.core import END_TOK, ModuleList
from typing import Iterator
from dachi.proc import _ai
from dachi import utils
import typing


class DummyAIModel(
    Process
):
    """APIAdapter allows one to adapt various WebAPI or other
    API for a consistent interface
    """

    target: str = 'Great!'
    proc: ModuleList = None

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
        resp = Resp(
            msg=Msg(
            role='assistant', 
            content=self.target
        ))
        resp.data['response'] = self.target
        resp.data['content'] = self.target
        return RespProc.run(resp, self.proc)
    
    def stream(
        self, 
        prompt: _ai.LLM_PROMPT, 
        **kwarg_override
    ) -> Iterator:

        cur_out = ''
        resp = Resp()
        
        for i, c in enumerate(self.target):
            cur_out += c
            is_last = i == len(self.target) - 1
        
            resp = resp.spawn(
                msg=Msg(
                role='assistant', content=self.target
            ))
            resp.data['response'] = self.target
            resp.data['content'] = c
            yield RespProc.run(resp, self.proc, is_last=False, is_streamed=True)
    
        resp = resp.spawn(msg=Msg(
            role='assistant', content=self.target
        ))
        resp.data['response'] = END_TOK
        resp.data['content'] = ''
        
        yield RespProc.run(resp, self.proc, is_last=True, is_streamed=True)
        
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


async def astream(
    msg: str
) -> typing.AsyncIterator[typing.Dict]:
    """Use to test astream
    """

    response = f'Hi! {msg}'
    for c in response:
        yield {'content': c}


class TextResp(_ai.RespProc):

    name: str = 'content'
    from_: str = _ai.RESPONSE_FIELD
    
    def post(
        self, 
        resp: Resp, 
        result, 
        delta_store, 
        streamed = False, 
        is_last = True
    ):
        
        resp.out[self.name] = delta_store.get(
            'content', ''
        )
        resp.msg.content = delta_store.get(
            'content', ''
        )

    def delta(
        self, 
        resp,
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=True
    ) -> typing.Any: 
        
        if resp is not _ai.END_TOK:
            resp = resp['content']
            print(resp)
            utils.acc(
                delta_store, 
                'content',
                resp
            )
            return resp
        return ''


class DeltaResp(_ai.RespProc):

    # from_: str = 'response'
    name: str = 'content'
    from_: str = _ai.RESPONSE_FIELD

    def post(
        self, 
        resp, 
        result, 
        delta_store, 
        streamed = False, 
        is_last = True
    ):
        resp.out[self.name] = delta_store.get('content', '')

    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=True
    ) -> typing.Any: 
        if resp is not _ai.END_TOK:
            return resp['content']
        return ''


class TestLLM:

    def test_llm_executes_forward(self):
        res = _ai.llm_forward(forward, 'Jack')
        assert res.data['response'] == {'content': 'Hi! Jack'}

    def test_llm_executes_forward_with_processor(self):
        res = _ai.llm_forward(
            forward, 'Jack', 
            _proc=TextResp()
        )
        assert res.data['response'] == {'content': 'Hi! Jack'}
        assert res.data['content'] == 'Hi! Jack'

    def test_llm_executes_stream_with_processor(self):
        responses = []
        contents = []
        for r in _ai.llm_stream(
            stream, 'Jack', _proc=TextResp()
        ):
            responses.append(r.data['response'])
            contents.append(r.out['content'])
        assert contents[0] == 'H'
        assert contents[-1] == 'Hi! Jack'

    def test_llm_executes_stream_with_two_processors(self):
        responses = []
        contents = []
        deltas = []
        for r in _ai.llm_stream(
            stream, 'Jack', _proc=[TextResp(), DeltaResp(name='delta')]
        ):
            print('R: ', type(r))
            responses.append(r)
            contents.append(r.msg.content)
            deltas.append(r.data['content'])
        assert contents[0] == 'H'
        assert contents[-1] == 'Hi! Jack'
        assert deltas[-1] is ''
        assert responses[-1].msg.content == 'Hi! Jack'
