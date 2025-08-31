# 1st party
import typing as t
from abc import abstractmethod

# 3rd party
import pydantic

# local
from ..core import (
    Msg, END_TOK, Resp,
    BaseModule, RespDelta
)
from ._process import AsyncProcess
from ._resp import RespProc, RESPONSE_FIELD

# TODO: MOVE OUT OF HERE
S = t.TypeVar('S', bound=pydantic.BaseModel)

from ._process import Process, AsyncProcess, AsyncStreamProcess, StreamProcess
from ..core import BaseDialog

# TODO: Update Assistant / LLM
LLM_PROMPT: t.TypeAlias = t.Union[t.Iterable[Msg], Msg]
S = t.TypeVar('S', bound=pydantic.BaseModel)


def _prepare(proc, kwargs):

    if isinstance(proc, RespProc):
        proc = [proc]
    elif proc is None:
        proc = []
    for r in proc:
        if isinstance(proc, RespProc):
            kwargs.update(r.prep())
    return proc


def llm_forward(
    f: t.Callable, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    _proc: t.List[RespProc] | RespProc=None, 
    _role: str='assistant',
    **kwargs
) -> Resp:
    """
    Executes a given function with specified arguments and processes the response.
    Args:
        f (t.Callable): The function to be executed.
        *args: Variable length argument list to be passed to the function.
        _resp_proc (t.List[Response], optional): A list of Response objects to process the function's response. Defaults to None.
        _role (str, optional): The role to be assigned to the message. Defaults to 'assistant'.
        **kwargs: Additional keyword arguments to be passed to the function.
    Returns:
        tuple: A tuple containing the final message (Msg) and the last value processed by the Response objects.
    """
    _proc = _prepare(_proc, kwargs)
    
    if _adapt is None:
        _adapt = DefaultAdapter()

    kwargs.update(_adapt.to_input(*args, **kwargs))
    result = f(
        *args, **kwargs
    )
    resp = _adapt.from_output(result)

    for r in _proc:
        resp = r(resp)
    return resp


async def llm_aforward(
    f, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    _proc: t.List[RespProc] | RespProc=None, 
    _role: str='assistant',
    **kwargs
) -> Resp:
    """
    Asynchronously forwards a function call with additional response processing.
    Args:
        f (Callable): The function to be called.
        *args: Positional arguments to pass to the function.
        _resp_proc (List[Response], optional): A list of Response objects to process the function's response. Defaults to None.
        _role (str, optional): The role to be assigned to the message. Defaults to 'assistant'.
        **kwargs: Additional keyword arguments to pass to the function.
    Returns:
        Tuple[Msg, Any]: A tuple containing the processed message and the final value from the response processing.
    """
    _proc = _prepare(_proc, kwargs)
    
    if _adapt is None:
        _adapt = DefaultAdapter()

    kwargs.update(_adapt.to_input(*args, **kwargs))
    result = await f(
        *args, **kwargs
    )
    resp = _adapt.from_output(result)

    for r in _proc:
        if isinstance(r, AsyncProcess):
            resp = await r.aforward(resp)
        else:
            resp = r(resp)

    return resp


def llm_stream(
    f: t.Callable, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    _proc: t.List[RespProc] | RespProc=None, 
    _role: str='assistant',
    **kwargs
) -> t.Iterator[Resp]:
    """
    Streams responses from a language model function, allowing for intermediate processing.
    Args:
        f (t.Callable): The language model function to call.
        *args: Positional arguments to pass to the language model function.
        _resp_proc (t.List[Response], optional): A list of Response objects for processing the model's output. Defaults to None.
        _role (str, optional): The role to assign to the message. Defaults to 'assistant'.
        **kwargs: Additional keyword arguments to pass to the language model function.
    Yields:
        Tuple[Msg, Any]: A tuple containing the message object and the processed value from the response.
    """
    _proc = _prepare(_proc, kwargs)
    prev_message: Msg | None = None
    resp = None
    
    if _adapt is None:
        _adapt = DefaultAdapter()
        
    kwargs.update(_adapt.to_input(*args, **kwargs))
    for response in f(
        *args, **kwargs
    ):
        resp = _adapt.from_streamed(response, resp)
        # msg = Msg(role=_role)
        # if prev_message is not None:
        #     msg.delta = prev_message.delta

        # if resp is None:
        #     resp = Resp(msg=msg)
        # else:
        #     resp = resp.spawn(
        #         msg=msg
        #     )
        resp.data[RESPONSE_FIELD] = response

        for r in _proc:
            resp = r(resp, True, False)
        prev_message = resp
        yield resp
    
    msg = Msg(role=_role)
    if prev_message is not None:
        msg.delta = prev_message.delta
    resp.data[RESPONSE_FIELD] = END_TOK

    for r in _proc:

        resp = r(resp, True, True)

    yield resp


async def llm_astream(
    f: t.Callable, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    _proc: t.List[RespProc] | RespProc=None, 
    _role: str='assistant',
    **kwargs
) -> t.AsyncIterator[Resp]:
    """

    Args:
        f (t.Callable): The function to run
        _resp_proc (t.List[Response], optional): The processes to use for responding
        _role (str, optional): The role for message. Defaults to 'assistant'.

    Returns:
        t.AsyncIterator: 

    Yields:
        t.AsyncIterator: The Message and the results
    """
    _proc = _prepare(_proc, kwargs)
    prev_message: Msg | None = None
    resp = None
    
    if _adapt is None:
        _adapt = DefaultAdapter()
        
    kwargs.update(_adapt.to_input(*args, **kwargs))
    async for response in await f(
        *args, **kwargs
    ):
        resp = _adapt.from_streamed(response, resp)
        # msg = Msg(role=_role)
        # if prev_message is not None:
        #     msg.delta = prev_message.delta

        # if resp is None:
        #     resp = Resp(msg=msg)
        # else:
        #     resp = resp.spawn(
        #         msg=msg
        #     )

        resp.data[RESPONSE_FIELD] = response

        for r in _proc:
            if isinstance(r, AsyncProcess):
                resp = await r(resp, True, False)
            else:
                resp = r(resp, True, False)

        prev_message = resp.msg
        yield resp
    
    msg = Msg(role=_role)
    if prev_message is not None:
        msg.delta = prev_message.delta
    resp.data[RESPONSE_FIELD] = END_TOK

    for r in _proc:

        resp = r(resp, True, True)

    yield resp


class AIAdapt(BaseModule):
    """
    Use to adapt the message from the standard format
    to the format required by the LLM
    """

    @abstractmethod
    def from_output(
        self, 
        output: t.Dict,
        inp: Msg | BaseDialog | str | None = None,
    ) -> Resp:
        pass

    @abstractmethod
    def from_streamed(
        self, 
        output: t.Dict,
        inp: Msg | BaseDialog | str | None = None,
        prev_resp: Resp | None=None
    ) -> Resp:
        pass

    @abstractmethod
    def to_input(
        self, 
        inp: Msg | BaseDialog, 
        **kwargs
    ) -> t.Dict:
        """Convert the input message to the format required by the LLM.

        Args:
            msg (Msg | BaseDialog): The input message to convert.

        Returns:
            t.Dict: The converted message in the required format.
        """
        pass


class LLM(Process, AsyncProcess, StreamProcess, AsyncStreamProcess):
    """
    Adapter for Large Language Models (LLMs).
    """
    def forward(self, inp: Msg | BaseDialog, out=None,**kwargs) -> Resp:
        raise NotImplementedError

    async def aforward(
        self, inp: Msg | BaseDialog, out=None,
        tools=None, **kwargs
    ) -> Resp:
        raise NotImplementedError

    def stream(self, inp: Msg | BaseDialog, out=None, tools=None, **kwargs) -> t.Iterator[Resp]:
        raise NotImplementedError

    def astream(self, inp: Msg | BaseDialog, out=None, tools=None, **kwargs) -> t.AsyncIterator[Resp]:
        raise NotImplementedError


class DefaultAdapter(AIAdapt):
    """
    Default/Null adapter that passes data through without transformation.
    
    Used when no specific adapter is needed, providing minimal processing
    while maintaining the AIAdapt interface.
    """
    
    def to_input(self, inp: Msg | BaseDialog, **kwargs) -> t.Dict:
        """Pass through kwargs without transformation."""
        return kwargs
    
    def from_output(self, output: t.Dict) -> Resp:
        """Create a basic Resp object from the output."""
        resp = Resp(msg=Msg(role='assistant'))
        resp.data['response'] = output
        return resp
    
    def from_streamed(self, output: t.Dict, prev_resp: Resp | None = None) -> Resp:
        """Handle streaming by creating or updating Resp objects."""
        if prev_resp is None:
            resp = Resp(msg=Msg(role='assistant'))
        else:
            resp = prev_resp.spawn(msg=Msg(role='assistant'))
        
        resp.data['response'] = output
        return resp


class Op(AsyncProcess, Process, StreamProcess, AsyncStreamProcess):
    """
    A basic operation that applies a function to the input.
    """

    llm: LLM
    base_out: t.Any
    tools: t.List[t.Any]

    def forward(self, inp: S, out=None, **kwargs) -> S:
        out = out or self.base_out
        return self.llm.forward(inp, tools=self.tools, op=out, **kwargs).out

    async def aforward(self, inp: S, out=None, **kwargs) -> S:
        out = out or self.base_out
        resp = await self.llm.aforward(inp, tools=self.tools, op=out, **kwargs)
        return resp.out

    def stream(self, inp: S, out=None, **kwargs) -> t.Iterator[S]:
        out = out or self.base_out
        for resp in self.llm.stream(inp, tools=self.tools, op=out, **kwargs):

            yield resp.out

    async def astream(self, inp: S, out=None, **kwargs) -> t.AsyncIterator[S]:
        out = out or self.base_out
        async for resp in await self.llm.astream(inp, tools=self.tools, op=out, **kwargs):
            yield resp.out

