# 1st party
import typing

# 3rd party
import pydantic

# local
from ..core import (
    Msg, END_TOK, Resp
)
from ._msg import Msg
from ._process import AsyncProcess
from ._msg import RespConv, RespProc

# TODO: MOVE OUT OF HERE
LLM_PROMPT = typing.Union[typing.Iterable[Msg], Msg]
S = typing.TypeVar('S', bound=pydantic.BaseModel)

# TODO: Update Assistant / LLM
LLM_PROMPT = typing.Union[typing.Iterable[Msg], Msg]
S = typing.TypeVar('S', bound=pydantic.BaseModel)
# TODO: MOVE OUT OF HERE


def _prepare(proc, kwargs):

    if isinstance(proc, RespProc):
        proc = [proc]
    elif proc is None:
        proc = []
    for r in proc:
        if isinstance(proc, RespConv):
            kwargs.update(r.prep())
    return proc


def llm_forward(
    f: typing.Callable, 
    *args, 
    _proc: typing.List[RespProc] | RespProc=None, 
    _role: str='assistant',
    **kwargs
) -> Resp:
    """
    Executes a given function with specified arguments and processes the response.
    Args:
        f (typing.Callable): The function to be executed.
        *args: Variable length argument list to be passed to the function.
        _resp_proc (typing.List[Response], optional): A list of Response objects to process the function's response. Defaults to None.
        _role (str, optional): The role to be assigned to the message. Defaults to 'assistant'.
        **kwargs: Additional keyword arguments to be passed to the function.
    Returns:
        tuple: A tuple containing the final message (Msg) and the last value processed by the Response objects.
    """
    _proc = _prepare(_proc, kwargs)

    result = f(
        *args, **kwargs
    )
    resp = Resp(
        msg=Msg(role=_role)
    )
    resp.data = result
    for r in _proc:
        resp = r(resp)
    return resp


async def llm_aforward(
    f, 
    *args, 
    _proc: typing.List[RespProc] | RespProc=None, 
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

    result = await f(
        *args, **kwargs
    )
    resp = Resp(
        msg=Msg(role=_role)
    )
    resp.data = result
    for r in _proc:
        if isinstance(_proc, AsyncProcess):
            resp = await r.aforward(resp)
        else:
            resp = r(resp)

    return resp


def llm_stream(
    f: typing.Callable, 
    *args, 
    _proc: typing.List[RespProc] | RespProc=None, 
    _role: str='assistant',
    **kwargs
) -> typing.Iterator[Resp]:
    """
    Streams responses from a language model function, allowing for intermediate processing.
    Args:
        f (typing.Callable): The language model function to call.
        *args: Positional arguments to pass to the language model function.
        _resp_proc (typing.List[Response], optional): A list of Response objects for processing the model's output. Defaults to None.
        _role (str, optional): The role to assign to the message. Defaults to 'assistant'.
        **kwargs: Additional keyword arguments to pass to the language model function.
    Yields:
        Tuple[Msg, Any]: A tuple containing the message object and the processed value from the response.
    """
    _proc = _prepare(_proc, kwargs)
    prev_message: Msg | None = None
    for response in f(
        *args, **kwargs
    ):
        msg = Msg(role=_role)
        if prev_message is not None:
            msg.delta = prev_message.delta

        resp = Resp(msg=msg)
        resp.data = response

        for r in _proc:
            msg = r(resp, True, False)
        prev_message = msg
        yield msg
    
    msg = Msg(role=_role)
    if prev_message is not None:
        msg.delta = prev_message.delta
    resp.data = END_TOK

    for r in _proc:

        resp = r(resp, True, True)

    yield resp


async def llm_astream(
    f: typing.Callable, 
    *args, 
    _proc: typing.List[RespProc] | RespProc=None, 
    _role: str='assistant',
    **kwargs
) -> typing.AsyncIterator[Resp]:
    """

    Args:
        f (typing.Callable): The function to run
        _resp_proc (typing.List[Response], optional): The processes to use for responding
        _role (str, optional): The role for message. Defaults to 'assistant'.

    Returns:
        typing.AsyncIterator: 

    Yields:
        typing.AsyncIterator: The Message and the results
    """
    _proc = _prepare(_proc, kwargs)
    prev_message: Msg | None = None
    async for response in await f(
        *args, **kwargs
    ):
        msg = Msg(role=_role)
        if prev_message is not None:
            msg.delta = prev_message.delta

        resp = Resp(msg=msg)
        resp.data = response

        for r in _proc:
            if isinstance(_proc, AsyncProcess):
                resp = await r(resp, True, False)
            else:
                resp = r(resp, True, False)

        prev_message = msg
        yield resp
    
    msg = Msg(role=_role)
    if prev_message is not None:
        msg.delta = prev_message.delta
    resp.data = END_TOK

    for r in _proc:

        resp = r(resp, True, True)

    yield resp
