# 1st party
import typing
from typing import Self

# 3rd party
import pydantic

# local
from ..msg._messages import (
    Msg, BaseDialog,
    END_TOK, to_list_input
)
from ..msg._msg import MsgProc
from ._asst import Assistant
from ..utils import (
    coalesce, UNDEFINED,
)
from..msg._resp import RespConv


S = typing.TypeVar('S', bound=pydantic.BaseModel)

# TODO: MOVE OUT OF HERE
LLM_PROMPT = typing.Union[typing.Iterable[Msg], Msg]


class LLM(Assistant):

    """
    LLM is a class that serves as an adapter for Language Model (LLM) functions, enabling the execution of various LLM operations such as forwarding, asynchronous forwarding, streaming, and asynchronous streaming. It provides a flexible interface to handle different types of LLM interactions.
    """
    def __init__(
        self, 
        forwardf: typing.Callable=None,
        aforwardf: typing.Callable=None,
        streamf: typing.Callable=None,
        astreamf: typing.Callable=None,
        procs: typing.List[MsgProc]=None,
        kwargs: typing.Dict=None,
        message_arg: str='messages',
        role: str='assistant',
    ):
        """Wrap the processes in an LLM. Can also inherit from LLM

        Args:
            forward (optional): Define the forward function Raises a Runtime error if not defined and called. Defaults to None.
            aforward (optional): Define the astream function. Will call forward if not defined. Defaults to None.
            stream (optional): Define the stream function. Will call forward if not defined. Defaults to None.
            astream (optional): Define the astream function. LLM astream will call stream as a backup if not defined. Defaults to None.
            responsresp_procse_processors (typing.List[Response], optional): . Defaults to None.
            kwargs (typing.Dict, optional): . Defaults to None.
            message_arg (str, optional): . Defaults to 'messages'.
            role_name (str, optional): . Defaults to 'assistant'.
        """
        super().__init__(role=role)
        self._kwargs = kwargs or {}
        if isinstance(procs, MsgProc):
            procs = [procs]
        self.procs = procs or []
        self._message_arg = message_arg
        self._base_aforwardf = aforwardf
        self._base_streamf = streamf
        self._base_astreamf = astreamf
        self._base_forwardf = forwardf
        self._set_val(forwardf, '_forwardf')
        self._set_val(aforwardf, '_aforwardf')
        self._set_val(streamf, '_streamff')
        self._set_val(astreamf, '_astreamf')

    def forward(
        self, 
        msg: Msg | BaseDialog, 
        *args, 
        _proc: typing.List[MsgProc]=None,
        **kwargs
    ) -> Msg:
        """
        Processes the given message and additional arguments.
        This method should be implemented by subclasses to define the specific
        behavior for handling the message.
        Args:
            msg: The message to be processed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        _proc = _proc if _proc is not None else self.procs
        kwargs = {
            **self._kwargs, 
            **kwargs, 
            self._message_arg:to_list_input(msg)
        }
        if self._forward is not None:
            return llm_forward(
                self._forward, *args, **kwargs, 
                _resp_proc=_proc,
                _role=self._role_name
            )
        raise RuntimeError(
            'The forward has not been defined for the LLM'
        )
    
    async def aforward(
        self, 
        msg: Msg | BaseDialog, 
        *args, 
        _proc: typing.List[MsgProc]=None,
        **kwargs
    ) -> Msg:
        """
        Asynchronous version of the forward method.
        This method calls the synchronous forward method with the provided
        message and any additional arguments or keyword arguments.
        Args:
            msg: The message to be forwarded.
            *args: Additional positional arguments to be passed to the forward method.
            **kwargs: Additional keyword arguments to be passed to the forward method.
        Returns:
            The result of the forward method.
        """
        _proc = _proc if _proc is not None else self.procs
        if self._aforward is not None:
            kwargs = {
                **self._kwargs, 
                **kwargs, 
                self._message_arg: to_list_input(msg)
            }
            return await llm_aforward(
                self._aforward, *args, **kwargs, 
                _proc=_proc,
                _role=self._role_name
            )
        else:
            return self.forward(
                msg, **kwargs
            )
    
    def stream(
        self, 
        msg: Msg | BaseDialog, 
        *args, 
        _proc: typing.List[MsgProc]=None,
        **kwargs
    ) -> typing.Iterator[Msg]:
        """
        Streams the assistant output for a given message.
        Args:
            msg: The message to be processed by the assistant.
            *args: Additional positional arguments to be passed to the forward method.
            **kwargs: Additional keyword arguments to be passed to the forward method.
        Yields:
            The output from the forward method.
        """
        _proc = _proc if _proc is not None else self.procs
        if self._stream is not None:
            kwargs = {
                **self._kwargs, 
                **kwargs, 
                self._message_arg: to_list_input(msg)
            }
            for v in llm_stream(
                self._stream, *args, **kwargs, 
                _proc=_proc,
                _role=self._role_name, 
                _delim=self._delim
            ):
                yield v
        else:
            yield self.forward(
                msg, **kwargs
            )

    # op.asst(...)()

    async def astream(
        self, 
        msg: Msg | BaseDialog, 
        *args, 
        _proc: typing.List[MsgProc]=None,
        **kwargs
    ) -> typing.AsyncIterator[Msg]:
        """
        Asynchronous streaming function to get the Assistant's output.
        This function yields the output of the `stream` function with the given 
        message and additional arguments.
        Args:
            msg (str): The message to be processed by the stream function.
            *args: Variable length argument list to be passed to the stream function.
            **kwargs: Arbitrary keyword arguments to be passed to the stream function.
        Yields:
            The output of the `stream` function.
        """
        _proc = _proc if _proc is not None else self.procs
        if self._astream is not None:
            
            kwargs = {
                **self._kwargs, 
                **kwargs, 
                self._message_arg: to_list_input(msg)
            }
            async for v in await llm_astream(
                self._stream, *args, **kwargs, 
                _resp_proc=_proc,
                _role=self._role_name, 
                _delim=self._delim
            ):
                yield v
        else:
            for v in self.stream(
                msg, **kwargs
            ):
                yield v

    def spawn(self, 
        kwargs: typing.Dict=UNDEFINED,
        message_arg: str=UNDEFINED,
        role_name: str=UNDEFINED,
    ) -> Self:
        return LLM(
            self._base_forwardf,
            self._base_aforwardf,
            self._base_streamf,
            self._base_astreamf,
            coalesce(kwargs, self._kwargs),
            coalesce(message_arg, self._message_arg),
            coalesce(role_name, self._role_name)
        )


def _prepare(proc, kwargs):

    if isinstance(proc, MsgProc):
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
    _proc: typing.List[MsgProc]=None, 
    _role: str='assistant',
    _response_name: str='response',
    **kwargs
):
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
    msg = Msg(role=_role)

    _proc = _prepare(_proc, kwargs)

    response = f(
        *args, **kwargs
    )
    msg.m[_response_name] = response
    for r in _proc:
        msg = r(msg)
    return msg


async def llm_aforward(
    f, 
    *args, 
    _proc: typing.List[MsgProc]=None, 
    _role: str='assistant',
    _response_name: str='response',
    **kwargs
):
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
    msg = Msg(
        role=_role
    )
    _proc = _prepare(_proc, kwargs)

    response = await f(
        *args, **kwargs
    )
    msg['meta'][_response_name] = response

    for r in _proc:
        msg = r(msg)
    return msg



def llm_stream(
    f: typing.Callable, 
    *args, 
    _proc: typing.List[MsgProc]=None, 
    _role: str='assistant',
    _response_name: str='response',
    **kwargs
):
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
    delta_stores = [{} for _ in range(len(_proc))]
    for response in f(
        *args, **kwargs
    ):
        msg = Msg(role=_role)
        msg['meta'][_response_name] = response

        for r, delta_store in zip(_proc, delta_stores):
            msg = r(msg, delta_store, True, False)
        
        yield msg
    
    msg = Msg(role=_role)
    msg['meta'][_response_name] = END_TOK

    for r, delta_store in zip(_proc, delta_stores):
        msg = r(msg, delta_store, True, True)

    yield msg


async def llm_astream(
    f: typing.Callable, 
    *args, 
    _resp_proc: typing.List[MsgProc]=None, 
    _role: str='assistant',
    _response_name: str='response',
    **kwargs
) -> typing.AsyncIterator:
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

    delta_stores = [{} for _ in range(len(_resp_proc))]
    async for response in await f(
        *args, **kwargs
    ):
        msg = Msg(role=_role)
        msg['meta'][_response_name] = response
        for r, delta_store in zip(_resp_proc, delta_stores):
            msg = r(msg, delta_store, True, False)
        
        yield msg

    msg = Msg(role=_role)
    msg['meta'][_response_name] = END_TOK
    for r, delta_store in zip(_resp_proc, delta_stores):
        msg = r(msg, delta_store, True, True)

    yield msg
