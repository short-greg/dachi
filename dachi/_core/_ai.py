# 1st party
import typing
from abc import ABC, abstractmethod
import typing
from ._process import Module
import pydantic
from ._messages import Msg, RespProc, ListDialog, BaseDialog
from ..utils._f_utils import (
    is_async_function, is_async_function, 
    is_generator_function
)
from ._core import END_TOK


LLM_PROMPT = typing.Union[typing.Iterable[Msg], Msg]
LLM_RESPONSE = typing.Tuple[Msg, typing.Any]


class ToolOption(pydantic.BaseModel):
    """
    Represents an option for a tool, encapsulating the tool's name, 
    the function to be executed, and any additional keyword arguments.
    Attributes:
        name (str): The name of the tool.
        f (typing.Callable[[typing.Any], typing.Any]): The function to be executed by the tool.
        kwargs (typing.Dict): A dictionary of additional keyword arguments to be passed to the function.
    """

    name: str
    f: typing.Callable[[typing.Any], typing.Any]
    kwargs: typing.Dict

    def to_input(self) -> typing.Dict:
        """
        Converts the instance's keyword arguments into a dictionary of arguments.
        Returns:
            dict: A dictionary containing the keyword arguments of the instance.
        """
        return {
            **self.kwargs
        }


class ToolSet(object):
    """A set of tools that the LLM can use
    """
    def __init__(self, tools: typing.List[ToolOption], **kwargs):
        """The set of tools

        Args:
            tools (typing.List[ToolOption]): The list of tools
        """
        self.tools = {
            tool.name: tool
            for tool in tools
        }
        self.kwargs = kwargs

    def add(self, option: ToolOption):
        """Add a tool to the set

        Args:
            option (ToolOption): The option to add
        """
        self.tools[option.name] = option

    def remove(self, option: ToolOption):
        """Remove a tool from the tool set

        Args:
            option (ToolOption): The option to add
        """
        del self.tools[option.name]

    def to_input(self):
        return list(
            tool.to_input() for _, tool in self.tools.items()
        )
    
    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self) -> typing.Iterator:
        """
        Returns an iterator over the tools in the collection.
        Yields:
            tool: Each tool in the collection.
        """

        for _, tool in self.tools.items():
            yield tool

    def __getitem__(self, name):
        """
        Retrieve a tool by its name.
        Args:
            name (str): The name of the tool to retrieve.
        Returns:
            object: The tool associated with the given name.
        Raises:
            KeyError: If the tool with the specified name does not exist.
        """
        return self.tools[name]


class ToolCall(pydantic.BaseModel):
    """A response from the LLM that a tool was called
    """
    option: ToolOption = pydantic.Field(
        description="The tool that was chosen."
    )
    args: typing.Dict[str, typing.Any] = pydantic.Field(
        description="The arguments to the tool."
    )

    def forward(self) -> typing.Any:
        """Call the tool

        Raises:
            NotImplementedError: If the function is async
            NotImplementedError: If the function is a generator function

        Returns:
            typing.Any: The result of the call
        """
        # Check if valid to use with forward
        if is_async_function(self.option.f):
            raise NotImplementedError
        if is_generator_function(self.option.f):
            raise NotImplementedError
        return self.option.f(**self.args)

    async def aforward(self) -> typing.Any:
        """Call the tool 

        Raises:
            NotImplementedError: If the function is a generator

        Returns:
            typing.Any: The result of the call
        """
        if is_async_function(self.option.f):
            return await self.option.f(**self.args)
        if is_generator_function(self.option.f):
            raise NotImplementedError
        return self.option.f(**self.args)

    def stream(self) -> typing.Iterator:
        """Stream the tool

        Raises:
            NotImplementedError: The result

        Yields:
            Iterator[typing.Iterator]: The result of the call
        """
        if is_async_function(self.option.f):
            raise NotImplementedError
        elif is_generator_function(self.option.f):
            for k in self.option.f(**self.args):
                yield k
        else:
            yield self.option.f(**self.args)
        
    async def astream(self):
        """Stream the tool

        Yields:
            Iterator[typing.Iterator]: The result of the call
        """
        if is_generator_function(self.option.f) and is_async_function(self.option.f):
            async for k in await self.option.f(**self.args):
                yield k
        elif is_generator_function(self.option.f):
            for k in await self.option.f(**self.args):
                yield k
        elif is_async_function(self.option.f):
            yield await self.option.f(**self.args)
        else:
            yield self.option.f(**self.args)


def exclude_role(messages: typing.Iterable[Msg], *role: str) -> typing.List[Msg]:
    """
    Filter messages by excluding specified roles.
    This function takes an iterable of messages and one or more role strings, returning
    a new list containing only messages whose roles are not in the specified roles to exclude.
    Args:
        messages (typing.Iterable[Msg]): An iterable of message objects
        *role (str): Variable number of role strings to exclude
    Returns:
        typing.List[Msg]: A list of messages excluding those with specified roles
    Example:
        >>> messages = [Msg(role="user", content="hi"), Msg(role="system", content="hello")]
        >>> exclude_role(messages, "system")
        [Msg(role="user", content="hi")]
    """
    exclude = set(role)
    return [message for message in messages
        if message.role not in exclude]


def include_role(messages: typing.Iterable[Msg], *role: str) -> typing.List[Msg]:
    """Filter the iterable of messages by a particular role

    Args:
        messages (typing.Iterable[Msg]): 

    Returns:
        typing.List[Msg]: 
    """
    include = set(role)
    return [message for message in messages
        if message.role in include]


def to_dialog(prompt: typing.Union[BaseDialog, Msg]) -> BaseDialog:
    """Convert a prompt to a dialog

    Args:
        prompt (typing.Union[Dialog, Msg]): The prompt to convert
    """
    if isinstance(prompt, Msg):
        prompt = ListDialog([prompt])

    return prompt

# 1st party
import typing
from abc import abstractmethod, ABC
from typing import Self
from itertools import chain
import inspect
from functools import wraps

# 3rd party
import pydantic

# local
from ._process import (
    Module, Trainable, 
    AsyncModule, StreamModule, AsyncStreamModule
)
from ._messages import Msg, BaseDialog
from ._ai import (
    llm_aforward, llm_astream, 
    llm_forward, llm_stream, RespProc, ToMsg
)
from ._core import Renderable
from ..utils import is_primitive

from ._read import TextProc
from ..utils import (
    is_async_function, 
    is_generator_function,
)
from ..utils._utils import str_formatter


S = typing.TypeVar('S', bound=pydantic.BaseModel)

# TODO: MOVE OUT OF HERE

class LLMBase(ABC):

    """
    LLM is a class that serves as an adapter for Language Model (LLM) functions, enabling the execution of various LLM operations such as forwarding, asynchronous forwarding, streaming, and asynchronous streaming. It provides a flexible interface to handle different types of LLM interactions.
    """
    def __init__(
        self, 
        resp_procs: typing.List[RespProc]=None,
        kwargs: typing.Dict=None,
        message_arg: str='messages',
        role_name: str='assistant'
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
        super().__init__()
        self._kwargs = kwargs or {}
        self.resp_procs = resp_procs or []
        self._message_arg = message_arg
        self._role_name = role_name


class LLM(LLMBase, Module):
    
    def __init__(self, forward=None, resp_procs = None, kwargs = None, message_arg = 'messages', role_name = 'assistant'):
        super().__init__(resp_procs, kwargs, message_arg, role_name)
        self._forward = forward

    def forward(self, dialog: BaseDialog, **kwarg_overrides) -> typing.Tuple[Msg, typing.Any]:
        """
        Processes the given dialog and returns a response message and additional data.
        Args:
            dialog (Dialog): The dialog object containing the input data.
            **kwarg_overrides: Additional keyword arguments to override default settings.
        Returns:
            typing.Tuple[Msg, typing.Any]: A tuple containing the response message and any additional data.
        Raises:
            RuntimeError: If the forward function is not defined for the LLM.
        """
        kwargs = {
            **self._kwargs, 
            **kwarg_overrides, 
            self._message_arg: dialog.to_list_input()
        }
        if self._forward is not None:
            return llm_forward(
                self._forward, **kwargs, 
                _resp_proc=self.resp_procs,
                _role=self._role_name
            )
        raise RuntimeError(
            'The forward has not been defined for the LLM'
        )


class AsyncLLM(LLMBase, AsyncModule):

    def __init__(self, aforward=None, resp_procs = None, kwargs = None, message_arg = 'messages', role_name = 'assistant'):
        super().__init__(resp_procs, kwargs, message_arg, role_name)
        self._aforward = aforward

    async def aforward(self, dialog: BaseDialog, **kwarg_overrides) -> typing.Tuple[Msg, typing.Any]:
        """
        Asynchronously forwards a dialog to the specified handler.
        This method checks if an asynchronous forward handler (`self._aforward`) is defined.
        If it is, it merges the provided keyword arguments with the instance's keyword arguments
        and the dialog input, then calls the asynchronous forward handler.
        If no asynchronous forward handler is defined, it falls back to the synchronous `forward` method.
        Args:
            dialog (Dialog): The dialog object containing the input to be forwarded.
            **kwarg_overrides: Additional keyword arguments to override the instance's keyword arguments.
        Returns:
            typing.Tuple[Msg, typing.Any]: A tuple containing the message and any additional data returned by the handler.
        """
        if self._aforward is not None:
            kwargs = {
                **self._kwargs, 
                **kwarg_overrides, 
                self._message_arg: dialog.to_list_input()
            }
            return await llm_aforward(
                self._aforward, **kwargs, 
                _resp_proc=self.resp_procs,
                _role=self._role_name
            )
        else:
            return self.forward(
                dialog, **kwarg_overrides
            )


class StreamLLM(LLMBase, StreamModule):

    def __init__(self, stream=None, resp_procs = None, kwargs = None, message_arg = 'messages', role_name = 'assistant'):
        super().__init__(resp_procs, kwargs, message_arg, role_name)
        self._stream = stream

    def stream(self, dialog: BaseDialog, **kwarg_overrides) -> typing.Iterator[typing.Tuple[Msg, typing.Any]]:
        """
        Streams responses from the language model based on the provided dialog.
        This method streams responses from the language model if the `_stream` attribute is set.
        It merges the default keyword arguments (`_kwargs`) with any overrides provided in `kwarg_overrides`,
        and uses the `dialog.to_input()` as the value for the `_message_arg` key. The responses are yielded
        one by one from the `llm_stream` function.
        If `_stream` is not set, it forwards the dialog to another method and yields the result.
        Args:
            dialog (Dialog): The dialog object containing the input message.
            **kwarg_overrides: Additional keyword arguments to override the default ones.
        Yields:
            typing.Tuple[Msg, typing.Any]: A tuple containing the message and any additional data.
        """
        if self._stream is not None:
            kwargs = {
                **self._kwargs, 
                **kwarg_overrides, 
                self._message_arg: dialog.to_list_input()
            }
            for v in llm_stream(
                self._stream, **kwargs, 
                _resp_proc=self.resp_procs,
                _role=self._role_name
            ):
                yield v
        else:
            yield self.forward(
                dialog, **kwarg_overrides
            )


class AsyncStreamLLM(LLMBase, AsyncStreamModule):

    def __init__(self, astream=None, resp_procs = None, kwargs = None, message_arg = 'messages', role_name = 'assistant'):
        super().__init__(resp_procs, kwargs, message_arg, role_name)
        self._astream = astream

    async def astream(self, dialog: BaseDialog, **kwarg_overrides) -> typing.AsyncIterator[typing.Tuple[Msg, typing.Any]]:
        """
        Asynchronously streams messages based on the provided dialog.
        This method checks if an asynchronous stream (`_astream`) is available. If it is, it uses the `llm_astream` function
        to asynchronously yield messages. If not, it falls back to a synchronous stream method.
        Args:
            dialog (Dialog): The dialog object containing the input message.
            **kwarg_overrides: Additional keyword arguments to override the default ones.
        Yields:
            typing.Tuple[Msg, typing.Any]: A tuple containing the message and any additional data.
        """
        if self._astream is not None:
            
            kwargs = {
                **self._kwargs, 
                **kwarg_overrides, 
                self._message_arg: dialog.to_list_input()
            }
            async for v in await llm_astream(
                self._stream, **kwargs, 
                _resp_proc=self.resp_procs,
                _role=self._role_name
            ):
                yield v
        else:
            for v in self.stream(
                dialog, **kwarg_overrides
            ):
                yield v


# class LLM(Module):
#     """
#     LLM is a class that serves as an adapter for Language Model (LLM) functions, enabling the execution of various LLM operations such as forwarding, asynchronous forwarding, streaming, and asynchronous streaming. It provides a flexible interface to handle different types of LLM interactions.
#     """
#     def __init__(
#         self, 
#         forward=None,
#         aforward=None,
#         stream=None,
#         astream=None,
#         resp_procs: typing.List[RespProc]=None,
#         kwargs: typing.Dict=None,
#         message_arg: str='messages',
#         role_name: str='assistant'
#     ):
#         """Wrap the processes in an LLM. Can also inherit from LLM

#         Args:
#             forward (optional): Define the forward function Raises a Runtime error if not defined and called. Defaults to None.
#             aforward (optional): Define the astream function. Will call forward if not defined. Defaults to None.
#             stream (optional): Define the stream function. Will call forward if not defined. Defaults to None.
#             astream (optional): Define the astream function. LLM astream will call stream as a backup if not defined. Defaults to None.
#             responsresp_procse_processors (typing.List[Response], optional): . Defaults to None.
#             kwargs (typing.Dict, optional): . Defaults to None.
#             message_arg (str, optional): . Defaults to 'messages'.
#             role_name (str, optional): . Defaults to 'assistant'.
#         """
#         super().__init__()
#         self._forward = forward
#         self._aforward = aforward
#         self._stream = stream
#         self._astream = astream
#         self._kwargs = kwargs or {}
#         self.resp_procs = resp_procs or []
#         self._message_arg = message_arg
#         self._role_name = role_name

#     def forward(self, dialog: BaseDialog, **kwarg_overrides) -> typing.Tuple[Msg, typing.Any]:
#         """
#         Processes the given dialog and returns a response message and additional data.
#         Args:
#             dialog (Dialog): The dialog object containing the input data.
#             **kwarg_overrides: Additional keyword arguments to override default settings.
#         Returns:
#             typing.Tuple[Msg, typing.Any]: A tuple containing the response message and any additional data.
#         Raises:
#             RuntimeError: If the forward function is not defined for the LLM.
#         """
#         kwargs = {
#             **self._kwargs, 
#             **kwarg_overrides, 
#             self._message_arg: dialog.to_list_input()
#         }
#         if self._forward is not None:
#             return llm_forward(
#                 self._forward, **kwargs, 
#                 _resp_proc=self.resp_procs,
#                 _role=self._role_name
#             )
#         raise RuntimeError(
#             'The forward has not been defined for the LLM'
#         )
    
#     async def aforward(self, dialog: BaseDialog, **kwarg_overrides) -> typing.Tuple[Msg, typing.Any]:
#         """
#         Asynchronously forwards a dialog to the specified handler.
#         This method checks if an asynchronous forward handler (`self._aforward`) is defined.
#         If it is, it merges the provided keyword arguments with the instance's keyword arguments
#         and the dialog input, then calls the asynchronous forward handler.
#         If no asynchronous forward handler is defined, it falls back to the synchronous `forward` method.
#         Args:
#             dialog (Dialog): The dialog object containing the input to be forwarded.
#             **kwarg_overrides: Additional keyword arguments to override the instance's keyword arguments.
#         Returns:
#             typing.Tuple[Msg, typing.Any]: A tuple containing the message and any additional data returned by the handler.
#         """
#         if self._aforward is not None:
#             kwargs = {
#                 **self._kwargs, 
#                 **kwarg_overrides, 
#                 self._message_arg: dialog.to_list_input()
#             }
#             return await llm_aforward(
#                 self._aforward, **kwargs, 
#                 _resp_proc=self.resp_procs,
#                 _role=self._role_name
#             )
#         else:
#             return self.forward(
#                 dialog, **kwarg_overrides
#             )

#     def stream(self, dialog: BaseDialog, **kwarg_overrides) -> typing.Iterator[typing.Tuple[Msg, typing.Any]]:
#         """
#         Streams responses from the language model based on the provided dialog.
#         This method streams responses from the language model if the `_stream` attribute is set.
#         It merges the default keyword arguments (`_kwargs`) with any overrides provided in `kwarg_overrides`,
#         and uses the `dialog.to_input()` as the value for the `_message_arg` key. The responses are yielded
#         one by one from the `llm_stream` function.
#         If `_stream` is not set, it forwards the dialog to another method and yields the result.
#         Args:
#             dialog (Dialog): The dialog object containing the input message.
#             **kwarg_overrides: Additional keyword arguments to override the default ones.
#         Yields:
#             typing.Tuple[Msg, typing.Any]: A tuple containing the message and any additional data.
#         """
#         if self._stream is not None:
#             kwargs = {
#                 **self._kwargs, 
#                 **kwarg_overrides, 
#                 self._message_arg: dialog.to_list_input()
#             }
#             for v in llm_stream(
#                 self._stream, **kwargs, 
#                 _resp_proc=self.resp_procs,
#                 _role=self._role_name
#             ):
#                 yield v
#         else:
#             yield self.forward(
#                 dialog, **kwarg_overrides
#             )
    
#     async def astream(self, dialog: BaseDialog, **kwarg_overrides) -> typing.AsyncIterator[typing.Tuple[Msg, typing.Any]]:
#         """
#         Asynchronously streams messages based on the provided dialog.
#         This method checks if an asynchronous stream (`_astream`) is available. If it is, it uses the `llm_astream` function
#         to asynchronously yield messages. If not, it falls back to a synchronous stream method.
#         Args:
#             dialog (Dialog): The dialog object containing the input message.
#             **kwarg_overrides: Additional keyword arguments to override the default ones.
#         Yields:
#             typing.Tuple[Msg, typing.Any]: A tuple containing the message and any additional data.
#         """
#         if self._astream is not None:
            
#             kwargs = {
#                 **self._kwargs, 
#                 **kwarg_overrides, 
#                 self._message_arg: dialog.to_list_input()
#             }
#             async for v in await llm_astream(
#                 self._stream, **kwargs, 
#                 _resp_proc=self.resp_procs,
#                 _role=self._role_name
#             ):
#                 yield v
#         else:
#             for v in self.stream(
#                 dialog, **kwarg_overrides
#             ):
#                 yield v


def llm_forward(
    f: typing.Callable, 
    *args, 
    _resp_proc: typing.List[RespProc]=None, 
    _role: str='assistant',
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

    if isinstance(_resp_proc, RespProc):
        kwargs.update(_resp_proc.prep())
    elif _resp_proc is not None:
        for r in _resp_proc:
            kwargs.update(r.prep())

    response = f(
        *args, **kwargs
    )
    msg['meta']['response'] = response

    if _resp_proc is None:
        return msg
    
    if isinstance(_resp_proc, RespProc):
        return msg, _resp_proc(response, msg)

    return msg, tuple(
        r(response, msg) for r in _resp_proc
    )


async def llm_aforward(
    f, 
    *args, 
    _resp_proc: typing.List[RespProc]=None, 
    _role: str='assistant',
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
    if isinstance(_resp_proc, RespProc):
        kwargs.update(_resp_proc.prep())
    elif _resp_proc is not None:
        for r in _resp_proc:
            kwargs.update(r.prep())

    print('f: ', f)
    response = await f(
        *args, **kwargs
    )
    msg['meta']['response'] = response

    if _resp_proc is None:
        return msg
    
    print(response)
    if isinstance(_resp_proc, RespProc):
        return msg, _resp_proc(response, msg)

    return msg, tuple(
        r(response, msg) for r in _resp_proc
    )


def process_response(response, msg, resp_proc, delta: typing.Dict):

    if resp_proc is None:
        return msg
    if isinstance(resp_proc, RespProc):
        return msg, resp_proc.delta(response, msg, delta)
    return msg, tuple(
        r.delta(response, msg, delta_i) for r, delta_i in zip(resp_proc, delta)
    )


def llm_stream(
    f: typing.Callable, 
    *args, 
    _resp_proc: typing.List[RespProc]=None, 
    _role: str='assistant',
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
    _resp_proc = _resp_proc or []

    if isinstance(_resp_proc, RespProc):
        delta = {}
        kwargs.update(_resp_proc.prep())
    elif _resp_proc is None:
        delta = None
    else:
        delta = [{} for _ in range(len(_resp_proc))]
        for r in _resp_proc:
            kwargs.update(r.prep())
    for response in f(
        *args, **kwargs
    ):
        msg = Msg(role=_role)
        msg['meta']['response'] = response

        yield process_response(
            response, msg, _resp_proc, delta
        ) 
    
    msg = Msg(role=_role)
    msg['meta']['response'] = END_TOK

    yield process_response(
        END_TOK, msg, _resp_proc, delta
    ) 


async def llm_astream(
    f: typing.Callable, 
    *args, 
    _resp_proc: typing.List[RespProc]=None, 
    _role: str='assistant',
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
    _resp_proc = _resp_proc or []

    if isinstance(_resp_proc, RespProc):
        delta = {}
        kwargs.update(_resp_proc.prep())
    elif _resp_proc is None:
        delta = None
    else:
        delta = [{} for _ in range(len(_resp_proc))]
        for r in _resp_proc:
            kwargs.update(r.prep())

    async for response in await f(
        *args, **kwargs
    ):
        msg = Msg(role=_role)
        msg['meta']['response'] = response
        yield process_response(
            response, msg, _resp_proc, delta
        ) 

    msg = Msg(role=_role)
    msg['meta']['response'] = END_TOK

    yield process_response(
        END_TOK, msg, _resp_proc, delta
    )


class ToMsg(ABC):
    """Converts the input to a message
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Msg:
        pass


class ToText(ToMsg):
    """Converts the input to a text message
    """

    def __init__(self, role: str='system', field: str='content'):
        """Converts an input to a text message

        Args:
            role (str): The role for the message
            field (str, optional): The name of the field for the text. Defaults to 'content'.
        """
        self.role = role
        self.field = field

    def __call__(self, text: str) -> Msg:
        """Create a text message

        Args:
            text (str): The text for the message

        Returns:
            Msg: Converts to a text message
        """
        return Msg(
            role=self.role, 
            **{self.field: text}
        )
