# 1st party
import typing
import json

from typing import Self
from abc import ABC, abstractmethod

# 3rd party
import pydantic

# local
from ._messages import (
    Msg, BaseDialog, 
    END_TOK
)
from ._convert import OutConv
from ..proc import Module, AsyncModule, StreamModule, AsyncStreamModule

from ..utils import (
    to_async_function, 
    to_async_function, to_async_function, 
    is_generator_function,
    coalesce, UNDEFINED
)
from ._convert import RespConv, Delim


S = typing.TypeVar('S', bound=pydantic.BaseModel)

# TODO: MOVE OUT OF HERE


LLM_PROMPT = typing.Union[typing.Iterable[Msg], Msg]
LLM_RESPONSE = typing.Tuple[Msg, typing.Any]

from abc import ABC, abstractmethod
import typing

from ..proc import (
    Module, AsyncModule, StreamModule, AsyncStreamModule
)
from ._messages import (
    Msg, BaseDialog
)


class Assist(Module, ABC):
    """
    An abstract base class that defines a framework for geting responses 
    from an API designed to assist, such as a large language model (LLM).
    and it enforces the implementation of the `forward` method in subclasses. 
    The `forward` method is intended to handle retrieval of messages 
    and responses, making it suitable for real-time or incremental data processing.
    Subclasses must implement the `forward` method to define the specific logic 
    for interacting with the API and retrieving responses.
    """


    @abstractmethod
    def forward(
        self, msg: Msg | BaseDialog, *args, **kwargs
    ) -> typing.Tuple[Msg, typing.Any]:
        pass


class AsyncAssist(AsyncModule, ABC):

    """
    An abstract base class that defines a framework for geting responses 
    from an API designed to assist, such as a large language model (LLM).
    and it enforces the implementation of the `astream` method in subclasses. 
    The `aforward` method is intended to handle asynchronous retrieval of messages 
    and responses, making it suitable for real-time or incremental data processing.
    Subclasses must implement the `aforward` method to define the specific logic 
    for interacting with the API and retrieving responses.
    """

    @abstractmethod
    async def aforward(
        self, msg: Msg | BaseDialog, *args, **kwargs
    ) -> typing.Tuple[Msg, typing.Any]:
        pass


class StreamAssist(StreamModule, ABC):
    """
    An abstract base class that defines a framework for streaming responses 
    from an API designed to assist, such as a large language model (LLM).
    and it enforces the implementation of the `stream` method in subclasses. 
    The `stream` method is intended to handle streaming of messages 
    and responses, making it suitable for real-time or incremental data processing.
    Subclasses must implement the `stream` method to define the specific logic 
    for interacting with the API and streaming responses.
    """

    @abstractmethod
    def stream(
        self, msg: Msg | BaseDialog, *args, **kwargs
    ) -> typing.Iterator[typing.Tuple[Msg, typing.Any]]:
        pass


class AsyncStreamAssist(AsyncStreamModule, ABC):
    """
    An abstract base class that defines a framework for streaming responses 
    from an API designed to assist, such as a large language model (LLM).
    and it enforces the implementation of the `astream` method in subclasses. 
    The `astream` method is intended to handle asynchronous streaming of messages 
    and responses, making it suitable for real-time or incremental data processing.
    Subclasses must implement the `astream` method to define the specific logic 
    for interacting with the API and streaming responses.
    """


    @abstractmethod
    async def astream(
        self, msg: Msg | BaseDialog, *args, **kwargs
    ) -> typing.AsyncIterator[typing.Tuple[Msg, typing.Any]]:
        pass


class Assistant(
    Assist, AsyncAssist, StreamAssist, 
    AsyncStreamAssist
):
    """
    Assistant class for wrapping AI or chat functionality.
    This class provides a flexible interface for handling messages and streaming outputs.
    It allows for optional method overrides for synchronous and asynchronous processing.
    """
    def _set_val(self, val, label: str):
        if val is not None:
            object.__setattr__(self, label, val)

    def __init__(
        self, 
        forward=None, 
        aforward=None,
        stream=None,
        astream=None
    ):
        """
        Initialize the instance with optional method overrides.
        Args:
            forward (callable, optional): A callable to override the default 'forward' method.
            aforward (callable, optional): A callable to override the default 'aforward' method.
            stream (callable, optional): A callable to override the default 'stream' method.
            astream (callable, optional): A callable to override the default 'astream' method.
        """
        super().__init__()
        self._forward = forward
        self._aforward = aforward
        self._stream = stream
        self._astream = astream
        self._set_val(forward, 'forward')
        self._set_val(aforward, 'aforward')
        self._set_val(stream, 'stream')
        self._set_val(astream, 'astream')
    
    def forward(self, msg: Msg | BaseDialog, *args, **kwargs) -> typing.Tuple[Msg, typing.Any]:
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
        raise NotImplementedError(
            ''
        )
    
    async def aforward(self, msg: Msg | BaseDialog, *args, **kwargs) -> typing.Tuple[Msg, typing.Any]:
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
        return self.forward(msg, *args, **kwargs)
    
    def stream(self, msg, *args, **kwargs) -> typing.Iterator[typing.Tuple[Msg, typing.Any]]:
        """
        Streams the assistant output for a given message.
        Args:
            msg: The message to be processed by the assistant.
            *args: Additional positional arguments to be passed to the forward method.
            **kwargs: Additional keyword arguments to be passed to the forward method.
        Yields:
            The output from the forward method.
        """
        yield self.forward(msg, *args, **kwargs)
    
    async def astream(self, msg, *args, **kwargs) -> typing.AsyncIterator[typing.Tuple[Msg, typing.Any]]:
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
        yield self.stream(msg, *args, **kwargs)

    def spawn(self, *args, **kwargs) -> typing.Self:
        
        return Assistant(
            self._forward,
            self._aforward,
            self._stream,
            self._astream
        )


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
        if to_async_function(self.option.f):
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
        if to_async_function(self.option.f):
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
        if to_async_function(self.option.f):
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
        if is_generator_function(self.option.f) and to_async_function(self.option.f):
            async for k in await self.option.f(**self.args):
                yield k
        elif is_generator_function(self.option.f):
            for k in await self.option.f(**self.args):
                yield k
        elif to_async_function(self.option.f):
            yield await self.option.f(**self.args)
        else:
            yield self.option.f(**self.args)


class ToolBuilder(object):

    def __init__(self):
        
        self._index = None
        self._name = ''
        self._args = ''
        self._tools = []

    def update(self, index, name, args):        
        
        if index != self._index:
            if self._index is not None:
                result = ToolCall(
                    option=self.tools[self._name],
                    args=json.loads(self._args)
                )
                self._tools.append(result)
            self._index = index
            self._name = name
            self._args = args
            return {
                'name': self._name,
                'args': self._args
            }
        self._args += args
        return None


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
        resp_procs: typing.List[RespConv]=None,
        kwargs: typing.Dict=None,
        message_arg: str='messages',
        role_name: str='assistant',
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
        self._base_aforwardf = aforwardf
        self._base_streamf = streamf
        self._base_astreamf = astreamf
        self._base_forwardf = forwardf
        self._set_val(forwardf, '_forwardf')
        self._set_val(aforwardf, '_aforwardf')
        self._set_val(streamf, '_streamff')
        self._set_val(astreamf, '_astreamf')

    def forward(self, msg: Msg | BaseDialog, *args, **kwargs) -> typing.Tuple[Msg, typing.Any]:
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
        kwargs = {
            **self._kwargs, 
            **kwargs, 
            self._message_arg:msg.to_list_input()
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
    
    async def aforward(self, msg: Msg | BaseDialog, *args, **kwargs) -> typing.Tuple[Msg, typing.Any]:
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
        if self._aforward is not None:
            kwargs = {
                **self._kwargs, 
                **kwargs, 
                self._message_arg: msg.to_list_input()
            }
            return await llm_aforward(
                self._aforward, **kwargs, 
                _resp_proc=self.resp_procs,
                _role=self._role_name
            )
        else:
            return self.forward(
                msg, **kwargs
            )
    
    def stream(self, msg: Msg | BaseDialog, *args, **kwargs) -> typing.Iterator[typing.Tuple[Msg, typing.Any]]:
        """
        Streams the assistant output for a given message.
        Args:
            msg: The message to be processed by the assistant.
            *args: Additional positional arguments to be passed to the forward method.
            **kwargs: Additional keyword arguments to be passed to the forward method.
        Yields:
            The output from the forward method.
        """
        if self._stream is not None:
            kwargs = {
                **self._kwargs, 
                **kwargs, 
                self._message_arg: msg.to_list_input()
            }
            for v in llm_stream(
                self._stream, **kwargs, 
                _resp_proc=self.resp_procs,
                _role=self._role_name, 
                _delim=self._delim
            ):
                yield v
        else:
            yield self.forward(
                msg, **kwargs
            )

    async def astream(self, msg: Msg | BaseDialog, *args, **kwargs) -> typing.AsyncIterator[typing.Tuple[Msg, typing.Any]]:
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
        if self._astream is not None:
            
            kwargs = {
                **self._kwargs, 
                **kwargs, 
                self._message_arg: msg.to_list_input()
            }
            async for v in await llm_astream(
                self._stream, **kwargs, 
                _resp_proc=self.resp_procs,
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


def llm_forward(
    f: typing.Callable, 
    *args, 
    _resp_proc: typing.List[RespConv]=None, 
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

    if isinstance(_resp_proc, RespConv):
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
    
    if isinstance(_resp_proc, RespConv):
        return msg, _resp_proc(response, msg)

    return msg, tuple(
        r(response, msg) for r in _resp_proc
    )


async def llm_aforward(
    f, 
    *args, 
    _resp_proc: typing.List[RespConv]=None, 
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
    if isinstance(_resp_proc, RespConv):
        kwargs.update(_resp_proc.prep())
    elif _resp_proc is not None:
        for r in _resp_proc:
            kwargs.update(r.prep())
    response = await f(
        *args, **kwargs
    )
    msg['meta']['response'] = response
    if _resp_proc is None:
        return msg
    
    if isinstance(_resp_proc, RespConv):
        return msg, _resp_proc(response, msg)

    return msg, tuple(
        r(response, msg) for r in _resp_proc
    )


def llm_stream(
    f: typing.Callable, 
    *args, 
    _resp_proc: typing.List[RespConv]=None, 
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

    if isinstance(_resp_proc, RespConv):
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
    _resp_proc: typing.List[RespConv]=None, 
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

    if isinstance(_resp_proc, RespConv):
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


def process_response(response, msg, resp_proc, delta: typing.Dict):
    """
    Processes a response from the LLM (Language Model).
    Args:
        response: The response from the LLM.
        msg: The message to be processed.
        resp_proc: A converter that processes the response. It can be an instance of RespConv or a list of such instances.
        delta (typing.Dict): A dictionary containing additional data for processing.
    Returns:
        If resp_proc is None, returns the original message.
        If resp_proc is an instance of RespConv, returns a tuple containing the message and the processed delta.
        If resp_proc is a list of RespConv instances, returns a tuple containing the message and a tuple of processed deltas.
    """
    if resp_proc is None:
        return msg
    if isinstance(resp_proc, RespConv):
        return msg, resp_proc.delta(response, msg, delta)
    return msg, tuple(
        r.delta(response, msg, delta_i) for r, delta_i in zip(resp_proc, delta)
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


class Op(Module, AsyncModule, StreamModule, AsyncStreamModule):
    """
    A class that facilitates the process of converting input into a message, 
    interacting with a language model (assistant), and transforming the 
    assistant's response into a final output.
    This class supports both synchronous and asynchronous operations, as well 
    as streaming responses.
    """

    def __init__(self, assistant: Assistant, to_msg: ToMsg, delim: typing.Callable | Delim | typing.Any | None, out: typing.Optional[OutConv]=None):
        """
        Initializes the class to facilitate interaction with a language model assistant by
        adapting inputs and outputs.
        Args:
            assistant (Assistant): The assistant instance responsible for processing messages.
            to_msg (ToMsg): A callable that adapts inputs into the format expected by the assistant.
            out (typing.Optional[OutConv]): An optional callable to process the assistant's output.
                Defaults to a no-op lambda function that returns the output unchanged.
        This constructor sets up the necessary components to streamline the use of language
        models by defining how inputs are transformed for the assistant and how outputs are
        handled after processing.
        """
        super().__init__()
        self.assistant = assistant
        self.to_msg = to_msg
        self.out = out or (lambda x: x)
        self.delim = delim

    def forward(self, *args, _out=None, _asst=None, **kwargs) -> typing.Any:
        """
        Executes the assistant with the provided arguments and returns the processed output.
        Args:
            *args: Positional arguments to be passed to the `to_msg` method for message creation.
            _out (callable, optional): A callable to process the assistant's response. Defaults to `self.out`.
            _asst (typing.Dict or object, optional): A dictionary to spawn a new assistant instance or an existing assistant object. 
            
            If None, defaults to `self.assistant`.
            **kwargs: Additional keyword arguments to be passed to the `to_msg` method.
        Returns:
            The processed output of the assistant's response, as handled by the `_out` callable.
        Raises:
            Any exceptions raised during the assistant's execution or message processing.
        """
        _out = _out or self.out
        msg = self.to_msg(*args, **kwargs)
        if isinstance(_asst, typing.Dict):
            _asst = self.assistant.spawn(**_asst)
        elif _asst is None:
            _asst = self.assistant

        _, resp = _asst(msg)
        return _out(resp) 
        
    async def aforward(self, *args, _out=None, _asst=None, **kwargs) -> typing.Any:
        """
        Asynchronously executes the assistant with the provided arguments and returns the processed output.
        Args:
            *args: Positional arguments to be passed to the `to_msg` method for message creation.
            _out (callable, optional): A callable to process the assistant's response. Defaults to `self.out`.
            _asst (typing.Dict or object, optional): A dictionary to spawn a new assistant instance or an existing assistant object. 
                If None, defaults to `self.assistant`.
            **kwargs: Additional keyword arguments to be passed to the `to_msg` method.
        Returns:
            The processed output of the assistant's response, as handled by the `_out` callable.
        Raises:
            Any exceptions raised during the assistant's execution or message processing.
        """
        _out = _out or self.out
        if isinstance(_asst, typing.Dict):
            _asst = self.assistant.spawn(**_asst)
        elif _asst is None:
            _asst = self.assistant

        msg = self.to_msg(*args, **kwargs)
        _, resp = await _asst.aforward(msg)
        return _out(resp) 
    
    def stream(self, *args, _out=None, _asst=None, **kwargs) -> typing.Iterator:
        """
        Streams the assistant with the provided arguments and returns the processed output.
        Args:
            *args: Positional arguments to be passed to the `to_msg` method for message creation.
            _out (callable, optional): A callable to process the assistant's response. Defaults to `self.out`.
            _asst (typing.Dict or object, optional): A dictionary to spawn a new assistant instance or an existing assistant object. 
            
            If None, defaults to `self.assistant`.
            **kwargs: Additional keyword arguments to be passed to the `to_msg` method.
        Returns:
            The processed output of the assistant's response, as handled by the `_out` callable.
        Raises:
            Any exceptions raised during the assistant's execution or message processing.
        """
        _out = _out or self.out
        delta_store = {}
        delim_store = {}
        if isinstance(_asst, typing.Dict):
            _asst = self.assistant.spawn(**_asst)
        elif _asst is None:
            _asst = self.assistant
        msg = self.to_msg(*args, **kwargs)
        for msg, resp in _asst.stream(msg):
            resp = self.delim(msg, resp, delim_store)
            if resp is not UNDEFINED:
                yield self.out(msg, resp, delta_store)
    
    async def astream(self, *args, _out=None, _asst=None, **kwargs) -> typing.AsyncIterator:
        _out = _out or self.out
        delta_store = {}
        delim_store = {}
        if isinstance(_asst, typing.Dict):
            _asst = self.assistant.spawn(**_asst)
        elif _asst is None:
            _asst = self.assistant
        msg = self.to_msg(*args, **kwargs)
        async for msg, resp in await _asst.astream(msg):
            resp = self.delim(msg, resp, delim_store)
            if resp is not UNDEFINED:
                yield self.out.delta(msg, resp, delta_store)

    def spawn(
        self, to_msg: ToMsg=UNDEFINED, assistant: Assistant=UNDEFINED, out: OutConv=UNDEFINED
    ):
        to_msg = coalesce(to_msg, self.to_msg)
        if isinstance(assistant, typing.Dict):
            assistant = self.assistant.spawn(**assistant)
        else:
            assistant = coalesce(assistant, self.assistant)
        out = coalesce(out, self.out)
        return Op(
            assistant, to_msg, out
        )
