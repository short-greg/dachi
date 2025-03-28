# 1st party
import typing
from typing import Self
import json
from collections import deque
from abc import ABC, abstractmethod
import typing

# 3rd party
import pydantic

# local
from ..msg._messages import (
    Msg, BaseDialog, 
     ToMsg,
)
from ._asst import Assistant
from ._out import OutConv
from ..proc import (
    Module, AsyncModule, 
    StreamModule, AsyncStreamModule
)
from ..utils import (
    coalesce, UNDEFINED,
    Args
)
from._resp import RespConv
from ._parse import (
    Parser, CharDelimParser, NullParser
)


S = typing.TypeVar('S', bound=pydantic.BaseModel)

# TODO: MOVE OUT OF HERE


LLM_PROMPT = typing.Union[typing.Iterable[Msg], Msg]
LLM_RESPONSE = typing.Tuple[Msg, typing.Any]


class Op(Module, AsyncModule, StreamModule, AsyncStreamModule):
    """
    A class that facilitates the process of converting input into a message, 
    interacting with a language model (assistant), and transforming the 
    assistant's response into a final output.
    This class supports both synchronous and asynchronous operations, as well 
    as streaming responses.
    """

    def __init__(self, assistant: Assistant, to_msg: ToMsg, parser: typing.Callable | Parser | typing.Any | None, out: typing.Optional[OutConv]=None):
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
        if isinstance(parser, str):
            self.parser = CharDelimParser(parser)
        elif self.parser is None:
            self.parser = NullParser()
        else:
            self.parser = parser

    def forward(self, *args, _out=None, _messages: typing.List[Msg]=None, **kwargs) -> typing.Any:
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

        if _messages is not None:
            _messages.append(msg)
            msg = _messages
        resp_msg, resp = self.assistant(msg)

        if _messages is not None:
            _messages.append(resp_msg)
        resp = self.parser(resp)
        return _out(resp) 
        
    async def aforward(
        self, *args, _out=None, 
        _messages: typing.List[Msg]=None, **kwargs
    ) -> typing.Any:
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
        msg = self.to_msg(*args, **kwargs)

        _messages = _messages or []

        resp_msg, resp = await self.assistant.aforward(msg + _messages)

        if _messages is not None:
            _messages.append(resp_msg)
        resp = self.parser(resp)

        return _out(resp) 
    
    def stream(self, *args, _out=None,
        _messages: typing.List[Msg]=None, **kwargs) -> typing.Iterator:
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
        queue = deque()
        msg = self.to_msg(*args, **kwargs)

        _messages = _messages or []
            
        resp_msg = None
        for resp_msg, resp in self.assistant.stream(msg + _messages):
            parsed = self.parser(resp, delim_store)
            if parsed is not None:
                queue.extend(parsed)
            if len(queue) > 0:
                yield self.out(queue.popleft(), delta_store)
        
        for q in queue:
            yield self.out(msg, queue.popleft(), delta_store)
        if _messages is not None and resp_msg is not None:
            _messages.append(resp_msg)

    
    async def astream(
        self, *args, _out=None, 
        _messages: typing.List[Msg]=None, **kwargs
    ) -> typing.AsyncIterator:
        """
        Asynchronously streams the assistant with the provided arguments and returns the processed output.
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
        _messages = _messages or []
        delta_store = {}
        delim_store = {}
        msg = self.to_msg(*args, **kwargs)
        queue = deque()
            
        async for resp_msg, resp in await self._asst.astream(msg + _messages):
            parsed = self.parser(resp, delim_store)
            if parsed is not None:
                queue.extend(parsed)
            if len(queue) > 0:
                yield self.out(queue.popleft(), delta_store)
        for q in queue:
            yield self.out(queue.popleft(), delta_store)
        if _messages is not None and resp_msg is not None:
            _messages.append(resp_msg)

    def asst(self, *args, **kwargs) -> Self:
        """Create a new operation with specified
        args

        Returns:
            Self: 
        """
        asst = self.assistant.spawn(*args, **kwargs)
        return self.spawn(
            assistant=asst
        )

    def spawn(
        self, to_msg: ToMsg=UNDEFINED, assistant: Assistant=UNDEFINED, 
        parser: Parser=UNDEFINED, out: OutConv=UNDEFINED,
    ):
        """
        Spawns a new `Op` instance based on the updated arguments.
        This method creates a new operation by combining the provided arguments
        with the existing attributes of the current instance. If any argument is
        not provided, it defaults to the corresponding attribute of the instance.
        Args:
            to_msg (ToMsg, optional): The message transformation logic. Defaults to `UNDEFINED` 
                and falls back to the instance's `to_msg` attribute if not provided.
            assistant (Assistant or dict, optional): The assistant instance or a dictionary 
                of parameters to spawn a new assistant. Defaults to `UNDEFINED` and falls 
                back to the instance's `assistant` attribute if not provided.
            parser (Parser, optional): The parser instance. Defaults to `UNDEFINED` and 
                falls back to the instance's `parser` attribute if not provided.
            out (OutConv, optional): The output conversion logic. Defaults to `UNDEFINED` 
                and falls back to the instance's `out` attribute if not provided.
        Returns:
            Op: A new `Op` instance initialized with the resolved arguments.
        """
        to_msg = coalesce(to_msg, self.to_msg)
        assistant = coalesce(assistant, self.assistant)
        parser = coalesce(parser, self.parser)
        out = coalesce(out, self.out)
        return Op(
            assistant, to_msg, out
        )

class Threaded(
    Module, AsyncModule, StreamModule,
    AsyncStreamModule
):
    
    def __init__(
        self, op: Op, op_args: Args,
        router: typing.Dict[str, ToMsg],
        dialog: BaseDialog=None
    ):
        super().__init__()
        self.op = op
        self.router = router
        self.dialog = dialog
        self.op_args = op_args

    def forward(self, role: str, *args, **kwargs) -> typing.Any:
        
        msg = self.router[role](*args, **kwargs)
        self.dialog.append(msg)

        return self.op.forward(
            *self.op_args.args, 
            **self.op_args.kwargs,
            _messages=self.dialog
        )

    async def aforward(self, role: str, *args, **kwargs) -> typing.Any:

        msg = self.router[role](*args, **kwargs)
        self.dialog.append(msg)

        return await self.op.aforward(
            *self.op_args.args, 
            **self.op_args.kwargs,
            _messages=self.dialog
        )

    def stream(self, role: str, *args, **kwargs):
        msg = self.router[role](*args, **kwargs)
        self.dialog.append(msg)

        for r in self.op.stream(
            *self.op_args.args, 
            **self.op_args.kwargs,
            _messages=self.dialog
        ):
            yield r
    
    async def astream(self, role: str, *args, **kwargs):
        msg = self.router[role](*args, **kwargs)
        self.dialog.append(msg)

        async for r in self.op.astream(
            *self.op_args.args, 
            **self.op_args.kwargs,
            _messages=self.dialog
        ):
            yield r
