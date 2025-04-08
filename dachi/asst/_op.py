# TODO: Update this once LLM and convs are done

# 1st party
import typing
from typing import Self
import typing

# 3rd party
import pydantic

# local
from ..msg._messages import (
    Msg, BaseDialog, 
)
from ._asst import Assistant
from ._msg import ToMsg, MR, FromMsg
from ..proc import (
    Module, AsyncModule, 
    StreamModule, AsyncStreamModule
)
from ..utils import (
    coalesce, UNDEFINED,
    Args
)

S = typing.TypeVar('S', bound=pydantic.BaseModel)

# TODO: MOVE OUT OF HERE


LLM_PROMPT = typing.Union[typing.Iterable[Msg], Msg]
LLM_RESPONSE = typing.Tuple[Msg, typing.Any]

# TODO: Need to specify whether to use meta or not somehow
# in out


class Op(Module, AsyncModule, StreamModule, AsyncStreamModule):
    """
    A class that facilitates the process of converting input into a message, 
    interacting with a language model (assistant), and transforming the 
    assistant's response into a final output.
    This class supports both synchronous and asynchronous operations, as well 
    as streaming responses.
    """

    def __init__(self, assistant: Assistant, to_msg: ToMsg, out: str | typing.List[str] | FromMsg):
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
        if not isinstance(out, FromMsg):
            out = FromMsg(out)
        else:
            print(out.key)
        self.out = out
        # if isinstance(parser, str):
        #     self.parser = CharDelimParser(parser)
        # elif parser is None:
        #     self.parser = NullParser()
        # else:
        #     self.parser = parser


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

        _messages = _messages or []
        resp_msg = self.assistant(
            msg + _messages
        )

        if _messages is not None:
            _messages.append(resp_msg)
        return self.out(resp_msg, _out)
        #resp_msg = self.parser(resp_msg)
        #return _out(resp_msg) 
        
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
        msg = self.to_msg(*args, **kwargs)

        _messages = _messages or []

        resp_msg = await self.assistant.aforward(msg + _messages)

        if _messages is not None:
            _messages.append(resp_msg)
        # resp_msg = self.parser(resp_msg)

        return self.out(resp_msg, _out)
        # return _out(resp_msg) 
    
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
        msg = self.to_msg(*args, **kwargs)

        _messages = _messages or []
        
        resp_msg = None

        for resp_msg in self.assistant.stream(
            msg + _messages
        ):
            # self.parser(resp_msg, parse_delta)
            # self.out(resp_msg, out_delta)
            yield self.out(resp_msg, _out)
        
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
        msg = self.to_msg(*args, **kwargs)

        parse_delta = {}
        out_delta = {}
        async for resp_msg in await self.assistant.astream(
            msg + _messages
        ):
            self.parser(resp_msg, parse_delta)
            self.out(resp_msg, out_delta)
            yield self.out(resp_msg, _out)
        
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
        self, to_msg: ToMsg=UNDEFINED, assistant: Assistant=UNDEFINED, out: str | MR | typing.List[str | MR]=UNDEFINED,
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
        out = coalesce(out, self.out)
        # parser = coalesce(parser, self.parser)
        return Op(
            assistant, to_msg, out
        )


class Threaded(
    Module, AsyncModule, StreamModule,
    AsyncStreamModule
):
    """A Threaded Op. Use to keep the Op 
    """
    
    def __init__(
        self, op: Op, op_args: Args,
        router: typing.Dict[str, ToMsg],
        dialog: BaseDialog=None
    ):
        """Create an Op that has a thread

        Args:
            op (Op): The Op
            op_args (Args): The arguments
            router (typing.Dict[str, ToMsg]): Routes the input to a type of message
            dialog (BaseDialog, optional): The type of dialog to use. Defaults to None.
        """
        super().__init__()
        self.op = op
        self.router = router
        self.dialog = dialog
        self.op_args = op_args

    def forward(self, route: str, *args, **kwargs) -> typing.Any:
        """Execute the op

        Args:
            route (str): The name of the route

        Returns:
            typing.Any: The result of the op
        """
        msg = self.router[route](*args, **kwargs)
        self.dialog.append(msg)

        return self.op.forward(
            *self.op_args.args, 
            **self.op_args.kwargs,
            _messages=self.dialog
        )

    async def aforward(self, route: str, *args, **kwargs) -> typing.Any:
        """Execute the op asynchronously

        Args:
            route (str): The name of the route for the message

        Returns:
            typing.Any: The result of the op
        """

        msg = self.router[route](*args, **kwargs)
        self.dialog.append(msg)

        return await self.op.aforward(
            *self.op_args.args, 
            **self.op_args.kwargs,
            _messages=self.dialog
        )

    def stream(self, route: str, *args, **kwargs) -> typing.Iterator:
        """Stream the Operation

        Args:
            route (str): The name of the route for the message

        Yields:
            typing.Any: The result of of the op
        """
        msg = self.router[route](*args, **kwargs)
        self.dialog.append(msg)

        for r in self.op.stream(
            *self.op_args.args, 
            **self.op_args.kwargs,
            _messages=self.dialog
        ):
            yield r
    
    async def astream(self, route: str, *args, **kwargs) -> typing.AsyncIterator:
        """Asynchronously stream

        Args:
            route (str): The name of the route for the message

        Yields:
            typing.Any: The result of of the op
        """
        msg = self.router[route](*args, **kwargs)
        self.dialog.append(msg)

        async for r in self.op.astream(
            *self.op_args.args, 
            **self.op_args.kwargs,
            _messages=self.dialog
        ):
            yield r
