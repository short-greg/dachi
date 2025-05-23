# TODO: Update this once LLM and convs are done
# 1st party
import typing
from typing import Self
# 3rd party
import pydantic
# local
from ..msg._messages import (
    Msg, BaseDialog, ListDialog
)
from ._asst import Assistant
from ..msg._msg import ToMsg, KeyGet, FromMsg
from ..msg._out import OutConv
from ..msg._out import conv_to_out
from ..proc import (
    Module, AsyncModule, 
    StreamModule, AsyncStreamModule
)
from ..utils import (
    coalesce, UNDEFINED
)
from .. import utils


S = typing.TypeVar('S', bound=pydantic.BaseModel)


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
    def __init__(
        self, assistant: Assistant, 
        to_msg: ToMsg, 
        out: str | typing.List[str] | FromMsg, 
        filter_undefined: bool=True,
        follow_up: bool=False
    ):
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
        self.out = out
        self.filter_undefined = filter_undefined
        self.follow_up = follow_up

    def forward(
        self, *args,  
        _out: OutConv=None,
        _messages: typing.List[Msg]=None, 
        **kwargs
    ) -> typing.Any:
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
        msg = self.to_msg(*args, **kwargs)
        _out = conv_to_out(_out)

        _messages = _messages or []
        _messages.append(msg)

        while True:
            resp_msg = self.assistant(
                _messages
            )
            _messages.append(resp_msg)
            if not self.follow_up or len(resp_msg.follow_up) == 0:
                break
            _messages.extend(resp_msg.follow_up)

        res = self.out(resp_msg)
        if _out is None:
            return res
        return _out.delta(res, {})
        
    async def aforward(
        self, *args, 
        _out: OutConv=None, 
        _messages: typing.List[Msg]=None,
        **kwargs
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
        _out = conv_to_out(_out)
        _messages = _messages or []
        _messages.append(msg)
        while True:
            resp_msg = await self.assistant.aforward(
                _messages
            )
            _messages.append(resp_msg)
            if not self.follow_up or len(resp_msg.follow_up) == 0:
                break
            _messages.extend(resp_msg.follow_up)

        res = self.out(resp_msg)
        if _out is None:
            return res
        return _out.delta(res, {})
    
    def stream(
        self, *args, 
        _out: OutConv=None,
        _messages: typing.List[Msg]=None, 
        **kwargs
    ) -> typing.Iterator:
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
        msg = self.to_msg(*args, **kwargs)

        _messages = _messages or []
        
        _out = conv_to_out(_out)
        _messages.append(msg)
        resp_msg = None

        delta_store = {}
        while True: 
            
            for resp_msg in self.assistant.stream(
                _messages
            ):
                if self.filter_undefined:
                    res, filtered = self.out.filter(resp_msg)

                    res = self.out(resp_msg)
                    if _out is not None:
                        res = _out.delta(
                            res, delta_store
                        )
                        filtered = filtered or res == UNDEFINED
                    if not filtered:
                        yield res
                else:
                    res = self.out(resp_msg)
                    if _out is not None:
                        res = _out.delta(res, {})
                    yield res
            _messages.append(resp_msg)
            if not self.follow_up or len(resp_msg.follow_up) == 0:
                break
            _messages.extend(resp_msg.follow_up)

    async def astream(
        self, 
        *args, 
        _out: OutConv=None, 
        _messages: typing.List[Msg]=None, **kwargs
    ) -> typing.AsyncIterator:
        """
        Asynchronously streams the assistant with the provided arguments and returns the processed output.
        Args:
            *args: Positional arguments to be passed to the `to_msg` method for message creation..
            _asst (typing.Dict or object, optional): A dictionary to spawn a new assistant instance or an existing assistant object. 
            
            If None, defaults to `self.assistant`.
            **kwargs: Additional keyword arguments to be passed to the `to_msg` method.
        Returns:
            The processed output of the assistant's response, as handled by the `_out` callable.
        Raises:
            Any exceptions raised during the assistant's execution or message processing.
        """
        _messages = _messages or []
        msg = self.to_msg(*args, **kwargs)
        _out = conv_to_out(_out)

        _messages.append(msg)
        delta_store = {}
        while True:
            async for resp_msg in await self.assistant.astream(
                _messages
            ):
                if self.filter_undefined:
                    res, filtered = self.out.filter(
                        resp_msg
                    )
                    if _out is not None:
                        res = _out.delta(
                            res, delta_store
                        )
                        filtered = filtered or res == UNDEFINED

                    if not filtered:
                        yield res
                else:
                    res = self.out(resp_msg)
                    if _out is not None:
                        res = _out.delta(res)
                    yield res
            
            _messages.append(resp_msg)
            if not self.follow_up or len(resp_msg.follow_up) == 0:
                break
            _messages.extend(resp_msg.follow_up)

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
        self, 
        to_msg: ToMsg=UNDEFINED, 
        assistant: Assistant=UNDEFINED, 
        out: str | KeyGet | typing.List[str | KeyGet]=UNDEFINED, filter_undefined: bool=UNDEFINED
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
        filter_undefined = coalesce(filter_undefined, self.filter_undefined)
        return Op(
            assistant, to_msg, out, filter_undefined
        )


class Threaded(
    Module, AsyncModule, StreamModule,
    AsyncStreamModule
):
    """A Threaded Op. Use to keep the Op 
    """
    def __init__(
        self, assistant: Assistant, 
        router: typing.Dict[str, ToMsg],
        out: str | typing.List[str] | FromMsg, 
        dialog: BaseDialog=None,
        filter_undefined: bool=True,
        follow_up: bool=False
    ):
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
        self.router = router
        if not isinstance(out, FromMsg):
            out = FromMsg(out)
        self.out = out
        self.dialog = dialog or ListDialog()
        self.filter_undefined = filter_undefined
        self.follow_up = follow_up

    def forward(
        self, 
        route: str, 
        *args, 
        _out: OutConv=None,
        **kwargs
    ) -> typing.Any:
        """
        Executes the assistant with the provided arguments and returns the processed output.
        Args:
            *args: Positional arguments to be passed to the `to_msg` method for message creation..
            _asst (typing.Dict or object, optional): A dictionary to spawn a new assistant instance or an existing assistant object. 
            
            If None, defaults to `self.assistant`.
            **kwargs: Additional keyword arguments to be passed to the `to_msg` method.
        Returns:
            The processed output of the assistant's response, as handled by the `_out` callable.
        Raises:
            Any exceptions raised during the assistant's execution or message processing.
        """
        msg = self.router[route](*args, **kwargs)
        _out = conv_to_out(_out)
        self.dialog = self.dialog.append(msg)

        while True:
            resp_msg = self.assistant(
                list(self.dialog)
            )
            if not self.follow_up or len(resp_msg.follow_up) == 0:
                break
            self.dialog = self.dialog.append(resp_msg.follow_up)

        res = self.out(resp_msg)
        if _out is None:
            return res
        return _out.delta(res, {})

    async def aforward(
        self, 
        route: str, 
        *args, 
        _out: OutConv=None,
        **kwargs
    ) -> typing.Any:
        """
        Asynchronously executes the assistant with the provided arguments and returns the processed output.
        Args:
            *args: Positional arguments to be passed to the `to_msg` method for message creation.`.
            _asst (typing.Dict or object, optional): A dictionary to spawn a new assistant instance or an existing assistant object. 
                If None, defaults to `self.assistant`.
            **kwargs: Additional keyword arguments to be passed to the `to_msg` method.
        Returns:
            The processed output of the assistant's response, as handled by the `_out` callable.
        Raises:
            Any exceptions raised during the assistant's execution or message processing.
        """
        msg = self.router[route](*args, **kwargs)
        _out = conv_to_out(_out)
        self.dialog = self.dialog.append(msg)
        while True:
            resp_msg = await self.assistant.aforward(
                list(self.dialog)
            )
            self.dialog = self.dialog.append(resp_msg.follow_up)
            if not self.follow_up or len(resp_msg.follow_up) == 0:
                break
            self.dialog.extend(resp_msg.follow_up)
        resp_msg = await self.assistant.aforward(
            list(self.dialog)
        )
        self.dialog = self.dialog.append(resp_msg)
        res = self.out(resp_msg)
        if _out is None:
            return res
        return _out.delta(res, {})

    def stream(
        self, 
        route: str, 
        *args, 
        _out: OutConv=None,
        **kwargs
    ) -> typing.Iterator:
        """Stream the Operation

        Args:
            route (str): The name of the route for the message

        Yields:
            typing.Any: The result of of the op
        """
        msg = self.router[route](*args, **kwargs)
        self.dialog = self.dialog.append(msg)
        _out = conv_to_out(_out)
        
        resp_msg = None
        delta_store = {}
        while True:
            for resp_msg in self.assistant.stream(
                list(self.dialog)
            ):
                if self.filter_undefined:
                    res, filtered = self.out.filter(
                        resp_msg
                    )
                    if _out is not None:
                        res = _out.delta(
                            res, delta_store
                        )
                        filtered = filtered or res == UNDEFINED

                    if not filtered:
                        yield res
                else:
                    res = self.out(resp_msg)
                    if _out is not None:
                        res = _out.delta(res, delta_store)
                    yield res
            
            self.dialog = self.dialog.append(resp_msg)
            if not self.follow_up or len(resp_msg.follow_up) == 0:
                break
            self.dialog.extend(resp_msg.follow_up)

    async def astream(
        self, 
        route: str, 
        *args, 
        _out: OutConv=None,
        **kwargs
    ) -> typing.AsyncIterator:
        """Asynchronously stream

        Args:
            route (str): The name of the route for the message

        Yields:
            typing.Any: The result of of the op
        """
        msg = self.router[route](*args, **kwargs)
        _out = conv_to_out(_out)
        self.dialog = self.dialog.append(msg)
        resp_msg = None

        delta_store = {}
        while True:
            async for resp_msg in await self.assistant.astream(
                list(self.dialog)
            ):
                if self.filter_undefined:
                    res, filtered = self.out.filter(
                        resp_msg
                    )
                    if _out is not None:
                        res = _out.delta(
                            res, delta_store
                        )
                        filtered = filtered or res == UNDEFINED

                    if not filtered:
                        yield res
                else:
                    res = self.out(resp_msg)
                    if _out is not None:
                        res = _out.delta(res, delta_store)
                    yield res
            
            self.dialog = self.dialog.append(resp_msg)
            if not self.follow_up or len(resp_msg.follow_up) == 0:
                break
            self.dialog.extend(resp_msg.follow_up)

    def asst(self, *args, **kwargs) -> Self:
        """Create a new operation with specified
        args

        Returns:
            Self: 
        """
        asst = self.assistant.spawn(
            *args, **kwargs
        )
        return self.spawn(
            assistant=asst
        )

    def spawn(
        self, to_msg: ToMsg=UNDEFINED, assistant: Assistant=UNDEFINED, router: typing.Dict[str, ToMsg]=UNDEFINED, dialog: BaseDialog=UNDEFINED, out: str | KeyGet | typing.List[str | KeyGet]=UNDEFINED, filter_undefined: bool=UNDEFINED
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
        dialog = coalesce(dialog, self.dialog)
        router = coalesce(router, self.router)
        filter_undefined = coalesce(filter_undefined, self.filter_undefined)
        return Threaded(
            assistant, to_msg, out, dialog, filter_undefined
        )

    def asst(self, *args, **kwargs) -> 'Op':
        """Use to spawn a new Op with a different assistant

        Returns:
            Op: a new op
        """
        return self.spawn(
            assistant=self.assistant.spawn(*args, **kwargs)
        )

