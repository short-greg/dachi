# 1st party
import typing
from abc import ABC, abstractmethod
import typing

# 3rd party
import pydantic

# local
from ..msg._messages import (
    Msg, BaseDialog
)
from ..proc import (
    Module, AsyncModule, 
    StreamModule, AsyncStreamModule
)

S = typing.TypeVar('S', bound=pydantic.BaseModel)

# TODO: MOVE OUT OF HERE


LLM_PROMPT = typing.Union[typing.Iterable[Msg], Msg]
LLM_RESPONSE = typing.Tuple[Msg, typing.Any]


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
