# 1st party
import typing
from abc import ABC, abstractmethod
import typing
from .._core._core import (
    Module
)
from .._core import Msg, ListDialog, Dialog


LLM_PROMPT = typing.Union[Dialog, Msg]


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
    include = set(role)
    return [message for message in messages
        if message.role in include]


class LLM(Module, ABC):
    """APIAdapter allows one to adapt various WebAPI or otehr
    API for a consistent interface
    """
    @abstractmethod
    def forward(
        self, prompt, 
        **kwarg_override
    ) -> Msg:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        pass

    def stream(
        self, prompt, 
        **kwarg_override
    ) -> typing.Iterator[Msg]:
        """API that allows for streaming the response

        Args:
            prompt (AIPrompt): Data to pass to the API

        Returns:
            typing.Iterator: Data representing the streamed response
            Uses 'delta' for the difference. Since the default
            behavior doesn't truly stream. This must be overridden 

        Yields:
            typing.Dict: The data
        """
        yield self.forward(prompt, **kwarg_override)
    
    async def aforward(
        self, prompt, **kwarg_override
    ) -> Msg:
        """Run this query for asynchronous operations
        The default behavior is simply to call the query

        Args:
            data: Data to pass to the API

        Returns:
            typing.Any: 
        """
        return self.forward(prompt, **kwarg_override)

    async def astream(
        self, prompt, **kwarg_override
    ) -> typing.AsyncIterator[Msg]:
        """Run this query for asynchronous streaming operations
        The default behavior is simply to call the query

        Args:
            prompt (AIPrompt): The data to pass to the API

        Yields:
            typing.Dict: The data returned from the API
        """
        result = self.forward(prompt, **kwarg_override)
        yield result

    def __call__(self, prompt: LLM_PROMPT, **kwarg_override) -> Msg:
        """Execute the AIModel

        Args:
            prompt (AIPrompt): The prompt

        Returns:
            AIResponse: Get the response from the AI
        """
        return self.forward(prompt, **kwarg_override)
    
    def get(self, x: typing.Union[str, typing.Callable], dx: typing.Union[str, typing.Callable]):
        """
        Retrieves a value based on the provided parameters.

            x (typing.Union[str, typing.Callable]): The key or a callable to determine the key.
            dx (typing.Union[str, typing.Callable]): The default value or a callable to determine the default value.

            Get: An instance of the Get class initialized with the provided parameters.
        """
        return Get(self, x, dx)


class Get(Module):
    """Use to convert a message response to a 
    """
    def __init__(self, llm: LLM, x: typing.Union[str, typing.Callable], dx: typing.Union[str, typing.Callable]):
        """Initialize the AI class instance.

            llm (LLM): Language Learning Model instance to be used for processing.
            x (Union[str, Callable]): Input content or function that returns content to be processed.
            dx (Union[str, Callable]): Delta or differential content/function that modifies or supplements the input.

        """
        super().__init__()
        self._llm = llm
        self._dx = dx
        self._x = x

    def forward(self, prompt, **kwarg_override) -> typing.Any:
        """
        Forward the prompt through the LLM and extract/process the response.
        Args:
            prompt: Input prompt to be processed by the LLM
            **kwarg_override: Optional keyword arguments to override default LLM parameters
        Returns:
            typing.Any: Either a direct string extraction from the LLM response using self._x as a key,
                       or the result of processing the response through self._x as a callable function.
        Yields:
            typing.Any: The processed response, either extracted or transformed based on self._x configuration
  
        """
        msg = self._llm.forward(prompt, **kwarg_override)
        if isinstance(self._x, str):
            yield msg[self._x]
        else:
            yield self._x(msg)
    
    async def aforward(self, prompt, **kwarg_override) -> typing.Any:
        """
        Asynchronously forwards a prompt to the language model and yields processed results.
        This method sends a prompt to the underlying language model and processes its response
        according to the configured extraction pattern (_x).
        Parameters
        ----------
        prompt : Any
            The input prompt to be sent to the language model.
        **kwarg_override : dict
            Optional keyword arguments to override default parameters of the language model.
        Yields
        ------
        Any
            The processed response from the language model. If _x is a string, yields the value
            corresponding to that key from the response. If _x is a callable, yields the result
            of applying that function to the response.
        Returns
        -------
        list
            A single-element list containing the extraction pattern (_x).
        Notes
        -----
        The extraction pattern (_x) determines how the response is processed:
        - If _x is a string: extracts the value at that key from the response dictionary
        - If _x is a callable: applies the function to the entire response
        """
        
        msg = self._llm.aforward(prompt, **kwarg_override)
        if isinstance(self._x, str):
            yield msg[self._x]
        else:
            yield self._x(msg)
        return [self._x]
    
    def stream(self, prompt, **kwarg_override) -> typing.Iterator:
        """ Stream the results from the LLM and extract/process the response.
        Args:
            prompt: Input prompt to be processed by the LLM
            **kwarg_override: Optional keyword arguments to override default LLM parameters
        Returns:
            typing.Iterator: An iterator that yields the processed response, either extracted or transformed based on self._x configuration
        Yields: 
            typing.Any: The processed response, either extracted or transformed based on self._x configuration
        """ 
        for msg in self._llm.stream(prompt, **kwarg_override):
            if isinstance(self._dx, str):
                yield msg[self._dx]
            else:
                yield self._dx(msg)
    
    async def astream(self, prompt, **kwarg_override) -> typing.AsyncIterator:
        """Asynchronously stream the results from the LLM and extract/process the response.
        Args:
            prompt: Input prompt to be processed by the LLM
            **kwarg_override: Optional keyword arguments to override default LLM parameters
        Returns:
            typing.AsyncIterator: An asynchronous iterator that yields the processed response, either extracted or transformed based on self._x configuration
        Yields:
            typing.Any: The processed response, either extracted or transformed based on self._x configuration
        """
        async for msg in self._llm.astream(prompt, **kwarg_override):
            if isinstance(self._dx, str):
                yield msg[self._dx]
            else:
                yield self._dx(msg)


TO_MSG = typing.Callable[[typing.Any], Msg]
FROM_MSG = typing.Callable[[Msg], typing.Any]


class ToMsg(ABC):
    """Converts inputs to a message"""

    @abstractmethod
    def __call__(self, *args, **kwds) -> Msg:
        pass


class StdToMsg(ToMsg):
    """Converts the inputs to a standard message"""

    def __call__(self, **kwargs) -> Msg:
        """Convert the inputs to a message"""
        return Msg(**kwargs)


def to_dialog(prompt: typing.Union[Dialog, Msg]) -> Dialog:
    """Convert a prompt to a dialog

    Args:
        prompt (typing.Union[Dialog, Msg]): The prompt to convert
    """
    if isinstance(prompt, Msg):
        prompt = ListDialog([prompt])

    return prompt
