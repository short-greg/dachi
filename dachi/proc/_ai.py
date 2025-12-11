import typing as t
from abc import abstractmethod, ABC

import pydantic

from ._process import (
    Process,
    AsyncProcess,
    StreamProcess,
    AsyncStreamProcess,
)
from dachi.core import Inp

class LangModel(Process, AsyncProcess, StreamProcess, AsyncStreamProcess):
    """A simple LLM process that echoes input with a prefix.
    """

    @abstractmethod
    def forward(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        """Synchronous LLM response.
        
        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response.
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing the LLM's response string (str, List[Inp], Any). The first element is the response text,
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), and the third element is the raw response from the LLM
        
        """
        pass

    @abstractmethod
    async def aforward(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        """Asynchronous LLM response.
        
        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the
            appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response.
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.
        Returns:
            A tuple containing the LLM's response string (str, List[Inp], Any). The first element is the response text,
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), and the third element is the raw response from the LLM
        
        """
        pass

    @abstractmethod
    def stream(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Iterator[t.Tuple[str, t.List[Inp], t.Any]]:
        """Streaming synchronous LLM response.
        
        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the
            appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response. 
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.
        Returns:
            An iterator yielding tuples containing the LLM's response string (str, List[Inp], Any). The first element is the response text,
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), the message for the current call will not be added until the stream is complete, and the third element is the raw response from the LLM, could be a chunk object
        """
        pass

    @abstractmethod
    async def astream(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.AsyncIterator[t.Tuple[str, t.List[Inp], t.Any]]:
        """Streaming asynchronous LLM response.
        
        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the
            appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response. 
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.
        Returns:
            An async iterator yielding tuples containing the LLM's response string (str, List[Inp
            , Any). The first element is the response text,
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), the message for the current call will not be added until the stream is complete, and the third element is the raw response from the LLM, could be a chunk object
        """
        pass


LANG_MODEL = t.TypeVar("LANG_MODEL", bound=LangModel)
