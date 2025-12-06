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

class LangModel(Process, AsyncProcess, StreamProcess, AsyncStreamProcess, ABC):
    """A simple LLM process that echoes input with a prefix.
    """

    @abstractmethod
    def forward(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Any=None,
        **kwargs
    ) -> t.Tuple[str, t.List[Inp]]:
        """Synchronous LLM response."""
        pass

    @abstractmethod
    async def aforward(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Any=None,
        **kwargs
    ) -> t.Tuple[str, t.List[Inp]]:
        """Asynchronous LLM response."""
        pass

    @abstractmethod
    def stream(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Any=None,
        **kwargs
    ) -> t.Iterator[t.Tuple[str, t.List[Inp]]]:
        """Streaming synchronous LLM response."""
        pass

    @abstractmethod
    async def astream(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Any=None,
        **kwargs
    ) -> t.AsyncIterator[t.Tuple[str, t.List[Inp]]]:
        """Streaming asynchronous LLM response."""
        pass


LANG_MODEL = t.TypeVar("LANG_MODEL", bound=LangModel)
