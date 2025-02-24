from abc import ABC, abstractmethod
import typing

from ..proc import (
    Module, AsyncModule, StreamModule, AsyncStreamModule
)
from ._messages import (
    Msg, BaseDialog
)


class AssistantBase(Module, ABC):

    @abstractmethod
    def forward(self, msg: Msg | BaseDialog, *args, **kwargs) -> typing.Tuple[Msg, typing.Any]:
        pass


class AsyncAssistantBase(AsyncModule, ABC):

    @abstractmethod
    async def aforward(self, msg: Msg | BaseDialog, *args, **kwargs) -> typing.Tuple[Msg, typing.Any]:
        pass


class StreamAssistantBase(StreamModule, ABC):

    @abstractmethod
    def stream(
        self, msg: Msg | BaseDialog, *args, **kwargs
    ) -> typing.Iterator[typing.Tuple[Msg, typing.Any]]:
        pass


class AsyncStreamAssistantBase(AsyncStreamModule, ABC):

    @abstractmethod
    async def astream(
        self, msg: Msg | BaseDialog, *args, **kwargs
    ) -> typing.AsyncIterator[typing.Tuple[Msg, typing.Any]]:
        pass
