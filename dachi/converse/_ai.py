from abc import ABC, abstractmethod
from typing import Any, Coroutine

from . import _structs
from .. import process
from ._structs import Message, MessageList


class PromptModel(process.Module, ABC):

    @abstractmethod
    def forward(self, message: Message) -> Message:
        pass

    @abstractmethod
    def async_forward(self, *args, **kwargs) -> Coroutine[Any, Any, Any]:
        return super().async_forward(*args, **kwargs)


class ChatModel(process.Module):
    
    @abstractmethod
    def forward(self, message: MessageList) -> Message:
        pass

    @abstractmethod
    async def async_forward(self, *args, **kwargs) -> Any:
        return await super().async_forward(*args, **kwargs)

