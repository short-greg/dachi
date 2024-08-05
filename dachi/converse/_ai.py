from abc import ABC, abstractmethod
from typing import Any

from ._convert import Module
from ._structs import Message, MessageList


class PromptModel(Module, ABC):

    @abstractmethod
    def forward(self, message: Message) -> Message:
        pass

    @abstractmethod
    def async_forward(self, message: Message) -> Message: # Coroutine[Any, Any, Any]:
        return self.forward(message)


class ChatModel(Module):
    
    @abstractmethod
    def forward(self, messages: MessageList) -> Message:
        pass

    @abstractmethod
    async def async_forward(self, messages: MessageList) -> Any:
        return self.forward(messages)


class Prompt(Module):

    pass
