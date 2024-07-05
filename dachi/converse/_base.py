from abc import ABC, abstractmethod
from typing import Any

from . import _structs
from .. import process
from ._structs import Message, MessageList


class PromptCompletion(process.Module, ABC):

    @abstractmethod
    def forward(self, message: Message) -> Message:
        pass


class ChatCompletion(process.Module):
    
    @abstractmethod
    def forward(self, message: MessageList) -> Message:
        pass
