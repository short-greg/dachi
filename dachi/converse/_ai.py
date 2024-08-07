from abc import ABC, abstractmethod
from typing import Any

from .._core import Module
from .._core._structs_doc import Message
from ..adapt import AIModel
import typing


class Assistant(Module, ABC):

    def __init__(self, model: AIModel):
        self.model = model

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    def stream_text(self, message: Message) -> typing.Iterator[str]:

        streamer = self.stream_forward(message)
        for _, dx in streamer:
            yield dx.content['text']


class Prompt(Assistant):

    def __init__(self, model: AIModel, init_messages: typing.List[Message]):

        self.init_messages = init_messages
        self.model = model

    def forward(self, message: Message) -> Message:

        response = self.model.query([*self.init_messages, message])
        return response.message

    def stream_iter(self, message: Message) -> typing.Iterator[
        typing.Tuple[Message, Message]
    ]:
        # default behavior doesn't actually stream
        for response in self.model.stream_query([
            *self.init_messages, message
        ]):
            yield response.message, response.delta

    async def async_forward(self, message: Message) -> typing.Any:

        response = await self.model.async_query(
            [*self.init_messages, message]
        )
        return response.message


class Chat(Assistant):

    def __init__(self, model: AIModel, init_messages: typing.List[Message]):

        self.messages = init_messages
        self.model = model

    def forward(self, message: Message) -> Message:

        self.messages.append(message)

        response = self.model.query(self.messages)
        self.messages.append(response.message)
        return response.message

    def stream_iter(self, message: Message) -> typing.Iterator[
        typing.Tuple[Message, Message]
    ]:
        # default behavior doesn't actually stream
        self.messages.append(message)

        for response in self.model.stream_query(self.messages):
            yield response.message, response.delta

        self.messages.append(response)

    async def async_forward(self, message: Message) -> typing.Any:

        self.messages.append(message)

        response = await self.model.async_query(
            [*self.init_messages, message]
        )
        self.messages.append(response.message)
        return response.message



# 1) Streamble
# 2) Async


# class PromptModel(Module, ABC):

#     @abstractmethod
#     def forward(self, message: Message) -> Message:
#         pass

#     @abstractmethod
#     def async_forward(self, message: Message) -> Message: # Coroutine[Any, Any, Any]:
#         return self.forward(message)

#     @abstractmethod
#     def stream_forward(self, messages: MessageList) -> Any:
#         pass

#     @abstractmethod
#     async def async_stream_forward(self, messages: MessageList) -> Any:
#         pass


# class ChatModel(Module):
    
#     @abstractmethod
#     def forward(self, messages: MessageList) -> Message:
#         pass

#     async def async_forward(self, messages: MessageList) -> Any:
#         return self.forward(messages)

#     @abstractmethod
#     def stream_forward(self, messages: MessageList) -> Any:
#         pass

#     @abstractmethod
#     async def async_stream_forward(self, messages: MessageList) -> Any:
#         pass


