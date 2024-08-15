from abc import ABC, abstractmethod
from typing import Any

from ._process import Module
from ._core import Message, TextMessage
from ._ai import AIModel
import typing


class Assistant(Module, ABC):

    def __init__(self, model: AIModel):
        self.model = model

    @abstractmethod
    def forward(self, message: Message) -> Message:
        pass

    def stream_text(self, message: Message) -> typing.Iterator[str]:

        streamer = self.stream_forward(message)
        for partial in streamer:
            
            yield partial.dx['text']


class Prompt(Assistant):

    def __init__(self, model: AIModel, init_messages: typing.List[Message]):

        self.init_messages = init_messages
        self.model = model
        self.role = 'assistant'

    def forward(self, message: Message) -> Message:

        response = self.model.query([*self.init_messages, message])
        return TextMessage(self.role, response.message)

    def stream_iter(self, message: Message) -> typing.Iterator[
        typing.Tuple[Message, Message]
    ]:
        for response in self.model.stream_query([
            *self.init_messages, message
        ]):
            yield TextMessage(self.role, response.message), TextMessage(self.role, response.delta)

    async def async_forward(self, message: Message) -> typing.Any:

        response = await self.model.async_query(
            [*self.init_messages, message]
        )
        return TextMessage(self.role, response.message)


class Chat(Assistant):

    def __init__(self, model: AIModel, init_messages: typing.List[Message]):

        self.messages = init_messages
        self.model = model
        self.role = 'assistant'

    def forward(self, message: Message) -> Message:

        self.messages.append(message)

        response = self.model.query(self.messages)
        self.messages.append(response.message)
        return TextMessage(source=self.role, text=response.message)

    def stream_iter(self, message: Message) -> typing.Iterator[
        typing.Tuple[Message, Message]
    ]:
        self.messages.append(message)

        for response in self.model.stream_query(self.messages):
            cur_message = TextMessage(source=self.role, text=response.message)
            cur_dx = TextMessage(source=self.role, text=response.delta)
            yield cur_message, cur_dx 
        else:
            self.messages.append(cur_message)

    async def async_forward(self, message: Message) -> typing.Any:

        self.messages.append(message)

        response = await self.model.async_query(
            [*self.init_messages, message]
        )
        self.messages.append(response.message)
        return TextMessage(source=self.role, text=response.message)

    def loop(self, include: typing.Callable[[Message], bool]=None) -> typing.Iterator[Message]:

        for message in self.messages:
            if include is None or include(message):
                yield message
