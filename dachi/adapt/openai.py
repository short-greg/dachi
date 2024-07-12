from typing import Coroutine
import pkg_resources

from ._structs import Message, MessageList
from ._base import PromptModel, ChatModel

# TODO: add utility for this
required = {'openai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if len(missing) > 0:
    raise RuntimeError(f'To use this module openai must be installed.')

import openai


class OpenAIPrompModel(PromptModel):

    def __init__(self, model: str='gpt-4-turbo', name: str='Assistant'):
        super().__init__()
        self.model = model
        self.name = name

    def forward(self, message: Message) -> Message:
        
        client = openai.OpenAI()
        result = client.chat.completions.create(
            model=self.model,
            messages=message.dump()
        )
        return Message.Text(
            self.name,
            result.choices[0].message
        )
    
    async def async_forward(self, message: Message) -> Message:

        client = openai.AsyncOpenAI()
        result = client.chat.completions.create(
            model=self.model,
            messages=message.dump()
        )
        return Message.Text(
            self.name,
            result.choices[0].message
        )


class OpenAIChatModel(ChatModel):

    def __init__(self, model: str='gpt-4-turbo', name: str='Assistant'):
        super().__init__()
        self.model = model
        self.name = name

    def forward(self, messages: MessageList) -> Message:
        
        client = openai.OpenAI()
        result = client.chat.completions.create(
            model=self.model,
            messages=messages.dump()
        )
        return Message.Text(
            self.name,
            result.choices[0].message
        )
    
    async def async_forward(self, messages: Message) -> Message:

        client = openai.AsyncOpenAI()
        result = client.chat.completions.create(
            model=self.model,
            messages=messages.dump()
        )
        return Message.Text(
            self.name,
            result.choices[0].message
        )
