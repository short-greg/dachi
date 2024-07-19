from typing import AsyncIterator, Coroutine, Dict
import pkg_resources
import asyncio

from ._core import APIAdapter

# TODO: add utility for this
required = {'openai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if len(missing) > 0:
    raise RuntimeError(f'To use this module openai must be installed.')

import openai


class OpenAIChatAdapter(APIAdapter):

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.kwargs = kwargs

    def query(self, data, **kwarg_override) -> Dict:

        kwargs = {
            **self._kwargs,
            **kwarg_override
        }
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=data,
            **kwargs
        )
        return {
            'message': response.choices[0].message,
            'response': response
        }
    
    def stream_query(self, data, **kwarg_override) -> pkg_resources.Iterator[Dict]:

        kwargs = {
            **self._kwargs,
            **kwarg_override
        }
        
        client = openai.OpenAI()
        query = client.chat.completions.create(
            model=self.model,
            messages=data,
            stream=True,
            **kwargs
        )
        message = ''
        for chunk in query:
            delta = chunk.choices[0].delta
            message = message + delta
            yield {
                'message': message,
                'delta': delta,
                'response': chunk
            }
    
    async def async_query(self, data, bulk: bool=False, **kwarg_override) -> Dict:
        client = openai.AsyncOpenAI()

        kwargs = {
            **self._kwargs,
            **kwarg_override
        }

        response = await client.chat.completions.create(
            model=self.model,
            messages=data,
            **kwargs
        )

        return {
            'message': response.choices[0].message,
            'response': response
        }
    
    async def async_stream_query(self, data, **kwarg_override) -> AsyncIterator[Dict]:

        kwargs = {
            **self._kwargs,
            **kwarg_override
        }
        client = openai.AsyncOpenAI()
        query = await client.chat.completions.create(
            model=self.model,
            messages=data,
            stream=True,
            **kwargs
        )

        async for chunk in query:
            delta = chunk.choices[0].delta
            message = message + delta
            yield {
                'message': message,
                'delta': delta,
                'response': chunk
            }


# class OpenAIPrompModel(PromptModel):

#     def __init__(self, model: str='gpt-4-turbo', name: str='Assistant'):
#         super().__init__()
#         self.model = model
#         self.name = name

#     def forward(self, message: Message) -> Message:
        
#         client = openai.OpenAI()
#         result = client.chat.completions.create(
#             model=self.model,
#             messages=message.dump()
#         )
#         return Message.Text(
#             self.name,
#             result.choices[0].message
#         )
    
#     async def async_forward(self, message: Message) -> Message:

#         client = openai.AsyncOpenAI()
#         result = client.chat.completions.create(
#             model=self.model,
#             messages=message.dump()
#         )
#         return Message.Text(
#             self.name,
#             result.choices[0].message
#         )


# class OpenAIChatModel(ChatModel):

#     def __init__(self, model: str='gpt-4-turbo', name: str='Assistant'):
#         super().__init__()
#         self.model = model
#         self.name = name

#     def forward(self, messages: MessageList) -> Message:
        
#         client = openai.OpenAI()
#         result = client.chat.completions.create(
#             model=self.model,
#             messages=messages.dump()
#         )
#         return Message.Text(
#             self.name,
#             result.choices[0].message
#         )
    
#     async def async_forward(self, messages: Message) -> Message:

#         client = openai.AsyncOpenAI()
#         result = client.chat.completions.create(
#             model=self.model,
#             messages=messages.dump()
#         )
#         return Message.Text(
#             self.name,
#             result.choices[0].message
#         )
