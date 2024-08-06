from typing import AsyncIterator, Dict
import pkg_resources
import openai
import typing
from .._core import Message

from ._core import AIModel, Response

# TODO: add utility for this
required = {'openai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if len(missing) > 0:
    raise RuntimeError(f'To use this module openai must be installed.')


class OpenAIChatModel(AIModel):

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
        return Response(
            response.choices[0].message,
            response
        )

    def stream_query(self, data, **kwarg_override) -> typing.Iterator[typing.Tuple[Message, Message, typing.Dict]]:

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
        cur_message = ''
        for chunk in query:
            delta = chunk.choices[0].delta

            cur_message = cur_message + delta
            yield Response(
                cur_message, chunk, delta
            )
    
    async def async_query(self, data, bulk: bool=False, **kwarg_override) -> typing.Tuple[str, typing.Dict]:
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

        return Response(
            response.choices[0].message,
            response
        )

    async def async_stream_query(self, data, **kwarg_override) -> AsyncIterator[Response]:

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

        cur_message = ''

        async for chunk in query:
            delta = chunk.choices[0].delta
            cur_message = cur_message + delta
            yield Response(
                cur_message,
                chunk,
                delta
            )


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
