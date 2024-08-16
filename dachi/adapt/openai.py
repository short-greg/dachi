from typing import AsyncIterator, Dict
import pkg_resources
import openai
import typing

from .._core import Message
from .._core import AIModel, AIResponse, Instruction

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

    def convert(self, messages: typing.List[Message]) -> typing.List[typing.Dict]:
        """Convert the messages to a format to use for the OpenAI API

        Args:
            messages (typing.List[Message]): The messages

        Returns:
            typing.List[typing.Dict]: The input to the API
        """
        result = []
        for message in messages:
            text = message.text
            if isinstance(text, Instruction):
                text = text.render()
            result.append({
                'role': message.source,
                'content': text
            })

        return result

    def forward(self, messages: typing.List[Message], **kwarg_override) -> Dict:

        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        client = openai.OpenAI()

        response = client.chat.completions.create(
            model=self.model,
            messages=self.convert(messages),
            **kwargs
        )
        return AIResponse(
            response.choices[0].message.content,
            response
        )

    def stream_iter(
        self, messages: typing.List[Message], 
        **kwarg_override
    ) -> typing.Iterator[typing.Tuple[Message, Message, typing.Dict]]:

        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        
        client = openai.OpenAI()
        query = client.chat.completions.create(
            model=self.model,
            messages=self.convert(messages),
            stream=True,
            **kwargs
        )
        cur_message = ''
        for chunk in query:
            delta = chunk.choices[0].delta.content

            if delta is None:
                delta = ''

            cur_message = cur_message + delta
            yield AIResponse(
                cur_message, chunk, None, delta
            )
    
    async def async_forward(
        self, messages: typing.List[Message], 
        **kwarg_override
    ) -> typing.Tuple[str, typing.Dict]:
        client = openai.AsyncOpenAI()

        kwargs = {
            **self.kwargs,
            **kwarg_override
        }

        response = await client.chat.completions.create(
            model=self.model,
            messages=self.convert(messages),
            **kwargs
        )

        return AIResponse(
            response.choices[0].message.content,
            response
        )

    async def async_stream_iter(self, messages: typing.List[Message], **kwarg_override) -> AsyncIterator[AIResponse]:

        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        client = openai.AsyncOpenAI()
        query = await client.chat.completions.create(
            model=self.model,
            messages=self.convert(messages),
            stream=True,
            **kwargs
        )

        cur_message = ''
        async for chunk in query:
            delta = chunk.choices[0].delta.content
            if delta is None:
                delta = ''
            cur_message = cur_message + delta
            yield AIResponse(
                cur_message,
                chunk,
                None,
                delta
            )
