from typing import AsyncIterator, Dict
import pkg_resources
import typing
from typing import Dict, List
import anthropic

from .._core import Message, AIPrompt
from .._core import AIModel, AIResponse, TextMessage
import anthropic
from typing import Dict, Iterator, AsyncIterator


# TODO: add utility for this
required = {'anthropic'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed


if len(missing) > 0:

    raise RuntimeError(f'To use this module openai must be installed.')


class AnthropicModel(AIModel):
    """Adapter for calling Anthropic's Messages API"""

    def __init__(
        self, model_name: str = 'claude-3.5-sonnet', 
        client_kwargs: typing.Dict=None, 
        **kwargs
    ):
        self.client_kwargs = client_kwargs or {}
        self.model_name = model_name
        self.kwargs = kwargs

    def convert(self, message: Message) -> Dict:
        """Convert a Message to the format needed for Anthropic's Messages API"""
        return {
            "role": message.source,
            "content": message.text
        }

    def forward(
        self, prompt: AIPrompt, **kwarg_override
    ) -> AIResponse:
        """Run a standard query to the Anthropic Messages API"""
        
        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        kwargs['max_tokens'] = kwargs.get('max_tokens', 100)
        client = anthropic.Anthropic(**self.client_kwargs)
        messages = self.convert_messages(prompt.aslist())
        response = client.messages.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        response_message = TextMessage(
            source="assistant", text=response['completion']
        )
        return AIResponse(message=response_message, source=response)

    def stream_forward(
        self, prompt: AIPrompt, **kwarg_override
    ) -> Iterator[AIResponse]:
        """Stream response from Anthropic's Messages API"""

        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        kwargs['max_tokens'] = kwargs.get('max_tokens', 100)
        client = anthropic.Anthropic(**self.client_kwargs)
        messages = self.convert_messages(prompt.aslist())
        response_generator = client.messages.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in response_generator:
            response_message = TextMessage(
                source="assistant", text=chunk['completion']
            )
            yield AIResponse(message=response_message, source=chunk)

    async def async_forward(
        self, prompt: AIPrompt, **kwarg_override
    ) -> AIResponse:
        """Run an asynchronous query to the Anthropic Messages API"""
        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        kwargs['max_tokens'] = kwargs.get('max_tokens', 100)
        client = anthropic.Anthropic(**self.client_kwargs)
        messages = self.convert_messages(prompt.aslist())
        response = await client.messages.acreate(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        response_message = TextMessage(source="assistant", text=response['completion'])
        return AIResponse(message=response_message, source=response)

    async def async_stream_forward(
        self, prompt: AIPrompt, **kwarg_override
    ) -> AsyncIterator[AIResponse]:
        """Run asynchronous streaming query to Anthropic's Messages API"""
        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        kwargs['max_tokens'] = kwargs.get('max_tokens', 100)
        client = anthropic.Anthropic(**self.client_kwargs)
        messages = self.convert_messages(prompt.aslist())
        async for chunk in client.messages.acreate(
            model=self.model_name,
            messages=messages,
            stream=True,
            **kwargs
        ):
            response_message = TextMessage(source="assistant", text=chunk['completion'])
            yield AIResponse(message=response_message, source=chunk)


class AnthropicMessage(TextMessage):
    """A text message made to work with Anthropic. Especially makes
    using prompt more straightforward
    """
    
    def prompt(self, model: typing.Union[AIModel, str], **kwarg_overrides) -> AIResponse:
        """prompt the model

        Args:
            model (typing.Union[AIModel, str]): The model to execute

        Returns:
            AIResponse: The response from the model
        """
        if isinstance(model, str):
            model = AnthropicModel(model, **kwarg_overrides)

        return super().prompt(model, **kwarg_overrides)
