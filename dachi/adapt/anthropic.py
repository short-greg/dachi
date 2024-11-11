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

    def __init__(self, api_key: str, model_name: str = 'claude-3.5-sonnet'):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

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
        messages = self.convert_messages(prompt.aslist())
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens_to_sample=kwarg_override.get("max_tokens", 100)
        )
        response_message = TextMessage(
            source="assistant", text=response['completion']
        )
        return AIResponse(message=response_message, source=response)

    def stream_forward(
        self, prompt: AIPrompt, **kwarg_override
    ) -> Iterator[AIResponse]:
        """Stream response from Anthropic's Messages API"""
        messages = self.convert_messages(prompt.aslist())
        response_generator = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens_to_sample=kwarg_override.get("max_tokens", 100),
            stream=True
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
        messages = self.convert_messages(prompt.aslist())
        response = await self.client.messages.acreate(
            model=self.model_name,
            messages=messages,
            max_tokens_to_sample=kwarg_override.get("max_tokens", 100)
        )
        response_message = TextMessage(source="assistant", text=response['completion'])
        return AIResponse(message=response_message, source=response)

    async def async_stream_forward(
        self, prompt: AIPrompt, **kwarg_override
    ) -> AsyncIterator[AIResponse]:
        """Run asynchronous streaming query to Anthropic's Messages API"""
        messages = self.convert_messages(prompt.aslist())
        async for chunk in self.client.messages.acreate(
            model=self.model_name,
            messages=messages,
            max_tokens_to_sample=kwarg_override.get("max_tokens", 100),
            stream=True
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


class AnthropicEmbeddingModel(AIModel):
    """Adapter for calling Anthropic's Embedding API"""

    def __init__(self, api_key: str, model_name: str = 'claude-embedding'):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def convert(self, message: Message) -> Dict:
        """Convert a Message to the format needed for Anthropic's Embedding API"""
        if isinstance(message, TextMessage):
            return {
                "role": message.source,
                "content": message.text
            }
        else:
            raise TypeError("Unsupported message type")

    def forward(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
        """Run a standard query to the Anthropic Embedding API"""
        # Extract text content from messages
        messages = self.convert_messages(prompt.aslist())
        texts_to_embed = [msg['content'] for msg in messages]
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts_to_embed
        )
        
        # Create an AIResponse for each embedded message
        responses = []
        for text, embedding in zip(texts_to_embed, response['embeddings']):
            response_message = TextMessage(source="embedding_result", text=text)
            responses.append(AIResponse(message=response_message, source=embedding))

        return responses  # Returning a list of AIResponse objects for each embedded input

    async def async_forward(self, prompt: AIPrompt, **kwarg_override) -> List[AIResponse]:
        """Run an asynchronous query to the Anthropic Embedding API"""
        messages = self.convert_messages(prompt.aslist())
        texts_to_embed = [msg['content'] for msg in messages]
        
        response = await self.client.embeddings.acreate(
            model=self.model_name,
            input=texts_to_embed
        )
        
        # Create an AIResponse for each embedded message
        responses = []
        for text, embedding in zip(texts_to_embed, response['embeddings']):
            response_message = TextMessage(
                source="embedding_result", text=text)
            responses.append(AIResponse(message=response_message, source=embedding))

        return responses
