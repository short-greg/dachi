import google.generativeai as genai
from abc import ABC
from typing import Dict, List
from .._core import Message, AIPrompt
from .._core import AIModel, AIResponse, TextMessage

from typing import List, Dict

from typing import Dict, Iterator, AsyncIterator, List

class GeminiModel(AIModel):
    """Adapter for calling Google's Gemini Messages API"""

    def __init__(
        self, model_name: str = 'gemini-1.5-pro',
        client_kwargs: Dict = None,
        **kwargs
    ):
        genai.configure(**(client_kwargs or {}))  # Initialize API configuration
        self.model_name = model_name
        self.kwargs = kwargs

    def convert(self, message: Message) -> Dict:
        """Convert a Message to the format needed for Gemini's Messages API"""
        return {
            "role": message.source,
            "content": message.text
        }

    def forward(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
        """Run a standard query to the Gemini Messages API"""
        # Combine default and override keyword arguments
        kwargs = {**self.kwargs, **kwarg_override}
        kwargs['max_tokens'] = kwargs.get('max_tokens', 100)

        # Convert messages for Gemini
        messages = self.convert_messages(prompt.aslist())
        
        # Call Gemini's chat API
        response = genai.chat(
            model=self.model_name,
            messages=messages,
            **kwargs
        )

        # Extract and return the response
        response_text = response['candidates'][0]['content']
        response_message = TextMessage(source="assistant", text=response_text)
        return AIResponse(message=response_message, source=response)

    def stream_forward(self, prompt: AIPrompt, **kwarg_override) -> Iterator[AIResponse]:
        """Stream response from Gemini's Messages API (synchronous)"""
        kwargs = {**self.kwargs, **kwarg_override}
        kwargs['max_tokens'] = kwargs.get('max_tokens', 100)

        messages = self.convert_messages(prompt.aslist())
        response_generator = genai.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
            **kwargs
        )

        for chunk in response_generator:
            response_text = chunk['content']
            response_message = TextMessage(source="assistant", text=response_text)
            yield AIResponse(message=response_message, source=chunk)

    async def async_forward(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
        """Run an asynchronous query to the Gemini Messages API"""
        kwargs = {**self.kwargs, **kwarg_override}
        kwargs['max_tokens'] = kwargs.get('max_tokens', 100)

        messages = self.convert_messages(prompt.aslist())

        # Asynchronously call Gemini's chat API
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: genai.chat(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
        )

        response_text = response['candidates'][0]['content']
        response_message = TextMessage(source="assistant", text=response_text)
        return AIResponse(message=response_message, source=response)

    async def async_stream_forward(self, prompt: AIPrompt, **kwarg_override) -> AsyncIterator[AIResponse]:
        """Run asynchronous streaming query to Gemini's Messages API"""
        kwargs = {**self.kwargs, **kwarg_override}
        kwargs['max_tokens'] = kwargs.get('max_tokens', 100)

        messages = self.convert_messages(prompt.aslist())

        # Asynchronously stream response
        loop = asyncio.get_event_loop()
        async for chunk in loop.run_in_executor(
            None, 
            lambda: genai.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                **kwargs
            )
        ):
            response_text = chunk['content']
            response_message = TextMessage(source="assistant", text=response_text)
            yield AIResponse(message=response_message, source=chunk)


class GeminiEmbeddingModel(AIModel):
    """Adapter for calling Google's Gemini Embedding API"""

    def __init__(self, api_key: str, model_name: str = 'gemini-1.5-embed'):
        # Initialize Google Generative AI client with the API key
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def convert(self, message: Message) -> str:
        """Convert a Message to the format needed for Gemini's Embedding API"""
        if isinstance(message, TextMessage):
            return message.text
        else:
            raise TypeError("Unsupported message type")

    def forward(self, prompt: AIPrompt, **kwarg_override) -> List[AIResponse]:
        """Run a query to the Gemini Embedding API"""
        # Extract text content from messages
        texts_to_embed = [self.convert(message) for message in prompt.aslist()]

        # Send request to Google Gemini Embedding API
        response = genai.embed(
            model=self.model_name,
            texts=texts_to_embed
        )

        # Generate AIResponse objects
        return [
            AIResponse(message=TextMessage(source="embedding_result", text=text), source={"embedding": embedding})
            for text, embedding in zip(texts_to_embed, response.embeddings)
        ]

    async def async_forward(self, prompt: AIPrompt, **kwarg_override) -> List[AIResponse]:
        """Run an asynchronous query to the Gemini Embedding API"""
        # Use asyncio to perform async embedding retrieval with a blocking method
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.forward(prompt, **kwarg_override)
        )
