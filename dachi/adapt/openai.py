from typing import AsyncIterator, Dict
import pkg_resources
import openai
import typing
from typing import Dict, List

from .._core import Message, AIPrompt
from .._core import AIModel, AIResponse, Cue, TextMessage, render

from typing import Dict, List


# TODO: add utility for this
required = {'openai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed


if len(missing) > 0:

    raise RuntimeError(f'To use this module openai must be installed.')


class OpenAIChatModel(AIModel):
    """A model that uses OpenAI's Chat API
    """

    def __init__(self, model: str, client_kwargs: typing.Dict=None, **kwargs) -> None:
        """Create an OpenAIChat model

        Args:
            model (str): The name of the model
        """
        super().__init__()
        self.client_kwargs = client_kwargs or {}
        self.model = model
        self.kwargs = kwargs

    def convert(self, message: Message) -> typing.Dict:
        """Convert the messages to a format to use for the OpenAI API

        Args:
            messages (typing.List[Message]): The messages

        Returns:
            typing.List[typing.Dict]: The input to the API
        """
        text = message['text']
        if isinstance(text, Cue):
            text = text.render()
        return {
            'role': message.source,
            'content': text
        }

    def forward(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
        """Execute the model

        Args:
            prompt (AIPrompt): The message to send the model

        Returns:
            AIResponse: the response from the model
        """
        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        client = openai.OpenAI(**self.client_kwargs)

        response = client.chat.completions.create(
            model=self.model,
            messages=self.convert_messages(prompt.aslist()),
            **kwargs
        )
        p = prompt.reader()
        text = response.choices[0].message.content
        message = TextMessage('assistant', text)
        
        val = p.read(text)
        
        return AIResponse(
            message,
            response,
            val
        )

    def stream_forward(
        self, prompt: AIPrompt, 
        **kwarg_override
    ) -> typing.Iterator[typing.Tuple[AIResponse, AIResponse]]:
        """Stream the model

        Args:
            prompt (AIPrompt): the model prmopt

        Yields:
            Iterator[typing.Iterator[typing.Tuple[AIResponse, AIResponse]]]: the responses from the model
        """
        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        
        client = openai.OpenAI(**self.client_kwargs)
        query = client.chat.completions.create(
            model=self.model,
            messages=self.convert_messages(prompt.aslist()),
            stream=True,
            **kwargs
        )
        cur_message = ''
        p = prompt.reader()
        for chunk in query:
            delta = chunk.choices[0].delta.content

            if delta is None:
                delta = ''
            
            cur_message = cur_message + delta
            
            message = TextMessage('assistant', cur_message)
            dx = TextMessage('assistant', delta)

            dx_val = p.read(delta)
            yield AIResponse(
                message, chunk, p.read(cur_message)
            ), AIResponse(
                dx, chunk, dx_val
            )
    
    async def async_forward(
        self, prompt: AIPrompt, 
        **kwarg_override
    ) -> typing.Tuple[str, typing.Dict]:
        """

        Args:
            prompt (AIPrompt): The 

        Returns:
            typing.Tuple[str, typing.Dict]: _description_
        """
        client = openai.AsyncOpenAI(**self.client_kwargs)

        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        p = prompt.reader()

        response = await client.chat.completions.create(
            model=self.model,
            messages=self.convert(prompt),
            **kwargs
        )
        text = response.choices[0].message.content
        return AIResponse(
            TextMessage('assistant', text),
            response,
            p.read(text)
        )

    async def async_stream_forward(self, prompt: AIPrompt, **kwarg_override) -> AsyncIterator[typing.Tuple[AIResponse, AIResponse]]:
        """Stream the model asyncrhonously

        Args:
            prompt (AIPrompt): The prompt for the model

        Yields:
            Iterator[AsyncIterator[typing.Tuple[AIResponse, AIResponse]]]: The response from the model
        """

        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        client = openai.AsyncOpenAI(**self.client_kwargs)
        query = await client.chat.completions.create(
            model=self.model,
            messages=self.convert(prompt.aslist()),
            stream=True,
            **kwargs
        )
        p = prompt.reader()

        cur_message = ''
        async for chunk in query:
            delta = chunk.choices[0].delta.content
            if delta is None:
                delta = ''
            cur_message = cur_message + delta

            yield AIResponse(
                TextMessage('assistant', cur_message),
                chunk,
                p.stream_read(cur_message),
            ), AIResponse(
                TextMessage('assistant', delta),
                chunk,
                None
            )


class OpenAIMessage(TextMessage):
    """A text message made to work with OpenAI. Especially makes
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
            model = OpenAIChatModel(model, **kwarg_overrides)

        return super().prompt(model, **kwarg_overrides)


class OpenEmbeddingModel(AIModel):
    """
    """

    def __init__(self, model: str, **kwargs) -> None:
        """Create an OpenEmbeddingModel

        Args:
            model (str): The name of the model
        """
        super().__init__()
        self.model = model
        self.kwargs = kwargs

    def convert(self, message: Message) -> typing.Dict:
        """Convert the messages to a format to use for the OpenAI API

        Args:
            messages (typing.List[Message]): The messages

        Returns:
            typing.List[typing.Dict]: The input to the API
        """
        return list(
            render(text_i) for text_i in message['texts']
        )

    def forward(self, prompt: AIPrompt, **kwarg_override) -> Dict:
        """Execute the Embedding model

        Args:
            prompt (AIPrompt): 

        Returns:
            Dict: 
        """

        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        client = openai.OpenAI()

        response = client.embeddings.create(
            model=self.model,
            input=self.convert(prompt),
            **kwargs
        )
        embeddings = [e for e in response.data]
        
        return AIResponse(
            embeddings,
            response,
            embeddings
        )

    def stream_forward(
        self, prompt: AIPrompt, 
        **kwarg_override
    ) -> typing.Iterator[typing.Tuple[AIResponse, AIResponse]]:

        response = self.forward(prompt, **kwarg_override)
        yield response, response
        
    async def async_forward(
        self, prompt: AIPrompt, 
        **kwarg_override
    ) -> typing.Tuple[str, typing.Dict]:
        client = openai.AsyncOpenAI()

        kwargs = {
            **self.kwargs,
            **kwarg_override
        }
        response = client.embeddings.create(
            model=self.model,
            input=self.convert(prompt),
            **kwargs
        )
        embeddings = [e for e in response.data]
        
        return AIResponse(
            embeddings,
            response,
            embeddings
        )

    async def async_stream_forward(self, prompt: AIPrompt, **kwarg_override) -> AsyncIterator[typing.Tuple[AIResponse, AIResponse]]:

        response = await self.async_forward(prompt, **kwarg_override)
        yield response, response


class OpenAIEmbeddingModel(AIModel):
    """Adapter for calling OpenAI's Embedding API"""

    def __init__(self, api_key: str, model_name: str = 'text-embedding-ada-002'):
        openai.api_key = api_key
        self.model_name = model_name

    def convert(self, message: Message) -> str:
        """Convert a Message to the format needed for OpenAI's Embedding API"""
        if isinstance(message, TextMessage):
            return message.text
        else:
            raise TypeError("Unsupported message type")

    def forward(self, prompt: AIPrompt, **kwarg_override) -> List[AIResponse]:
        """Run a query to the OpenAI Embedding API"""
        # Convert messages and extract text
        texts_to_embed = [self.convert(message) for message in prompt.aslist()]

        # Send request to OpenAI Embedding API
        response = openai.Embedding.create(
            model=self.model_name,
            input=texts_to_embed
        )

        # Generate AIResponse objects
        return [
            AIResponse(message=TextMessage(source="embedding_result", text=text), source={"embedding": data['embedding']})
            for text, data in zip(texts_to_embed, response['data'])
        ]

    async def async_forward(self, prompt: AIPrompt, **kwarg_override) -> List[AIResponse]:
        """Run an asynchronous query to the OpenAI Embedding API"""
        return self.forward(prompt, **kwarg_override)  # Use sync forward for simplicity
