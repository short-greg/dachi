from typing import AsyncIterator, Dict
import typing
import pkg_resources
from typing import Dict, List
from openai import OpenAI
from openai import AssistantEventHandler
from typing import Dict, AsyncIterator, Iterator

from .._core import Message, AIPrompt
from .._core import AIModel, AIResponse, Cue, TextMessage, render

from typing import Dict, List


# TODO: add utility for this
required = {'openai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed


class OpenAIAssistantsModel(AIModel):
    """A model that uses OpenAI's Assistants API."""

    def __init__(self, assistant_id: str, thread_id: str = None, client_kwargs: Dict = None, **kwargs):
        """
        Initialize the OpenAIAssistantsModel.

        Args:
            assistant_id (str): The ID of the assistant.
            thread_id (str, optional): Optional thread ID. If not provided, a new thread will be created.
            client_kwargs (Dict, optional): Optional parameters for the OpenAI client.
            **kwargs: Additional parameters for the Assistants API.
        """
        super().__init__()
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.client_kwargs = client_kwargs or {}
        self.kwargs = kwargs
        self.client = OpenAI(**self.client_kwargs)

    def _ensure_thread(self):
        """Ensure a thread is created if none exists."""
        if not self.thread_id:
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id

    def convert(self, message: Message) -> Dict:
        """Convert a Message to the format needed for the Assistants API."""
        return {
            "role": message.source,
            "content": message.text
        }

    def forward(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
        """
        Send a single message to the Assistants API and run the thread.
        """
        if len(prompt.aslist()) != 1:
            raise ValueError("Only one message can be sent per call to forward.")

        self._ensure_thread()
        message_data = self.convert(prompt.aslist()[0])

        # Add the message to the thread
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            **message_data
        )

        # Create a run and poll for the response
        kwargs = {**self.kwargs, **kwarg_override}
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            **kwargs
        )

        if run.status != "completed":
            raise RuntimeError(f"Run did not complete successfully: {run.status}")

        # Retrieve the assistant's message from the thread
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
        assistant_message = next(
            msg for msg in messages if msg["role"] == "assistant"
        )

        text = assistant_message["content"]
        return AIResponse(TextMessage("assistant", text), run, prompt.reader().read(text))

    def stream_forward(self, prompt: AIPrompt, **kwarg_override) -> Iterator[AIResponse]:
        """
        Stream a single message response from the Assistants API.
        """
        if len(prompt.aslist()) != 1:
            raise ValueError("Only one message can be sent per call to stream_forward.")

        self._ensure_thread()
        message_data = self.convert(prompt.aslist()[0])

        # Add the message to the thread
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            **message_data
        )

        # Stream the run
        kwargs = {**self.kwargs, **kwarg_override}
        with self.client.beta.threads.runs.stream(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            **kwargs
        ) as stream:
            cur_message = ""
            for event in stream.events:
                if event.type == "text_delta":
                    delta = event.delta
                    cur_message += delta.value
                    yield AIResponse(
                        TextMessage("assistant", cur_message), event, None
                    )

    async def async_forward(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
        """
        Send a single message to the Assistants API asynchronously and poll the thread.
        """
        if len(prompt.aslist()) != 1:
            raise ValueError("Only one message can be sent per call to async_forward.")

        self._ensure_thread()
        message_data = self.convert(prompt.aslist()[0])

        # Add the message to the thread
        await self.client.beta.threads.messages.acreate(
            thread_id=self.thread_id,
            **message_data
        )

        # Create a run and poll for the response
        kwargs = {**self.kwargs, **kwarg_override}
        run = await self.client.beta.threads.runs.acreate_and_poll(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            **kwargs
        )

        if run.status != "completed":
            raise RuntimeError(f"Run did not complete successfully: {run.status}")

        # Retrieve the assistant's message from the thread
        messages = await self.client.beta.threads.messages.alist(thread_id=self.thread_id)
        assistant_message = next(
            msg for msg in messages if msg["role"] == "assistant"
        )

        text = assistant_message["content"]
        return AIResponse(TextMessage("assistant", text), run, prompt.reader().read(text))

    async def async_stream_forward(self, prompt: AIPrompt, **kwarg_override) -> AsyncIterator[AIResponse]:
        """
        Stream a single message response from the Assistants API asynchronously.
        """
        if len(prompt.aslist()) != 1:
            raise ValueError("Only one message can be sent per call to async_stream_forward.")

        self._ensure_thread()
        message_data = self.convert(prompt.aslist()[0])

        # Add the message to the thread
        await self.client.beta.threads.messages.acreate(
            thread_id=self.thread_id,
            **message_data
        )

        # Stream the run
        kwargs = {**self.kwargs, **kwarg_override}
        async with self.client.beta.threads.runs.astream(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            **kwargs
        ) as stream:
            cur_message = ""
            async for event in stream.events:
                if event.type == "text_delta":
                    delta = event.delta
                    cur_message += delta.value
                    yield AIResponse(
                        TextMessage("assistant", cur_message), event, None
                    )


class OpenAIAssistantMessage(TextMessage):
    """A text message made to work with OpenAI. Especially makes
    using prompt more straightforward
    """
    def __init__(self, source: str, text: typing.Union[str, 'Cue'], thread_id: str):
        """Create a text message with a source

        Args:
            source (str): the source of the message
            text (typing.Union[str, Cue]): the content of the message

        """
        super().__init__(
            source=source,
            data={
                'text': text
            }
        )
    
    def prompt(self, assistant_model: str, thread_id: str=None, **kwarg_overrides) -> AIResponse:
        """prompt the model

        Args:
            model (typing.Union[AIModel, str]): The model to execute

        Returns:
            AIResponse: The response from the model
        """
        model = OpenAIAssistantsModel(
            assistant_model, thread_id, **kwarg_overrides
        )

        return super().prompt(model, **kwarg_overrides)
