from llama_index.core.llms.llm import LLMChatComponent
from typing import AsyncIterator, Iterator, List, Dict, Tuple
from .._core import Message, AIPrompt, AIResponse, TextMessage, AIModel


class LlamaIndexAIModel(AIModel):
    """Adapter for LlamaIndex's LLMChatComponent functionality."""

    def __init__(self, llm_component: LLMChatComponent, **kwargs):
        """
        Initialize the LlamaIndexAIModel.

        Args:
            llm_component (LLMChatComponent): An instance of LLMChatComponent.
            kwargs (dict): Additional configurations.
        """
        super().__init__()
        self.llm_component = llm_component
        self.kwargs = kwargs

    def convert(self, message: Message) -> Dict:
        """
        Convert the Message to the format expected by LLMChatComponent.

        Args:
            message (Message): The message to convert.

        Returns:
            Dict: The format required by LLMChatComponent.
        """
        return {"content": message["text"], "role": message.source}

    def forward(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
        """
        Process a single prompt using LLMChatComponent.

        Args:
            prompt (AIPrompt): The prompt to process.

        Returns:
            AIResponse: The response from the LLMChatComponent.
        """
        kwargs = {**self.kwargs, **kwarg_override}
        converted_prompt = self.convert_messages(prompt.aslist())

        # Handle errors here (e.g., validate input, check for exceptions)
        response = self.llm_component.chat(prompt=converted_prompt, **kwargs)
        text = response["content"]

        return AIResponse(
            message=TextMessage(role="assistant", text=text),
            raw_response=response,
            processed_output=prompt.reader().read(text),
        )

    def stream(self, prompt: AIPrompt, **kwarg_override) -> Iterator[Tuple[AIResponse, AIResponse]]:
        """
        Stream responses using LLMChatComponent.

        Args:
            prompt (AIPrompt): The prompt to process.

        Yields:
            Iterator[Tuple[AIResponse, AIResponse]]: Streamed responses.
        """
        kwargs = {**self.kwargs, **kwarg_override}
        converted_prompt = self.convert_messages(prompt.aslist())

        cur_message = ""
        reader = prompt.reader()

        for chunk in self.llm_component.stream_chat(prompt=converted_prompt, **kwargs):
            delta = chunk.get("delta", "")
            cur_message += delta

            yield (
                AIResponse(
                    message=TextMessage(role="assistant", text=cur_message),
                    raw_response=chunk,
                    processed_output=reader.read(cur_message),
                ),
                AIResponse(
                    message=TextMessage(role="assistant", text=delta),
                    raw_response=chunk,
                    processed_output=reader.read(delta),
                ),
            )

    async def aforward(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
        """
        Asynchronous processing of a single prompt.

        Args:
            prompt (AIPrompt): The prompt to process.

        Returns:
            AIResponse: The response from the LLMChatComponent.
        """
        kwargs = {**self.kwargs, **kwarg_override}
        converted_prompt = self.convert_messages(prompt.aslist())

        # Handle errors here (e.g., validate input, check for exceptions)
        response = await self.llm_component.async_chat(prompt=converted_prompt, **kwargs)
        text = response["content"]

        return AIResponse(
            message=TextMessage(role="assistant", text=text),
            raw_response=response,
            processed_output=prompt.reader().read(text),
        )

    async def astream(self, prompt: AIPrompt, **kwarg_override) -> AsyncIterator[Tuple[AIResponse, AIResponse]]:
        """
        Asynchronously stream responses using LLMChatComponent.

        Args:
            prompt (AIPrompt): The prompt to process.

        Yields:
            AsyncIterator[Tuple[AIResponse, AIResponse]]: Streamed responses.
        """
        kwargs = {**self.kwargs, **kwarg_override}
        converted_prompt = self.convert_messages(prompt.aslist())

        cur_message = ""
        reader = prompt.reader()

        async for chunk in self.llm_component.async_stream_chat(prompt=converted_prompt, **kwargs):
            delta = chunk.get("delta", "")
            cur_message += delta

            yield (
                AIResponse(
                    message=TextMessage(role="assistant", text=cur_message),
                    raw_response=chunk,
                    processed_output=reader.read(cur_message),
                ),
                AIResponse(
                    message=TextMessage(role="assistant", text=delta),
                    raw_response=chunk,
                    processed_output=reader.read(delta),
                ),
            )