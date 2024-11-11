import google.generativeai as genai
from abc import ABC
from typing import Dict, List
from .._core import Message, AIPrompt
from .._core import AIModel, AIResponse, TextMessage


class GeminiChatAdapter(AIModel):
    """Adapter for interfacing with Gemini's Chat API using Google's Python SDK"""

    def __init__(
        self, api_key: str, model_name: str = 'gemini-1.5-pro'
    ):
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def convert(self, message: Message) -> Dict:
        """Convert a Message to the format required by Gemini's Chat API"""
        if isinstance(message, TextMessage):
            return {
                "role": message.source,
                "content": message.text
            }
        else:
            raise TypeError("Unsupported message type")

    def forward(
        self, prompt: AIPrompt, **kwarg_override
    ) -> AIResponse:
        """Send a request to Gemini's Chat API and return the response"""
        messages = [{'role': m['role'], 'content': m['content']} for m in self.convert_messages(prompt.aslist())]
        response = genai.chat(
            model=self.model_name,
            messages=messages,
            max_tokens=kwarg_override.get('max_tokens', 100)
        )
        response_text = response['responses'][0]['message']['content']
        response_message = TextMessage(source="assistant", text=response_text)
        return AIResponse(message=response_message, source=response)


class GeminiEmbeddingAdapter(AIModel):
    """Adapter for interfacing with Gemini's Embedding API using Google's Python SDK"""

    def __init__(self, api_key: str, model_name: str = 'gemini-1.5-pro'):
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def convert(self, message: Message) -> str:
        """Extract text content from a Message for embedding"""
        if isinstance(message, TextMessage):
            return message.text
        else:
            raise TypeError("Unsupported message type")

    def forward(self, prompt: AIPrompt, **kwarg_override) -> List[AIResponse]:
        """Send a request to Gemini's Embedding API and return the embeddings"""
        texts_to_embed = [self.convert(message) for message in prompt.aslist()]
        response = genai.embeddings(
            model=self.model_name,
            input=texts_to_embed
        )
        
        responses = []
        for text, embedding_data in zip(texts_to_embed, response['embeddings']):
            response_message = TextMessage(source="embedding_result", text=text)
            responses.append(AIResponse(message=response_message, source={"embedding": embedding_data}))
        
        return responses
