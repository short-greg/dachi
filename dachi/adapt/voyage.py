from typing import Dict
import pkg_resources
import voyageai
from typing import Dict, List
import asyncio


from .._core import Message, AIPrompt
from .._core import AIModel, AIResponse, TextMessage

from typing import Dict, List

# TODO: add utility for this
required = {'voyageai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed


import voyageai
from typing import List, Dict

class VoyageAIEmbeddingModel(AIModel):
    """Adapter for calling VoyageAI's Embedding API"""

    def __init__(
        self, model_name: str = 'voyage-2', 
        client_args: Dict=None
    ):
        # Initialize VoyageAI client with the API key
        self.client_args = client_args or {}
        self.model_name = model_name

    def convert(self, message: Message) -> str:
        """Convert a Message to the format needed for VoyageAI's Embedding API"""
        if isinstance(message, TextMessage):
            return message.text
        else:
            raise TypeError("Unsupported message type")

    def forward(
        self, prompt: AIPrompt, **kwarg_override
    ) -> List[AIResponse]:
        """Run a query to the VoyageAI Embedding API"""
        # Extract text content from messages

        texts_to_embed = [
            self.convert(message) for message in prompt.aslist()
        ]

        # Send request to VoyageAI Embedding API
        client = voyageai.Client(**self.client_args)
        response = client.embed(
            texts=texts_to_embed,
            model=self.model_name,
            input_type=kwarg_override.get(
                "input_type", "document"
            )  # Can be 'document' or 'query'
        )

        # Generate AIResponse objects
        return [
            AIResponse(message=TextMessage(source="embedding_result", text=text), source={"embedding": embedding})
            for text, embedding in zip(texts_to_embed, response.embeddings)
        ]

    async def async_forward(self, prompt: AIPrompt, **kwarg_override) -> List[AIResponse]:
        """Run an asynchronous query to the VoyageAI Embedding API"""
        # Use asyncio to perform async embedding retrieval with a blocking method
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.forward(prompt, **kwarg_override)
        )
