from .openai import OpenAIChatModel
import typing
import os


class GeminiChatModel(OpenAIChatModel):
    """A model that uses the Gemini API through the OpenAI library"""

    def __init__(
        self, model: str="gemini-1.5-flash", 
        client_kwargs: typing.Dict = None, **kwargs
    ):
        """Create a GeminiChat model using OpenAI's library

        Args:
            model (str): The Gemini model name
            api_key (str): The API key for accessing the Gemini API
        """
        client_kwargs = client_kwargs or {}
        api_key = client_kwargs.get('api_key', os.environ.get('GOOGLE_API_KEY'))
        # Update client_kwargs to include the Gemini API base URL
        client_kwargs['api_key'] = api_key
        client_kwargs['base_url'] = "https://generativelanguage.googleapis.com/v1beta/"
        
        # Call the parent class constructor with Gemini settings
        super().__init__(model=model, client_kwargs=client_kwargs, **kwargs)
