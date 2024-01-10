import threading
from openai import OpenAI

from typing import Any
from .comm import Query, UIInterface


class LLMQuery(Query):

    def __init__(self):
        """

        Args:
            store (DataStore): 
        """
        super().__init__()
        self.client = OpenAI()

    def prepare_response(self, contents):
        assert 'prompt' in contents
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=contents['prompt']
        )
        return response
    
    def prepare_post(self, contents) -> Any:
        thread = threading.Thread(self.prepare_response, args=[contents])
        thread.run()


class UIQuery(Query):

    def __init__(self, ui_interface: UIInterface):
        """

        Args:
            store (DataStore): 
        """
        super().__init__()
        self.ui_interface = ui_interface

    def prepare_response(self, contents) -> Any:
        """
        Args:
            contents: 

        Returns:
            Any: 
        """
        return contents

    def prepare_post(self, contents=None):
        thread = threading.Thread(self.ui_interface, args=[self.prepare_response])
        thread.run()
