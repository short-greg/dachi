import threading
from openai import OpenAI

from typing import Any
from dachi.comm import Query, DataStore
from dachi.comm._storage import DataStore
from .comm import IOHandler


class LLMQuery(Query):

    def __init__(self, store: DataStore):
        """

        Args:
            store (DataStore): 
        """
        super().__init__(store)
        self.client = OpenAI()

    def prepare_response(self, contents):
        assert 'prompt' in contents
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=contents['prompt']
        )
        return response
    
    def prepare_post(self, contents) -> Any:
        thread = threading.Thread(self.prepare_post, args=[contents])
        thread.run()


class UIQuery(Query):

    def __init__(self, ui_callback, store: DataStore):
        """

        Args:
            store (DataStore): 
        """
        super().__init__(store)
        # self.io_handler = io_handler
        self.ui_callback = ui_callback

    def prepare_response(self, contents) -> Any:
 
        return contents

    def prepare_post(self, contents):
        thread = threading.Thread(self.ui_callback, args=[contents, self.prepare_response])
        thread.run()
