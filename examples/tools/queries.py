import threading
from openai import OpenAI

from typing import Any
from .comm import UI
from dachi.storage import Conv
from dachi.comm import Query, Request
from functools import partial


class LLMQuery(Query):

    def __init__(self, termperature=0.0):
        """

        Args:
            store (DataStore): 
        """
        super().__init__()
        self.client = OpenAI()
        self.temperature = termperature

    def prepare_response(self, request: Request):
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=request.contents.as_messages(), temperature=self.temperature
        )
        message = response.choices[0].message.content
        self.respond(request, message)
    
    def prepare_post(self, request: Request) -> Any:
        thread = threading.Thread(target=self.prepare_response, args=[request])
        thread.start()

    def __call__(self, conv: Conv):

        request = Request(contents=conv)
        self.prepare_response(request)
        return request.response.choices[0].message.content


class UIQuery(Query):

    def __init__(self, ui_interface: UI):
        """

        Args:
            store (DataStore): 
        """
        super().__init__()
        self.ui_interface = ui_interface

    def prepare_post(self, request: Request):
        thread = threading.Thread(target=self.ui_interface.request_message, args=[partial(self.respond, request)])
        thread.start()
