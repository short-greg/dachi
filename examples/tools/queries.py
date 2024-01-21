import threading
from openai import OpenAI
from abc import abstractmethod

from typing import Any
from .comm import UI
from dachi.storage import Conv
from dachi.comm import Query, Request
from functools import partial


class LLMQuery(Query):

    @abstractmethod
    def prepare_response(self, request: Request):
        pass
    
    def exec_post(self, request: Request) -> Any:
        message = self.prepare_response(request)
        self.respond(request, message)

    def __call__(self, conv: Conv, asynchronous: bool=True):

        request = Request(contents=conv.as_messages())
        self.post(request, asynchronous)
        return request.response


class OpenAIQuery(LLMQuery):

    def __init__(self, temperature: float=0.0):
        """

        Args:
            store (DataStore): 
        """
        super().__init__()
        self.client = OpenAI()
        self.temperature = temperature

    def prepare_response(self, request: Request):
        return self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=request.contents, temperature=self.temperature
        ).choices[0].message.content


class UIQuery(Query):

    def __init__(self, ui_interface: UI):
        """

        Args:
            store (DataStore): 
        """
        super().__init__()
        self.ui_interface = ui_interface

    def exec_post(self, request: Request):
        self.ui_interface.request_message(partial(self.respond, request))
