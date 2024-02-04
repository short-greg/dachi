from openai import OpenAI
from abc import abstractmethod, ABC
from typing import Any
from ._ui import UI
from ..storage import MessageLister
from functools import partial
import threading
import typing
from ._requests import CALLBACK, Request


class Query(ABC):
    """Base class for creating a message that will return a response
    after completion
    """

    def __init__(self):
        """Run a Query

        Args:
            store (DataStore): The store that the query uses
        """
        self._on_post = []
        self._on_response = []
        self._processing = {}
    
    def register(
        self, 
        on_post: typing.Callable=None,
        on_response: typing.Callable=None
    ):
        """Register a callback 

        Args:
            on_post (typing.Union[str, typing.Callable], optional): Callback for when posting. Defaults to None.
            on_response (typing.Union[str, typing.Callable], optional): Callback for when response. Defaults to None.
        """
        if on_post is not None:
            self._on_post.append(on_post)
        if on_response is not None:
            self._on_response.append(on_response)

    def unregister(
        self, 
        on_post: typing.Callable=None,
        on_response: typing.Callable=None
    ):
        """Remove a callback from the query

        Args:
            on_post (typing.Callable, optional): The on_post callback to remove. Defaults to None.
            on_response (typing.Callable, optional): The on_resposne callback to remove. Defaults to None.
        """
        if on_post is not None:
            self._on_post.remove(on_post)
        if on_response is not None:
            self._on_response.remove(on_response)

    @abstractmethod
    def exec_post(self,  request):
        pass

    def post(
        self, request: Request, asynchronous: bool=True
    ):
        """Make a query for information. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the query
            on_post (typing.Union[str, typing.Callable], optional): Callback when the query was posted. If it is a string, it sets that value to true on storage. Defaults to None.
            on_response (typing.Union[str, typing.Callable], optional): The  . If it is a string, it will set the storage specified by the key to the reponse Defaults to None.

        """
        for on_post in self._on_post:
            on_post(request)
        request.post()

        if asynchronous:
            thread = threading.Thread(target=self.exec_post, args=[request])
            thread.start()
        else:
            self.exec_post(request)

    def respond(
        self, request: Request, response
    ):
        """Make a query for information. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the query
            on_post (typing.Union[str, typing.Callable], optional): Callback when the query was posted. If it is a string, it sets that value to true on storage. Defaults to None.
            on_response (typing.Union[str, typing.Callable], optional): The  . If it is a string, it will set the storage specified by the key to the reponse Defaults to None.

        """
        request.respond(response)
        for on_response in self._on_response:
            on_response(request)


# TODO: Update SIGNAL

class Signal(ABC):
    """A message that does not respond to the user
    """

    def __init__(self):
        """
        """
        self._on_post = []
    
    @abstractmethod
    def prepare_post(
        self, request: Request,
        on_post: CALLBACK=None, 
    ):
        """Sends a message. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the message
        """
        raise NotImplementedError

    def post(
        self, request: Request, asynchronous: bool=False
    ):
        """Sends a message. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the message
        """
        self.prepare_post(request)

        request.post()
        for on_post in self._on_post:
            on_post(request)

    def register(
        self, 
        on_post: typing.Union[str, typing.Callable]
    ):
        self._on_post.append(on_post)

    def unregister(
        self, 
        on_post: typing.Union[str, typing.Callable]
    ):
        self._on_post.remove(on_post)


class LLMQuery(Query):

    @abstractmethod
    def prepare_response(self, request: Request):
        pass
    
    def exec_post(self, request: Request) -> Any:
        message = self.prepare_response(request)
        self.respond(request, message)

    def __call__(self, conv: MessageLister, asynchronous: bool=True):
        request = Request(contents=conv.as_messages().as_dict_list())
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
        # exepct content to be a list of messages
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
