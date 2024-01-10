import typing
from abc import ABC, abstractmethod

from ._storage import DataStore
import uuid
from dataclasses import dataclass


CALLBACK = typing.Union[str, typing.Callable]


@dataclass
class Request(object):
    """Make a request to a query
    """

    contents: typing.Any = None
    on_post: typing.Callable = None
    on_response: typing.Callable = None

    def __post_init__(self):
        self.id = str(uuid.uuid4())
        self._posted: bool = False
        self._processing: bool = False
        self._response = None
        self._success = False

    def post(self):
        """Post the request
        """
        if self._posted:
            return

        self._posted = True
        self._processing = True
        if self.on_post is not None:
            self.on_post(self)

    @property
    def posted(self) -> bool:

        return self._posted
    
    def respond(self, response, success: bool=True):

        if self.processed:
            raise ValueError('Request has already been processed')

        self._processing = False
        self._response = response
        self._success = success
        if self.on_response is not None:
            self.on_response(self)

    @property
    def response(self) -> typing.Any:

        return self._response

    @property
    def processed(self) -> bool:

        return not self._processing and self._posted
    
    @property
    def success(self) -> bool:

        return self._success


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
    def prepare_response(self, request) -> typing.Any:
        pass

    @abstractmethod
    def prepare_post(self,  request):
        pass

    def post(
        self, request: Request
    ):
        """Make a query for information. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the query
            on_post (typing.Union[str, typing.Callable], optional): Callback when the query was posted. If it is a string, it sets that value to true on storage. Defaults to None.
            on_response (typing.Union[str, typing.Callable], optional): The  . If it is a string, it will set the storage specified by the key to the reponse Defaults to None.

        """
        self.prepare_post(request)
        request.post()
        for on_post in self._on_post:
            on_post(request)

    def respond(
        self, request: Request
    ):
        """Make a query for information. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the query
            on_post (typing.Union[str, typing.Callable], optional): Callback when the query was posted. If it is a string, it sets that value to true on storage. Defaults to None.
            on_response (typing.Union[str, typing.Callable], optional): The  . If it is a string, it will set the storage specified by the key to the reponse Defaults to None.

        """
        response = self.prepare_response(request)
        request.respond(response)
        for on_response in self._on_response:
            on_response(request)


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
        self, request: Request
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
