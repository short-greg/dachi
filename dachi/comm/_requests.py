import typing
from abc import ABC, abstractmethod

from ._storage import DataStore
import uuid
from dataclasses import dataclass


CALLBACK = typing.Union[str, typing.Callable]


# def callback(cb, contents, store: DataStore):

#     if cb is None:
#         return
#     if isinstance(cb, str):
#         store[cb] = contents
#     else:
#         cb(contents)


@dataclass
class Request(object):

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
        on_post: typing.Union[str, typing.Callable]=None,
        on_response: typing.Union[str, typing.Callable]=None
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
        on_post: typing.Union[str, typing.Callable]=None,
        on_response: typing.Union[str, typing.Callable]=None
    ):
        """Remove a callback from the query

        Args:
            on_post (typing.Union[str, typing.Callable], optional): The on_post callback to remove. Defaults to None.
            on_response (typing.Union[str, typing.Callable], optional): The on_resposne callback to remove. Defaults to None.
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


class InterComm(object):

    def __init__(self):
        """Create a set of communication channels to allow for more 'private' communication
        between tasks
        """
        self._channels: typing.Dict[str, DataStore] = {}

    def __getitem__(self, name: str) -> DataStore:
        """
        Args:
            name (str): Name of the DataStore

        Raises:
            KeyError: If name is incorrect
        Returns:
            DataStore: the DataStore specified by name
        """
        if name not in self._channels:
            raise KeyError(f'Name by {name} does not exist in channels')

        return self._channels[name]
    
    def get_or_add(self, name: str) -> DataStore:
        """Get the DataStore specified by name or add it

        Args:
            name (str): The name of the DataStore

        Returns:
            DataStore: The DataStore specified by name
        """

        if name not in self._channels:
            self._channels[name] = DataStore()
        
        return self._channels[name]

    def add(self, name: str) -> 'InterComm':
        """Add DataStore specified by name
        Args:
            name (str): Name of the DataStore

        Raises:
            ValueError: If it cannot be added
        Returns:
            InterComm: self
        """
        if name in self._channels:
            raise ValueError('Channel with name {name} already exists')
        
        return self._channels[name]
    
    def get(self, name: str) -> DataStore:
        """
        Args:
            name (str): Name of the DataStore

        Returns:
            DataStore: the DataStore specified by name
        """
        return self._channels.get(name)
