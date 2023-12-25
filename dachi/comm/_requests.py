import typing
from collections import OrderedDict
from abc import ABC, abstractmethod

from ._storage import DataStore
from ._base import Receiver

CALLBACK = typing.Union[str, typing.Callable]


def callback(cb, contents, store: DataStore):

    if cb is None:
        return
    if isinstance(cb, str):
        store[cb] = contents
    else:
        cb(contents)


class Query(ABC):

    def __init__(self, store: 'DataStore'):

        self._on_post = []
        self._on_response = []
        self._store = store
    
    def register(
        self, 
        on_post: typing.Union[str, typing.Callable]=None,
        on_response: typing.Union[str, typing.Callable]=None
    ):
        if on_post is not None:
            self._on_post.append(on_post)
        if on_response is not None:
            self._on_response.append(on_response)

    def unregester(
        self, 
        on_post: typing.Union[str, typing.Callable]=None,
        on_response: typing.Union[str, typing.Callable]=None
    ):
        if on_post is not None:
            self._on_post.remove(on_post)
        if on_response is not None:
            self._on_response.remove(on_response)

    @abstractmethod
    def prepare_response(self, contents) -> typing.Any:
        pass

    @abstractmethod
    def prepare_post(self,  contents):
        pass

    def post(
        self, contents, 
        on_post: CALLBACK=None, 
        on_response: CALLBACK=None
    ):
        
        """Make a query for information. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the query
            on_post (typing.Union[str, typing.Callable], optional): Callback when the query was posted. If it is a string, it sets that value to true on storage. Defaults to None.
            on_response (typing.Union[str, typing.Callable], optional): The  . If it is a string, it will set the storage specified by the key to the reponse Defaults to None.

        """
        self.prepare_post(contents, on_response)
        callback(on_post, contents, self._store)
        for on_post in self._on_post:
            callback(on_post, contents, self._store)

    def respond(
        self, contents,
        on_response: CALLBACK=None
    ):
        """Make a query for information. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the query
            on_post (typing.Union[str, typing.Callable], optional): Callback when the query was posted. If it is a string, it sets that value to true on storage. Defaults to None.
            on_response (typing.Union[str, typing.Callable], optional): The  . If it is a string, it will set the storage specified by the key to the reponse Defaults to None.

        """
        response = self.prepare_response(contents)
        callback(on_response, response, self._store)
        for on_response in self._on_response:
            callback(on_response, contents, self._store)


class Signal(ABC):

    def __init__(self, store: DataStore):

        self._on_post = []
        self._store = store
    
    @abstractmethod
    def prepare_post(
        self, store: DataStore, contents, 
        on_post: CALLBACK=None, 
    ):
        """Sends a message. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the message
        """
        raise NotImplementedError

    def post(
        self, store: DataStore, contents, 
        on_post: CALLBACK=None, 
    ):
        """Sends a message. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the message
            on_post (typing.Union[str, typing.Callable], optional): Callback when the query was posted. If it is a string, it sets that value to true on storage. Defaults to None.

        """
        self._base_post(store, contents, on_post)
        callback(on_post, contents, self._store)
        for on_post in self._on_post:
            callback(on_post, contents, self._store)

    def register(
        self, 
        on_post: typing.Union[str, typing.Callable]
    ):
        self._on_post.append(on_post)

    def unregester(
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
