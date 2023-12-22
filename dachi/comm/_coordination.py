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



class Server(object):

    def __init__(self):
        """Create a server to manage requests and store data
        """
        self._shared = DataStore()
        self._terminals = OrderedDict()
        self._register: typing.Dict[str, 'Terminal'] = {}
        self._queries: typing.Dict[str, Query] = {}
        self._signals: typing.Dict[str, Signal] = {}
        self._intercomm: InterComm = InterComm()

    @property
    def shared(self) -> DataStore:
        """
        Returns:
            DataStore: The globally shared data
        """
        return self._shared

    @property
    def signals(self) -> typing.Dict[str, Signal]:
        """
        Returns:
            typing.Dict[str, Signal]: All of the Signals
        """
        return self._signals
    
    @property
    def queries(self) -> typing.Dict[str, Query]:
        """
        Returns:
            typing.Dict[str, Query]: The queries the server has
        """
        return self._queries

    def signal(self, name: str, contents, on_post: CALLBACK=None):
        """Send a signal 

        Args:
            name (str): The name of the signal
            contents: The contents of the signal
            on_post (CALLBACK, optional): The callback when the signal is posted. Defaults to None.
        """
        self._signals[name].post(self.shared, contents, on_post)
    
    def query(self, name: str, contents, on_post: CALLBACK=None, on_response: CALLBACK=None):
        """Send a query

        Args:
            name (str): The name of the query
            contents (_type_): The contents of the query
            on_post (CALLBACK, optional): The callback when the query is posted. Defaults to None.
            on_response (CALLBACK, optional): The callback when the response is received. Defaults to None.
        """
        self._queries[name].post(self.shared, contents, on_post, on_response)

    @property
    def intercomm(self) -> typing.Dict[str, 'DataStore']:
        """Intercomm is used to have more exclusive communication between 

        Returns:
            typing.Dict[str, 'Terminal']: The dictionary of terminals
        """
        return self._intercomm

    def register(self, terminal: 'Terminal'):
        """
        Returns:
            typing.Dict[str, 'Terminal']: The registered terminals
        """
        # TODO: IMPLEMENT

        return self._register

    @property
    def registered(self) -> typing.Dict[str, 'Terminal']:
        """
        Returns:
            typing.Dict[str, 'Terminal']: The registered terminals
        """
        return self._register

    def state_dict(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: The server converted to a state dict
        """
        receivers = {}
        for k, v in self._receivers.items():
            cur = {}
            for callback, expires in v.items():
                if isinstance(callback, str):
                    cur[callback] = expires
            receivers[k] = cur

        return {
            'shared': self._shared.state_dict(),
            'registered': {id: None for id, _ in self._register.items()},
            'terminals': {id: terminal.state_dict() for id, terminal in self._terminals.items()},
            'receivers': receivers
        }
    
    def load_state_dict(self, state_dict) -> 'Server':
        """Load a state dict

        Args:
            state_dict (): The dictionary specifying the state

        Returns:
            Server: The updated Server
        """
        self._shared.load_state_dict(state_dict['shared'])
        self._terminals = {
            k: Terminal(self).load_state_dict(terminal_state) for k, terminal_state in state_dict['terminals']
        }
        self._register = {
            k: None for k, _ in state_dict['registered'].items()
        }
        # Update this to account for 
        self._receivers = {
            id: receiver for id, receiver in state_dict['receivers'].items()
        }
        return self


class Terminal(object):
    """A Terminal connects a task to the server. It has its own storage.
    """

    def __init__(self, server: Server) -> None:
        """Create the terminal to connect to the server

        Args:
            server (Server): the server to connect to
        """
        self._initialized = False
        self._server = server
        self._storage = DataStore()

    @property
    def initialized(self) -> bool:
        """
        Returns:
            bool: If the terminal has been initialized
        """
        return self._initialized
    
    def initialize(self):
        """Set the terminal to initialized"""
        self._initialized = True

    @property
    def storage(self) -> DataStore:
        """
        Returns:
            DataStore: The storage associated with the terminal
        """
        return self._storage
    
    @property
    def shared(self) -> DataStore:
        """
        Returns:
            DataStore: The globally shared data
        """
        return self._server.shared
    
    @property
    def server(self) -> Server:
        """
        Returns:
            Server: The server for the terminal
        """
        return self._server
    
    # @property
    # def parent(self) -> 'Terminal':
    #     return self._parent
    
    def reset(self):
        self._initialized = False
        self._storage.reset()

    def child(self, receiver: Receiver) -> 'Terminal':

        return self.server.register(receiver)

    def state_dict(self) -> typing.Dict:
        
        return {
            'initialized': self._initialized,
            'storage': self._storage.state_dict()
        }
    
    def load_state_dict(self, state_dict, server: Server=None):

        server = server or self._server
        self._storage = DataStore().load_state_dict(state_dict['storage'])
        self._initialized = state_dict['initialized']
        return self
