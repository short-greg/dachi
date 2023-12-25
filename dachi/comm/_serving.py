import typing
from collections import OrderedDict
from ._requests import Query, Signal, InterComm

from ._storage import DataStore
from ._base import Receiver

CALLBACK = typing.Union[str, typing.Callable]


class Server(object):

    def __init__(self):
        """Create a server to manage requests and store data
        """
        self._central = DataStore()
        self._shared: typing.Dict[str, DataStore] = {}
        self._terminals = OrderedDict()
        self._register: typing.Dict[str, 'Terminal'] = {}
        self._queries: typing.Dict[str, Query] = {}
        self._signals: typing.Dict[str, Signal] = {}
        self._intercomm: InterComm = InterComm()

    @property
    def central(self) -> DataStore:
        """
        Returns:
            DataStore: The globally shared data
        """
        return self._central

    def shared(self, name: str) -> DataStore:
        """
        Returns:
            DataStore: The globally shared data
        """
        return self._shared[name]

    def add_shared(self, name: str) -> DataStore:
        """
        name: str Add a DataStore
        """
        self._shared[name] = DataStore()

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
        self._signals[name].post(self.central, contents, on_post)
    
    def query(self, name: str, contents, on_post: CALLBACK=None, on_response: CALLBACK=None):
        """Send a query

        Args:
            name (str): The name of the query
            contents (_type_): The contents of the query
            on_post (CALLBACK, optional): The callback when the query is posted. Defaults to None.
            on_response (CALLBACK, optional): The callback when the response is received. Defaults to None.
        """
        self._queries[name].post(self.central, contents, on_post, on_response)

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
            'shared': {k: v.state_dict() for k, v in self._shared.items()},
            'central': self._central.state_dict(),
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
        for k, v in state_dict['shared'].items():
            self._shared[k].load_state_dict(v)
        self._central.load_state_dict(state_dict['central'])
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
    def central(self) -> DataStore:
        """
        Returns:
            DataStore: The globally shared data
        """
        return self._server.central

    def add_shared(self, name: str) -> typing.NoReturn:
        """Adds shared storage to the server

        Returns:
            name: The name of the shared store to include in the 
            server
        """
        self._server.add_shared(name)

    def shared(self, name: str) -> DataStore:
        """
        Args:
            name (str): The name of the shared store

        Returns:
            DataStore: The DataStore retrieved
        """
        return self._server.shared(name)

    @property
    def server(self) -> Server:
        """
        Returns:
            Server: The server for the terminal
        """
        return self._server
    
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
