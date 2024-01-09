# import typing
# import uuid
# from collections import OrderedDict
# from ._requests import Query, Signal, InterComm

# from ._storage import DataStore
# from ._base import Receiver

# # REMOVE
# CALLBACK = typing.Union[str, typing.Callable]
# NAME = typing.Union[str, typing.List[str]]


# class Server(object):
#     """Server manages and requests and data storage. There can be
#     a terminal for each server
#     """

#     def __init__(self):
#         """Create a server to manage requests and store data
#         """
#         self._central = DataStore()
#         self._shared: typing.Dict[str, DataStore] = {}
#         self._terminals = OrderedDict()
#         self._register: typing.Dict[str, 'Terminal'] = {}
#         self._queries: typing.Dict[str, Query] = {}
#         self._signals: typing.Dict[str, Signal] = {}
#         self._intercomm: InterComm = InterComm()

#     @property
#     def central(self) -> DataStore:
#         """
#         Returns:
#             DataStore: The globally shared data
#         """
#         return self._central

#     def shared(self, name: str) -> DataStore:
#         """
#         Returns:
#             DataStore: The globally shared data
#         """
#         return self._shared[name]

#     def add_shared(self, name: str) -> DataStore:
#         """
#         name: str Add a DataStore
#         """
#         self._shared[name] = DataStore()

#     @property
#     def signals(self) -> typing.Dict[str, Signal]:
#         """
#         Returns:
#             typing.Dict[str, Signal]: All of the Signals
#         """
#         return self._signals
    
#     @property
#     def queries(self) -> typing.Dict[str, Query]:
#         """
#         Returns:
#             typing.Dict[str, Query]: The queries the server has
#         """
#         return self._queries

#     def signal(self, name: str, contents, on_post: CALLBACK=None):
#         """Send a signal 

#         Args:
#             name (str): The name of the signal
#             contents: The contents of the signal
#             on_post (CALLBACK, optional): The callback when the signal is posted. Defaults to None.
#         """
#         self._signals[name].post(self.central, contents, on_post)
    
#     def query(self, name: str, contents, on_post: CALLBACK=None, on_response: CALLBACK=None):
#         """Send a query

#         Args:
#             name (str): The name of the query
#             contents (_type_): The contents of the query
#             on_post (CALLBACK, optional): The callback when the query is posted. Defaults to None.
#             on_response (CALLBACK, optional): The callback when the response is received. Defaults to None.
#         """
#         self._queries[name].post(self.central, contents, on_post, on_response)

#     @property
#     def intercomm(self) -> typing.Dict[str, 'DataStore']:
#         """Intercomm is used to have more exclusive communication between 

#         Returns:
#             typing.Dict[str, 'Terminal']: The dictionary of terminals
#         """
#         return self._intercomm

#     @property
#     def registered(self) -> typing.Dict[str, 'Terminal']:
#         """
#         Returns:
#             typing.Dict[str, 'Terminal']: The registered terminals
#         """
#         return self._register
    
#     def terminal(self) -> 'Terminal':
#         """Create a terminal from the server

#         Returns:
#             Terminal: The created terminal
#         """
#         terminal = Terminal(self)
#         self._register[terminal]
#         return terminal

#     def state_dict(self) -> typing.Dict:
#         """
#         Returns:
#             typing.Dict: The server converted to a state dict
#         """
#         receivers = {}
#         for k, v in self._receivers.items():
#             cur = {}
#             for callback, expires in v.items():
#                 if isinstance(callback, str):
#                     cur[callback] = expires
#             receivers[k] = cur

#         return {
#             'shared': {k: v.state_dict() for k, v in self._shared.items()},
#             'central': self._central.state_dict(),
#             'registered': {id: None for id, _ in self._register.items()},
#             'terminals': {id: terminal.state_dict() for id, terminal in self._terminals.items()},
#             'receivers': receivers
#         }
    
#     def load_state_dict(self, state_dict) -> 'Server':
#         """Load a state dict

#         Args:
#             state_dict (): The dictionary specifying the state

#         Returns:
#             Server: The updated Server
#         """
#         for k, v in state_dict['shared'].items():
#             self._shared[k].load_state_dict(v)
#         self._central.load_state_dict(state_dict['central'])
#         self._terminals = {
#             k: Terminal(self).load_state_dict(terminal_state) for k, terminal_state in state_dict['terminals']
#         }
#         self._register = {
#             k: None for k, _ in state_dict['registered'].items()
#         }
#         # Update this to account for 
#         self._receivers = {
#             id: receiver for id, receiver in state_dict['receivers'].items()
#         }
#         return self


# class Terminal(object):
#     """A Terminal connects a task to the server. It has its own storage.
#     """

#     def __init__(self, server: Server) -> None:
#         """Create the terminal to connect to the server

#         Args:
#             server (Server): the server to connect to
#         """
#         self._initialized = False
#         self._server = server
#         self._storage = DataStore()
#         self._id = str(uuid.uuid4())

#     @property
#     def initialized(self) -> bool:
#         """
#         Returns:
#             bool: If the terminal has been initialized
#         """
#         return self._initialized
    
#     def initialize(self):
#         """Set the terminal to initialized"""
#         self._initialized = True

#     @property
#     def storage(self) -> DataStore:
#         """
#         Returns:
#             DataStore: The storage associated with the terminal
#         """
#         return self._storage
    
#     @property
#     def central(self) -> DataStore:
#         """
#         Returns:
#             DataStore: The globally shared data
#         """
#         return self._server.central

#     def add_shared(self, name: str) -> typing.NoReturn:
#         """Adds shared storage to the server

#         Returns:
#             name: The name of the shared store to include in the 
#             server
#         """
#         self._server.add_shared(name)

#     def shared(self, name: str) -> DataStore:
#         """
#         Args:
#             name (str): The name of the shared store

#         Returns:
#             DataStore: The DataStore retrieved
#         """
#         return self._server.shared(name)

#     @property
#     def server(self) -> Server:
#         """
#         Returns:
#             Server: The server for the terminal
#         """
#         return self._server
    
#     def reset(self):
#         self._initialized = False
#         self._storage.reset()

#     def child(self, receiver: Receiver) -> 'Terminal':

#         return self.server.register(receiver)

#     def state_dict(self) -> typing.Dict:
        
#         return {
#             'initialized': self._initialized,
#             'storage': self._storage.state_dict()
#         }
    
#     def load_state_dict(self, state_dict, server: Server=None):

#         server = server or self._server
#         self._storage = DataStore().load_state_dict(state_dict['storage'])
#         self._initialized = state_dict['initialized']
#         return self



# def get_parent_dict(d: typing.Dict, name: NAME) -> typing.Tuple[typing.Dict, str]:
#     """Retrieve the parent dict for the item specified by name

#     Args:
#         d (typing.Dict): The base dictionary
#         name (NAME): the name of the item to retrieve. Can be list of names or a single name

#     Returns:
#         typing.Tuple[typing.Dict, str]: The parent dict
#     """
#     cur = d
#     if isinstance(name, typing.List):
#         for name_i in name[:-1]:
#             cur = cur[name_i]
#         name = name_i
#     return cur, name


# class Ref(object):
#     """A Ref is used to provide a way to access date in a 
#     storage. 
#     """

#     CENTRAL = 'CENTRAL'
#     LOCAL = 'LOCAL'

#     def __init__(self, name: str, store: str='CENTRAL', key: str=None, sub_name: str=None):
#         """
#         Args:
#             name (str): The name of the key
#             key (str, optional): The key to retrieve. Defaults to uuid.
#         """
#         self._key = key or str(uuid.uuid4())
#         self._name = name
#         self._store = store
#         self._sub_name = sub_name

#     def get_store(self, terminal: Terminal) -> DataStore:

#         if self._store == 'CENTRAL':
#             return terminal.central
#         if self._store == 'LOCAL':
#             return terminal.storage
#         return terminal.shared[self._store]

#     def get(self, terminal: Terminal, default) -> typing.Any:
#         """Get A value from the store

#         Args:
#             terminal (Terminal): The terminal to retrieve from
#             default: The value to return if not in store

#         Returns:
#             typing.Any: The value retrieved
#         """
#         base = self.get_store(terminal).get(self._key, default)
#         if self._sub_name is None:
#             return base
#         return base.get(self._sub_name)
    
#     def get_or_set(self, terminal: Terminal, value):
#         """Get or set the value

#         Args:
#             store (DataStore): The store to get or set from
#             value: Value to set if not set
#         """
#         store = self.get_store(terminal)
#         if self._sub_name is None:
#             return store.get_or_set(self._key, value)
#         data = store.get(self._key)
#         return data.get_or_set(self._sub_name, value)
    
#     def set(self, terminal: Terminal, value):
#         """Set the value indexed by the ref

#         Args:
#             terminal (Terminal): The terminal to set to
#             value : The value to set
#         """
#         store = self.get_store(terminal)
#         if self._sub_name is None:
#             store[self._key] = value
#         else:
#             data = store[self._key]
#             data.set(self._sub_name, value)
    
#     def load_state_dict(self, state_dict):
#         """Retrieve the definition for ref 

#         Args:
#             state_dict:  The state dict specified  
#         """
#         self._name = state_dict['name']
#         self._key = state_dict['key']
#         self._store = state_dict['terminal']

#     def state_dict(self) -> typing.Dict:
#         """
#         Returns:
#             typing.Dict: The state dict for the Ref
#         """
#         return {
#             'name': self._name,
#             'key': self._key,
#             'terminal': self._store
#         }


# def refer(names: typing.List[str], store: str='CENTRAL') -> typing.Tuple[Ref]:
#     """Create several refs

#     Args:
#         names (typing.List[str]): The names of the refs
#         terminal (terminal): str

#     Returns:
#         typing.Tuple[Ref]: Several refs
#     """
#     return tuple(
#         Ref(name, store) for name in names
#     )


# def sub_refer(name: str, sub_names: typing.List[str], store: str='CENTRAL') -> typing.Tuple[Ref]:    
#     """Create several refs that refer to values in an object

#     Args:
#         name (str):  The name of the main object to retrieve
#         sub_names (typing.List[str]): _description_
#         store (str, optional): The store to retrieve. Defaults to 'CENTRAL'.

#     Returns:
#         typing.Tuple[Ref]: The references
#     """
#     return tuple(
#         Ref(name, store, sub_name=sub_name) for sub_name in sub_names
#     )
