from enum import Enum
import typing
from dataclasses import dataclass
from collections import OrderedDict
from abc import ABC, abstractmethod

from ._storage import DataStore
from ._base import Receiver


CALLBACK = typing.Union[str, typing.Callable]


def callback(cb, contents, store: DataStore):

    if isinstance(cb, str):
        store[cb] = contents
    else:
        cb(contents)


class Query(ABC):

    def __init__(self, server: 'Server'):

        self._on_post = []
        self._on_response = []
        self._server = server
        self._base_respond = self.respond
        self._base_post = self.post
        self.respond = self._respond
        self.post = self._post
    
    def _respond(
        self, store: DataStore, contents,
        on_response: CALLBACK=None):
        
        self._base_respond(store, contents, on_response)
        for on_response in self._on_response:
            callback(on_response, contents, store)

    def _post(
        self, store: DataStore, contents, 
        on_post: CALLBACK=None, 
        on_response: CALLBACK=None
    ):
        
        self._base_post(store, contents, on_response)
        for on_post in self._on_post:
            callback(on_post, contents, store)

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
    def respond(
        self, store: DataStore, contents,
        on_response: CALLBACK=None
    ):
        raise NotImplementedError

    @abstractmethod
    def post(
        self, store: DataStore, contents, 
        on_post: CALLBACK=None, 
        on_response: CALLBACK=None
    ):
        """Make a query for information. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the query
            on_post (typing.Union[str, typing.Callable], optional): Callback when the query was posted. If it is a string, it sets that value to true on storage. Defaults to None.
            on_response (typing.Union[str, typing.Callable], optional): The  . If it is a string, it will set the storage specified by the key to the reponse Defaults to None.

        """
        raise NotImplementedError


class Signal(ABC):

    def __init__(self):

        self._on_post = []
        self._base_post = self.post
        self.post = self._post
    
    def _post(
        self, store: DataStore, contents, 
        on_post: CALLBACK=None, 
        on_response: CALLBACK=None
    ):
        
        self._base_post(store, contents, on_response)
        for on_post in self._on_post:
            callback(on_post, contents, store)

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
    
    @abstractmethod
    def post(
        self, store: DataStore, contents, on_post: CALLBACK=None
    ):
        """Sends a message. If the content cannot be processed it raises an ValueError

        Args:
            contents: The contents of the message
            on_post (typing.Union[str, typing.Callable], optional): Callback when the query was posted. If it is a string, it sets that value to true on storage. Defaults to None.

        """
        raise NotImplementedError


class Server(object):

    def __init__(self):
        """Create a server to manage requests and store data
        """
        self._shared = DataStore()
        self._terminals = OrderedDict()
        self._register = {}
        self._queries: typing.Dict[str, Query] = {}
        self._signals: typing.Dict[str, Signal] = {}
        # self._responses = ResponseHandler()
        # self._requests = RequestHandler()
        # self._receivers: typing.Dict[typing.Tuple[SignalType, str], typing.Dict[typing.Union[str, typing.Callable], bool]] = {}

    @property
    def shared(self) -> DataStore:
        return self._shared

    @property
    def signals(self) -> typing.Dict[str, Signal]:
        return self._signals
    
    @property
    def queries(self) -> typing.Dict[str, Query]:
        return self._queries

    def signal(self, name: str, contents, on_post: CALLBACK=None):

        self._signals[name].post(self.storage, contents, on_post)
    
    def query(self, name: str, contents, on_post: CALLBACK=None, on_response: CALLBACK=None):

        self._queries[name].post(self.storage, contents, on_post, on_response)

    def state_dict(self) -> typing.Dict:

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

    # def receive(self, signal_type: SignalType, name: str, receiver: typing.Union['Receiver', typing.Callable], expires: bool=True):

    #     is_receiver = isinstance(receiver, Receiver)
    #     if is_receiver and receiver.id not in self._register:
    #         raise ValueError(f'No task with id of {receiver.id} in the register')
    #     if (signal_type, name) not in self._receivers:
    #         self._receivers[(signal_type, name)] = {}
        
    #     if is_receiver:
    #         self._receivers[(signal_type, name)][receiver.id] = expires
    #     else:
    #         self._receivers[(signal_type, name)][receiver] = expires

    # def cancel_receive(self, signal_type: SignalType, name: str, receiver: typing.Union['Receiver', typing.Callable]):

    #     receiver = self._receivers.get((signal_type, name))
    #     if receiver is None:
    #         raise ValueError(f'No receiver for {signal_type} and {name}')

    #     del receiver[receiver.id]
    #     if len(receiver) == 0:
    #         del self._receivers[signal_type, name]

    # def has_receiver(self, signal_type: SignalType, name: str, receiver: 'Receiver'):

    #     receiver = self._receivers.get((signal_type, name))
    #     if receiver is None:
    #         return False
        
    #     return receiver.id in receiver

    # def receives_signal(self, signal: typing.Union[Signal, typing.Tuple[SignalType, str]]) -> bool:
    #     """Check if it receives a message

    #     Args:
    #         message (typing.Union[Message, typing.Tuple[MessageType, str]]): _description_

    #     Returns:
    #         bool: If it receives a message
    #     """
    #     if isinstance(signal, Signal):
    #         message_type, name = signal.message_type, signal.name
    #     else:
    #         message_type, name = signal
        
    #     return (message_type, name) in self._receivers

    # @property
    # def reponses(self) -> ResponseHandler:
    #     return self._responses
    
    # @property
    # def requests(self) -> RequestHandler:
    #     return self._requests
    
    # def send(self, signal: Signal):
    #     """Send a signal to the server
    #     Args:
    #         signal (Signal): The message to send to the server
    #     """
    #     receivers = self._receivers.get((signal.message_type, signal.name))
    #     if receivers is None:
    #         return
        
    #     updated = {}
    #     for task, expires in receivers.items():
    #         if isinstance(task, str):
    #             self._register[task].receive(signal)
    #         else:
    #             task(signal)
    #         if not expires:
    #             updated[task.id] = False
    #     self._receivers[(signal.message_type, signal.name)] = updated

    # def register(self, receiver: 'Receiver', overwrite: bool=False) -> 'Terminal':
    #     """Register a receiver with the server

    #     Args:
    #         receiver (receiver): The receiver to register
    #         overwrite (bool, optional): Overwrite the receiver if it already exist. Defaults to False.

    #     Raises:
    #         ValueError: If overwrite is false and not in reg

    #     Returns:
    #         Terminal: The terminal for the receiver
    #     """
    #     in_register = self._register.get(receiver.id)
    #     if in_register is None:
    #         self._register[receiver.id] = receiver
    #         terminal = Terminal(self)
    #         self._terminals[receiver.id] = terminal
    #     elif in_register and overwrite:
    #         self._register[receiver.id] = receiver
    #         terminal = Terminal(self)
    #         self._terminals[receiver.id] = terminal
    #     elif in_register:
    #         raise ValueError(f'The task in register {in_register} is different from this task')
    #     elif receiver is not in_register:
    #         raise ValueError(f'The task in register {in_register} is different from this task')
    #     return self._terminals[receiver.id]
    

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
    
    # @property
    # def responses(self) -> ResponseHandler:
    #     """
    #     Returns:
    #         ResponseHandler: The response handler for the server
    #     """
    #     return self._server.reponses
    
    # @property
    # def requests(self) -> RequestHandler:
    #     """
    #     Returns:
    #         RequestHandler: The request handler for the server
    #     """
    #     return self._server.requests
    
    @property
    def shared(self) -> DataStore:
        return self._server.shared
    
    @property
    def server(self) -> Server:
        return self._server
    
    @property
    def parent(self) -> 'Terminal':
        return self._parent
    
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



# @dataclass
# class Request(object):
#     """Create a request for another task to process
#     """
#     name: str
#     receiver: str
#     contents: typing.Any

#     def respond(self, sender: str, contents) -> 'Response':
#         """Respond to the request

#         Args:
#             sender (str): The sender of the response
#             contents: The contents of the response

#         Returns:
#             Response: The response
#         """
#         return Response(self, sender, contents)


# @dataclass
# class Response(object):
#     """Create a response to a request that has been processed
#     """
#     request: Request
#     sender: str
#     contents: typing.Any


# class RequestHandler(object):
#     """Manage the requests to send to other tasks
#     """

#     def __init__(self):
#         """Create a handler to manage the requests to send to other tasks
#         """
#         self._requests: typing.List[Request] = []
    
#     def add(self, request: Request):
#         """Add another request to the handler

#         Args:
#             request (Request): The request to add
#         """
#         self._requests.append(request)

#     def find(self, receiver: str=None, request_name: str=None) -> int:
#         """Find a request in the handler

#         Args:
#             receiver (str, optional): The receiver to find by. Defaults to None.
#             request_name (str, optional): The name of the request. Defaults to None.

#         Returns:
#             int: The position of the request
#         """
#         for i, request in enumerate(self._requests):

#             if (
#                 request_name is not None and request.name == request_name
#             ) and (
#                 receiver is not None and request.receiver == receiver
#             ):
#                 return i
#             elif request_name is None and  receiver is not None and request.receiver == receiver:
#                 return i
#             elif receiver is None and request_name is not None and request.name == request_name:
#                 return i
#         return None

#     def pop(self, receiver: str=None, request_name: str=None):
#         """Get the response and pop it

#         Args:
#             receiver (str, optional): The name of the receiver. Defaults to None.
#             request_name (str, optional): The naem of the request. Defaults to None.

#         Returns:
#             Response: The response if found else None
#         """
#         index = self.find(receiver, request_name)
#         if index is not None:
#             return self._requests.pop(index)
#         return None
    
#     def get(self, receiver: str=None, request_name: str=None):
#         """Get the response

#         Args:
#             receiver (str, optional): The name of the receiver. Defaults to None.
#             request_name (str, optional): The naem of the request. Defaults to None.

#         Returns:
#             Response: The response if found else None
#         """
#         index = self.find(receiver, request_name)
#         if index is not None:
#             return self._responses[index]
#         return None

#     def has_request(self, receiver: str=None, request_name: str=None) -> bool:
#         """
#         Args:
#             receiver (str, optional): The name of the receiver. Defaults to None.
#             request_name (str, optional): The naem of the request. Defaults to None.

#         Returns:
#             bool: if the quest is found
#         """
#         return self.find(receiver, request_name) is not None


# class ResponseHandler(object):
#     """Manage the responses to requests
#     """

#     def __init__(self):

#         self._responses: typing.List[Response] = []
    
#     def add(self, response: Response):

#         self._responses.append(response)

#     def find(self, receiver: str=None, request_name: str=None) -> int:
#         """Find the response if it exists

#         Args:
#             receiver (str, optional): The name of the receiver. Defaults to None.
#             request_name (str, optional): The naem of the request. Defaults to None.

#         Returns:
#             int: _description_
#         """
#         for i, response in enumerate(self._responses):

#             request = response.request
#             if (
#                 request_name is not None and response.name == request_name
#             ) and (
#                 receiver is not None and request.receiver == receiver
#             ):
#                 return i
#             elif request_name is None and  receiver is not None and request.receiver == receiver:
#                 return i
#             elif receiver is None and request_name is not None and request.name == request_name:
#                 return i
#         return None

#     def pop(self, receiver: str=None, request_name: str=None):
#         """Get the response and pop it

#         Args:
#             receiver (str, optional): The name of the receiver. Defaults to None.
#             request_name (str, optional): The naem of the request. Defaults to None.

#         Returns:
#             Response: The response if found else None
#         """
#         index = self.find(receiver, request_name)
#         if index is not None:
#             return self._requests.pop(index)
#         return None

#     def get(self, receiver: str=None, request_name: str=None) -> Response:
#         """
#         Args:
#             receiver (str, optional): The name of the receiver. Defaults to None.
#             request_name (str, optional): The naem of the request. Defaults to None.

#         Returns:
#             Response: The response if found else None
#         """
#         index = self.find(receiver, request_name)
#         if index is not None:
#             return self._responses[index]
#         return None

#     def has_response(self, receiver: str=None, request_name: str=None) -> bool:
#         """
#         Args:
#             receiver (str, optional): The name of the receiver. Defaults to None.
#             request_name (str, optional): The naem of the request. Defaults to None.

#         Returns:
#             bool: if the response is found
#         """
#         return self.find(receiver, request_name) is not None



# TODO: Decide whether to keep this

# class SignalType(Enum):

#     INPUT = 'input'
#     OUTPUT = 'output'
#     WAITING = 'waiting'
#     CUSTOM = 'custom'


# @dataclass
# class Signal(object):

#     message_type: SignalType
#     name: str
#     data: typing.Dict = None

