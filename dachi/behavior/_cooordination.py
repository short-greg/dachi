from enum import Enum
import typing
from dataclasses import dataclass
from collections import OrderedDict

from ._base import Behavior
from ._storage import DataStore


class MessageType(Enum):

    INPUT = 'input'
    OUTPUT = 'output'
    WAITING = 'waiting'
    CUSTOM = 'custom'


@dataclass
class Message(object):

    message_type: MessageType
    name: str
    data: typing.Dict = None


@dataclass
class Request(object):

    name: str
    receiver: str
    contents: typing.Any

    def respond(self, contents) -> 'Response':

        return Response(self, contents)


@dataclass
class Response(object):

    request: Request
    contents: typing.Any


class RequestHandler(object):
    """Manage the requests to send to other tasks
    """

    def __init__(self):
        """Create a handler to manage the requests to send to other tasks
        """
        self._requests: typing.List[Request] = []
    
    def add(self, request: Request):
        """Add another request to the handler

        Args:
            request (Request): The request to add
        """
        self._requests.append(request)

    def find(self, receiver: str=None, request_name: str=None) -> int:
        """Find a request in the handler

        Args:
            receiver (str, optional): The receiver to find by. Defaults to None.
            request_name (str, optional): The name of the request. Defaults to None.

        Returns:
            int: The position of the request
        """
        for i, request in enumerate(self._requests):

            if (
                request_name is not None and request.name == request_name
            ) and (
                receiver is not None and request.receiver == receiver
            ):
                return i
            elif request_name is None and  receiver is not None and request.receiver == receiver:
                return i
            elif receiver is None and request_name is not None and request.name == request_name:
                return i
        return None

    def pop(self, receiver: str=None, request_name: str=None):
        """Get the response and pop it

        Args:
            receiver (str, optional): The name of the receiver. Defaults to None.
            request_name (str, optional): The naem of the request. Defaults to None.

        Returns:
            Response: The response if found else None
        """
        index = self.find(receiver, request_name)
        if index is not None:
            return self._requests.pop(index)
        return None
    
    def get(self, receiver: str=None, request_name: str=None):
        """Get the response

        Args:
            receiver (str, optional): The name of the receiver. Defaults to None.
            request_name (str, optional): The naem of the request. Defaults to None.

        Returns:
            Response: The response if found else None
        """
        index = self.find(receiver, request_name)
        if index is not None:
            return self._responses[index]
        return None

    def has_request(self, receiver: str=None, request_name: str=None) -> bool:
        """
        Args:
            receiver (str, optional): The name of the receiver. Defaults to None.
            request_name (str, optional): The naem of the request. Defaults to None.

        Returns:
            bool: if the quest is found
        """
        return self.find(receiver, request_name) is not None


class ResponseHandler(object):
    """Manage the responses to requests
    """

    def __init__(self):

        self._responses: typing.List[Response] = []
    
    def add(self, response: Response):

        self._responses.append(response)

    def find(self, receiver: str=None, request_name: str=None) -> int:
        """Find the response if it exists

        Args:
            receiver (str, optional): The name of the receiver. Defaults to None.
            request_name (str, optional): The naem of the request. Defaults to None.

        Returns:
            int: _description_
        """
        for i, response in enumerate(self._responses):

            request = response.request
            if (
                request_name is not None and response.name == request_name
            ) and (
                receiver is not None and request.receiver == receiver
            ):
                return i
            elif request_name is None and  receiver is not None and request.receiver == receiver:
                return i
            elif receiver is None and request_name is not None and request.name == request_name:
                return i
        return None

    def pop(self, receiver: str=None, request_name: str=None):
        """Get the response and pop it

        Args:
            receiver (str, optional): The name of the receiver. Defaults to None.
            request_name (str, optional): The naem of the request. Defaults to None.

        Returns:
            Response: The response if found else None
        """
        index = self.find(receiver, request_name)
        if index is not None:
            return self._requests.pop(index)
        return None

    def get(self, receiver: str=None, request_name: str=None) -> Response:
        """
        Args:
            receiver (str, optional): The name of the receiver. Defaults to None.
            request_name (str, optional): The naem of the request. Defaults to None.

        Returns:
            Response: The response if found else None
        """
        index = self.find(receiver, request_name)
        if index is not None:
            return self._responses[index]
        return None

    def has_response(self, receiver: str=None, request_name: str=None) -> bool:
        """
        Args:
            receiver (str, optional): The name of the receiver. Defaults to None.
            request_name (str, optional): The naem of the request. Defaults to None.

        Returns:
            bool: if the response is found
        """
        return self.find(receiver, request_name) is not None


class Server(object):

    def __init__(self):
        """Create a server to manage requests and store data
        """
        self._shared = DataStore()
        self._terminals = OrderedDict()
        self._register = {}
        self._responses = ResponseHandler()
        self._requests = RequestHandler()
        self._receivers: typing.Dict[typing.Tuple[MessageType, str], typing.Dict[typing.Union[str, typing.Callable], bool]] = {}

    @property
    def shared(self) -> DataStore:
        return self._shared
    
    @property
    def reponses(self) -> ResponseHandler:
        return self._responses
    
    @property
    def requests(self) -> RequestHandler:
        return self._requests
    
    def send(self, message: Message):
        """Send a message to the server
        Args:
            message (Message): The message to send to the server
        """
        receivers = self._receivers.get((message.message_type, message.name))
        if receivers is None:
            return
        
        updated = {}
        for task, expires in receivers.items():
            if isinstance(task, str):
                self._register[task].receive(message)
            else:
                task(message)
            if not expires:
                updated[task.id] = False
        self._receivers[(message.message_type, message.name)] = updated

    def register(self, behavior: 'Behavior', overwrite: bool=False) -> 'Terminal':
        """Register a behavior with the server

        Args:
            behavior (Behavior): The behavior to register
            overwrite (bool, optional): Overwrite the behavior if it already exist. Defaults to False.

        Raises:
            ValueError: If overwrite is false and not in reg

        Returns:
            Terminal: The terminal for the behavior
        """
        in_register = self._register.get(behavior.id)
        if in_register is None:
            self._register[behavior.id] = behavior
            terminal = Terminal(self)
            self._terminals[behavior.id] = terminal
        elif in_register and overwrite:
            self._register[behavior.id] = behavior
            terminal = Terminal(self)
            self._terminals[behavior.id] = terminal
        elif in_register:
            raise ValueError(f'The task in register {in_register} is different from this task')
        elif behavior is not in_register:
            raise ValueError(f'The task in register {in_register} is different from this task')
        return self._terminals[behavior.id]
    
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

    def receive(self, message_type: MessageType, name: str, behavior: typing.Union['Behavior', typing.Callable], expires: bool=True):

        is_behavior = isinstance(behavior, Behavior)
        if is_behavior and behavior.id not in self._register:
            raise ValueError(f'No task with id of {behavior.id} in the register')
        if (message_type, name) not in self._receivers:
            self._receivers[(message_type, name)] = {}
        
        if is_behavior:
            self._receivers[(message_type, name)][behavior.id] = expires
        else:
            self._receivers[(message_type, name)][behavior] = expires

    def cancel_receive(self, message_type: MessageType, name: str, behavior: typing.Union['Behavior', typing.Callable]):

        receiver = self._receivers.get((message_type, name))
        if receiver is None:
            raise ValueError(f'No receiver for {message_type} and {name}')

        del receiver[behavior.id]
        if len(receiver) == 0:
            del self._receivers[message_type, name]

    def has_receiver(self, message_type: MessageType, name: str, behavior: 'Behavior'):

        receiver = self._receivers.get((message_type, name))
        if receiver is None:
            return False
        
        return behavior.id in receiver

    def receives_message(self, message: typing.Union[Message, typing.Tuple[MessageType, str]]) -> bool:
        """Check if it receives a message

        Args:
            message (typing.Union[Message, typing.Tuple[MessageType, str]]): _description_

        Returns:
            bool: If it receives a message
        """
        if isinstance(message, Message):
            message_type, name = message.message_type, message.name
        else:
            message_type, name = message
        
        return (message_type, name) in self._receivers


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
    def responses(self) -> ResponseHandler:
        """
        Returns:
            ResponseHandler: The response handler for the server
        """
        return self._server.reponses
    
    @property
    def requests(self) -> RequestHandler:
        """
        Returns:
            RequestHandler: The request handler for the server
        """
        return self._server.requests
    
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

    def child(self, behavior: 'Behavior') -> 'Terminal':

        return self.server.register(behavior)

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
