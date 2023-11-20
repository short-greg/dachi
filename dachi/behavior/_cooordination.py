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


class Server(object):

    def __init__(self):
        
        self._shared = DataStore()
        self._terminals = OrderedDict()
        self._register = {}
        self._receivers: typing.Dict[typing.Tuple[MessageType, str], typing.Dict[typing.Union[str, typing.Callable], bool]] = {}

    @property
    def shared(self) -> DataStore:
        return self._shared
    
    def send(self, message: Message):
        
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

        in_register = self._register.get(behavior.id)
        if in_register is None:
            self._register[behavior.id] = behavior
            terminal = Terminal(self)
            self._terminals[behavior.id] = terminal
        elif in_register and overwrite:
            self._register[behavior.id] = behavior
            terminal = Terminal(self)
            self._terminals[behavior.id] = terminal
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

    def receives_message(self, message: typing.Union[Message, typing.Tuple[MessageType, str]]):

        if isinstance(message, Message):
            message_type, name = message.message_type, message.name
        else:
            message_type, name = message
        
        return (message_type, name) in self._receivers


class Terminal(object):

    def __init__(self, server: Server) -> None:
        self._initialized = False
        self._server = server
        self._storage = DataStore()

    @property
    def initialized(self) -> bool:
        return self._initialized
    
    def initialize(self):

        self._initialized = True

    @property
    def storage(self) -> DataStore:
        return self._storage
    
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
