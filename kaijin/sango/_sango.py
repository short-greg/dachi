from enum import Enum
from abc import abstractmethod, ABC, abstractproperty
import typing
from typing import Any
from dataclasses import dataclass
from uuid import uuid4
from collections import OrderedDict

# tree.tick(comm)
# comm.share('', ...)
# comm.shared('')
# comm.get_or_store()
# comm.store
# comm.get
# comm.send()
# comm.receive
# init_comm() # add this hook
# comm.sub(3)


# How to deal with waiting
# How to continue running


class DataHook(ABC):

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def __call__(self, data: 'Data', prev_value):
        pass


class MetaHook(ABC):

    def __init__(self, hook: DataHook):
        super().__init__()
        self._hook = hook
    
    @property
    def name(self) -> str:
        return self._hook.name

    def __call__(self, data: 'Data', prev_value):
        self._hook(data, prev_value)


class IData(ABC):
    """Define the base class for data
    """

    def __init__(self, name: str):
        """Create the data

        Args:
            name (str): The name of the data
        """
        super().__init__()
        self._name = name

    @abstractmethod
    def register_hook(self, hook):
        pass

    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the data
        """
        return self._name

    @abstractproperty
    def value(self) -> typing.Any:
        pass

    @abstractmethod
    def update(self, value) -> typing.Any:
        pass


class Data(IData):
    """Definitions of data for the behavior tree
    """

    def __init__(self, name: str, value: typing.Any, check_f=None):
        """Create data for the behavior tree

        Args:
            name (str): The name of the data
            value (typing.Any): The value for the data
            check_f (optional): Validation function if necessary. Defaults to None.
        """
        super().__init__(name)
        self._check_f = check_f
        self.validate(value)
        self._value = value
        self._hooks = CompositeHook(name)

    def register_hook(self, hook: DataHook):
        """Register a hook for when the data is updated

        Args:
            hook: The hook to call
        """
        self._hooks.append(hook)

    def has_hook(self, hook: DataHook):
        """Get whether 

        Args:
            hook (DataHook): _description_

        Returns:
            _type_: _description_
        """
        return hook in self._hooks

    def remove_hook(self, hook: DataHook):
        """Get whether 

        Args:
            hook (DataHook): _description_

        """
        self._hooks.remove(hook)
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> typing.Any:
        return self._value
    
    def validate(self, value):
        if self._check_f is not None:
            valid, msg = self._check_f(value)
            if not valid:
                raise ValueError(msg)

    def update(self, value) -> typing.Any:
        self.validate(value)
        prev_value = self._value
        self._value = value
        self._hooks(self, prev_value)
        return value


class Synched(IData):

    def __init__(self, name: str, base_data: Data):

        super().__init__(name)
        self._base_data = base_data
        self._hooks: typing.Dict[DataHook, MetaHook] = {}

    def register_hook(self, hook: DataHook):
        """Register a hook for when the data is updated. Will be
        attached to the base data

        Args:
            hook (DataHook): The hook to call when updating
        """
        meta = MetaHook(hook)
        self._hooks[hook] = meta
        self._base_data.register_hook(meta)

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def synched_name(self) -> str:
        return self._base_data.name

    @property
    def value(self) -> typing.Any:
        return self._base_data.value

    def update(self, value) -> typing.Any:
        return self._base_data.update(value)

    def has_hook(self, hook: DataHook):
        """Get whether 

        Args:
            hook (DataHook): _description_

        Returns:
            _type_: _description_
        """
        return hook in self._hooks

    def remove_hook(self, hook: DataHook):
        """Get whether 

        Args:
            hook (DataHook): _description_

        """
        meta = self._hooks[hook]
        self._base_data.remove_hook(meta)
        del self._hooks[hook]
        

class CompositeHook(DataHook, list):
    """A hook that will run multiple hooks
    """

    def __init__(self, name: str, hooks: typing.List[DataHook]=None):
        """Create a composite hook to run multiple hook

        Args:
            name (str): The name of the hook
            hooks (typing.List[DataHook], optional): The hooks to run. Defaults to None.
        """
        super().__init__(name, hooks or [])

    def __call__(self, data: 'Data', prev_value):
        """Run the hooks

        Args:
            data (Data): The data that was updated
            prev_value (typing.Any): The previous value for the data
        """
        for hook in self:
            hook(data, prev_value)


class Status(Enum):

    RUNNING = 'running'
    WAITING = 'waiting'
    SUCCESS = 'success'
    FAILURE = 'failure'
    READY = 'ready'

    def is_done(self) -> bool:
        return self == Status.FAILURE or self == Status.SUCCESS


class DataStore(object):
    """
    """

    def __init__(self) -> None:
        super().__init__()
        self._data: typing.Dict[str, Data] = {}

    def add(self, key: str, value: typing.Any, check_f=None) -> None:
        """Add data to the data store

        Args:
            key (str): The name of the data
            value (typing.Any): The value for the data
            check_f (typing.Callable[[typing.Any], bool], optional): Add data to the datastore . Defaults to None.

        Raises:
            ValueError: 
        """
        if key in self._data:
            raise ValueError(f'Data with key {key} has already been added to the data store')
        self._data[key] = Data(key, value, check_f)
    
    def add_sync(self, key: str, data: Data):
        if key in self._data:
            raise ValueError()
        self._data[key] = Synched(key, data)
    
    def update(self, key: str, value: typing.Any):
        
        if key not in self._data:
            self.add(key, value)
        else:
            self._data[key].update(value)

    def get(self, key: str) -> typing.Any:

        result = self._data.get(key)
        if result is None:
            return None
        return result.value
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def __getitem__(self, key: str) -> typing.Any:
        return self._data[key].value
    
    def __setitem__(self, key: str, value) -> typing.Any:

        self.update(key, value)
        return value

    def register_hook(self, key: str, hook: DataHook):
        """Register a hook for when the data is updated. Will be
        attached to the base data

        Args:
            hook (DataHook): The hook to call when updating
        """
        self._data[key].register_hook(hook)
    
    def has_hook(self, key: str, hook: DataHook):
        """
        Args:
            key : The key to check if exists
            hook (DataHook): The hook to 

        Returns:
            bool: Whether the hook data specified by key has the hook
        """
        return self._data[key].has_hook(hook)

    def remove_hook(self, key: str, hook: DataHook):
        """Get whether 

        Args:
            hook (DataHook): _description_

        """
        self._data[key].remove_hook(hook)

    def reset(self):
        """Clear out the data in the data store
        """
        self._data.clear()        


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


# class MessageHandler(object):

#     def __init__(self):

#         self._receivers: typing.Dict[typing.Tuple, typing.Dict[typing.Callable, bool]] = dict()

#     def trigger(self, message: Message):
        
#         receivers = self._receivers.get((message.message_type, message.name))
#         if receivers is None:
#             return
        
#         updated = {}
#         for callback, expires in receivers.items():
#             callback(message)
#             if not expires:
#                 updated[callback] = False
#         self._receivers[(message.message_type, message.name)] = updated

#     def add_receiver(self, message_type: MessageType, name: str, callback, expires: bool=True):

#         if (message_type, name) not in self._receivers:
#             self._receivers[(message_type, name)] = {}
#         self._receivers[(message_type, name)][callback] = expires

#     def remove_receiver(self, message_type: MessageType, name: str, callback):

#         receiver = self._receivers.get((message_type, name))
#         if receiver is None:
#             raise ValueError(f'No receiver for {message_type} and {name}')

#         del receiver[callback]
#         if len(receiver) == 0:
#             del self._receivers[message_type, name]

#     def __contains__(self, message: typing.Union[Message, typing.Tuple[MessageType, str]]):

#         if isinstance(message, Message):
#             message_type, name = message.message_type, message.name
#         else:
#             message_type, name = message
        
#         return (message_type, name) in self._receivers

#     def has_receiver(self, message_type: MessageType, name: str, callback):

#         receiver = self._receivers.get((message_type, name))
#         if receiver is None:
#             return False
        
#         return callback in receiver

#     # Not sure how to revive te revive this
#     # the callbacks will likely be wrong
#     # can 

class Server(object):

    def __init__(self):
        
        self._shared = DataStore()
        # self._message_handler = MessageHandler()
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

    def register(self, task: 'Task', overwrite: bool=False) -> 'Terminal':

        in_register = self._register.get(task.id)
        if in_register is None:
            self._register[task.id] = task
            terminal = Terminal(self)
            self._terminals[task.id] = terminal
        elif in_register and overwrite:
            self._register[task.id] = task
            terminal = Terminal(self)
            self._terminals[task.id] = terminal
        elif task is not in_register:
            raise ValueError(f'The task in register {in_register} is different from this task')
        return self._terminals[task.id]
    
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

    def receive(self, message_type: MessageType, name: str, task: typing.Union['Task', typing.Callable], expires: bool=True):

        is_task = isinstance(task, Task)
        if is_task and task.id not in self._register:
            raise ValueError(f'No task with id of {task.id} in the register')
        if (message_type, name) not in self._receivers:
            self._receivers[(message_type, name)] = {}
        
        if is_task:
            self._receivers[(message_type, name)][task.id] = expires
        else:
            self._receivers[(message_type, name)][task] = expires

    def cancel_receive(self, message_type: MessageType, name: str, task: typing.Union['Task', typing.Callable]):

        receiver = self._receivers.get((message_type, name))
        if receiver is None:
            raise ValueError(f'No receiver for {message_type} and {name}')

        del receiver[task.id]
        if len(receiver) == 0:
            del self._receivers[message_type, name]

    def has_receiver(self, message_type: MessageType, name: str, task: 'Task'):

        receiver = self._receivers.get((message_type, name))
        if receiver is None:
            return False
        
        return task.id in receiver

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

    def child(self, task: 'Task') -> 'Terminal':

        return self.server.register(task)

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


class Task(ABC):

    def _tick_wrapper(self, f) -> typing.Callable[[Terminal], Status]:

        def _tick(terminal: Terminal, *args, **kwargs) -> Status:
            if not terminal.initialized:
                self.__init_terminal__(terminal)
                terminal.initialize()

            return f(terminal, *args, **kwargs)
        return _tick

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._id = str(uuid4())
        self.tick = self._tick_wrapper(self.tick)

    def __init_terminal__(self, terminal: Terminal):
        terminal.storage.add('status', Status.READY)

    @property
    def name(self) -> str:

        return self._name

    @abstractmethod    
    def tick(self, terminal: Terminal) -> Status:
        pass

    def clone(self) -> 'Task':
        return self.__class__(self.name)

    def state_dict(self) -> typing.Dict:

        return {
            'id': self._id,
            'name': self._name
        }
    
    def load_state_dict(self, state_dict: typing.Dict):

        self._id = state_dict['id']
        self._name = state_dict['name']

    def __call__(self, terminal: Terminal) -> Status:
    
        return self.tick(terminal)

    def reset(self, terminal):

        terminal.reset()
        self.__init_terminal__(terminal)

    def receive(self, message: Message):
        pass

    @property
    def id(self):
        return self._id


# 1) not sure how to recover the state


class Composite(Task):
    """Task composed of subtasks
    """
    def __init__(
        self, tasks: typing.List[Task], name: str=''
    ):
        super().__init__(name)
        self._tasks = tasks

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.storage.add('idx', 0)

    @property
    def n(self):
        """The number of subtasks"""
        return len(self._tasks)
    
    @property
    def tasks(self):
        """The subtasks"""
        return [*self._tasks]

    @abstractmethod
    def subtick(self, terminal: Terminal) -> Status:
        """Tick each subtask. Implement when implementing a new Composite task"""
        raise NotImplementedError

    def tick(self, terminal: Terminal) -> Status:
        status = self.subtick(terminal)
        terminal.storage['status'] = status
        return status
    
    def reset(self):
        super().reset()
        for task in self._tasks:
            task.reset()

    def state_dict(self) -> typing.Dict:

        return {
            'tasks': [
                task.state_dict()
                for task in self._tasks
            ],
            **super().state_dict()
        }
    
    def load_state_dict(self, state_dict: typing.Dict):

        super().load_state_dict(state_dict)
        for task, task_state_dict in zip(self._tasks, state_dict['tasks']):
            task.load_state_dict(task_state_dict)


class Sequence(Composite):

    def add(self, task: Task) -> 'Sequence':
        self._tasks.append(task)

    def subtick(self, terminal: Terminal) -> Status:
        
        idx = terminal.storage['idx']
        if terminal.storage['status'].is_done():
            return terminal.storage['status']
    
        task = self._tasks[idx]
        status = task.tick(terminal.child(task))

        if status == Status.FAILURE:
            return Status.FAILURE
        
        if status == Status.SUCCESS:
            terminal.storage['idx'] += 1
            status = Status.RUNNING
        if terminal.storage['idx'] >= len(self._tasks):
            status = Status.SUCCESS

        return status

    def clone(self) -> 'Sequence':
        return Sequence(
            self.name, [task.clone() for task in self._tasks]
        )
    

class Fallback(Composite):

    def add(self, task: Task) -> 'Sequence':
        self._tasks.append(task)

    def subtick(self, terminal: Terminal) -> Status:
        
        idx = terminal.storage['idx']
        status = self._tasks[idx].tick(terminal)

        if status == Status.SUCCESS:
            return Status.SUCCESS
        
        if status == Status.FAILURE:
            terminal.storage['idx'] = idx + 1
            status = Status.RUNNING
        if terminal.storage['idx'] >= len(self._tasks):
            status = Status.FAILURE
        
        return status

    def clone(self) -> 'Fallback':
        return Fallback(
            self.name, [task.clone() for task in self._tasks]
        )

class Parallel(Task):

    def __init__(self, name: str, tasks: typing.List[Task]):
        super().__init__(name, tasks)

    def add(self, task: Task):
        self._tasks.append(task)

    @abstractmethod
    def parallel(self, terminal: Terminal) -> Status:
        pass

    def clone(self) -> 'Task':
        return Parallel(
            self.name, [task.clone() for task in self._tasks]
        )


class Action(Task):

    @abstractmethod
    def act(self, terminal: Terminal):
        raise NotImplementedError

    def tick(self, terminal: Terminal) -> Status:

        if terminal.storage['status'].is_done():
            return terminal.storage['status']
        status = self.act(terminal)
        terminal.storage[status] = status
        return status


class Condition(Task):

    @abstractmethod
    def condition(self, terminal: Terminal) -> bool:
        pass

    def tick(self, terminal: Terminal) -> Status:
        
        if self.condition(terminal):
            return Status.SUCCESS
        return Status.FAILURE


# class LinearExecutor(CompositeExecutor):

#     def __init__(self, items: typing.List[Task]):
#         """initializer

#         Args:
#             items (typing.List[Task])
#         """
#         self._items = items
#         self._idx = 0

#     def reset(self, items: list=None):
#         """Reset the planner to the start state

#         Args:
#             items (list, optional): Updated items to set the planner to. Defaults to None.
#         """
#         self._idx = 0
#         self._items = items # coalesce(items, self._items)
    
#     @property
#     def idx(self):
#         return self._idx
    
#     @idx.setter
#     def idx(self, idx):
#         if not (0 <= idx <= len(self._items)):
#             raise IndexError(f"Index {idx} is out of range in {len(self._items)}")
#         self._idx = idx

#     def end(self) -> bool:
#         """Advance the planner

#         Returns:
#             bool
#         """
#         if self._idx == len(self._items):
#             return True
#         return False
    
#     def adv(self) -> bool:
#         """Advance the planner

#         Returns:
#             bool
#         """
#         if self._idx == len(self._items):
#             return False
#         self._idx += 1
#         return True
    
#     @property
#     def cur(self) -> Task:
#         """
#         Returns:
#             Task
#         """
#         if self._idx == len(self._items):
#             return None
#         return self._items[self._idx]

#     def rev(self) -> bool:
#         """Reverse the planner

#         Returns:
#             bool
#         """
#         if self._idx == 0:
#             return False
#         self._idx -= 1
#         return True

#     def clone(self):
#         """Clone the linear planner

#         Returns:
#             LinearPlanner
#         """
#         return LinearExecutor([*self._items])
    
#     def __len__(self) -> int:
#         return len(self._items)



# server.sync(terminal, []) <- shared fields

# class CompositeExecutor(object):

#     @abstractmethod
#     def reset(self, items: list=None):
#         """Reset the planner to the start state

#         Args:
#             items (list, optional): Updated items to set the planner to. Defaults to None.
#         """
#         pass

#     @abstractproperty
#     def idx(self):
#         pass
    
#     @idx.setter
#     def idx(self, idx):
#         pass

#     @abstractmethod
#     def end(self) -> bool:
#         """Advance the planner

#         Returns:
#             bool
#         """
#         pass

#     @abstractmethod
#     def adv(self) -> bool:
#         """Advance the planner

#         Returns:
#             bool
#         """
#         pass
    
#     @abstractproperty
#     def cur(self) -> Task:
#         """
#         Returns:
#             Task
#         """
#         pass

#     @abstractmethod
#     def rev(self) -> bool:
#         """Reverse the planner

#         Returns:
#             bool
#         """
#         pass

#     @abstractmethod
#     def clone(self):
#         """Clone the linear planner

#         Returns:
#             LinearPlanner
#         """
#         pass
    
#     @abstractmethod
#     def __len__(self) -> int:
#         pass
