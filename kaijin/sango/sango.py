from enum import Enum
from abc import abstractmethod, ABC, abstractproperty
import typing
from typing import Any
from dataclasses import dataclass


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
        self._value = value
        self._check_f = check_f
        self._hooks = CompositeHook(f'')

    def register_hook(self, hook: DataHook):
        """Register a hook for when the data is updated

        Args:
            hook: The hook to call
        """
        self._hooks.append(hook)

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> typing.Any:
        return self._value

    def update(self, value) -> typing.Any:
        if self._check_f is not None:
            valid, msg = self._check_f(value)
            if not valid:
                raise ValueError(msg)
        prev_value = self._value
        self._value = value
        self._hooks(self, prev_value)
        return value


class Synched(IData):

    def __init__(self, name: str, base_data: Data):

        super().__init__(name)
        self._base_data = base_data

    def register_hook(self, hook: DataHook):
        """Register a hook for when the data is updated. Will be
        attached to the base data

        Args:
            hook (DataHook): The hook to call when updating
        """
        self._base_data.register_hook(hook)

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def synched_name(self) -> str:
        return self._base_data.name

    @property
    def value(self) -> typing.Any:
        return self._value

    def update(self, value) -> typing.Any:
        return self._base_data.update(value)


class CompositeHook(DataHook, list):
    """A hook that will run multiple hooks
    """

    def __init__(self, name: str, hooks: typing.List[DataHook]=None):
        """Create a composite hook to run multiple hook

        Args:
            name (str): The name of the hook
            hooks (typing.List[DataHook], optional): The hooks to run. Defaults to None.
        """
        super().__init__(name, hooks)

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
            raise ValueError()
        self._data[key] = Data(key, value, check_f)
    
    def add_sync(self, key: str, data: Data):
        if key in self._data:
            raise ValueError()
        self._data[key] = Synched(key, data)
    
    def update(self, key: str, value: typing.Any):
        
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


class MessageHandler(object):

    def __init__(self):

        self._receivers: typing.Dict[typing.Tuple, typing.List[typing.Tuple[typing.Callable, bool]]] = {}

    def trigger(self, message: Message):
        
        receivers = self._receivers.get((message.message_type, message.name))
        if receivers is None:
            return
        
        updated = []
        for callback, expires in receivers:
            callback(message)
            if not expires:
                updated.append((callback, False))
        self._receivers[(message.message_type, message.name)] = updated

    def add_receiver(self, message_type: MessageType, name: str, callback, expires: bool=True):

        if (message_type, name) not in self._receivers:
            self._receivers = []
        self._receivers[(message_type, name)].append((callback, expires))

    def remove_receiver(self, message_type: MessageType, name: str, callback):

        updated = []

        for callback, expires in self._receivers[(message_type, name)]:
            if callback != callback:
                updated.append[(callback, expires)]
        self._receivers[(message_type, name)] = updated


class Server(object):

    def __init__(self):
        
        self._shared = DataStore()
        self._message_handler = MessageHandler()

    @property
    def shared(self) -> DataStore:
        return self._shared
    
    def send(self, message: Message):
        
        self._message_handler.trigger(message)

    def receive(self, message_type: MessageType, name: str, callback) -> typing.Any:
        
        self._message_handler.add_receiver(message_type, name, callback)

    def cancel_receive(self, message_type: MessageType, name: str, f):
        self._message_handler.cancel_receive(message_type, name, f)

# server.receive(message_type, name, callback)


class Terminal(object):

    def __init__(self, server: Server, parent: 'Terminal'=None) -> None:
        self._initialized = False
        self._server = server
        self._parent = parent
        self._data = DataStore()

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


class Task(ABC):

    def _tick_wrapper(self, f) -> typing.Callable[[Terminal], Status]:

        def _tick(self, terminal: Terminal, *args, **kwargs) -> Status:
            if not terminal.initialized:
                self.__init_terminal__(terminal)

            return f(terminal, *args, **kwargs)
        return _tick

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._status = Status.READY
        self.tick = self._tick_wrapper(self.tick)

    @abstractmethod
    def __init_terminal__(self, terminal: Terminal):
        pass

    @property
    def name(self) -> str:

        return self._name
    
    @property
    def status(self) -> Status:
        return self._status

    @abstractmethod    
    def tick(self, terminal: Terminal) -> Status:
        pass

    @abstractmethod
    def clone(self) -> 'Task':
        pass

    def state_dict(self) -> typing.Dict:

        return {
            'name': self._name
        }
    
    def load_state_dict(self, state_dict: typing.Dict):

        self._name = state_dict['name']
        self._status = state_dict['status']

    def __call__(self, terminal: Terminal) -> Status:
    
        return self.tick(terminal)


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

    # def iterate(self, filter: Filter=None, deep: bool=True):
    #     filter = filter or NullFilter()    
    #     for task in self._tasks:
    #         if filter.check(task):
    #             yield task
    #             if deep:
    #                 for subtask in task.iterate(filter, deep):
    #                     yield subtask


class Sequence(Composite):

    def add(self, task: Task) -> 'Sequence':
        self._tasks.append(task)

    def subtick(self, terminal: Terminal) -> Status:
        
        idx = terminal.storage['idx']
        status = self._tasks[idx].tick(terminal)

        if status == Status.FAILURE:
            return Status.FAILURE
        
        if status == Status.SUCCESS:
            terminal.storage['idx'] = idx + 1
            status = Status.RUNNING
        if terminal.storage['idx'] >= len(self._tasks):
            status = Status.FAILURE
        
        return status

    def clone(self) -> 'Task':
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

    def clone(self) -> 'Task':
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
    def act(self):
        raise NotImplementedError

    def tick(self, terminal: Terminal) -> Status:

        if self._status.is_done():
            return self._status
        status = self.act()
        terminal.storage['status'] = status
        return status


class Condition(Task):

    @abstractmethod
    def condition(self) -> bool:
        pass

    def tick(self) -> Status:
        
        if self.condition():
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
