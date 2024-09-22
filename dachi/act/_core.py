# 1st party
from enum import Enum
from abc import abstractmethod
import typing
from dataclasses import dataclass
from functools import reduce
from typing import Self

# local
from .._core import Storable


def get_or_spawn(state: typing.Dict, child: str) -> typing.Dict:
    """Get a child or spawn it if it does not exist

    Args:
        state (typing.Dict): The state
        child (str): The name of the child

    Returns:
        typing.Dict: The dictionary for the child
    """
    if child not in state:
        state[child] = {}
    return state[child]


def get_or_set(state: typing.Dict, key: str, val: typing.Any) -> typing.Any:
    """Use to get or set a value in a state dict

    Args:
        state (typing.Dict): The state dict to update
        key (str): The key to update
        val (typing.Any): The value to add

    Returns:
        typing.Any: 
    """
    if key not in state:
        state[key] = val
    return state[key]


class TaskStatus(Enum):
    """Status of a Behavior Tree task
    """
    # The task is currently in progress
    RUNNING = 'running'
    # The task is currently waiting for a response
    WAITING = 'waiting'
    # The task was successful
    SUCCESS = 'success'
    # The task failed
    FAILURE = 'failure'
    # The task has not been executed
    READY = 'ready'

    @property
    def is_done(self) -> bool:
        return self == TaskStatus.FAILURE or self == TaskStatus.SUCCESS
    
    @property
    def in_progress(self) -> bool:
        return self == TaskStatus.RUNNING or self == TaskStatus.WAITING
    
    @property
    def ready(self) -> bool:
        return self == TaskStatus.READY
    
    @property
    def failure(self) -> bool:
        return self == TaskStatus.FAILURE
    
    @property
    def success(self) -> bool:
        return self == TaskStatus.SUCCESS
    
    @property
    def running(self) -> bool:
        return self == TaskStatus.RUNNING
    
    @classmethod
    def from_bool(cls, val: bool) -> 'TaskStatus':
        return TaskStatus.SUCCESS if val is True else TaskStatus.FAILURE
    
    def __or__(self, other: 'TaskStatus') -> 'TaskStatus':
        """Compute the union of two statuses

        Returns:
            SangoStatus: The resulting status. Returns success if one status
            is success.
        """
        if self == other:
            return self
        
        if (
            (self.success or other.success)
        ):
            return TaskStatus.SUCCESS
        if self.running or other.running:
            return TaskStatus.RUNNING
        
        if self.waiting or other.waiting:
            return TaskStatus.WAITING
        
        if (self.success and not other.success):
            return other
        if (not self.failure and other.failure):
            return self
        
        raise ValueError(f'Invalid combination of statuses {self} and {other}')

    def __and__(self, other: 'TaskStatus') -> 'TaskStatus':
        """Compute the union of two statuses

        Returns:
            SangoStatus: The resulting status. Returns failure if one status
            is failure.
        """

        if self == other:
            return self
        
        if (
            (self.failure or other.failure)
        ):
            return TaskStatus.FAILURE
        if self.running or other.running:
            return TaskStatus.RUNNING
        
        if self.waiting or other.waiting:
            return TaskStatus.WAITING
        
        if (self.success and not other.success):
            return other
        if (not self.success and other.success):
            return self
        raise ValueError(f'Invalid combination of statuses {self} and {other}')

    def invert(self) -> 'TaskStatus':

        if self.success:
            return TaskStatus.FAILURE
        if self.failure:
            return TaskStatus.SUCCESS
        return self


WAITING = TaskStatus.WAITING
READY = TaskStatus.READY
SUCCESS = TaskStatus.SUCCESS
FAILURE = TaskStatus.FAILURE
RUNNING = TaskStatus.RUNNING

@dataclass
class TaskMessage:

    name: str
    data: typing.Any


class Task(Storable):
    """The base class for a task in the behavior tree
    """

    SUCCESS = TaskStatus.SUCCESS
    FAILURE = TaskStatus.FAILURE
    RUNNING = TaskStatus.RUNNING

    def __init__(self) -> None:
        """Create the task

        Args:
            name (str): The name of the task
        """
        super().__init__()
        self._status = TaskStatus.READY

    @abstractmethod    
    def tick(self) -> TaskStatus:
        raise NotImplementedError

    def __call__(self) -> TaskStatus:
        """

        Args:
            terminal (Terminal): _description_

        Returns:
            SangoStatus: _description_
        """
        return self.tick()

    def reset(self):
        """Reset the terminal

        """
        self._status = TaskStatus.READY
    
    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def id(self):
        return self._id


class Shared:
    """Allows for shared data between tasks
    """

    def __init__(
        self, data: typing.Any=None, 
        default: typing.Any=None,
        callbacks: typing.List[typing.Callable[[typing.Any], None]]=None
    ):
        """Create shard data

        Args:
            data (typing.Any, optional): The data. Defaults to None.
            default (typing.Any, optional): The default value of the shared. When resetting, will go back to this. Defaults to None.
            callbacks (typing.List[typing.Callable[[typing.Any], None]], optional): The callbacks to call on update. Defaults to None.
        """

        self._default = default
        self._data = data if data is not None else default
        self._callbacks = callbacks or []

    def register(self, callback) -> bool:
        """Register a callback to call on data updates

        Args:
            callback (function): The callback to register

        Returns:
            bool: True the callback was registered, False if already registered
        """

        if callback in self._callbacks:
            return False
        
        self._callbacks.append(callback)
        return True

    def unregister(self, callback) -> bool:
        """Unregister a callback to call on data updates

        Args:
            callback (function): The callback to unregister

        Returns:
            bool: True if the callback was removed, False if callback was not registered
        """

        if callback not in self._callbacks:
            return False
        
        self._callbacks.remove(callback)
        return True

    @property
    def data(self) -> typing.Any:
        """Get the data shared

        Returns:
            typing.Any: The shared data
        """
        return self._data

    @data.setter
    def data(self, data) -> typing.Any:
        """Update the data

        Args:
            data: The data to share

        Returns:
            typing.Any: The value for the data
        """
        self._data = data
        for callback in self._callbacks:
            callback(data)
        return data

    def reset(self):
        self.data = self._default


# TODO: FINISH!

# x, y, z, ..., STOP
# current index

class Buffer(object):

    def __init__(self) -> None:
        
        self._buffer = []
        self._opened = True
        
    def add(self, *data):
        
        if not self._opened:
            raise RuntimeError(
                'The buffer is currently closed so cannot add data to it.'
            )

        self._buffer.extend(data)

    def open(self):
        self._opened = True

    def close(self):

        self._opened = False

    def is_open(self):

        return self._opened
        
    def it(self) -> 'BufferIter':
        return BufferIter(self)

    # def reset(self, opened: bool=True):
    #     """Resets new buffer. Note that any iterators will
    #     still use the old buffer
    #     """
    #     self._opened = opened
    #     self._buffer = []

    def __getitem__(self, key) -> typing.Union[typing.Any, typing.List]:

        if isinstance(key, slice):
            return self._buffer[key]
        if isinstance(key, typing.Iterable):
            return [self._buffer[i] for i in key]
        return self._buffer[key]
        
    def __len__(self) -> int:
        """

        Returns:
            int: 
        """
        return len(self._buffer)


class BufferIter(object):

    def __init__(self, buffer: Buffer) -> None:
        """

        Args:
            buffer (typing.List): 
        """
        self._buffer = buffer
        self._i = 0

    def start(self):
        """Whether the iterator is at the start of the buffer
        """
        return self._i == 0

    def end(self):

        return self._i >= (len(self._buffer) - 1)
    
    def is_open(self):

        return self._buffer.is_open()

    def read(self) -> typing.Any:
        
        if self._i < (len(self._buffer) - 1):
            self._i += 1
            return self._buffer[self._i - 1]
        raise StopIteration('Reached the end of the buffer')

    def read_all(self) -> typing.List:
        
        if self._i < len(self._buffer):
            i = self._i
            self._i = len(self._buffer)
            return self._buffer[i:]
            
        return []

    def read_reduce(self, f, init_val=None) -> typing.Any:

        data = self.read_all()
        return reduce(
            f, data, init_val
        )

    def read_map(self, f) -> typing.List:

        data = self.read_all()
        return map(
            f, data
        )


class TaskFunc(object):

    def __init__(self, f: typing.Callable, is_method: bool=False, instance=None):
        
        super().__init__()
        self.f = f
        self._is_method = is_method
        self._instance = instance
    
    def __get__(self, instance, owner):

        if self._instance is not None and instance is self._instance:
            return self._instance
        self._instance = instance
        return self._instance
    
    def _exec(self, *args, **kwargs):

        if self._is_method:
            return self.f(self._instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    @abstractmethod
    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> TaskStatus:
        pass

    def l(self) -> typing.Callable[[], Self]:
    
        def _(_: typing.Dict) -> Self:
            return self.tick()
        
        return _


class Context(dict):

    def get_or_set(self, key, value):

        if key not in self:
            self[key] = value
            return value
        
        return self[key]


class ContextStorage(object):

    def __init__(self):

        object.__setattr__(self, '_data', {})

    @property
    def states(self) -> typing.Dict[str, Context]:
        return {**self._data}
    
    def remove(self, key):

        del self._data[key]
    
    def add(self, key):

        if key not in self._data:
            d = Context()
            self._data[key] = d
        
        return self._data[key]

    def __getattr__(self, key):

        if key not in self._data:
            d = Context()
            self._data[key] = d
        
        return self._data[key]


class ContextSpawner(object):

    def __init__(self, manager: ContextStorage, base_name: str):
        """_summary_

        Args:
            manager (StateManager): _description_
            base_name (str): _description_
        """
        self.manager = manager
        self.base_name = base_name

    def __getitem__(self, i: int) -> Context:

        name = f'{self.base_name}_{i}'
        return self.manager.add(name)
