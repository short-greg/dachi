# 1st party
from enum import Enum
from abc import abstractmethod
import typing
from dataclasses import dataclass
from functools import reduce
from typing import Self

# local
from .._core import Storable


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
