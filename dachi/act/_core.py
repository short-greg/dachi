# 1st party
from enum import Enum
from abc import abstractmethod
import typing

# local
from ..base import Storable
# TODO: Add in Action (For GOAP)


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
        """Get whether the task has succeeded or failed

        Returns:
            bool: Whether the task is "done"
        """
        return self == TaskStatus.FAILURE or self == TaskStatus.SUCCESS
    
    @property
    def in_progress(self) -> bool:
        """Get whether the task is in progress

        Returns:
            bool: whether the task is still running
        """
        return self == TaskStatus.RUNNING or self == TaskStatus.WAITING
    
    @property
    def ready(self) -> bool:
        """Get whether the task is ready to be executed

        Returns:
            bool: Whether the task is ready to be executed
        """
        return self == TaskStatus.READY
    
    @property
    def failure(self) -> bool:
        """Get whether the task has failed

        Returns:
            bool: whether the task has failed
        """
        return self == TaskStatus.FAILURE
    
    @property
    def success(self) -> bool:
        """Get whether the task 

        Returns:
            bool: Whether the task has succeeded
        """
        return self == TaskStatus.SUCCESS
    
    @property
    def running(self) -> bool:
        """Get whether the task is running

        Returns:
            bool: Whether the task is running or not
        """
        return self == TaskStatus.RUNNING
    
    @classmethod
    def from_bool(cls, val: bool) -> 'TaskStatus':
        """Convert a boolean to a TaskStatus

        Args:
            val (bool): The value to convert

        Returns:
            TaskStatus: The status
        """
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

# @dataclass
# class TaskMessage:

#     name: str
#     data: typing.Any


class Task(Storable):
    """The base class for a task in the behavior tree
    """

    # _status: TaskStatus = pydantic.PrivateAttr(default=TaskStatus.READY)

    SUCCESS: TaskStatus = TaskStatus.SUCCESS
    FAILURE: TaskStatus = TaskStatus.FAILURE
    READY: TaskStatus = TaskStatus.READY
    RUNNING: TaskStatus = TaskStatus.RUNNING

    def __init__(self):

        super().__init__()

        self._status = self.READY

    @abstractmethod    
    def tick(self, reset: bool=False) -> TaskStatus:
        raise NotImplementedError

    def __call__(self, reset: bool=False) -> TaskStatus:
        """Execute the task

        Returns:
            TaskStatus: The status of the task after execution
        """
        return self.tick(reset)

    def reset_status(self):
        """Reset the terminal

        """
        self._status = TaskStatus.READY
    
    @property
    def status(self) -> TaskStatus:
        """The status

        Returns:
            TaskStatus: 
        """
        return self._status

    @property
    def id(self):
        """Get the id for the task

        Returns:
            The id of the task 
        """
        return self._id
    
    def load_state_dict(self, state_dict: typing.Dict):
        """Load the state dict for the object

        Args:
            state_dict (typing.Dict): The state dict
        """
        all_items = {**self.__dict__}
        for k, v in all_items.items():
            if isinstance(v, Storable):
                self.__dict__[k] = v.load_state_dict(state_dict[k])
            else:
                self.__dict__[k] = state_dict[k]


class ToStatus(object):
    """Use to convert a value to a status
    """

    @abstractmethod
    def __call__(self, val) -> TaskStatus:
        """Convert the value to a status

        Args:
            val: The value to convert

        Returns:
            TaskStatus: The status
        """
        pass


TOSTATUS = ToStatus | typing.Callable[[typing.Any], TaskStatus]


def from_bool(status: bool) -> TaskStatus:
    """functional code to map from a boolean

    Args:
        status (bool): The status in boolean form

    Returns:
        TaskStatus: The task status
    """
    return TaskStatus.from_bool(status)


class State(Storable):
    """Use State creating a state machine
    """
    @abstractmethod
    def update(self, reset: bool=False) -> typing.Union['State', TaskStatus]:
        """Update the 

        Returns: bool
            typing.Union['State', TaskStatus]: The new State/Status
        """
        pass

    def __call__(self, reset: bool=False):
        return self.update(reset)


STATE_CALL = State | typing.Callable[[], State | TaskStatus]


class Router(object):
    """Use to route a value to a Task
    """
    @abstractmethod
    def __call__(self, val) -> TaskStatus | State:
        pass


ROUTE = Router | typing.Callable[[typing.Any], TaskStatus | State]
