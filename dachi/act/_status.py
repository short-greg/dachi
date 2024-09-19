# 1st party
from enum import Enum
import typing


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
