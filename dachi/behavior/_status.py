from enum import Enum


class TaskStatus(Enum):
    """Status of a Sango Behavior Tree task
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
