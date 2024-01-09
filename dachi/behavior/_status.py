from enum import Enum


class SangoStatus(Enum):
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
        return self == SangoStatus.FAILURE or self == SangoStatus.SUCCESS
    
    @property
    def in_progress(self) -> bool:
        return self == SangoStatus.RUNNING or self == SangoStatus.WAITING
    
    @property
    def ready(self) -> bool:
        return self == SangoStatus.READY
    
    @property
    def failure(self) -> bool:
        return self == SangoStatus.FAILURE
    
    @property
    def success(self) -> bool:
        return self == SangoStatus.SUCCESS
    
    @property
    def running(self) -> bool:
        return self == SangoStatus.RUNNING
    
    def __or__(self, other: 'SangoStatus') -> 'SangoStatus':
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
            return SangoStatus.SUCCESS
        if self.running or other.running:
            return SangoStatus.RUNNING
        
        if self.waiting or other.waiting:
            return SangoStatus.WAITING
        
        if (self.success and not other.success):
            return other
        if (not self.failure and other.failure):
            return self
        
        raise ValueError(f'Invalid combination of statuses {self} and {other}')

    def __and__(self, other: 'SangoStatus') -> 'SangoStatus':
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
            return SangoStatus.FAILURE
        if self.running or other.running:
            return SangoStatus.RUNNING
        
        if self.waiting or other.waiting:
            return SangoStatus.WAITING
        
        if (self.success and not other.success):
            return other
        if (not self.success and other.success):
            return self
        raise ValueError(f'Invalid combination of statuses {self} and {other}')
