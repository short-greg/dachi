from enum import Enum


# How to deal with waiting
# How to continue running

class Status(Enum):

    RUNNING = 'running'
    WAITING = 'waiting'
    SUCCESS = 'success'
    FAILURE = 'failure'
    READY = 'ready'

    @property
    def is_done(self) -> bool:
        return self == Status.FAILURE or self == Status.SUCCESS
    
    @property
    def in_progress(self) -> bool:
        return self == Status.RUNNING or self == Status.WAITING
    
    @property
    def ready(self) -> bool:
        return self == Status.READY
    
    @property
    def failure(self) -> bool:
        return self == Status.FAILURE
    
    @property
    def success(self) -> bool:
        return self == Status.SUCCESS
    
    @property
    def waiting(self) -> bool:
        return self == Status.WAITING
    
    @property
    def running(self) -> bool:
        return self == Status.RUNNING
    
    def __or__(self, other: 'Status') -> 'Status':

        if self == other:
            return self
        
        if (
            (self.success or other.success)
        ):
            return Status.SUCCESS
        if self.running or other.running:
            return Status.RUNNING
        
        if self.waiting or other.waiting:
            return Status.WAITING
        
        if (self.success and not other.success):
            return other
        if (not self.failure and other.failure):
            return self
        
        raise ValueError(f'Invalid combination of statuses {self} and {other}')

    def __and__(self, other: 'Status') -> 'Status':

        if self == other:
            return self
        
        if (
            (self.failure or other.failure)
        ):
            return Status.FAILURE
        if self.running or other.running:
            return Status.RUNNING
        
        if self.waiting or other.waiting:
            return Status.WAITING
        
        if (self.success and not other.success):
            return other
        if (not self.success and other.success):
            return self
        raise ValueError(f'Invalid combination of statuses {self} and {other}')
