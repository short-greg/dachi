from enum import Enum


# How to deal with waiting
# How to continue running

class SangoStatus(Enum):

    RUNNING = 'running'
    WAITING = 'waiting'
    SUCCESS = 'success'
    FAILURE = 'failure'
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
    def waiting(self) -> bool:
        return self == SangoStatus.WAITING
    
    @property
    def running(self) -> bool:
        return self == SangoStatus.RUNNING
    
    def __or__(self, other: 'SangoStatus') -> 'SangoStatus':

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
