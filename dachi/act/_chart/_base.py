from __future__ import annotations

from typing import Any, Dict, List, Union, Optional, Callable, Tuple
from enum import Enum
from abc import abstractmethod
import asyncio

from dachi.core import BaseModule, Attr


JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class ChartStatus(Enum):
    """State lifecycle status."""
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTING = "preempting"
    SUCCESS = "success"
    CANCELED = "canceled"
    FAILURE = "failure"

    def is_waiting(self) -> bool:
        return self == ChartStatus.WAITING

    def is_running(self) -> bool:
        return self == ChartStatus.RUNNING

    def is_preempting(self) -> bool:
        return self == ChartStatus.PREEMPTING

    def is_success(self) -> bool:
        return self == ChartStatus.SUCCESS

    def is_canceled(self) -> bool:
        return self == ChartStatus.CANCELED

    def is_failure(self) -> bool:
        return self == ChartStatus.FAILURE

    def is_completed(self) -> bool:
        """Returns True if in a final state (SUCCESS, FAILURE, or CANCELED)."""
        return self in (ChartStatus.SUCCESS, ChartStatus.FAILURE, ChartStatus.CANCELED)


# TODO: Determine if StatusResult is needed
# class StatusResult:
#     """Result of a status check."""
#     def __init__(self, status: ChartStatus, message: Optional[str] = None):
#         self.status = status
#         self.message = message


class InvalidTransition(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class ChartBase(BaseModule):
    """Base class for all state types."""
    name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = self.__class__.__name__
        self._status = Attr[ChartStatus](data=ChartStatus.WAITING)
        self._finish_callbacks: Dict[Callable, Tuple[tuple, dict]] = {}

    @property
    def status(self) -> ChartStatus:
        """Get current state status."""
        return self._status.get()

    def is_running(self) -> bool:
        return self._status.get().is_running()

    def is_completed(self) -> bool:
        return self._status.get().is_completed()
    
    def is_waiting(self) -> bool:
        return self._status.get().is_waiting()

    @abstractmethod
    def reset(self) -> None:
        """Reset state to WAITING. Must be implemented by subclasses."""
        pass

    def can_reset(self) -> bool:
        """Check if state can be reset."""
        return self._status.get().is_completed()

    async def finish(self) -> None:
        """Mark as finished and invoke finish callbacks"""
        for callback, (args, kwargs) in list(self._finish_callbacks.items()):
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)

    def register_finish_callback(self, callback: Callable, *args, **kwargs) -> None:
        """Register a callback to be called when finish() is invoked.

        Note: If the same callback is registered multiple times with different
        args/kwargs, only the most recent registration will be kept. This
        simplifies callback removal but means you cannot register the same
        callback with different arguments.
        """
        self._finish_callbacks[callback] = (args, kwargs)

    def unregister_finish_callback(self, callback: Callable) -> None:
        """Unregister a finish callback."""
        if callback in self._finish_callbacks:
            del self._finish_callbacks[callback]

    def get_status(self) -> ChartStatus:
        """Get current state status."""
        return self._status.get()
