from __future__ import annotations

from typing import Any, Dict, List, Union, Optional, Callable, Tuple
from enum import Enum
import asyncio

from dachi.core import BaseModule
import typing as t


JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class ChartStatus(Enum):
    """State lifecycle status."""
    RUNNING = "running" 
    COMPLETED = "completed"
    PREEMPTED = "preempted"
    CANCELED = "canceled"
    WAITING = "waiting"
    IDLE = "idle"

    def is_running(self) -> bool:
        return self == ChartStatus.RUNNING
    
    def is_completed(self) -> bool:
        return self == ChartStatus.COMPLETED
    
    def is_canceled(self) -> bool:
        return self == ChartStatus.CANCELED
    
    def is_preempted(self) -> bool:
        return self == ChartStatus.PREEMPTED

    def is_idle(self) -> bool:
        return self == ChartStatus.IDLE


class StatusResult:
    """Result of a status check."""
    def __init__(self, status: ChartStatus, message: Optional[str] = None):
        self.status = status
        self.message = message
    

class ChartBase(BaseModule):
    """Base class for all state types."""
    name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = self.__class__.__name__
        self._finish_callbacks = t.Tuple[List[Tuple[Callable, tuple, dict]], None] = []
    
    async def finish(self) -> None:
        """Mark region as finished and invoke finish callbacks"""
        self._status.set(ChartStatus.COMPLETED)
        for callback, args, kwargs in self._finish_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)

    def register_finish_callback(self, callback: Callable, *args, **kwargs) -> None:
        """Register a callback to be called when finish() is invoked."""
        if callback not in self._finish_callbacks:
            self._finish_callbacks.append((callback, args, kwargs))

    def unregister_finish_callback(self, callback: Callable) -> None:
        """Unregister a finish callback."""
        if callback in self._finish_callbacks:
            self._finish_callbacks.remove(callback)

    def get_status(self) -> ChartStatus:
        """Get current state status."""
        return self._status.get()
