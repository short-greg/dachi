# 1st Party
from dataclasses import dataclass
from abc import ABC
from typing import Any, Dict, List, Optional, Union, Literal, TypedDict, Callable
import typing as t
from collections import deque
from dachi.proc import AsyncProcess


class Payload(TypedDict, ABC, total=False):
    pass


class Event(TypedDict, total=False):
    type: str
    payload: Payload
    port: Optional[str]
    scope: Literal["chart", "parent", "self"] = "chart"
    source_region: Optional[str] = None
    source_state: Optional[str] = None
    epoch: Optional[int] = None
    meta: Dict[str, Any]
    ts: float
    
    # port: Optional[str] = None,
    # correlation_id: Optional[str]


class EventQueue:
    """
    A queue for events, supporting various enqueue and dequeue strategies.

    """

    def __init__(
        self,
        maxsize: int = 1024,
        overflow: Literal["drop_newest", "drop_oldest", "block"] = "drop_newest",
    ):
        self.queue: deque[Event] = deque(maxlen=maxsize)
        self.overflow = overflow

    # Non-blocking enqueue. Returns False if dropped/rejected.
    def post_nowait(
        self,
        event: Event | str,
    ) -> bool:
        """Add an event to the queue. Returns True if added, False if dropped."""
        if isinstance(event, str):
            event = Event(type=event)
        if len(self.queue) >= self.queue.maxlen:
            if self.overflow == "drop_newest":
                return False
            elif self.overflow == "drop_oldest":
                self.queue.popleft()
            elif self.overflow == "block":
                return False
        
        self.queue.append(event)
        return True

    # Blocking/awaiting enqueue (only meaningful if overflow="block").
    async def post(
        self,
        event: Event | str
    ) -> bool:
        """Add an event to the queue. Returns True if added, False if dropped."""
        return self.post_nowait(event)

    # Dequeue next envelope (FIFO).
    async def pop(self) -> Event:
        """Remove and return the next event from the queue. Raises IndexError if empty."""
        if not self.queue:
            raise IndexError("pop from an empty queue")
        return self.queue.popleft()

    def size(self) -> int:
        """Return the number of events in the queue."""
        return len(self.queue)
    
    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self.queue) == 0
    
    def capacity(self) -> int:
        """Return the maximum size of the queue."""
        return self.queue.maxlen

    def clear(self) -> None:
        """Clear all events from the queue."""
        self.queue.clear()


class Post:
    """Post an event to the event queue.
    """

    def __init__(
        self,
        queue: "EventQueue",
        *,
        source_region: Optional[str],
        source_state: Optional[str],
        epoch: Optional[int],
        quiescing: Callable[[], bool],
    ) -> None:
        self.queue = queue
        self.source_region = source_region
        self.source_state = source_state
        self.epoch = epoch
        self.quiescing = quiescing

    async def aforward(
        self,
        event: str,
        payload: Optional[Payload] = None,
        *,
        scope: Literal["chart", "parent"] = "chart",
        port: Optional[str] = None,
    ) -> bool:
        
        self.queue.post({
            "type": event,
            "payload": payload or {},
            "scope": scope,
            "port": port,
            "source_region": self.source_region,
            "source_state": self.source_state,
            "epoch": self.epoch,
            "meta": {},
            "ts": 0.0,  # TODO: timestamp
        })

    async def __call__(self, *args, **kwds):
        return await self.aforward(*args, **kwds)



class Timer:
    """A timer for scheduling events with delays.
    """
    def __init__(self, queue: "EventQueue", clock: "MonotonicClock") -> None: ...

    def start(
        self,
        tag: str,
        delay: float,
        *,
        owner_region: Optional[str],
        owner_state: Optional[str],
        scope: Literal["chart", "parent"] = "chart",
        payload: Optional[Dict[str, Any]] = None,
    ) -> str: ...

    def cancel(self, timer_id: str) -> bool: ...
    def cancel_owned(self, owner_region: str, owner_state: str) -> int: ...
    def list(self) -> List[Dict[str, Any]]: ...
    def snapshot(self) -> List[Dict[str, Any]]: ...
    def restore(self, items: List[Dict[str, Any]]) -> None: ...


class MonotonicClock:
    def now(self) -> float: ...
    async def sleep_until(self, when: float) -> None: ...

