# 1st Party
from abc import ABC
from typing import Any, Dict, List, Optional, Literal, TypedDict, Tuple, Callable
from collections import deque
import time
import asyncio

# Local
from dachi.proc import AsyncProcess


class Payload(TypedDict, total=False):
    pass


class Event(TypedDict, total=False):
    type: str
    payload: Payload
    port: Optional[str]
    scope: Literal["chart", "parent", "self"] = "chart"
    source: List[Tuple[str, str]]  # List of (region_name, state_name) pairs
    epoch: Optional[int] = None
    meta: Dict[str, Any]
    ts: float

    # port: Optional[str] = None,
    # correlation_id: Optional[str]


class EventQueue:
    """Event queue with serializable state."""
    
    def __init__(self, maxsize: int = 1024, overflow: Literal["drop_newest", "drop_oldest", "block"] = "drop_newest"):
        self.maxsize = maxsize
        self.overflow = overflow
        self.queue: deque[Event] = deque()
        self._callbacks: Dict[Callable, Tuple[Any, Any]] = {}
        self._posts = {}

    # Non-blocking enqueue. Returns False if dropped/rejected.
    def post_nowait(
        self,
        event: Event | str,
    ) -> bool:
        """Add an event to the queue. Returns True if added, False if dropped."""
        if isinstance(event, str):
            event = Event(type=event, ts=time.monotonic())
        
        if len(self.queue) >= self.maxsize:
            if self.overflow == "drop_newest":
                return False
            elif self.overflow == "drop_oldest":
                self.queue.popleft()
            elif self.overflow == "block":
                return False
        
        self.queue.append(event)
        for callback, (args, kwargs) in self._callbacks.items():
            callback(event, *args, **kwargs)
        return True

    # Blocking/awaiting enqueue (only meaningful if overflow="block").
    async def post(
        self,
        event: Event | str
    ) -> bool:
        """Add an event to the queue. Returns True if added, False if dropped."""
        return self.post_nowait(event)

    def pop_nowait(self) -> Event:
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
        return self.maxsize

    def clear(self) -> None:
        """Clear all events from the queue."""
        self.queue.clear()
    
    def state_dict(self) -> Dict[str, Any]:
        """Return serializable state."""
        return {
            "maxsize": self.maxsize,
            "overflow": self.overflow,
            "events": list(self.queue)  # Convert deque to list
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from dict."""
        self.maxsize = state["maxsize"]
        self.overflow = state["overflow"]
        self.queue = deque(state["events"])

    def register_callback(self, callback: Any, *args, **kwargs) -> None:
        """Register a callback to be called when an event is posted.
        Note: This is a placeholder for actual event loop integration.
        """
        self._callbacks[callback] = (args, kwargs)
    
    def unregister_callback(self, callback: Any) -> None:
        """Unregister a previously registered callback."""
        if callback in self._callbacks:
            del self._callbacks[callback]

    def child(self, region_name: str) -> 'Post':
        """Create a Post object for this queue."""
        if region_name in self._posts:
            return self._posts[region_name]
        post = Post(queue=self).child(region_name)
        self._posts[region_name] = post
        return post
    
    def clear_children(self) -> None:
        """Clear all events from the queue."""
        self.queue.clear()


class Post(AsyncProcess):
    """Post an event to the event queue.
    """

    queue: "EventQueue"
    source: List[Tuple[str, str]] = []  # List of (region_name, state_name) pairs
    epoch: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.preempting = lambda: False

    async def aforward(
        self,
        event: str,
        payload: Optional[Payload] = None,
        *,
        scope: Literal["chart", "parent"] = "chart",
        port: Optional[str] = None,
    ) -> bool:

        result = self.queue.post_nowait({
            "type": event,
            "payload": payload or {},
            "scope": scope,
            "port": port,
            "source": self.source,
            "epoch": self.epoch,
            "meta": {},
            "ts": time.monotonic(),
        })
        return result

    def child(self, region_name: str) -> "Post":
        """Create a child Post with extended source hierarchy for a new region.

        Args:
            region_name: Name of the child region

        Returns:
            New Post with extended source list and shared queue
        """
        return Post(
            queue=self.queue,
            source=self.source + [(region_name, None)],
            epoch=self.epoch
        )

    def sibling(self, state_name: str) -> "Post":
        """Create a sibling Post by setting the state name in the last source tuple.

        Args:
            state_name: Name of the state

        Returns:
            New Post with updated state in the last source tuple
        """
        if not self.source:
            raise ValueError("Cannot add state without a region in the source")
        region_name, _ = self.source[-1]
        return Post(
            queue=self.queue,
            source=self.source[:-1] + [(region_name, state_name)],
            epoch=self.epoch
        )

    async def __call__(self, *args, **kwds):
        return await self.aforward(*args, **kwds)


class Timer:
    """Runtime timer manager with serializable metadata."""
    
    def __init__(self, queue: "EventQueue", clock: "MonotonicClock"):
        self.queue = queue
        self.clock = clock
        self._timers: Dict[str, Dict[str, Any]] = {}
        self._next_id = 0

    def start(
        self,
        tag: str,
        delay: float,
        *,
        owner_region: Optional[str],
        owner_state: Optional[str],
        scope: Literal["chart", "parent"] = "chart",
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        timer_id = f"timer_{self._next_id}"
        self._next_id += 1

        when = self.clock.now() + delay

        async def _fire():
            await self.clock.sleep_until(when)
            if timer_id in self._timers:
                event = Event(
                    type="Timer",
                    payload={"tag": tag, "timer_id": timer_id, **(payload or {})},
                    scope=scope,
                    source=[(owner_region, owner_state)] if owner_region and owner_state else [],
                    ts=self.clock.now()
                )
                await self.queue.post(event)
                del self._timers[timer_id]

        self._timers[timer_id] = {
            "tag": tag,
            "when": when,
            "owner_region": owner_region,
            "owner_state": owner_state,
            "task": asyncio.create_task(_fire())
        }

        return timer_id

    def cancel(self, timer_id: str) -> bool:
        timer_info = self._timers.pop(timer_id, None)
        if timer_info:
            timer_info["task"].cancel()
            return True
        return False

    def cancel_owned(self, owner_region: str, owner_state: str) -> int:
        to_cancel = [
            tid for tid, info in self._timers.items()
            if info["owner_region"] == owner_region and info["owner_state"] == owner_state
        ]
        return sum(1 for tid in to_cancel if self.cancel(tid))

    def clear(self) -> None:
        """Cancel all active timers."""
        for timer_id in list(self._timers.keys()):
            self.cancel(timer_id)

    def list(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": timer_id,
                "tag": info["tag"],
                "remaining": max(0, info["when"] - self.clock.now())
            }
            for timer_id, info in self._timers.items()
        ]

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable timer metadata (no active tasks)."""
        return {
            "next_id": self._next_id,
            "timer_metadata": {
                tid: {k: v for k, v in info.items() if k != "task"}
                for tid, info in self._timers.items()
            }
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load timer metadata (tasks will need to be restarted)."""
        self._next_id = state["next_id"]
        # Note: Active tasks are not restored - they're runtime-only
        self._timers = {}  # Empty - timers need manual restart
    
    def snapshot(self) -> List[Dict[str, Any]]:
        """Legacy method - use state_dict() instead."""
        return []

    def restore(self, items: List[Dict[str, Any]]) -> None:
        """Legacy method - use load_state_dict() instead."""
        pass


class MonotonicClock:

    def now(self) -> float:
        return time.monotonic()
    
    async def sleep_until(self, when: float) -> None:
        delay = when - self.now()
        if delay > 0:
            await asyncio.sleep(delay)
