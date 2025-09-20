from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Literal, TypedDict

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class Event(TypedDict, total=False):
    type: str
    payload: JSON
    correlation_id: Optional[str]
    port: Optional[str]
    meta: Dict[str, Any]


from __future__ import annotations
from typing import Any, Optional, Union, Literal

JSON = Union[dict, list, str, int, float, bool, None]

class Post:
    def __init__(
        self,
        queue: "EventQueue",
        *,
        source_region: Optional[str],
        source_state: Optional[str],
        epoch: Optional[int],
        quiescing: "Callable[[], bool]",
    ) -> None: ...

    async def post(
        self,
        type_or_event: Union[str, "Event"],
        payload: JSON = None,
        *,
        scope: Literal["chart", "parent"] = "chart",
        port: Optional[str] = None,
    ) -> bool: ...

    async def post_up(
        self,
        type_or_event: Union[str, "Event"],
        payload: JSON = None,
    ) -> bool: ...



@dataclass
class Envelope:
    id: int
    ts: float
    event: Event
    scope: Literal["chart", "parent", "self"]
    source_region: Optional[str]
    source_state: Optional[str]
    epoch: Optional[int]


class EventQueue:
    def __init__(
        self,
        maxsize: int = 1024,
        overflow: Literal["drop_newest", "drop_oldest", "block"] = "drop_newest",
    ) -> None: ...

    # Non-blocking enqueue. Returns False if dropped/rejected.
    def post_nowait(
        self,
        type_or_event: Union[str, Event],
        payload: JSON = None,
        *,
        scope: Literal["chart", "parent", "self"] = "chart",
        port: Optional[str] = None,
        source_region: Optional[str] = None,
        source_state: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> bool: ...

    # Blocking/awaiting enqueue (only meaningful if overflow="block").
    async def post(
        self,
        type_or_event: Union[str, Event],
        payload: JSON = None,
        *,
        scope: Literal["chart", "parent", "self"] = "chart",
        port: Optional[str] = None,
        source_region: Optional[str] = None,
        source_state: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> bool: ...

    # Dequeue next envelope (FIFO).
    async def get(self) -> Envelope: ...

    # ---- Introspection / testing / snapshots ----
    def qsize(self) -> int: ...
    def empty(self) -> bool: ...
    def capacity(self) -> int: ...
    def snapshot(self) -> List[Envelope]: ...
    def load_snapshot(self, items: List[Envelope]) -> None: ...
    def drain(self, limit: Optional[int] = None) -> List[Envelope]: ...
    def clear(self) -> None: ...

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal

class Timer:
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


from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RegionStatus:
    name: str
    current_state: str
    is_final: bool
    quiescing: bool
    pending_target: Optional[str]

