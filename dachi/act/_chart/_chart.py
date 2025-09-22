from dachi.core import BaseModule
from enum import Enum, auto
from dataclasses import dataclass
import typing as t
from ._region import Region, RegionStatus
from ._event import Event, EventQueue, Envelope, Timer

from dataclasses import dataclass


class ChartLifecycle(Enum):
    IDLE = auto()       # constructed, never started
    RUNNING = auto()    # event loop task alive
    FINISHED = auto()   # reached final (all top-level regions final, none quiescing)
    STOPPED = auto()    # manually stopped before finish
    ERROR = auto()      # loop crashed


@dataclass
class ChartStatus:
    lifecycle: ChartLifecycle
    running: bool                # convenience: lifecycle == RUNNING
    finished: bool               # lifecycle == FINISHED
    started_at: float | None
    finished_at: float | None
    queue_size: int
    regions: list[dict]          # {name, current_state, is_final, quiescing, pending_target}


JSON = t.Union[t.Dict[str, t.Any], t.List[t.Any], str, int, float, bool, None]


class StateChart(BaseModule):
    # ----- Spec fields (serialized) -----
    name: str
    regions: t.List["Region"]
    checkpoint_policy: t.Literal["yield", "hard"] = "yield"
    queue_maxsize: int = 1024
    queue_overflow: t.Literal["drop_newest", "drop_oldest", "block"] = "drop_newest"
    emit_enforcement: t.Literal["none", "warn", "error"] = "warn"
    auto_finish: bool = True

    # ----- Runtime fields (non-serialized/internal) -----
    _lifecycle: "ChartLifecycle"
    _started_at: t.Optional[float]
    _finished_at: t.Optional[float]
    _queue: "EventQueue"
    _timer: "Timer"

    def __post_init__(self) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def join(self, timeout: t.Optional[float] = None) -> bool: ...

    def post(
        self,
        type_or_event: t.Union[str, "Event"],
        payload: JSON = None,
        *,
        scope: t.Literal["chart", "parent"] = "chart",
        port: t.Optional[str] = None,
    ) -> bool: ...

    def post_up(
        self,
        type_or_event: t.Union[str, "Event"],
        payload: JSON = None,
    ) -> bool: ...

    def is_running(self) -> bool: ...
    def is_finished(self) -> bool: ...
    def status(self) -> "ChartStatus": ...

    async def step(self, evt: t.Optional["Event"] = None) -> None: ...
    def active_states(self) -> t.Dict[str, str]: ...
    def queue_size(self) -> int: ...
    def list_timers(self) -> t.List[t.Dict[str, t.Any]]: ...



# @dataclass
# class Snapshot:
#     lifecycle: "ChartLifecycle"
#     started_at: Optional[float]
#     finished_at: Optional[float]
#     queue_items: List["Envelope"]
#     regions: List[Dict[str, Any]]      # per-region runtime flags (current, last, quiescing, pending_target, pending_reason)
#     timers: List[Dict[str, Any]]       # Timer.snapshot()
