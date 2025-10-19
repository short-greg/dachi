from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Union, Literal
from ._base import ChartBase, ChartStatus
from dataclasses import dataclass

from dachi.core import Attr, ModuleList
from dachi.core._scope import Scope, Ctx
from ._event import Event, EventQueue, Timer, MonotonicClock, ChartEventHandler, EventPost
from ._region import Region, ValidationResult


@dataclass
class ChartSnapshot:
    status: ChartStatus
    running: bool
    finished: bool
    started_at: Optional[float]
    finished_at: Optional[float]
    queue_size: int
    regions: List[Dict[str, Any]]


JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class StateChart(ChartBase, ChartEventHandler):
    name: str
    regions: ModuleList  # ModuleList[Region]
    checkpoint_policy: Literal["yield", "hard"] = "yield" # currently not used
    queue_maxsize: int = 1024
    queue_overflow: Literal["drop_newest", "drop_oldest", "block"] = "drop_newest"
    emit_enforcement: Literal["none", "warn", "error"] = "warn" # currently not used
    auto_finish: bool = True # currently not used

    def __post_init__(self) -> None:
        """Initialize the state chart and its status.
        Raises:
            ValueError: If no regions are defined.
        """
        super().__post_init__()

        if isinstance(self.regions, list):
            self.regions = ModuleList(items=self.regions)

        self._status = Attr[ChartStatus](data=ChartStatus.WAITING)
        self._started_at = Attr[Optional[float]](data=None)
        self._finished_at = Attr[Optional[float]](data=None)
        self._stopping = Attr[bool](data=False)

        self._regions_completed = Attr[Dict[str, bool]](data={
            region.name: region.is_completed() for region in self.regions
        })
        self._queue = EventQueue(
            maxsize=self.queue_maxsize,
            overflow=self.queue_overflow
        )
        self._queue.register_callback(self._process_event_callback)
        self._clock = MonotonicClock()
        self._timer = Timer(queue=self._queue, clock=self._clock)
        self._scope = Scope(name=self.name)
        self._event_loop_task: Optional[asyncio.Task] = None

    def reset(self):
        """
        Reset the chart to its initial state.
        """
        if not self.can_reset():
            raise RuntimeError(f"Cannot reset chart in {self._status.get()} state")
        super().reset()
        
        self._status.set(ChartStatus.WAITING)
        self._started_at.set(None)
        self._finished_at.set(None)
        self._regions_completed.set({
            region.name: region.is_completed() for region in self.regions
        })
        self._queue.clear()
        self._timer.clear()
        for region in self.regions:
            region.reset()

    def can_reset(self):
        return self._status.get().is_completed()
    
    def can_start(self):
        return self._status.get().is_waiting()
    
    def can_stop(self):
        return self._status.get().is_running()

    def __getitem__(self, region_name: str) -> "Region":
        """Get region by name.

        Args:
            region_name: Name of the region to retrieve

        Returns:
            The region instance

        Raises:
            KeyError: If region not found
        """
        for region in self.regions:
            if region.name == region_name:
                return region
        raise KeyError(f"Region '{region_name}' not found in chart '{self.name}'")

    async def finish_region(self, region: str) -> None:
        """Handle completion of a region. This is a callback registered with each region.
        Check if all regions are completed and finish the chart if so.

        Args:
            region (str): The name of the region that finished
            post: Post object for this StateChart
            ctx: Ctx object for this StateChart
        Raises:
            ValueError: If the region is unknown
        """
        # get the region
        if region not in self._regions_completed.get():
            raise ValueError(f"Unknown region '{region}' finished")
        region_obj = None
        for r in self.regions:
            if r.name == region:
                region_obj = r
                break

        if region_obj is None:
            raise ValueError(f"Unknown region '{region}' finished")

        region_obj.unregister_finish_callback(self.finish_region)
        completed = self._regions_completed.get()
        completed[region] = True
        self._regions_completed.set(completed)
        
        if all(self._regions_completed.get().values()):
            self._finished_at.set(self._clock.now())
            if self._stopping.get():
                self._status.set(ChartStatus.CANCELED)
            else:
                self._status.set(ChartStatus.SUCCESS)
            await self.finish()

    async def start(self) -> None:
        """Start the state chart by starting all regions."""
        if not self.can_start():
            raise RuntimeError(f"Cannot start chart in {self._status.get()} state")

        self._status.set(ChartStatus.RUNNING)
        self._started_at.set(self._clock.now())

        for i, region in enumerate(self.regions):
            region.register_finish_callback(
                self.finish_region, 
                region.name,
            )
            post = self._queue.child(region.name)
            ctx = self._scope.ctx(i)
            await region.start(post, ctx)

    def _process_event_callback(self, event: Event) -> None:
        """Sync callback that schedules async event processing.

        Called when an event is posted to the queue. If there's a running event
        loop, pops the event and schedules it for async processing. Otherwise,
        leaves it in the queue to be processed later (e.g., when chart.start()
        is called in an async context).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop - defer processing until chart is started
            return

        event_to_process = self._queue.pop_nowait()
        loop.create_task(self.handle_event(event_to_process))

    async def handle_event(self, event: Event, post: Optional[EventPost]=None, ctx: Optional[Ctx]=None) -> None:
        """
        Handle an incoming event by dispatching it to all running regions.
        """
        # TODO: use 
        if self._status.get() == ChartStatus.RUNNING:
            for i, region in enumerate(self.regions):
                if region.status.is_running():
                    post = self._queue.child(region.name)
                    ctx = self._scope.ctx(i)
                    await region.handle_event(event, post, ctx)

    async def stop(self) -> None:
        """Stop the state chart by stopping all running regions.

        This initiates stopping but returns immediately. The chart will complete
        asynchronously via finish_region() callbacks.
        """
        if not self.can_stop():
            raise RuntimeError(
                f"Cannot stop chart in {self._status.get()} state"
            )

        self._stopping.set(True)
        for i, region in enumerate(self.regions):
            if region.can_stop():
                post = self._queue.child(region.name)
                ctx = self._scope.ctx(i)
                await region.stop(post, ctx, preempt=True)

    def post(
        self,
        type_or_event: Union[str, Event],
        payload: JSON = None,
        *,
        scope: Literal["chart", "parent"] = "chart",
        port: Optional[str] = None,
    ) -> bool:
        if isinstance(type_or_event, str):
            event = Event(
                type=type_or_event,
                payload=payload or {},
                scope=scope,
                port=port,
                ts=self._clock.now()
            )
        else:
            event = type_or_event

        return self._queue.post_nowait(event)

    def post_up(
        self,
        type_or_event: Union[str, Event],
        payload: JSON = None,
    ) -> bool:
        return self.post(type_or_event, payload, scope="parent")
    
    def snapshot(self) -> ChartSnapshot:
        status = self._status.get()
        return ChartSnapshot(
            status=status,
            running=status == ChartStatus.RUNNING,
            finished=status.is_completed(),
            started_at=self._started_at.get(),
            finished_at=self._finished_at.get(),
            queue_size=self._queue.size(),
            regions=[
                {
                    "name": r.name,
                    "current_state": r.current_state,
                    "status": r.status.value,
                    "pending_target": r._pending_target.get()
                }
                for r in self.regions
            ]
        )

    def active_states(self) -> Dict[str, str]:
        return {r.name: r.status.is_running() for r in self.regions}

    def queue_size(self) -> int:
        return self._queue.size()

    def list_timers(self) -> List[Dict[str, Any]]:
        return self._timer.list()

    def validate(self, raise_on_error: bool = True) -> List[ValidationResult]:
        """Validate all regions in the chart.

        Delegates validation to each region and optionally raises on first error.

        Args:
            raise_on_error: If True, raise RegionValidationError on first failure

        Returns:
            List of ValidationResult, one per region

        Raises:
            RegionValidationError: If raise_on_error=True and any region is invalid
        """
        results = []
        for region in self.regions:
            result = region.validate()
            results.append(result)
            if raise_on_error and not result.is_valid():
                result.raise_if_invalid()
        return results

# @dataclass
# class Snapshot:
#     lifecycle: "ChartLifecycle"
#     started_at: Optional[float]
#     finished_at: Optional[float]
#     queue_items: List["Envelope"]
#     regions: List[Dict[str, Any]]      # per-region runtime flags (current, last, quiescing, pending_target, pending_reason)
#     timers: List[Dict[str, Any]]       # Timer.snapshot()

# class ChartStatus(Enum):
#     IDLE = auto()
#     RUNNING = auto()
#     FINISHED = auto()
#     STOPPED = auto()
#     ERROR = auto()

