from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Union, Literal
from ._base import ChartBase, ChartStatus
from dataclasses import dataclass

from dachi.core import Attr, ModuleList
from dachi.core._scope import Scope
from ._region import Region
from ._event import Event, EventQueue, Timer, MonotonicClock, Post


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


class StateChart(ChartBase):
    name: str
    regions: ModuleList
    checkpoint_policy: Literal["yield", "hard"] = "yield"
    queue_maxsize: int = 1024
    queue_overflow: Literal["drop_newest", "drop_oldest", "block"] = "drop_newest"
    emit_enforcement: Literal["none", "warn", "error"] = "warn"
    auto_finish: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        self._status = Attr[ChartStatus](data=ChartStatus.IDLE)
        self._started_at = Attr[Optional[float]](data=None)
        self._finished_at = Attr[Optional[float]](data=None)

        self._modules_completed = {
            region.name: region.is_completed() for region in self.regions
        }
        self._queue = EventQueue(
            maxsize=self.queue_maxsize, 
            overflow=self.queue_overflow
        )
        self._queue.register_callback(self.handle_event)
        self._clock = MonotonicClock()
        self._timer = Timer(queue=self._queue, clock=self._clock)
        self._scope = Scope(name=self.name)
        self._event_loop_task: Optional[asyncio.Task] = None
        self._region_tasks: Dict[str, asyncio.Task] = {}
        self._post = Post(queue=self._queue, source=(self.name, None))

    async def finish_region(self, region: str) -> None:
        """Handle completion of a region's task."""
        # get the region
        if region not in self._regions_completed:
            # TODO: Decide whether to raise error or just log
            return 
        region_obj = None
        for r in self.regions:
            if r.name == region:
                region_obj = r
                break
        
        if region_obj is None:
            # TODO: Decide whether to raise error or just log
            return
        region_obj.unregister_finish_callback(self.finish_region)
        self._regions_completed[region] = True
        if all(self._modules_completed.values()):
            self._status.set(ChartStatus.COMPLETED)
            self._region_tasks = {}
            await self.finish()

    async def start(self) -> None:
        if self._status.get() != ChartStatus.IDLE:
            raise RuntimeError(f"Cannot start chart in {self._status.get()} state")

        self._status.set(ChartStatus.RUNNING)
        self._started_at.set(self._clock.now())

        self._region_tasks = {}
        for i, region in enumerate(self.regions):
            region.register_finish_callback(self.finish_region, region.name)
            task = asyncio.create_task(region.start(self._post.child(region.name), self._scope.child(i)))
            self._region_tasks[region.name] = task

    async def handle_event(self, event: Event) -> None:
        if self._status.get() == ChartStatus.RUNNING:
            for region in self.regions:
                if region.status != RegionStatus.FINAL:
                    await region.handle_event(event)

    async def stop(self) -> None:
        if self._status.get() not in {ChartStatus.RUNNING, ChartStatus.ERROR}:
            return

        for region in self.regions:
            if region.status == RegionStatus.ACTIVE:
                await region.stop()
        
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

    def is_running(self) -> bool:
        return self._status.get() == ChartStatus.RUNNING

    def is_finished(self) -> bool:
        return self._status.get() == ChartStatus.FINISHED

    def snapshot(self) -> ChartSnapshot:
        status = self._status.get()
        return ChartSnapshot(
            status=status,
            running=status == ChartStatus.RUNNING,
            finished=status == ChartStatus.FINISHED,
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
        return {r.name: r.current_state for r in self.regions}

    def queue_size(self) -> int:
        return self._queue.size()

    def list_timers(self) -> List[Dict[str, Any]]:
        return self._timer.list()

    def _check_all_final(self) -> bool:
        if not self.auto_finish:
            return False
        return all(r.is_final() for r in self.regions)

    async def _enter_state(self, region: Region, state_name: str) -> None:
        try:
            state = region._states[state_name]
        except (KeyError, IndexError):
            raise ValueError(f"State {state_name} not found in region {region.name}")

        region._current_state.set(state_name)
        state.enter()

        post = Post(
            queue=self._queue,
            source=(region.name, state_name)
        )
        ctx = self._scope.ctx()

        task = asyncio.create_task(state.run(post, ctx))
        task_key = f"{region.name}:{state_name}"
        self._region_tasks[task_key] = task

        if state.is_final():
            region._status.set(RegionStatus.FINAL)

    async def _transition_region(self, region: Region, target_state: str) -> None:
        current_state_name = region.current_state
        try:
            current_state = region._states[current_state_name]
        except (KeyError, IndexError):
            current_state = None

        if current_state:
            task_key = f"{region.name}:{current_state_name}"
            if task_key in self._region_tasks:
                task = self._region_tasks[task_key]
                if not task.done():
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                del self._region_tasks[task_key]

            current_state.exit()
            self._timer.cancel_owned(region.name, current_state_name)

        region._last_active_state.set(current_state_name)

        await self._enter_state(region, target_state)

    async def _preempt_region(self, region: Region, target_state: str) -> None:
        region._status.set(RegionStatus.PREEMPTING)
        region._pending_target.set(target_state)

        current_state_name = region.current_state
        try:
            current_state = region._states[current_state_name]
        except (KeyError, IndexError):
            current_state = None

        if current_state:
            current_state.request_termination()

            task_key = f"{region.name}:{current_state_name}"
            if task_key in self._region_tasks:
                task = self._region_tasks[task_key]
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self._region_tasks[task_key]

            current_state.exit()
            self._timer.cancel_owned(region.name, current_state_name)

        region._status.set(RegionStatus.ACTIVE)
        region._last_active_state.set(current_state_name)
        region._pending_target.set(None)

        await self._enter_state(region, target_state)


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

