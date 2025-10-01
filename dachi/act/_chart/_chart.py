from __future__ import annotations
import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum, auto
from dataclasses import dataclass

from dachi.core import BaseModule, Attr
from dachi.core._scope import Scope
from ._region import Region, RegionStatus
from ._event import Event, EventQueue, Timer, MonotonicClock, Post


class ChartStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    FINISHED = auto()
    STOPPED = auto()
    ERROR = auto()


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


class StateChart(BaseModule):
    name: str
    regions: List[Region]
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

        self._queue = EventQueue(maxsize=self.queue_maxsize, overflow=self.queue_overflow)
        self._clock = MonotonicClock()
        self._timer = Timer(queue=self._queue, clock=self._clock)

        self._scope = Scope(name=self.name)

        self._event_loop_task: Optional[asyncio.Task] = None
        self._region_tasks: Dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        if self._status.get() != ChartStatus.IDLE:
            raise RuntimeError(f"Cannot start chart in {self._status.get()} state")

        self._status.set(ChartStatus.RUNNING)
        self._started_at.set(self._clock.now())

        for region in self.regions:
            region._status.set(RegionStatus.ACTIVE)
            await self._enter_state(region, region.initial)

        self._event_loop_task = asyncio.create_task(self._event_loop())

    async def stop(self) -> None:
        if self._status.get() not in {ChartStatus.RUNNING, ChartStatus.ERROR}:
            return

        self._status.set(ChartStatus.STOPPED)
        self._finished_at.set(self._clock.now())

        if self._event_loop_task:
            self._event_loop_task.cancel()
            try:
                await self._event_loop_task
            except asyncio.CancelledError:
                pass

        for task in list(self._region_tasks.values()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._region_tasks.clear()

        for region in self.regions:
            try:
                state = region.states[region.current_state]
                state.exit()
            except (KeyError, IndexError):
                pass

    async def join(self, timeout: Optional[float] = None) -> bool:
        if not self._event_loop_task:
            return True

        try:
            await asyncio.wait_for(self._event_loop_task, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

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

    async def step(self, evt: Optional[Event] = None) -> None:
        if evt:
            self._queue.post_nowait(evt)

        if self._queue.empty():
            return

        event = self._queue.pop_nowait()

        for region in self.regions:
            if region.status == RegionStatus.PREEMPTING:
                continue

            decision = region.decide(event)

            if decision["type"] == "stay":
                continue
            elif decision["type"] == "immediate":
                await self._transition_region(region, decision["target"])
            elif decision["type"] == "preempt":
                await self._preempt_region(region, decision["target"])

    def active_states(self) -> Dict[str, str]:
        return {r.name: r.current_state for r in self.regions}

    def queue_size(self) -> int:
        return self._queue.size()

    def list_timers(self) -> List[Dict[str, Any]]:
        return self._timer.list()

    async def _event_loop(self) -> None:
        try:
            while self._status.get() == ChartStatus.RUNNING:
                if not self._queue.empty():
                    await self.step()
                else:
                    await asyncio.sleep(0.01)

                if self._check_all_final():
                    self._status.set(ChartStatus.FINISHED)
                    self._finished_at.set(self._clock.now())
                    break
        except Exception:
            self._status.set(ChartStatus.ERROR)
            raise

    def _check_all_final(self) -> bool:
        if not self.auto_finish:
            return False
        return all(r.is_final() for r in self.regions)

    async def _enter_state(self, region: Region, state_name: str) -> None:
        try:
            state = region.states[state_name]
        except (KeyError, IndexError):
            raise ValueError(f"State {state_name} not found in region {region.name}")

        region._current_state.set(state_name)
        state.enter()

        post = Post(
            queue=self._queue,
            source_region=region.name,
            source_state=state_name
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
            current_state = region.states[current_state_name]
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
            current_state = region.states[current_state_name]
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
