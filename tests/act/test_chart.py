import asyncio
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dachi.act._chart._chart import StateChart, ChartStatus, ChartSnapshot
from dachi.act._chart._region import Region, RegionStatus, Rule
from dachi.act._chart._state import State, StreamState, FinalState, StateStatus
from dachi.act._chart._event import Event
from dachi.core import Scope


class IdleState(State):
    async def execute(self, post, **inputs):
        return None


class ActiveState(State):
    async def execute(self, post, **inputs):
        await post("WorkDone", {"result": "success"})
        return {"completed": True}


class SlowStreamState(StreamState):
    async def astream(self, post, **inputs):
        yield {"step": 1}
        await asyncio.sleep(0.05)
        yield {"step": 2}
        await asyncio.sleep(0.05)
        yield {"step": 3}


class DoneState(FinalState):
    pass


class TestChartLifecycle:

    def test_chart_initializes_in_idle(self):
        region = Region(name="main", initial="idle", rules=[])
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])

        assert chart._status.get() == ChartStatus.IDLE
        assert chart._started_at.get() is None
        assert chart._finished_at.get() is None

    @pytest.mark.asyncio
    async def test_start_sets_lifecycle_to_running(self):
        region = Region(name="main", initial="idle", rules=[])
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])
        await chart.start()

        assert chart._status.get() == ChartStatus.RUNNING
        assert chart._started_at.get() is not None

        await chart.stop()

    @pytest.mark.asyncio
    async def test_start_enters_initial_states_in_all_regions(self):
        region1 = Region(name="r1", initial="idle", rules=[])
        region1._states["idle"] = IdleState()

        region2 = Region(name="r2", initial="idle", rules=[])
        region2._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region1, region2])
        await chart.start()

        assert region1.current_state == "idle"
        assert region2.current_state == "idle"
        assert region1.status == RegionStatus.ACTIVE
        assert region2.status == RegionStatus.ACTIVE

        await chart.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_lifecycle_to_stopped(self):
        region = Region(name="main", initial="idle", rules=[])
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])
        await chart.start()
        await chart.stop()

        assert chart._status.get() == ChartStatus.STOPPED
        assert chart._finished_at.get() is not None

    @pytest.mark.asyncio
    async def test_stop_cancels_running_tasks(self):
        region = Region(name="main", initial="slow", rules=[])
        region._states["slow"] = SlowStreamState()

        chart = StateChart(name="test", regions=[region])
        await chart.start()
        await asyncio.sleep(0.02)

        await chart.stop()

        assert len(chart._region_tasks) == 0

    @pytest.mark.asyncio
    async def test_auto_finish_when_all_regions_final(self):
        region = Region(name="main", initial="idle", rules=[
            Rule(event_type="finish", target="done")
        ])
        region._states["idle"] = IdleState()
        region._states["done"] = DoneState()

        chart = StateChart(name="test", regions=[region], auto_finish=True)
        await chart.start()

        chart.post("finish")
        await asyncio.sleep(0.05)

        assert chart._status.get() == ChartStatus.FINISHED
        await chart.stop()

    @pytest.mark.asyncio
    async def test_join_waits_for_completion(self):
        region = Region(name="main", initial="idle", rules=[
            Rule(event_type="finish", target="done")
        ])
        region._states["idle"] = IdleState()
        region._states["done"] = DoneState()

        chart = StateChart(name="test", regions=[region], auto_finish=True)
        await chart.start()
        chart.post("finish")

        completed = await chart.join(timeout=1.0)

        assert completed is True
        await chart.stop()


class TestChartEventProcessing:

    def test_post_adds_event_to_queue(self):
        region = Region(name="main", initial="idle", rules=[])
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])

        result = chart.post("TestEvent", {"data": "value"})

        assert result is True
        assert chart.queue_size() == 1

    def test_post_returns_false_when_queue_full(self):
        region = Region(name="main", initial="idle", rules=[])
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region], queue_maxsize=2, queue_overflow="drop_newest")

        chart.post("Event1")
        chart.post("Event2")
        result = chart.post("Event3")

        assert result is False
        assert chart.queue_size() == 2

    @pytest.mark.asyncio
    async def test_step_processes_single_event(self):
        region = Region(name="main", initial="idle", rules=[
            Rule(event_type="go", target="active")
        ])
        region._states["idle"] = IdleState()
        region._states["active"] = ActiveState()

        chart = StateChart(name="test", regions=[region])
        await chart.start()

        chart.post("go")
        await chart.step()

        assert region.current_state == "active"

        await chart.stop()

    @pytest.mark.asyncio
    async def test_step_with_no_events_does_nothing(self):
        region = Region(name="main", initial="idle", rules=[])
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])
        await chart.start()

        initial_state = region.current_state
        await chart.step()

        assert region.current_state == initial_state

        await chart.stop()

    @pytest.mark.asyncio
    async def test_event_triggers_immediate_transition(self):
        region = Region(name="main", initial="idle", rules=[
            Rule(event_type="activate", target="active")
        ])
        region._states["idle"] = IdleState()
        region._states["active"] = ActiveState()

        chart = StateChart(name="test", regions=[region])
        await chart.start()

        await chart.step(Event(type="activate", ts=0.0))

        assert region.current_state == "active"

        await chart.stop()


class TestChartRegionCoordination:

    @pytest.mark.asyncio
    async def test_enter_state_creates_run_task(self):
        region = Region(name="main", initial="active", rules=[])
        region._states["active"] = ActiveState()

        chart = StateChart(name="test", regions=[region])
        await chart.start()

        task_key = "main:active"
        assert task_key in chart._region_tasks

        await chart.stop()

    @pytest.mark.asyncio
    async def test_transition_exits_current_enters_target(self):
        region = Region(name="main", initial="idle", rules=[
            Rule(event_type="go", target="active")
        ])
        region._states["idle"] = IdleState()
        active = ActiveState()
        region._states["active"] = active

        chart = StateChart(name="test", regions=[region])
        await chart.start()

        chart.post("go")
        await chart.step()
        await asyncio.sleep(0.01)

        assert region.current_state == "active"
        assert active.get_status() == StateStatus.RUNNING

        await chart.stop()

    @pytest.mark.asyncio
    async def test_multiple_regions_run_independently(self):
        region1 = Region(name="r1", initial="idle", rules=[
            Rule(event_type="go1", target="active")
        ])
        region1._states["idle"] = IdleState()
        region1._states["active"] = ActiveState()

        region2 = Region(name="r2", initial="idle", rules=[
            Rule(event_type="go2", target="active")
        ])
        region2._states["idle"] = IdleState()
        region2._states["active"] = ActiveState()

        chart = StateChart(name="test", regions=[region1, region2])
        await chart.start()

        chart.post("go1")
        await chart.step()

        assert region1.current_state == "active"
        assert region2.current_state == "idle"

        await chart.stop()


class TestChartPreemption:

    @pytest.mark.asyncio
    async def test_preempt_requests_termination(self):
        region = Region(name="main", initial="slow", rules=[
            Rule(event_type="stop", target="idle")
        ])
        slow = SlowStreamState()
        region._states["slow"] = slow
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])
        await chart.start()
        await asyncio.sleep(0.01)

        chart.post("stop")
        await chart.step()
        await asyncio.sleep(0.01)

        assert region.current_state == "idle"
        assert slow.get_status() in [StateStatus.PREEMPTED, StateStatus.COMPLETED]

        await chart.stop()

    @pytest.mark.asyncio
    async def test_preempt_waits_for_stream_checkpoint(self):
        region = Region(name="main", initial="slow", rules=[
            Rule(event_type="cancel", target="idle")
        ])
        region._states["slow"] = SlowStreamState()
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])
        await chart.start()
        await asyncio.sleep(0.03)

        chart.post("cancel")
        await chart.step()

        await asyncio.sleep(0.05)
        assert region.current_state == "idle"

        await chart.stop()


class TestChartStatus:

    def test_snapshot_returns_status(self):
        region = Region(name="main", initial="idle", rules=[])
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])
        snapshot = chart.snapshot()

        assert isinstance(snapshot, ChartSnapshot)
        assert snapshot.status == ChartStatus.IDLE
        assert snapshot.running is False
        assert snapshot.finished is False

    @pytest.mark.asyncio
    async def test_snapshot_includes_region_data(self):
        region = Region(name="main", initial="idle", rules=[])
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])
        await chart.start()

        snapshot = chart.snapshot()

        assert len(snapshot.regions) == 1
        assert snapshot.regions[0]["name"] == "main"
        assert snapshot.regions[0]["current_state"] == "idle"

        await chart.stop()

    def test_is_running_convenience_method(self):
        region = Region(name="main", initial="idle", rules=[])
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])

        assert chart.is_running() is False

    def test_is_finished_convenience_method(self):
        region = Region(name="main", initial="idle", rules=[])
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])

        assert chart.is_finished() is False

    def test_active_states_returns_region_map(self):
        region1 = Region(name="r1", initial="idle", rules=[])
        region1._states["idle"] = IdleState()

        region2 = Region(name="r2", initial="active", rules=[])
        region2._states["active"] = ActiveState()

        chart = StateChart(name="test", regions=[region1, region2])

        active = chart.active_states()

        assert active == {"r1": "idle", "r2": "active"}

    def test_queue_size_tracking(self):
        region = Region(name="main", initial="idle", rules=[])
        region._states["idle"] = IdleState()

        chart = StateChart(name="test", regions=[region])

        chart.post("Event1")
        chart.post("Event2")

        assert chart.queue_size() == 2
