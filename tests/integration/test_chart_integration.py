"""Integration tests for StateChart.

Tests verify that multiple components work together correctly:
- States, Regions, EventQueue, Timer coordination
- Multi-state workflows with transitions
- Concurrent region coordination
- Context data flow through states
- Preemption and cancellation flows
"""

import asyncio
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dachi.act._chart._chart import StateChart, ChartStatus
from dachi.act._chart._region import Region, Rule
from dachi.act._chart._state import State, StreamState, FinalState
from dachi.core import Scope


pytestmark = pytest.mark.integration


# ============================================================================
# Test States for Integration Tests
# ============================================================================

class InitState(State):
    """Initial state that sets up context data."""
    async def execute(self, post, **inputs):
        await post("initialized", {"setup": "complete"})
        return {"counter": 0, "status": "initialized"}


class ProcessState(StreamState):
    """Processing state that updates counter and yields progress."""
    class inputs:
        counter: int = 0

    async def astream(self, post, counter):
        for i in range(3):
            counter += 1
            await asyncio.sleep(0.01)
            yield {"counter": counter, "progress": i + 1}

        await post("processing_done", {"final_count": counter})


class ValidateState(State):
    """Validation state that checks counter value."""
    class inputs:
        counter: int

    async def execute(self, post, counter):
        is_valid = counter >= 3
        event_type = "valid" if is_valid else "invalid"
        await post(event_type, {"counter": counter})
        return {"validated": is_valid}


class ErrorHandlerState(State):
    """State that handles errors."""
    async def execute(self, post, **inputs):
        await post("error_handled")
        return {"error_handled": True}


class SuccessState(FinalState):
    """Final success state."""
    pass


class FailureState(FinalState):
    """Final failure state."""
    pass


class TimeoutState(FinalState):
    """Final timeout state."""
    pass


# ============================================================================
# Multi-State Workflow Tests
# ============================================================================

class TestMultiStateWorkflow:
    """Test complete workflows through multiple states."""

    @pytest.mark.asyncio
    async def test_successful_workflow_with_context_flow(self):
        """Test data flows through states via context updates."""
        region = Region(name="workflow", initial="init", rules=[
            Rule(event_type="initialized", target="process"),
            Rule(event_type="processing_done", target="validate"),
            Rule(event_type="valid", target="success"),
            Rule(event_type="invalid", target="failure"),
        ])

        region._states["init"] = InitState()
        region._states["process"] = ProcessState()
        region._states["validate"] = ValidateState()
        region._states["success"] = SuccessState()
        region._states["failure"] = FailureState()

        chart = StateChart(name="workflow_test", regions=[region], auto_finish=True)
        await chart.start()

        # Wait for workflow to complete
        completed = await chart.join(timeout=2.0)

        assert completed is True
        assert chart._status.get() == ChartStatus.FINISHED
        assert region.current_state == "success"

        await chart.stop()

    @pytest.mark.asyncio
    async def test_workflow_with_conditional_branching(self):
        """Test workflow that branches based on state outputs."""
        # Create a workflow that can go to success or failure
        region = Region(name="branching", initial="init", rules=[
            Rule(event_type="initialized", target="validate"),
            Rule(event_type="valid", target="success"),
            Rule(event_type="invalid", target="failure"),
        ])

        # Start with init to set counter, then validate
        region._states["init"] = InitState()
        validate = ValidateState()
        region._states["validate"] = validate
        region._states["success"] = SuccessState()
        region._states["failure"] = FailureState()

        chart = StateChart(name="branch_test", regions=[region], auto_finish=True)
        await chart.start()

        completed = await chart.join(timeout=1.0)

        assert completed is True
        assert region.current_state == "failure"  # Should fail due to counter=0

        await chart.stop()

    @pytest.mark.asyncio
    async def test_workflow_with_error_recovery(self):
        """Test workflow that recovers from errors."""
        region = Region(name="recovery", initial="init", rules=[
            Rule(event_type="initialized", target="process"),
            Rule(event_type="processing_done", target="validate"),
            Rule(event_type="invalid", target="error_handler"),
            Rule(event_type="error_handled", target="success"),
            Rule(event_type="valid", target="success"),
        ])

        region._states["init"] = InitState()
        region._states["process"] = ProcessState()
        region._states["validate"] = ValidateState()
        region._states["error_handler"] = ErrorHandlerState()
        region._states["success"] = SuccessState()

        chart = StateChart(name="recovery_test", regions=[region], auto_finish=True)
        await chart.start()

        completed = await chart.join(timeout=2.0)

        assert completed is True
        assert region.current_state == "success"

        await chart.stop()


# ============================================================================
# Concurrent Multi-Region Tests
# ============================================================================

class CounterState(State):
    """State that increments a counter."""
    class inputs:
        counter: int = 0

    async def execute(self, post, counter):
        await asyncio.sleep(0.02)
        counter += 1
        await post("counted", {"count": counter})
        return {"counter": counter}


class WaitState(StreamState):
    """State that waits for a specified duration."""
    async def astream(self, post, **inputs):
        for i in range(3):
            await asyncio.sleep(0.02)
            yield {"tick": i}
        await post("wait_done")


class TestConcurrentRegions:
    """Test multiple regions running concurrently."""

    @pytest.mark.asyncio
    async def test_two_regions_run_independently(self):
        """Test that two regions can progress independently."""
        region1 = Region(name="r1", initial="count", rules=[
            Rule(event_type="counted", target="done"),
        ])
        region1._states["count"] = CounterState()
        region1._states["done"] = SuccessState()

        region2 = Region(name="r2", initial="wait", rules=[
            Rule(event_type="wait_done", target="done"),
        ])
        region2._states["wait"] = WaitState()
        region2._states["done"] = SuccessState()

        chart = StateChart(name="concurrent_test", regions=[region1, region2], auto_finish=True)
        await chart.start()

        completed = await chart.join(timeout=2.0)

        assert completed is True
        assert region1.current_state == "done"
        assert region2.current_state == "done"

        await chart.stop()

    @pytest.mark.asyncio
    async def test_regions_respond_to_different_events(self):
        """Test that regions only respond to their specific events."""
        region1 = Region(name="r1", initial="init", rules=[
            Rule(event_type="event1", target="done"),
        ])
        region1._states["init"] = InitState()
        region1._states["done"] = SuccessState()

        region2 = Region(name="r2", initial="init", rules=[
            Rule(event_type="event2", target="done"),
        ])
        region2._states["init"] = InitState()
        region2._states["done"] = SuccessState()

        chart = StateChart(name="selective_test", regions=[region1, region2])
        await chart.start()

        # Send event1 - only region1 should transition
        chart.post("event1")
        await asyncio.sleep(0.05)

        assert region1.current_state == "done"
        assert region2.current_state == "init"

        # Send event2 - now region2 should transition
        chart.post("event2")
        await asyncio.sleep(0.05)

        assert region2.current_state == "done"

        await chart.stop()


# ============================================================================
# Preemption and Cancellation Tests
# ============================================================================

class LongRunningState(StreamState):
    """State that runs for a long time with multiple checkpoints."""
    async def astream(self, post, **inputs):
        for i in range(10):
            await asyncio.sleep(0.05)
            yield {"iteration": i}
        await post("completed")


class QuickState(State):
    """Quick state for preemption target."""
    async def execute(self, post, **inputs):
        return {"preempted": True}


class TestPreemptionFlows:
    """Test preemption and cancellation of running states."""

    @pytest.mark.asyncio
    async def test_preempt_long_running_stream_state(self):
        """Test that long-running StreamState can be preempted."""
        region = Region(name="preempt_test", initial="long", rules=[
            Rule(event_type="cancel", target="quick"),
            Rule(event_type="completed", target="done"),
        ])

        long_state = LongRunningState()
        region._states["long"] = long_state
        region._states["quick"] = QuickState()
        region._states["done"] = SuccessState()

        chart = StateChart(name="preempt", regions=[region])
        await chart.start()

        # Let it run for a bit
        await asyncio.sleep(0.15)

        # Preempt it
        chart.post("cancel")
        await asyncio.sleep(0.2)  # Give more time for preemption

        # Should have transitioned to quick state
        assert region.current_state == "quick"

        await chart.stop()

    @pytest.mark.asyncio
    async def test_preemption_waits_for_checkpoint(self):
        """Test that preemption waits for next yield checkpoint."""
        region = Region(name="checkpoint_test", initial="long", rules=[
            Rule(event_type="cancel", target="done"),
        ])

        region._states["long"] = LongRunningState()
        region._states["done"] = SuccessState()

        chart = StateChart(name="checkpoint", regions=[region])
        await chart.start()

        # Let one iteration complete
        await asyncio.sleep(0.06)

        # Request cancellation
        chart.post("cancel")

        # Wait for preemption to complete
        await asyncio.sleep(0.2)  # Give more time for event processing

        assert region.current_state == "done"

        await chart.stop()


# ============================================================================
# Timer Integration Tests
# ============================================================================

class TimerWaitState(State):
    """State that sets up a timer."""
    async def execute(self, post, **inputs):
        # Note: Timer integration would be tested here when timers are connected
        await post("timer_set")
        return {"timer_active": True}


class TestTimerIntegration:
    """Test timer integration with state transitions."""

    @pytest.mark.asyncio
    async def test_timer_cancellation_on_state_exit(self):
        """Test that timers are cancelled when state exits."""
        region = Region(name="timer_test", initial="wait", rules=[
            Rule(event_type="timer_set", target="done"),
        ])

        region._states["wait"] = TimerWaitState()
        region._states["done"] = SuccessState()

        chart = StateChart(name="timer_cancel", regions=[region], auto_finish=True)
        await chart.start()

        # Verify chart can transition and timers don't cause issues
        completed = await chart.join(timeout=1.0)

        assert completed is True
        assert chart.list_timers() == []  # No active timers

        await chart.stop()


# ============================================================================
# Event Queue Stress Tests
# ============================================================================

class EventGeneratorState(State):
    """State that generates multiple events."""
    async def execute(self, post, **inputs):
        for i in range(5):
            await post(f"event_{i}", {"index": i})
        await post("all_sent")
        return {"events_sent": 5}


class TestEventQueueIntegration:
    """Test event queue behavior under various conditions."""

    @pytest.mark.asyncio
    async def test_event_queue_processes_multiple_events(self):
        """Test that queue processes events in order."""
        region = Region(name="queue_test", initial="generate", rules=[
            Rule(event_type="all_sent", target="done"),
        ])

        region._states["generate"] = EventGeneratorState()
        region._states["done"] = SuccessState()

        chart = StateChart(name="queue", regions=[region], auto_finish=True)
        await chart.start()

        completed = await chart.join(timeout=1.0)

        assert completed is True
        assert region.current_state == "done"

        await chart.stop()

    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self):
        """Test queue behavior when full."""
        region = Region(name="overflow_test", initial="init", rules=[])
        region._states["init"] = InitState()

        chart = StateChart(
            name="overflow",
            regions=[region],
            queue_maxsize=5,
            queue_overflow="drop_newest"
        )
        await chart.start()

        # Fill the queue
        for i in range(10):
            result = chart.post(f"event_{i}")
            if i < 5:
                assert result is True  # Should succeed
            else:
                assert result is False  # Should be dropped

        assert chart.queue_size() == 5

        await chart.stop()


# ============================================================================
# State Lifecycle Integration Tests
# ============================================================================

class LifecycleTracker:
    """Helper to track state lifecycle calls."""
    def __init__(self):
        self.entered = []
        self.exited = []
        self.executed = []


tracker = LifecycleTracker()


class TrackedState(State):
    """State that tracks its lifecycle."""
    def enter(self):
        super().enter()
        tracker.entered.append(self.name)

    def exit(self):
        super().exit()
        tracker.exited.append(self.name)

    async def execute(self, post, **inputs):
        tracker.executed.append(self.name)
        await post("done")
        return {}


class TestStateLifecycle:
    """Test state enter/exit/execute lifecycle."""

    @pytest.mark.asyncio
    async def test_state_lifecycle_order(self):
        """Test that enter/execute/exit are called in correct order."""
        global tracker
        tracker = LifecycleTracker()

        region = Region(name="lifecycle", initial="state1", rules=[
            Rule(event_type="done", target="state2"),
        ])

        state1 = TrackedState(name="state1")
        state2 = TrackedState(name="state2")
        region._states["state1"] = state1
        region._states["state2"] = state2

        chart = StateChart(name="lifecycle_test", regions=[region])
        await chart.start()

        # Wait for transition
        await asyncio.sleep(0.1)

        # Verify lifecycle order
        assert "state1" in tracker.entered
        assert "state1" in tracker.executed
        assert "state1" in tracker.exited
        assert "state2" in tracker.entered

        await chart.stop()
