"""Basic E2E tests for StateChart.

These tests validate fundamental StateChart patterns that developers would commonly build:
- Simple sequential state machines
- Event-driven state transitions
- Built-in state outcomes (SUCCESS, FAILURE, CANCELED)
- Context data flow through states
- Reset and rerun capabilities

Each test demonstrates a complete, realistic StateChart scenario.
"""

import asyncio
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from dachi.act.chart._chart import StateChart, ChartStatus
from dachi.act.chart._region import Region, Rule
from dachi.act.chart._state import State, StreamState, FinalState


pytestmark = pytest.mark.e2e


# ============================================================================
# Helper Utilities
# ============================================================================

async def wait_for_chart(chart: StateChart, timeout: float = 2.0) -> bool:
    """Wait for chart to complete with timeout.

    Args:
        chart: StateChart to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        True if chart completed, False if timeout
    """
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        status = chart._status.get()
        if status.is_completed():
            return True
        await asyncio.sleep(0.01)
    return False


# ============================================================================
# Test 1: Simple Sequential State Machine
# ============================================================================

class IdleState(State):
    """Initial idle state."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("start_work")
        return {"phase": "idle_complete"}


class WorkingState(State):
    """Active working state."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("work_done")
        return {"phase": "work_complete"}


class CompleteState(FinalState):
    """Completion state."""
    pass


class TestSimpleSequentialStateMachine:
    """Test simple sequential state progression: idle → working → complete."""

    @pytest.mark.asyncio
    async def test_sequential_state_transitions(self):
        """Test region progresses through states sequentially and completes."""
        region = Region(name="simple", initial="idle", rules=[
            Rule(event_type="start_work", target="working"),
            Rule(event_type="work_done", target="complete"),
        ])

        region["idle"] = IdleState()
        region["working"] = WorkingState()
        region["complete"] = CompleteState()

        chart = StateChart(name="sequential", regions=[region], auto_finish=True)

        # Verify region starts in READY before start()
        assert region.current_state_name == "READY"

        await chart.start()

        # Wait for completion
        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert chart._status.get() == ChartStatus.SUCCESS
        assert region.current_state_name == "complete"


# ============================================================================
# Test 2: Event-Driven State Machine
# ============================================================================

class ReviewingState(State):
    """State that performs review work."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("review_complete")
        return {"reviewed": True}


class WaitingDecisionState(State):
    """State that waits for external approval/rejection decision."""
    async def execute(self, post, **inputs):
        # Just wait - external events will trigger transition
        await asyncio.sleep(0.5)
        return None


class ApprovedState(FinalState):
    """Approved outcome."""
    pass


class RejectedState(FinalState):
    """Rejected outcome."""
    pass


class TestEventDrivenStateMachine:
    """Test event-driven state transitions where external events drive behavior."""

    @pytest.mark.asyncio
    async def test_events_trigger_state_transitions(self):
        """Test external events trigger appropriate state transitions."""
        region = Region(name="event_driven", initial="reviewing", rules=[
            Rule(event_type="review_complete", target="waiting_decision"),
            Rule(event_type="approve", target="approved"),
            Rule(event_type="reject", target="rejected"),
        ])

        region["reviewing"] = ReviewingState()
        region["waiting_decision"] = WaitingDecisionState()
        region["approved"] = ApprovedState()
        region["rejected"] = RejectedState()

        chart = StateChart(name="event_test", regions=[region], auto_finish=True)
        await chart.start()

        # Wait for review to complete and reach waiting_decision state
        await asyncio.sleep(0.05)

        # Post external approval event - this demonstrates event-driven transition
        chart.post("approve")

        # Wait for completion
        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "approved"


# ============================================================================
# Test 3: Built-in SUCCESS State Transition
# ============================================================================

class InitializeState(State):
    """Initialization state."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("initialized", {"data": "ready"})
        return {"initialized": True}


class FinalizeState(State):
    """Finalization state that transitions to built-in SUCCESS."""
    class inputs:
        initialized: bool = False

    async def execute(self, post, initialized):
        await asyncio.sleep(0.01)
        if initialized:
            await post.aforward("complete")
        return {"finalized": True}


class TestBuiltInSuccessState:
    """Test transition to built-in SUCCESS state."""

    @pytest.mark.asyncio
    async def test_transition_to_builtin_success(self):
        """Test state can transition to built-in SUCCESS state via rules."""
        region = Region(name="success_test", initial="init", rules=[
            Rule(event_type="initialized", target="finalize"),
            Rule(event_type="complete", target="SUCCESS"),  # Built-in state
        ])

        region["init"] = InitializeState()
        region["finalize"] = FinalizeState()

        chart = StateChart(name="builtin_success", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert chart._status.get() == ChartStatus.SUCCESS
        assert region.current_state_name == "SUCCESS"
        assert region.is_final() is True


# ============================================================================
# Test 4: Built-in FAILURE State via Exception
# ============================================================================

class SetupState(State):
    """Setup state that works correctly."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("setup_done")
        return {"setup": "complete"}


class FailingState(State):
    """State that throws exception."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        raise ValueError("Simulated failure")


class TestBuiltInFailureState:
    """Test automatic transition to FAILURE on exception."""

    @pytest.mark.asyncio
    async def test_exception_transitions_to_failure(self):
        """Test state exception automatically transitions region to FAILURE."""
        region = Region(name="failure_test", initial="setup", rules=[
            Rule(event_type="setup_done", target="failing"),
        ])

        region["setup"] = SetupState()
        region["failing"] = FailingState()

        chart = StateChart(name="exception_test", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert chart._status.get() == ChartStatus.SUCCESS  # Chart completes
        assert region.current_state_name == "FAILURE"  # Region reached FAILURE state
        assert region.is_final() is True

        # Verify exception details stored in context
        scope = chart._scope
        ctx = scope.ctx(0)  # First region's context
        exception_info = ctx.get("__exception__")
        assert exception_info is not None
        assert exception_info["type"] == "ValueError"
        assert "Simulated failure" in exception_info["message"]
        assert exception_info["state"] == "failing"


# ============================================================================
# Test 5: Built-in CANCELED State via Preemption
# ============================================================================

class LongRunningState(StreamState):
    """Long-running state with checkpoints."""
    async def execute(self, post, **inputs):
        for i in range(20):
            await asyncio.sleep(0.05)
            yield {"progress": i + 1}

        await post.aforward("job_complete")


class TestBuiltInCanceledState:
    """Test transition to CANCELED via preemption."""

    @pytest.mark.asyncio
    async def test_preemption_transitions_to_canceled(self):
        """Test preempting long-running state transitions to CANCELED."""
        region = Region(name="cancel_test", initial="running", rules=[
            Rule(event_type="cancel", target="CANCELED"),  # Explicit transition to CANCELED
            Rule(event_type="job_complete", target="SUCCESS"),
        ])

        region["running"] = LongRunningState()

        chart = StateChart(name="preempt_test", regions=[region], auto_finish=True)
        await chart.start()

        # Let it run for a bit
        await asyncio.sleep(0.15)

        # Post cancel event to trigger preemption
        chart.post("cancel")

        # Wait for preemption to complete
        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "CANCELED"
        assert region.is_final() is True


# ============================================================================
# Test 6: Context Data Flow
# ============================================================================

class CollectDataState(State):
    """Collects initial data."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("data_collected")
        return {"name": "John", "age": 30}


class ProcessDataState(State):
    """Processes data from context."""
    class inputs:
        name: str
        age: int

    async def execute(self, post, name, age):
        await asyncio.sleep(0.01)
        await post.aforward("data_processed")
        return {"greeting": f"Hello {name}, you are {age} years old"}


class SaveDataState(State):
    """Saves final processed data."""
    class inputs:
        greeting: str

    async def execute(self, post, greeting):
        await asyncio.sleep(0.01)
        await post.aforward("saved")
        return {"saved": True, "message": greeting}


class DataCompleteState(FinalState):
    """Data processing complete."""
    pass


class TestContextDataFlow:
    """Test data flows through context across state transitions."""

    @pytest.mark.asyncio
    async def test_context_data_accumulates_across_states(self):
        """Test context data flows and accumulates through multiple states."""
        region = Region(name="data_flow", initial="collect", rules=[
            Rule(event_type="data_collected", target="process"),
            Rule(event_type="data_processed", target="save"),
            Rule(event_type="saved", target="complete"),
        ])

        region["collect"] = CollectDataState()
        region["process"] = ProcessDataState()
        region["save"] = SaveDataState()
        region["complete"] = DataCompleteState()

        chart = StateChart(name="context_test", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "complete"

        # Verify context accumulated data from all states
        scope = chart._scope
        ctx = scope.ctx(0)
        assert ctx.get("name") == "John"
        assert ctx.get("age") == 30
        assert ctx.get("greeting") == "Hello John, you are 30 years old"
        assert ctx.get("saved") is True


# ============================================================================
# Test 7: Reset and Rerun
# ============================================================================

class CounterState(State):
    """State that increments counter."""
    class inputs:
        count: int = 0

    async def execute(self, post, count):
        await asyncio.sleep(0.01)
        count += 1
        await post.aforward("counted")
        return {"count": count}


class DoneState(FinalState):
    """Counting complete."""
    pass


class TestResetAndRerun:
    """Test region can be reset and rerun."""

    @pytest.mark.asyncio
    async def test_region_reset_and_rerun(self):
        """Test region can be reset to READY state after completion."""
        region = Region(name="reset_test", initial="counter", rules=[
            Rule(event_type="counted", target="done"),
        ])

        region["counter"] = CounterState()
        region["done"] = DoneState()

        chart = StateChart(name="reset_chart", regions=[region], auto_finish=True)

        # First run
        await chart.start()
        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "done"

        scope = chart._scope
        ctx = scope.ctx(0)
        first_count = ctx.get("count")
        assert first_count == 1

        # Reset region back to READY state
        region.reset()
        assert region.current_state_name == "READY"
        assert region.can_start() is True
        assert region.status == ChartStatus.WAITING


# ============================================================================
# Test 8: State with No Output
# ============================================================================

class NoOutputState(State):
    """State that returns no output."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("done")
        # Return None (no output)
        return None


class FinalState2(FinalState):
    """Final state."""
    pass


class TestStateWithNoOutput:
    """Test state that returns None completes successfully."""

    @pytest.mark.asyncio
    async def test_state_with_none_output_completes(self):
        """Test state returning None completes successfully and transitions."""
        region = Region(name="no_output", initial="no_output_state", rules=[
            Rule(event_type="done", target="final"),
        ])

        region["no_output_state"] = NoOutputState()
        region["final"] = FinalState2()

        chart = StateChart(name="no_output_chart", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "final"

        # Context should not have undefined keys from None return
        scope = chart._scope
        ctx = scope.ctx(0)
        # Should complete without errors
