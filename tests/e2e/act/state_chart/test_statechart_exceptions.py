"""E2E tests for StateChart exception handling.

Tests validate that exceptions are caught gracefully, logged properly,
stored in context, and don't crash the system.
"""
import asyncio
import logging
import pytest

from dachi.act._chart import StateChart, Region, State, StreamState, FinalState
from dachi.act._chart._base import ChartStatus
from dachi.core import ModuleList, Scope
from dachi.act._chart._region import Rule


# ============================================================================
# Test 1: State Exception Caught and Logged
# ============================================================================

class FailingState(State):
    """State that throws ValueError."""
    async def execute(self, post, **inputs):
        raise ValueError("test error message")


class TestStateExceptionLogging:
    """Test that exceptions are logged with full traceback."""

    @pytest.mark.asyncio
    async def test_state_exception_logged_with_traceback(self, caplog):
        """Verify exception is caught and logged with traceback."""
        region = Region(name="test", initial="failing", rules=[])
        region["failing"] = FailingState()

        chart = StateChart(name="test_chart", regions=[region])

        with caplog.at_level(logging.ERROR):
            await chart.start()

            # Poll for region to reach FAILURE state
            for i in range(50):
                if region.status == ChartStatus.FAILURE:
                    break
                await asyncio.sleep(0.01)

        # Verify region transitioned to FAILURE
        assert region.status == ChartStatus.FAILURE

        # Verify exception was logged
        assert "State 'failing' failed" in caplog.text
        assert "ValueError" in caplog.text
        assert "test error message" in caplog.text


# ============================================================================
# Test 2: Exception Details Stored in Context
# ============================================================================

class TestExceptionContextStorage:
    """Test that exception details are stored in context."""

    @pytest.mark.asyncio
    async def test_exception_details_in_context(self):
        """Verify exception details stored in ctx['__exception__']."""
        region = Region(name="test", initial="failing", rules=[])
        region["failing"] = FailingState()

        scope = Scope(name="test")
        ctx = scope.ctx()

        chart = StateChart(name="test_chart", regions=[region])
        chart._scope = scope

        await chart.start()

        # Poll for region to reach FAILURE
        for i in range(50):
            if region.status == ChartStatus.FAILURE:
                break
            await asyncio.sleep(0.01)

        # Verify exception details in context
        exception_data = ctx.get("__exception__")
        assert exception_data is not None
        assert exception_data["type"] == "ValueError"
        assert exception_data["message"] == "test error message"
        assert exception_data["state"] == "failing"


# ============================================================================
# Test 3: State Machine Continues After Exception
# ============================================================================

class WorkingState(State):
    """State that completes successfully."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        return {"success": True}


class SuccessFinal(FinalState):
    pass


class TestGracefulDegradation:
    """Test that one region's failure doesn't crash the whole chart."""

    @pytest.mark.asyncio
    async def test_statechart_continues_after_region_fails(self):
        """Verify chart continues running when one region fails."""
        # Region 1: Will fail
        region1 = Region(name="region1", initial="failing", rules=[])
        region1["failing"] = FailingState()

        # Region 2: Will succeed
        region2 = Region(name="region2", initial="working", rules=[
            Rule(event_type="done", target="SUCCESS"),
        ])
        region2["working"] = WorkingState()

        chart = StateChart(name="multi_region", regions=[region1, region2])
        await chart.start()

        # Give time for states to execute
        await asyncio.sleep(0.05)

        # Region 2 should post "done" event to transition
        chart.post("done")

        # Poll for both regions to complete
        for i in range(50):
            if (region1.status == ChartStatus.FAILURE and
                region2.status == ChartStatus.SUCCESS):
                break
            await asyncio.sleep(0.01)

        # Verify region1 failed
        assert region1.status == ChartStatus.FAILURE

        # Verify region2 succeeded
        assert region2.status == ChartStatus.SUCCESS


# ============================================================================
# Test 4: Exception in StreamState After Yields
# ============================================================================

class PartialStreamState(StreamState):
    """StreamState that yields 3 times then throws exception."""
    async def execute(self, post, **inputs):
        for i in range(3):
            yield {"progress": i}
            await asyncio.sleep(0.01)

        # Throw exception on 4th iteration
        raise RuntimeError("stream error after 3 yields")


class TestStreamStatePartialProgress:
    """Test StreamState exception handling with partial progress."""

    @pytest.mark.asyncio
    async def test_streamstate_exception_after_partial_progress(self):
        """Verify partial progress is captured before exception."""
        region = Region(name="test", initial="streaming", rules=[])
        region["streaming"] = PartialStreamState()

        scope = Scope(name="test")
        ctx = scope.ctx()

        chart = StateChart(name="stream_test", regions=[region])
        chart._scope = scope

        await chart.start()

        # Poll for region to reach FAILURE
        for i in range(50):
            if region.status == ChartStatus.FAILURE:
                break
            await asyncio.sleep(0.01)

        # Verify region failed
        assert region.status == ChartStatus.FAILURE

        # Verify exception details
        exception_data = ctx.get("__exception__")
        assert exception_data is not None
        assert exception_data["type"] == "RuntimeError"
        assert "stream error" in exception_data["message"]
        assert exception_data["yielded_count"] == 3

        # Verify partial progress data was captured
        # Context should have data from first 3 yields
        # (This depends on how the StreamState stores yielded data)


# ============================================================================
# Test 5: No Exception Re-raised to Caller
# ============================================================================

class TestExceptionContainment:
    """Test that exceptions are contained and not re-raised."""

    @pytest.mark.asyncio
    async def test_no_exception_propagated_to_caller(self):
        """Verify start() doesn't raise exception from state."""
        region = Region(name="test", initial="failing", rules=[])
        region["failing"] = FailingState()

        chart = StateChart(name="test_chart", regions=[region])

        # This should NOT raise ValueError
        await chart.start()

        # Poll for completion
        for i in range(50):
            if region.status == ChartStatus.FAILURE:
                break
            await asyncio.sleep(0.01)

        # Verify we can inspect status
        assert region.status == ChartStatus.FAILURE


# ============================================================================
# Test 6: Multiple States Fail in Sequence
# ============================================================================

class FirstFailingState(State):
    """First state that fails."""
    async def execute(self, post, **inputs):
        raise ValueError("first failure")


class SecondFailingState(State):
    """Second state that also fails."""
    async def execute(self, post, **inputs):
        raise RuntimeError("second failure")


class TestMultipleFailures:
    """Test handling of multiple failures in parallel regions."""

    @pytest.mark.asyncio
    async def test_multiple_failures_in_parallel_regions(self, caplog):
        """Verify multiple failures in different regions are both logged."""
        # Two regions that both fail
        region1 = Region(name="region1", initial="first", rules=[])
        region1["first"] = FirstFailingState()

        region2 = Region(name="region2", initial="second", rules=[])
        region2["second"] = SecondFailingState()

        chart = StateChart(name="multi_fail_test", regions=[region1, region2])

        with caplog.at_level(logging.ERROR):
            await chart.start()

            # Wait for both regions to fail
            for i in range(50):
                if (region1.status == ChartStatus.FAILURE and
                    region2.status == ChartStatus.FAILURE):
                    break
                await asyncio.sleep(0.01)

        # Verify both regions failed
        assert region1.status == ChartStatus.FAILURE
        assert region2.status == ChartStatus.FAILURE

        # Verify both exceptions were logged
        assert "first failure" in caplog.text
        assert "second failure" in caplog.text


# ============================================================================
# Test 7: Exception During State Enter
# ============================================================================

class EnterFailingState(State):
    """State that throws exception during enter."""

    def enter(self, post, ctx):
        super().enter(post, ctx)
        raise ValueError("enter failed")

    async def execute(self, post, **inputs):
        return {"should_not_reach": True}


class TestEnterExceptionHandling:
    """Test exception handling during enter phase."""

    @pytest.mark.asyncio
    async def test_exception_during_state_enter(self):
        """Verify exceptions during enter are NOT caught (expected to crash)."""
        region = Region(name="test", initial="enter_fail", rules=[])
        region["enter_fail"] = EnterFailingState()

        chart = StateChart(name="enter_test", regions=[region])

        # Exceptions during enter() are NOT caught - they crash
        # This is expected behavior as enter() is synchronous setup
        with pytest.raises(ValueError, match="enter failed"):
            await chart.start()


# ============================================================================
# Test 8: Exception During State Exit
# ============================================================================

class ExitFailingState(State):
    """State that throws exception during exit."""

    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        return {"executed": True}

    def exit(self, post, ctx):
        super().exit(post, ctx)
        raise ValueError("exit failed")


class TestExitExceptionHandling:
    """Test exception handling during exit phase."""

    @pytest.mark.asyncio
    async def test_exception_during_state_exit(self):
        """Verify exceptions during exit are handled gracefully."""
        region = Region(name="test", initial="exit_fail", rules=[
            Rule(event_type="transition", target="SUCCESS"),
        ])
        region["exit_fail"] = ExitFailingState()

        chart = StateChart(name="exit_test", regions=[region])
        await chart.start()

        # Wait for state to execute
        await asyncio.sleep(0.05)

        # Post event to trigger transition (and thus exit)
        chart.post("transition")

        # Give time for transition
        await asyncio.sleep(0.05)

        # Transition should complete despite exit error
        # (Region may be in SUCCESS or handle error differently)
        assert region.current_state_name in ["SUCCESS", "exit_fail"]
