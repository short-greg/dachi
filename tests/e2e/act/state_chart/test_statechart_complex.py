"""Complex State Machine Scenario E2E tests for StateChart.

These tests validate complex scenarios that combine multiple StateChart features:
- Multi-step state machines with conditional branching
- StreamState with preemption and progress tracking
- State re-entry and retry loops
- Parallel regions with synchronization via CompositeState
- Cross-region event coordination
- Context persistence through reset
- Nested hierarchical states

Each test demonstrates a realistic complex state machine scenario.
"""

import asyncio
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from dachi.act.chart._chart import StateChart, ChartStatus
from dachi.act.chart._region import Region, Rule
from dachi.act.chart._state import State, StreamState, FinalState
from dachi.act.chart._composite import CompositeState
from dachi.core import ModuleList

pytestmark = pytest.mark.e2e


# ============================================================================
# Helper Utilities
# ============================================================================

async def wait_for_chart(chart: StateChart, timeout: float = 3.0) -> bool:
    """Wait for chart to complete with timeout."""
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        status = chart._status.get()
        if status.is_completed():
            return True
        await asyncio.sleep(0.01)
    return False


# ============================================================================
# Test 1: Form Validation with Retry Loop
# ============================================================================

class CollectInputState(State):
    """Collect user input."""
    class inputs:
        retry_count: int = 0

    async def execute(self, post, retry_count):
        await asyncio.sleep(0.01)
        # Simulate collecting input - use retry_count to eventually succeed
        value = 10 * (retry_count + 1)
        retry_count += 1
        await post.aforward("input_collected")
        return {"value": value, "retry_count": retry_count}


class ValidateInputState(State):
    """Validate input with conditional branching."""
    class inputs:
        value: int
        retry_count: int

    async def execute(self, post, value, retry_count):
        await asyncio.sleep(0.01)
        is_valid = value >= 30  # Needs 3 retries to succeed

        if is_valid:
            await post.aforward("valid")
        else:
            await post.aforward("invalid")

        return {"is_valid": is_valid}


class ProcessDataState(State):
    """Process valid data."""
    class inputs:
        value: int

    async def execute(self, post, value):
        await asyncio.sleep(0.01)
        result = value * 2
        await post.aforward("processed")
        return {"result": result}


class TestFormValidationWithRetry:
    """Test form validation with retry loop - combines conditional + self-transition + context."""

    @pytest.mark.asyncio
    async def test_validation_retries_until_valid_input(self):
        """Test form retries collection on invalid input until valid."""
        region = Region(name="form", initial="collect", rules=[
            Rule(event_type="input_collected", target="validate"),
            Rule(event_type="valid", target="process"),
            Rule(event_type="invalid", target="collect"),  # Retry loop
            Rule(event_type="processed", target="SUCCESS"),
        ])

        region["collect"] = CollectInputState()
        region["validate"] = ValidateInputState()
        region["process"] = ProcessDataState()

        chart = StateChart(name="form_validation", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=2.0)

        assert completed is True
        assert chart._status.get() == ChartStatus.SUCCESS
        assert region.current_state_name == "SUCCESS"

        # Verify retries happened
        scope = chart._scope
        ctx = scope.ctx(0)
        assert ctx.get("retry_count") >= 2  # Took multiple tries
        assert ctx.get("is_valid") is True
        assert ctx.get("result") == 60  # value=30, result=60


# ============================================================================
# Test 2: Long-Running Task with Cancellation
# ============================================================================

class LongTaskStreamState(StreamState):
    """Long-running task with progress checkpoints."""
    async def execute(self, post, **inputs):
        for i in range(10):
            await asyncio.sleep(0.02)
            yield {"progress": i + 1, "status": "running"}

        await post.aforward("task_complete")
        yield {"status": "complete"}


class TestLongRunningTaskCancellation:
    """Test StreamState preemption with progress tracking."""

    @pytest.mark.asyncio
    async def test_task_cancelled_mid_execution_saves_partial_progress(self):
        """Test cancelling StreamState saves partial progress in context."""
        region = Region(name="task", initial="running", rules=[
            Rule(event_type="cancel", target="CANCELED"),
            Rule(event_type="task_complete", target="SUCCESS"),
        ])

        region["running"] = LongTaskStreamState()

        chart = StateChart(name="long_task", regions=[region], auto_finish=True)
        await chart.start()

        # Let it run for a bit
        await asyncio.sleep(0.06)

        # Cancel the task
        chart.post("cancel")

        # Wait for cancellation
        await asyncio.sleep(0.1)

        assert region.current_state_name == "CANCELED"

        # Verify partial progress saved
        scope = chart._scope
        ctx = scope.ctx(0)
        progress = ctx.get("progress", 0)
        assert 1 <= progress < 10, f"Expected partial progress, got {progress}"


# ============================================================================
# Test 3: Multi-Step Editor Lifecycle with State Re-entry
# ============================================================================

class OpenDocumentState(State):
    """Open document."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("opened")
        return {"doc_id": "doc123", "opened": True}


class EditingStreamState(StreamState):
    """Editing with incremental changes."""
    class inputs:
        edit_count: int = 0

    async def execute(self, post, edit_count):
        # Simulate a few edits
        for i in range(2):
            await asyncio.sleep(0.01)
            edit_count += 1
            yield {"edit_count": edit_count, "content": f"Edit {edit_count}"}

        await post.aforward("save_requested")


class SavingState(State):
    """Save document."""
    class inputs:
        edit_count: int

    async def execute(self, post, edit_count):
        await asyncio.sleep(0.01)
        await post.aforward("saved")
        return {"saved": True, "save_count": 1}


class PublishingState(State):
    """Publish document."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("published")
        return {"published": True}


class TestEditorLifecycle:
    """Test editor state machine with re-entry - combines StreamState + multiple transitions."""

    @pytest.mark.asyncio
    async def test_editor_open_edit_save_publish_cycle(self):
        """Test document editor lifecycle with editing and saving."""
        region = Region(name="editor", initial="opening", rules=[
            Rule(event_type="opened", target="editing"),
            Rule(event_type="save_requested", target="saving"),
            Rule(event_type="saved", target="publishing"),
            Rule(event_type="published", target="SUCCESS"),
        ])

        region["opening"] = OpenDocumentState()
        region["editing"] = EditingStreamState()
        region["saving"] = SavingState()
        region["publishing"] = PublishingState()

        chart = StateChart(name="editor", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=2.0)

        assert completed is True
        assert region.current_state_name == "SUCCESS"

        # Verify lifecycle completed
        scope = chart._scope
        ctx = scope.ctx(0)
        assert ctx.get("opened") is True
        assert ctx.get("edit_count") >= 2
        assert ctx.get("saved") is True
        assert ctx.get("published") is True


# ============================================================================
# Test 4: Request with Retry Counter
# ============================================================================

class SendRequestState(State):
    """Send request."""
    class inputs:
        attempt: int = 0

    async def execute(self, post, attempt):
        await asyncio.sleep(0.01)
        attempt += 1
        # Simulate timeout by not receiving response
        await post.aforward("timeout")
        return {"attempt": attempt, "request_sent": True}


class CheckRetryState(State):
    """Check if should retry."""
    class inputs:
        attempt: int

    async def execute(self, post, attempt):
        await asyncio.sleep(0.01)

        if attempt < 3:
            await post.aforward("retry")
        else:
            await post.aforward("give_up")

        return {"attempt": attempt}


class TestRequestWithRetry:
    """Test request-response with retry counter - combines conditional + retry + context."""

    @pytest.mark.asyncio
    async def test_request_retries_three_times_then_gives_up(self):
        """Test request retries multiple times before giving up."""
        region = Region(name="request", initial="send", rules=[
            Rule(event_type="timeout", target="check_retry"),
            Rule(event_type="retry", target="send"),  # Loop back
            Rule(event_type="give_up", target="FAILURE"),
        ])

        region["send"] = SendRequestState()
        region["check_retry"] = CheckRetryState()

        chart = StateChart(name="request_retry", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=2.0)

        assert completed is True
        assert region.current_state_name == "FAILURE"

        # Verify retries
        scope = chart._scope
        ctx = scope.ctx(0)
        assert ctx.get("attempt") == 3


# ============================================================================
# Test 5: Parallel Regions with Data Merging
# ============================================================================

class FetchDataAState(State):
    """Fetch data from source A."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("fetched_a")
        return {"data_a": [1, 2, 3]}


class FetchDataBState(State):
    """Fetch data from source B."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.03)
        await post.aforward("fetched_b")
        return {"data_b": [4, 5, 6]}


class MergeDataState(State):
    """Merge data from both sources."""
    async def execute(self, post, **inputs):
        # Read from composite's child contexts
        await asyncio.sleep(0.01)
        await post.aforward("merged")
        return {"merged": True}


class TestParallelDataFetching:
    """Test parallel regions with synchronization - combines CompositeState + context."""

    @pytest.mark.asyncio
    async def test_parallel_fetch_then_merge(self):
        """Test two regions fetch data in parallel, then merge."""
        # Create two parallel fetching regions
        region_a = Region(name="fetch_a", initial="fetch", rules=[
            Rule(event_type="fetched_a", target="SUCCESS"),
        ])
        region_a["fetch"] = FetchDataAState()

        region_b = Region(name="fetch_b", initial="fetch", rules=[
            Rule(event_type="fetched_b", target="SUCCESS"),
        ])
        region_b["fetch"] = FetchDataBState()

        # Composite for parallel execution
        composite = CompositeState(
            name="parallel_fetch",
            regions=ModuleList(vals=[region_a, region_b])
        )

        # Main region orchestrates
        main_region = Region(name="main", initial="composite", rules=[
            Rule(event_type="both_done", target="merge"),
            Rule(event_type="merged", target="SUCCESS"),
        ])
        main_region["composite"] = composite
        main_region["merge"] = MergeDataState()

        chart = StateChart(name="parallel_fetch", regions=[main_region])
        await chart.start()

        # Wait for both to complete
        await asyncio.sleep(0.1)

        # Both regions should complete
        assert region_a.current_state_name == "SUCCESS"
        assert region_b.current_state_name == "SUCCESS"
        assert composite._run_completed.get() is True

        # Post event to trigger merge
        chart.post("both_done")
        await asyncio.sleep(0.05)

        assert main_region.current_state_name == "SUCCESS"

        # Verify data was fetched
        scope = chart._scope
        main_ctx = scope.ctx(0)
        composite_ctx = main_ctx.child(0)  # Composite context
        ctx_a = composite_ctx.child(0)
        ctx_b = composite_ctx.child(1)

        assert ctx_a.get("data_a") == [1, 2, 3]
        assert ctx_b.get("data_b") == [4, 5, 6]


# ============================================================================
# Test 6: Cross-Region Event Coordination
# ============================================================================

class ProducerState(State):
    """Producer that signals when done."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("producer_ready")
        return {"producer_done": True}


class WaitingState(State):
    """Consumer waiting for producer signal."""
    async def execute(self, post, **inputs):
        # Wait for event from producer
        await asyncio.sleep(0.5)
        return None


class ConsumerState(State):
    """Consumer processes after producer ready."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("consumer_done")
        return {"consumer_done": True}


class TestCrossRegionCoordination:
    """Test cross-region event coordination - combines multiple regions + events."""

    @pytest.mark.asyncio
    async def test_consumer_waits_for_producer_signal(self):
        """Test consumer region waits for producer region's signal."""
        # Producer region
        producer_region = Region(name="producer", initial="producing", rules=[
            Rule(event_type="producer_ready", target="SUCCESS"),
        ])
        producer_region["producing"] = ProducerState()

        # Consumer region waits for producer_ready event
        consumer_region = Region(name="consumer", initial="waiting", rules=[
            Rule(event_type="producer_ready", target="consuming"),
            Rule(event_type="consumer_done", target="SUCCESS"),
        ])
        consumer_region["waiting"] = WaitingState()
        consumer_region["consuming"] = ConsumerState()

        chart = StateChart(
            name="producer_consumer",
            regions=[producer_region, consumer_region],
            auto_finish=True
        )
        await chart.start()

        completed = await wait_for_chart(chart, timeout=2.0)

        assert completed is True
        assert producer_region.current_state_name == "SUCCESS"
        assert consumer_region.current_state_name == "SUCCESS"

        # Verify coordination happened
        scope = chart._scope
        ctx_producer = scope.ctx(0)
        ctx_consumer = scope.ctx(1)
        assert ctx_producer.get("producer_done") is True
        assert ctx_consumer.get("consumer_done") is True


# ============================================================================
# Test 7: Context Persistence Through Reset
# ============================================================================

class Step1State(State):
    """First step."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("step1_done")
        return {"step1": "complete"}


class Step2State(State):
    """Second step."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("step2_done")
        return {"step2": "complete"}


class Step3FailingState(State):
    """Third step that fails first time."""
    class inputs:
        retry: bool = False

    async def execute(self, post, retry):
        await asyncio.sleep(0.01)

        if not retry:
            # Fail on first attempt
            raise ValueError("Step 3 failed")
        else:
            # Succeed on retry
            await post.aforward("step3_done")
            return {"step3": "complete"}


class TestContextPersistenceThroughReset:
    """Test context persists through reset - combines multi-step + exceptions + reset."""

    @pytest.mark.asyncio
    async def test_context_preserved_after_reset_and_restart(self):
        """Test accumulated context survives reset and can be used on restart."""
        region = Region(name="multi_step", initial="step1", rules=[
            Rule(event_type="step1_done", target="step2"),
            Rule(event_type="step2_done", target="step3"),
            Rule(event_type="step3_done", target="SUCCESS"),
        ])

        region["step1"] = Step1State()
        region["step2"] = Step2State()
        region["step3"] = Step3FailingState()

        chart = StateChart(name="multi_step", regions=[region], auto_finish=True)

        # First run - will fail at step3
        await chart.start()
        await asyncio.sleep(0.1)

        # Should have failed
        assert region.current_state_name == "FAILURE"

        # Context should have step1 and step2 data
        scope = chart._scope
        ctx = scope.ctx(0)
        assert ctx.get("step1") == "complete"
        assert ctx.get("step2") == "complete"

        # Reset chart (which should reset regions)
        chart.reset()

        # Update context to retry step3
        ctx["retry"] = True

        # Second run - should succeed this time
        await chart.start()
        completed = await wait_for_chart(chart, timeout=2.0)

        assert completed is True
        assert region.current_state_name == "SUCCESS"

        # Verify all steps completed
        assert ctx.get("step1") == "complete"
        assert ctx.get("step2") == "complete"
        assert ctx.get("step3") == "complete"
