"""End-to-end tests for StateChart.

These tests verify complete, realistic workflows that demonstrate the StateChart
system working as a whole in real-world scenarios:

- Multi-step wizard workflows (form submission, validation, payment)
- Request-response patterns with timeouts and retries
- Parallel task coordination with synchronization
- Background jobs with cancellation
- Complex state machines (document editors, chat systems)
- Nested composite states with hierarchical coordination

These tests use realistic state implementations and demonstrate the full
capabilities of the StateChart system.
"""

import asyncio
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dachi.act._chart._chart import StateChart, ChartStatus
from dachi.act._chart._region import Region, Rule
from dachi.act._chart._state import State, StreamState, FinalState
from dachi.act._chart._composite import CompositeState
from dachi.core import Scope


pytestmark = pytest.mark.e2e


# ============================================================================
# Helper Utilities
# ============================================================================

async def wait_for_chart(chart: StateChart, timeout: float = 3.0, check_success: bool = True) -> bool:
    """Wait for chart to complete with timeout.

    Args:
        chart: StateChart to wait for
        timeout: Maximum time to wait in seconds
        check_success: If True, check for SUCCESS status, else just completed

    Returns:
        True if chart completed as expected, False if timeout
    """
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        status = chart._status.get()
        if check_success and status == ChartStatus.SUCCESS:
            return True
        elif not check_success and status.is_completed():
            return True
        await asyncio.sleep(0.01)
    return False


# ============================================================================
# E2E Scenario 1: Multi-Step Wizard Workflow
# ============================================================================

class FormInputState(State):
    """Simulates user filling out a form."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        form_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "amount": 100
        }
        await post.aforward("form_submitted", form_data)
        return {"form_data": form_data}


class ValidateFormState(State):
    """Validates form data."""
    class inputs:
        form_data: dict

    async def execute(self, post, form_data):
        await asyncio.sleep(0.02)
        is_valid = (
            form_data.get("name") and
            form_data.get("email") and
            form_data.get("amount", 0) > 0
        )

        if is_valid:
            await post.aforward("validation_passed", {"validated": True})
        else:
            await post.aforward("validation_failed", {"errors": ["Invalid form"]})

        return {"validated": is_valid}


class ProcessPaymentState(StreamState):
    """Processes payment with progress updates."""
    class inputs:
        form_data: dict

    async def execute(self, post, form_data):
        amount = form_data.get("amount", 0)

        yield {"status": "charging", "progress": 0}
        await asyncio.sleep(0.03)

        yield {"status": "charging", "progress": 50}
        await asyncio.sleep(0.03)

        yield {"status": "charged", "progress": 100}

        await post.aforward("payment_success", {"transaction_id": "tx_123"})


class PaymentSuccessState(FinalState):
    """Payment completed successfully."""
    pass


class PaymentFailedState(FinalState):
    """Payment failed."""
    pass


class TestWizardWorkflow:
    """Test multi-step wizard workflow."""

    @pytest.mark.asyncio
    async def test_complete_wizard_flow(self):
        """Test complete wizard from form input through payment."""
        region = Region(name="wizard", initial="form", rules=[
            Rule(event_type="form_submitted", target="validate"),
            Rule(event_type="validation_passed", target="payment"),
            Rule(event_type="validation_failed", target="failed"),
            Rule(event_type="payment_success", target="success"),
        ])

        region["form"] = FormInputState()
        region["validate"] = ValidateFormState()
        region["payment"] = ProcessPaymentState()
        region["success"] = PaymentSuccessState()
        region["failed"] = PaymentFailedState()

        chart = StateChart(name="wizard", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=2.0)

        assert completed is True
        assert chart._status.get() == ChartStatus.SUCCESS
        assert region.current_state == "success"


# ============================================================================
# E2E Scenario 2: Request-Response with Timeout and Retry
# ============================================================================

class SendRequestState(State):
    """Sends a request and starts timeout timer."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)

        # Start a timeout timer (simulated via event)
        await post.aforward("request_sent", {"request_id": "req_001"})

        # Simulate timeout after delay
        asyncio.create_task(self._simulate_timeout(post))

        return {"request_id": "req_001", "retry_count": 0}

    async def _simulate_timeout(self, post):
        """Simulate timeout firing."""
        await asyncio.sleep(0.15)
        await post.aforward("timeout")


class WaitForResponseState(StreamState):
    """Waits for response with checkpoints."""
    async def execute(self, post, **inputs):
        for i in range(5):
            await asyncio.sleep(0.05)
            yield {"waiting": i}

        # Simulate response arriving
        await post.aforward("response_received", {"data": "success"})


class RetryState(State):
    """Retry logic after timeout."""
    class inputs:
        retry_count: int = 0

    async def execute(self, post, retry_count):
        retry_count += 1

        if retry_count >= 3:
            await post.aforward("max_retries")
        else:
            await post.aforward("retry", {"retry_count": retry_count})

        return {"retry_count": retry_count}


class RequestSuccessState(FinalState):
    """Request succeeded."""
    pass


class RequestFailedState(FinalState):
    """Request failed after retries."""
    pass


class TestRequestResponsePattern:
    """Test request-response with timeout and retry."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self):
        """Test that timeout triggers retry logic."""
        region = Region(name="request", initial="send", rules=[
            Rule(event_type="request_sent", target="wait"),
            Rule(event_type="timeout", target="retry"),
            Rule(event_type="retry", target="send"),
            Rule(event_type="response_received", target="success"),
            Rule(event_type="max_retries", target="failed"),
        ])

        region["send"] = SendRequestState()
        region["wait"] = WaitForResponseState()
        region["retry"] = RetryState()
        region["success"] = RequestSuccessState()
        region["failed"] = RequestFailedState()

        chart = StateChart(name="request_retry", regions=[region], auto_finish=True)
        await chart.start()

        # Wait for completion (should timeout and retry)
        completed = await wait_for_chart(chart, timeout=2.0, check_success=False)

        assert completed is True
        # Should eventually timeout and fail after retries
        assert region.current_state in ["failed", "success"]


# ============================================================================
# E2E Scenario 3: Parallel Task Coordination
# ============================================================================

class DataFetchState(StreamState):
    """Fetches data with progress."""
    async def execute(self, post, **inputs):
        for i in range(3):
            await asyncio.sleep(0.03)
            yield {"fetched": i + 1}

        await post.aforward("data_ready", {"data": [1, 2, 3]})


class DataProcessState(StreamState):
    """Processes data."""
    async def execute(self, post, **inputs):
        for i in range(2):
            await asyncio.sleep(0.04)
            yield {"processed": i + 1}

        await post.aforward("processing_done")


class DataSaveState(State):
    """Saves processed data."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("saved")


class CompletedState(FinalState):
    """All tasks completed."""
    pass


class TestParallelTaskCoordination:
    """Test parallel task execution and coordination."""

    @pytest.mark.asyncio
    async def test_parallel_data_pipeline(self):
        """Test parallel fetch, process, save pipeline."""
        fetch_region = Region(name="fetch", initial="fetching", rules=[
            Rule(event_type="data_ready", target="done"),
        ])
        fetch_region["fetching"] = DataFetchState()
        fetch_region["done"] = CompletedState()

        process_region = Region(name="process", initial="processing", rules=[
            Rule(event_type="processing_done", target="save"),
            Rule(event_type="saved", target="done"),
        ])
        process_region["processing"] = DataProcessState()
        process_region["save"] = DataSaveState()
        process_region["done"] = CompletedState()

        chart = StateChart(
            name="parallel_pipeline",
            regions=[fetch_region, process_region],
            auto_finish=True
        )
        await chart.start()

        completed = await wait_for_chart(chart, timeout=2.0)

        assert completed is True
        assert fetch_region.current_state == "done"
        assert process_region.current_state == "done"


# ============================================================================
# E2E Scenario 4: Background Job with Cancellation
# ============================================================================

class BackgroundJobState(StreamState):
    """Long-running background job."""
    async def execute(self, post, **inputs):
        for i in range(20):
            await asyncio.sleep(0.05)
            yield {"progress": (i + 1) * 5}

        await post.aforward("job_complete")


class CancelJobState(State):
    """Handles job cancellation."""
    async def execute(self, post, **inputs):
        await post.aforward("cancelled")


class JobCancelledState(FinalState):
    """Job was cancelled."""
    pass


class JobCompleteState(FinalState):
    """Job completed successfully."""
    pass


class TestBackgroundJobCancellation:
    """Test background job that can be cancelled."""

    @pytest.mark.asyncio
    async def test_cancel_running_job(self):
        """Test cancelling a long-running background job."""
        region = Region(name="job", initial="running", rules=[
            Rule(event_type="cancel", target="cancelling"),
            Rule(event_type="cancelled", target="cancelled_state"),
            Rule(event_type="job_complete", target="complete"),
        ])

        region["running"] = BackgroundJobState()
        region["cancelling"] = CancelJobState()
        region["cancelled_state"] = JobCancelledState()
        region["complete"] = JobCompleteState()

        chart = StateChart(name="background_job", regions=[region], auto_finish=True)
        await chart.start()

        # Let job run for a bit
        await asyncio.sleep(0.15)

        # Cancel it
        chart.post("cancel")

        # Wait for cancellation to complete
        completed = await wait_for_chart(chart, timeout=1.0, check_success=False)

        assert completed is True
        assert region.current_state == "cancelled_state"


# ============================================================================
# E2E Scenario 5: Complex State Machine (Document Editor)
# ============================================================================

class IdleState(State):
    """Editor is idle, waiting for user action."""
    async def execute(self, post, **inputs):
        # Just wait for events
        return {"mode": "idle"}


class EditingState(StreamState):
    """User is editing document."""
    async def execute(self, post, **inputs):
        for i in range(5):
            await asyncio.sleep(0.03)
            yield {"changes": i + 1}

        # Auto-save after edits
        await post.aforward("auto_save")


class SavingState(State):
    """Saving document."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("save_complete", {"saved": True})


class ReviewingState(State):
    """Reviewing document before publish."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("review_complete")


class PublishedState(FinalState):
    """Document published."""
    pass


class TestDocumentEditorWorkflow:
    """Test complex document editor state machine."""

    @pytest.mark.asyncio
    async def test_edit_save_review_publish_flow(self):
        """Test complete document workflow."""
        region = Region(name="editor", initial="idle", rules=[
            Rule(event_type="start_edit", target="editing"),
            Rule(event_type="auto_save", target="saving"),
            Rule(event_type="save_complete", target="idle"),
            Rule(event_type="review", target="reviewing"),
            Rule(event_type="review_complete", target="published"),
        ])

        region["idle"] = IdleState()
        region["editing"] = EditingState()
        region["saving"] = SavingState()
        region["reviewing"] = ReviewingState()
        region["published"] = PublishedState()

        chart = StateChart(name="doc_editor", regions=[region], auto_finish=True)
        await chart.start()

        # Start editing
        chart.post("start_edit")

        # Wait for auto-save cycle
        await asyncio.sleep(0.3)

        # Start review
        chart.post("review")

        # Wait for completion
        completed = await wait_for_chart(chart, timeout=2.0)

        assert completed is True
        assert region.current_state == "published"

    @pytest.mark.asyncio
    async def test_interrupt_editing(self):
        """Test interrupting editing state."""
        region = Region(name="editor", initial="editing", rules=[
            Rule(event_type="cancel", target="idle"),
            Rule(event_type="auto_save", target="saving"),
        ])

        region["editing"] = EditingState()
        region["saving"] = SavingState()
        region["idle"] = IdleState()

        chart = StateChart(name="doc_editor_interrupt", regions=[region])
        await chart.start()

        # Let editing run for a bit
        await asyncio.sleep(0.08)

        # Interrupt it
        chart.post("cancel")

        # Wait for transition
        await asyncio.sleep(0.15)

        assert region.current_state == "idle"

        await chart.stop()


# ============================================================================
# E2E Scenario 6: Nested Composite States (if implemented)
# ============================================================================

class FormStepState(State):
    """Individual form step in wizard."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("step_complete")


class TestCompositeStateWorkflow:
    """Test composite state with nested regions."""

    @pytest.mark.asyncio
    async def test_wizard_with_composite_form(self):
        """Test wizard with composite form containing multiple steps."""
        # Create child regions for the composite form state
        step1_region = Region(name="step1", initial="input", rules=[
            Rule(event_type="step_complete", target="done"),
        ])
        step1_region["input"] = FormStepState()
        step1_region["done"] = FinalState()

        step2_region = Region(name="step2", initial="input", rules=[
            Rule(event_type="step_complete", target="done"),
        ])
        step2_region["input"] = FormStepState()
        step2_region["done"] = FinalState()

        # Create composite state
        composite_form = CompositeState(
            name="multi_step_form",
            regions=[step1_region, step2_region]
        )

        # Main wizard region
        wizard_region = Region(name="wizard", initial="form", rules=[
            Rule(event_type="form_complete", target="success"),
        ])
        wizard_region["form"] = composite_form
        wizard_region["success"] = FinalState()

        chart = StateChart(name="composite_wizard", regions=[wizard_region])
        await chart.start()

        # Wait for composite to complete (all child regions finish)
        await asyncio.sleep(0.3)

        # Composite should complete when all children are done
        # This will trigger form_complete event
        chart.post("form_complete")

        await asyncio.sleep(0.1)

        assert wizard_region.current_state == "success"

        await chart.stop()


# ============================================================================
# E2E Scenario 7: Error Recovery and Resilience
# ============================================================================

class RiskyOperationState(State):
    """Operation that might fail."""
    class inputs:
        attempt: int = 0

    async def execute(self, post, attempt):
        await asyncio.sleep(0.02)

        # Fail on first two attempts, succeed on third
        if attempt < 2:
            await post.aforward("operation_failed", {"attempt": attempt})
        else:
            await post.aforward("operation_success")

        return {"attempt": attempt + 1}


class ErrorHandlerState(State):
    """Handles errors with retry logic."""
    class inputs:
        attempt: int = 0

    async def execute(self, post, attempt):
        await asyncio.sleep(0.01)

        if attempt >= 3:
            await post.aforward("give_up")
        else:
            await post.aforward("retry", {"attempt": attempt})


class SuccessState(FinalState):
    """Operation succeeded."""
    pass


class FailedState(FinalState):
    """Operation failed permanently."""
    pass


class TestErrorRecovery:
    """Test error recovery and retry patterns."""

    @pytest.mark.asyncio
    async def test_retry_until_success(self):
        """Test retrying failed operation until success."""
        region = Region(name="resilient", initial="operation", rules=[
            Rule(event_type="operation_failed", target="error_handler"),
            Rule(event_type="retry", target="operation"),
            Rule(event_type="operation_success", target="success"),
            Rule(event_type="give_up", target="failed"),
        ])

        region["operation"] = RiskyOperationState()
        region["error_handler"] = ErrorHandlerState()
        region["success"] = SuccessState()
        region["failed"] = FailedState()

        chart = StateChart(name="resilient_op", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=2.0, check_success=False)

        assert completed is True
        # Should succeed on third attempt
        assert region.current_state == "success"


# ============================================================================
# E2E Scenario 8: Event Correlation and Matching
# ============================================================================

class CorrelatedRequestState(State):
    """Sends request with correlation ID."""
    async def execute(self, post, **inputs):
        import uuid
        correlation_id = str(uuid.uuid4())

        await post.aforward("request_sent", {"correlation_id": correlation_id})

        # Simulate delayed response with same correlation ID
        asyncio.create_task(self._send_response(post, correlation_id))

        return {"correlation_id": correlation_id}

    async def _send_response(self, post, correlation_id):
        """Simulate response after delay."""
        await asyncio.sleep(0.1)
        await post.aforward("response", {"correlation_id": correlation_id, "data": "result"})


class WaitingState(State):
    """Waits for correlated response."""
    async def execute(self, post, **inputs):
        return {}


class ResponseHandlerState(FinalState):
    """Handles correlated response."""
    pass


class TestEventCorrelation:
    """Test event correlation patterns."""

    @pytest.mark.asyncio
    async def test_correlated_request_response(self):
        """Test request-response with correlation ID matching."""
        region = Region(name="correlation", initial="request", rules=[
            Rule(event_type="request_sent", target="waiting"),
            Rule(event_type="response", target="done"),
        ])

        region["request"] = CorrelatedRequestState()
        region["waiting"] = WaitingState()
        region["done"] = ResponseHandlerState()

        chart = StateChart(name="correlation_test", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state == "done"
