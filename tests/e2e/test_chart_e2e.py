"""End-to-end tests for StateChart.

These tests simulate realistic, complete usage scenarios:
- Multi-step wizard/form workflows
- Request-response patterns with timeout and retry
- Parallel task coordination with synchronization
- Long-running background jobs with cancellation
- Complex state machine patterns

Run with: pytest tests/e2e/ -m e2e
"""

import asyncio
import pytest
import sys
import os
from typing import Dict, Any
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dachi.act._chart._chart import StateChart, ChartStatus
from dachi.act._chart._region import Region, Rule
from dachi.act._chart._state import State, StreamState, FinalState


pytestmark = pytest.mark.e2e


# ============================================================================
# E2E Scenario 1: Multi-Step Wizard Workflow
# ============================================================================

class CollectUserInfoState(State):
    """Collect user information."""
    async def execute(self, post, **inputs):
        # Simulate user input collection
        user_data = {
            "name": "John Doe",
            "email": "john@example.com"
        }
        await post("user_info_collected", user_data)
        return {"user": user_data}


class ValidateEmailState(State):
    """Validate email format."""
    class inputs:
        user: Dict[str, str]

    async def execute(self, post, user):
        email = user.get("email", "")
        is_valid = "@" in email and "." in email

        if is_valid:
            await post("email_valid")
        else:
            await post("email_invalid")

        return {"email_validated": is_valid}


class CollectPaymentInfoState(State):
    """Collect payment information."""
    async def execute(self, post, **inputs):
        payment_data = {
            "card_number": "****1234",
            "expiry": "12/25"
        }
        await post("payment_collected", payment_data)
        return {"payment": payment_data}


class ProcessPaymentState(StreamState):
    """Process payment with progress updates."""
    class inputs:
        payment: Dict[str, str]

    async def astream(self, post, payment):
        yield {"status": "authorizing"}
        await asyncio.sleep(0.05)

        yield {"status": "processing"}
        await asyncio.sleep(0.05)

        yield {"status": "completed"}
        await post("payment_success")


class ConfirmationState(FinalState):
    """Final confirmation state."""
    pass


class RetryEmailState(State):
    """Retry email collection."""
    async def execute(self, post, **inputs):
        await post("retry_complete")
        return {"retried": True}


class TestWizardWorkflow:
    """E2E test for multi-step wizard."""

    @pytest.mark.asyncio
    async def test_complete_wizard_flow_success_path(self):
        """Test complete wizard flow with all steps succeeding."""
        region = Region(name="wizard", initial="collect_user", rules=[
            Rule(event_type="user_info_collected", target="validate_email"),
            Rule(event_type="email_valid", target="collect_payment"),
            Rule(event_type="payment_collected", target="process_payment"),
            Rule(event_type="payment_success", target="confirmation"),
        ])

        region._states["collect_user"] = CollectUserInfoState()
        region._states["validate_email"] = ValidateEmailState()
        region._states["collect_payment"] = CollectPaymentInfoState()
        region._states["process_payment"] = ProcessPaymentState()
        region._states["confirmation"] = ConfirmationState()

        chart = StateChart(name="wizard", regions=[region], auto_finish=True)
        await chart.start()

        completed = await chart.join(timeout=3.0)

        assert completed is True
        assert chart._status.get() == ChartStatus.FINISHED
        assert region.current_state == "confirmation"

        await chart.stop()

    @pytest.mark.asyncio
    async def test_wizard_flow_with_retry(self):
        """Test wizard flow with validation failure and retry."""
        region = Region(name="wizard_retry", initial="collect_user", rules=[
            Rule(event_type="user_info_collected", target="validate_email"),
            Rule(event_type="email_invalid", target="retry_email"),
            Rule(event_type="retry_complete", target="confirmation"),
            Rule(event_type="email_valid", target="confirmation"),
        ])

        # Use a modified validator that fails
        class FailingEmailValidator(State):
            async def execute(self, post, **inputs):
                await post("email_invalid")
                return {"email_validated": False}

        region._states["collect_user"] = CollectUserInfoState()
        region._states["validate_email"] = FailingEmailValidator()
        region._states["retry_email"] = RetryEmailState()
        region._states["confirmation"] = ConfirmationState()

        chart = StateChart(name="wizard_retry", regions=[region], auto_finish=True)
        await chart.start()

        completed = await chart.join(timeout=2.0)

        assert completed is True
        assert region.current_state == "confirmation"

        await chart.stop()


# ============================================================================
# E2E Scenario 2: Request-Response with Timeout and Retry
# ============================================================================

class SendRequestState(State):
    """Send an HTTP-like request."""
    async def execute(self, post, **inputs):
        request_id = "req-123"
        await post("request_sent", {"request_id": request_id})
        return {"request_id": request_id, "retry_count": 0}


class WaitForResponseState(StreamState):
    """Wait for response with timeout capability."""
    class inputs:
        request_id: str

    async def astream(self, post, request_id):
        # Simulate waiting for response
        for i in range(5):
            await asyncio.sleep(0.05)
            yield {"waiting": i}

        # Timeout after waiting
        await post("timeout")


class SimulateResponseState(State):
    """Simulate receiving a response."""
    async def execute(self, post, **inputs):
        # Simulate successful response
        await post("response_received", {"status": "ok"})
        return {"response": "success"}


class RetryRequestState(State):
    """Retry the request."""
    class inputs:
        retry_count: int = 0

    async def execute(self, post, retry_count):
        retry_count += 1
        if retry_count < 3:
            await post("retry", {"retry_count": retry_count})
        else:
            await post("max_retries_exceeded")

        return {"retry_count": retry_count}


class SuccessResponseState(FinalState):
    """Successful response received."""
    pass


class FailedRequestState(FinalState):
    """Request failed after retries."""
    pass


class TestRequestResponsePattern:
    """E2E test for request-response with timeout."""

    @pytest.mark.asyncio
    async def test_request_response_with_manual_response(self):
        """Test request-response where response arrives in time."""
        region = Region(name="request", initial="send", rules=[
            Rule(event_type="request_sent", target="wait"),
            Rule(event_type="response_received", target="success"),
            Rule(event_type="timeout", target="retry"),
            Rule(event_type="retry", target="send"),
            Rule(event_type="max_retries_exceeded", target="failed"),
        ])

        region._states["send"] = SendRequestState()
        region._states["wait"] = WaitForResponseState()
        region._states["retry"] = RetryRequestState()
        region._states["success"] = SuccessResponseState()
        region._states["failed"] = FailedRequestState()

        chart = StateChart(name="req_resp", regions=[region], auto_finish=True)
        await chart.start()

        # Wait a bit then send response
        await asyncio.sleep(0.05)
        chart.post("response_received", {"status": "ok"})

        completed = await chart.join(timeout=2.0)

        assert completed is True
        assert region.current_state == "success"

        await chart.stop()

    @pytest.mark.asyncio
    async def test_request_times_out_and_retries(self):
        """Test request that times out and gets retried."""
        region = Region(name="timeout_retry", initial="send", rules=[
            Rule(event_type="request_sent", target="wait"),
            Rule(event_type="timeout", target="retry"),
            Rule(event_type="retry", target="success"),  # Simplified for test
        ])

        region._states["send"] = SendRequestState()
        region._states["wait"] = WaitForResponseState()
        region._states["retry"] = RetryRequestState()
        region._states["success"] = SuccessResponseState()

        chart = StateChart(name="timeout", regions=[region], auto_finish=True)
        await chart.start()

        # Let it timeout and retry
        completed = await chart.join(timeout=3.0)

        assert completed is True

        await chart.stop()


# ============================================================================
# E2E Scenario 3: Parallel Task Coordination
# ============================================================================

class DataFetchTask(StreamState):
    """Simulate fetching data from an API."""
    async def astream(self, post, **inputs):
        await asyncio.sleep(0.1)
        yield {"data": "fetched"}
        await post("data_ready")


class ImageProcessTask(StreamState):
    """Simulate processing images."""
    async def astream(self, post, **inputs):
        for i in range(3):
            await asyncio.sleep(0.04)
            yield {"processed": i + 1}
        await post("images_ready")


class CacheWarmupTask(State):
    """Simulate cache warmup."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.05)
        await post("cache_ready")
        return {"cache": "warmed"}


class AggregateResultsState(State):
    """Aggregate results from all tasks."""
    async def execute(self, post, **inputs):
        await post("all_complete")
        return {"aggregated": True}


class TestParallelTaskCoordination:
    """E2E test for parallel task execution."""

    @pytest.mark.asyncio
    async def test_three_parallel_tasks_with_aggregation(self):
        """Test multiple parallel regions that synchronize at the end."""
        # Three parallel regions
        data_region = Region(name="data", initial="fetch", rules=[
            Rule(event_type="data_ready", target="done"),
        ])
        data_region._states["fetch"] = DataFetchTask()
        data_region._states["done"] = SuccessResponseState()

        image_region = Region(name="images", initial="process", rules=[
            Rule(event_type="images_ready", target="done"),
        ])
        image_region._states["process"] = ImageProcessTask()
        image_region._states["done"] = SuccessResponseState()

        cache_region = Region(name="cache", initial="warmup", rules=[
            Rule(event_type="cache_ready", target="done"),
        ])
        cache_region._states["warmup"] = CacheWarmupTask()
        cache_region._states["done"] = SuccessResponseState()

        chart = StateChart(
            name="parallel",
            regions=[data_region, image_region, cache_region],
            auto_finish=True
        )
        await chart.start()

        # All should complete
        completed = await chart.join(timeout=3.0)

        assert completed is True
        assert data_region.current_state == "done"
        assert image_region.current_state == "done"
        assert cache_region.current_state == "done"

        await chart.stop()


# ============================================================================
# E2E Scenario 4: Long-Running Background Job with Cancellation
# ============================================================================

class BackgroundJobState(StreamState):
    """Long-running background job with progress reporting."""
    async def astream(self, post, **inputs):
        total_steps = 20
        for step in range(total_steps):
            await asyncio.sleep(0.05)
            progress = (step + 1) / total_steps * 100
            yield {"progress": progress, "step": step + 1}

        await post("job_completed")


class MonitorState(StreamState):
    """Monitor job progress."""
    async def astream(self, post, **inputs):
        # Simulate monitoring
        for i in range(3):
            await asyncio.sleep(0.1)
            yield {"check": i + 1}

        # Decide to cancel after monitoring
        await post("cancel_job")


class CancelledState(FinalState):
    """Job was cancelled."""
    pass


class CompletedState(FinalState):
    """Job completed successfully."""
    pass


class TestBackgroundJobCancellation:
    """E2E test for long-running job with cancellation."""

    @pytest.mark.asyncio
    async def test_background_job_runs_to_completion(self):
        """Test job completes when not cancelled."""
        region = Region(name="job", initial="running", rules=[
            Rule(event_type="job_completed", target="completed"),
        ])

        region._states["running"] = BackgroundJobState()
        region._states["completed"] = CompletedState()

        chart = StateChart(name="bg_job", regions=[region], auto_finish=True)
        await chart.start()

        completed = await chart.join(timeout=5.0)

        assert completed is True
        assert region.current_state == "completed"

        await chart.stop()

    @pytest.mark.asyncio
    async def test_background_job_can_be_cancelled(self):
        """Test job can be cancelled mid-execution."""
        region = Region(name="job_cancel", initial="running", rules=[
            Rule(event_type="cancel_job", target="cancelled"),
            Rule(event_type="job_completed", target="completed"),
        ])

        region._states["running"] = BackgroundJobState()
        region._states["cancelled"] = CancelledState()
        region._states["completed"] = CompletedState()

        chart = StateChart(name="cancel_test", regions=[region], auto_finish=True)
        await chart.start()

        # Let it run a bit
        await asyncio.sleep(0.15)

        # Cancel it
        chart.post("cancel_job")

        completed = await chart.join(timeout=2.0)

        assert completed is True
        assert region.current_state == "cancelled"

        await chart.stop()

    @pytest.mark.asyncio
    async def test_job_with_parallel_monitoring(self):
        """Test job running with separate monitoring region."""
        job_region = Region(name="job", initial="running", rules=[
            Rule(event_type="cancel_job", target="cancelled"),
            Rule(event_type="job_completed", target="completed"),
        ])
        job_region._states["running"] = BackgroundJobState()
        job_region._states["cancelled"] = CancelledState()
        job_region._states["completed"] = CompletedState()

        monitor_region = Region(name="monitor", initial="monitoring", rules=[
            Rule(event_type="cancel_job", target="done"),
        ])
        monitor_region._states["monitoring"] = MonitorState()
        monitor_region._states["done"] = SuccessResponseState()

        chart = StateChart(
            name="monitored_job",
            regions=[job_region, monitor_region],
            auto_finish=True
        )
        await chart.start()

        completed = await chart.join(timeout=3.0)

        assert completed is True
        # Monitor should trigger cancellation
        assert job_region.current_state == "cancelled"
        assert monitor_region.current_state == "done"

        await chart.stop()


# ============================================================================
# E2E Scenario 5: Complex State Machine (Document Editor)
# ============================================================================

class IdleEditorState(State):
    """Editor in idle state."""
    async def execute(self, post, **inputs):
        return {"status": "idle"}


class EditingState(StreamState):
    """User is editing."""
    async def astream(self, post, **inputs):
        # Simulate editing session
        for i in range(3):
            await asyncio.sleep(0.05)
            yield {"edits": i + 1}

        await post("save_requested")


class SavingState(State):
    """Saving document."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.05)
        success = True  # Simulate save

        if success:
            await post("save_success")
        else:
            await post("save_failed")

        return {"saved": success}


class ErrorState(State):
    """Handle save error."""
    async def execute(self, post, **inputs):
        await post("error_handled")
        return {"error_handled": True}


class TestComplexStateMachine:
    """E2E test for complex state machine patterns."""

    @pytest.mark.asyncio
    async def test_document_editor_workflow(self):
        """Test complete document editor state machine."""
        region = Region(name="editor", initial="idle", rules=[
            Rule(event_type="start_editing", target="editing"),
            Rule(event_type="save_requested", target="saving"),
            Rule(event_type="save_success", target="idle"),
            Rule(event_type="save_failed", target="error"),
            Rule(event_type="error_handled", target="idle"),
        ])

        region._states["idle"] = IdleEditorState()
        region._states["editing"] = EditingState()
        region._states["saving"] = SavingState()
        region._states["error"] = ErrorState()

        chart = StateChart(name="editor", regions=[region])
        await chart.start()

        # Start editing
        chart.post("start_editing")

        # Wait for editing to complete and save
        await asyncio.sleep(0.5)

        # Should be back to idle after save
        assert region.current_state == "idle"

        await chart.stop()

    @pytest.mark.asyncio
    async def test_editor_can_be_interrupted_during_edit(self):
        """Test that editing can be interrupted."""
        region = Region(name="interrupt_test", initial="idle", rules=[
            Rule(event_type="start_editing", target="editing"),
            Rule(event_type="interrupt", target="idle"),
        ])

        region._states["idle"] = IdleEditorState()
        region._states["editing"] = EditingState()

        chart = StateChart(name="interrupt", regions=[region])
        await chart.start()

        # Start editing
        chart.post("start_editing")
        await asyncio.sleep(0.05)

        # Interrupt it
        chart.post("interrupt")
        await asyncio.sleep(0.1)

        # Should be back to idle
        assert region.current_state == "idle"

        await chart.stop()
