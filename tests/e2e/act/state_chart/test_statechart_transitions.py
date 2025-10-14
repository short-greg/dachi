"""Transition E2E tests for StateChart.

These tests validate different types of state transitions:
- Conditional transitions based on context data
- Automatic transitions on state completion
- Self-transitions (state loops back to itself)
- Multiple event handling per state
- Event filtering and rule precedence
- Transition chains and data flow

Each test demonstrates a realistic transition pattern.
"""

import asyncio
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from dachi.act._chart._chart import StateChart, ChartStatus
from dachi.act._chart._region import Region, Rule
from dachi.act._chart._state import State, StreamState, FinalState
from dachi.core import Scope


pytestmark = pytest.mark.e2e


# ============================================================================
# Helper Utilities
# ============================================================================

async def wait_for_chart(chart: StateChart, timeout: float = 2.0) -> bool:
    """Wait for chart to complete with timeout."""
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        status = chart._status.get()
        if status.is_completed():
            return True
        await asyncio.sleep(0.01)
    return False


# ============================================================================
# Test 1: Conditional Transition with Guard
# ============================================================================

class ValidateInputState(State):
    """Validates input and returns validation result."""
    class inputs:
        value: int = 0

    async def execute(self, post, value):
        await asyncio.sleep(0.01)
        is_valid = value > 0 and value < 100

        if is_valid:
            await post.aforward("valid")
        else:
            await post.aforward("invalid")

        return {"value": value, "is_valid": is_valid}


class ProcessValidState(State):
    """Processes valid input."""
    class inputs:
        value: int

    async def execute(self, post, value):
        await asyncio.sleep(0.01)
        result = value * 2
        await post.aforward("processed")
        return {"result": result}


class HandleInvalidState(State):
    """Handles invalid input."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("handled")
        return {"error": "Invalid input"}


class SuccessState(FinalState):
    """Success outcome."""
    pass


class FailureState(FinalState):
    """Failure outcome."""
    pass


class TestConditionalTransition:
    """Test conditional transitions based on context data."""

    @pytest.mark.asyncio
    async def test_conditional_branch_on_validation_result(self):
        """Test state branches to different targets based on validation."""
        region = Region(name="validator", initial="validate", rules=[
            Rule(event_type="valid", target="process"),
            Rule(event_type="invalid", target="handle_error"),
            Rule(event_type="processed", target="success"),
            Rule(event_type="handled", target="failure"),
        ])

        region["validate"] = ValidateInputState()
        region["process"] = ProcessValidState()
        region["handle_error"] = HandleInvalidState()
        region["success"] = SuccessState()
        region["failure"] = FailureState()

        chart = StateChart(name="conditional", regions=[region], auto_finish=True)

        # Set valid input in context
        scope = chart._scope
        ctx = scope.ctx(0)
        ctx["value"] = 50

        await chart.start()
        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "success"
        assert ctx.get("result") == 100

    @pytest.mark.asyncio
    async def test_conditional_branch_on_invalid_input(self):
        """Test state branches to error handler for invalid input."""
        region = Region(name="validator", initial="validate", rules=[
            Rule(event_type="valid", target="process"),
            Rule(event_type="invalid", target="handle_error"),
            Rule(event_type="processed", target="success"),
            Rule(event_type="handled", target="failure"),
        ])

        region["validate"] = ValidateInputState()
        region["process"] = ProcessValidState()
        region["handle_error"] = HandleInvalidState()
        region["success"] = SuccessState()
        region["failure"] = FailureState()

        chart = StateChart(name="conditional", regions=[region], auto_finish=True)

        # Set invalid input in context
        scope = chart._scope
        ctx = scope.ctx(0)
        ctx["value"] = 150  # Out of range

        await chart.start()
        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "failure"
        assert ctx.get("error") == "Invalid input"


# ============================================================================
# Test 2: Automatic Transition on Completion
# ============================================================================

class InitState(State):
    """Initialization state that auto-transitions."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("initialized")
        return {"initialized": True}


class ProcessState(State):
    """Processing state that auto-transitions."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("processed")
        return {"processed": True}


class CompleteState(FinalState):
    """Completion state."""
    pass


class TestAutomaticTransition:
    """Test automatic transitions without external events."""

    @pytest.mark.asyncio
    async def test_states_auto_transition_on_completion(self):
        """Test states automatically transition via posted events."""
        region = Region(name="auto", initial="init", rules=[
            Rule(event_type="initialized", target="process"),
            Rule(event_type="processed", target="complete"),
        ])

        region["init"] = InitState()
        region["process"] = ProcessState()
        region["complete"] = CompleteState()

        chart = StateChart(name="auto_transition", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "complete"


# ============================================================================
# Test 3: Self-Transition (Loop)
# ============================================================================

class RetryState(State):
    """State that retries up to N times."""
    class inputs:
        attempt: int = 0

    async def execute(self, post, attempt):
        await asyncio.sleep(0.02)
        attempt += 1

        if attempt < 3:
            await post.aforward("retry")
        else:
            await post.aforward("done")

        return {"attempt": attempt}


class DoneState(FinalState):
    """Done after retries."""
    pass


class TestSelfTransition:
    """Test state that transitions back to itself (loop)."""

    @pytest.mark.asyncio
    async def test_state_loops_back_to_itself(self):
        """Test state re-enters itself multiple times before completing."""
        region = Region(name="loop", initial="retry", rules=[
            Rule(event_type="retry", target="retry"),  # Self-transition
            Rule(event_type="done", target="done"),
        ])

        region["retry"] = RetryState()
        region["done"] = DoneState()

        chart = StateChart(name="self_loop", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=2.0)

        assert completed is True
        assert region.current_state_name == "done"

        scope = chart._scope
        ctx = scope.ctx(0)
        assert ctx.get("attempt") == 3


# ============================================================================
# Test 4: Multiple Event Handlers
# ============================================================================

class MultiEventState(State):
    """State that responds to multiple different events."""
    async def execute(self, post, **inputs):
        # Just wait for events
        await asyncio.sleep(0.5)
        return None


class ApprovedPath(FinalState):
    """Approved path."""
    pass


class RejectedPath(FinalState):
    """Rejected path."""
    pass


class CanceledPath(FinalState):
    """Canceled path."""
    pass


class TestMultipleEventHandlers:
    """Test state that handles multiple different event types."""

    @pytest.mark.asyncio
    async def test_state_responds_to_different_events(self):
        """Test same state transitions differently based on event type."""
        region = Region(name="multi", initial="waiting", rules=[
            Rule(event_type="approve", target="approved"),
            Rule(event_type="reject", target="rejected"),
            Rule(event_type="cancel", target="canceled"),
        ])

        region["waiting"] = MultiEventState()
        region["approved"] = ApprovedPath()
        region["rejected"] = RejectedPath()
        region["canceled"] = CanceledPath()

        chart = StateChart(name="multi_event", regions=[region], auto_finish=True)
        await chart.start()

        # Wait for state to be active
        await asyncio.sleep(0.05)

        # Post one of the events
        chart.post("reject")

        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "rejected"


# ============================================================================
# Test 5: Event Ignored When No Rule Matches
# ============================================================================

class WaitingState(State):
    """State waiting for specific event."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.2)
        # Timeout - post completion event
        await post.aforward("timeout")
        return None


class TimeoutState(FinalState):
    """Timed out."""
    pass


class TestEventFiltering:
    """Test unmatched events are ignored without error."""

    @pytest.mark.asyncio
    async def test_unmatched_event_ignored_gracefully(self):
        """Test posting unrelated event doesn't crash state machine."""
        region = Region(name="filter", initial="waiting", rules=[
            Rule(event_type="timeout", target="timeout"),
            Rule(event_type="approved", target="timeout"),  # Valid event
        ])

        region["waiting"] = WaitingState()
        region["timeout"] = TimeoutState()

        chart = StateChart(name="filter_test", regions=[region], auto_finish=True)
        await chart.start()

        await asyncio.sleep(0.05)

        # Post unrelated events that don't match any rules
        chart.post("unknown_event")
        chart.post("another_unmatched")

        # These should be ignored, machine continues normally
        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "timeout"


# ============================================================================
# Test 6: State-Dependent vs State-Independent Rules
# ============================================================================

class StateA(State):
    """First state."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("next")
        return {"from": "A"}


class StateB(State):
    """Second state."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("next")
        return {"from": "B"}


class StateC(FinalState):
    """Third state."""
    pass


class StateD(FinalState):
    """Alternate final state."""
    pass


class TestRulePrecedence:
    """Test state-dependent rules take precedence over state-independent."""

    @pytest.mark.asyncio
    async def test_state_dependent_rule_overrides_independent(self):
        """Test state-dependent rule has higher precedence."""
        region = Region(name="precedence", initial="a", rules=[
            Rule(event_type="next", target="d"),  # State-independent (lower precedence)
            Rule(event_type="next", when_in="a", target="b"),  # State-dependent (higher)
            Rule(event_type="next", when_in="b", target="c"),  # State-dependent (higher)
        ])

        region["a"] = StateA()
        region["b"] = StateB()
        region["c"] = StateC()
        region["d"] = StateD()

        chart = StateChart(name="precedence_test", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        # Should follow state-dependent rules: a → b → c
        # Not state-independent rule: a → d
        assert region.current_state_name == "c"


# ============================================================================
# Test 7: Transition Chain
# ============================================================================

class QuickState1(State):
    """Fast state 1."""
    async def execute(self, post, **inputs):
        await post.aforward("step1")
        return {"step": 1}


class QuickState2(State):
    """Fast state 2."""
    async def execute(self, post, **inputs):
        await post.aforward("step2")
        return {"step": 2}


class QuickState3(State):
    """Fast state 3."""
    async def execute(self, post, **inputs):
        await post.aforward("step3")
        return {"step": 3}


class ChainComplete(FinalState):
    """Chain complete."""
    pass


class TestTransitionChain:
    """Test multiple rapid transitions in sequence."""

    @pytest.mark.asyncio
    async def test_rapid_transition_chain(self):
        """Test states can transition rapidly through multiple states."""
        region = Region(name="chain", initial="s1", rules=[
            Rule(event_type="step1", target="s2"),
            Rule(event_type="step2", target="s3"),
            Rule(event_type="step3", target="complete"),
        ])

        region["s1"] = QuickState1()
        region["s2"] = QuickState2()
        region["s3"] = QuickState3()
        region["complete"] = ChainComplete()

        chart = StateChart(name="chain_test", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "complete"

        # Verify all states executed
        scope = chart._scope
        ctx = scope.ctx(0)
        assert ctx.get("step") == 3


# ============================================================================
# Test 8: Transition with Data Transformation
# ============================================================================

class CollectState(State):
    """Collects raw data."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("collected")
        return {"raw_data": [1, 2, 3, 4, 5]}


class TransformState(State):
    """Transforms data."""
    class inputs:
        raw_data: list

    async def execute(self, post, raw_data):
        await asyncio.sleep(0.01)
        transformed = [x * 2 for x in raw_data]
        await post.aforward("transformed")
        return {"transformed_data": transformed}


class AggregateState(State):
    """Aggregates transformed data."""
    class inputs:
        transformed_data: list

    async def execute(self, post, transformed_data):
        await asyncio.sleep(0.01)
        total = sum(transformed_data)
        await post.aforward("aggregated")
        return {"total": total}


class DataComplete(FinalState):
    """Data pipeline complete."""
    pass


class TestDataTransformationPipeline:
    """Test data transforms through state transitions."""

    @pytest.mark.asyncio
    async def test_data_transforms_through_states(self):
        """Test each state transforms data for next state."""
        region = Region(name="pipeline", initial="collect", rules=[
            Rule(event_type="collected", target="transform"),
            Rule(event_type="transformed", target="aggregate"),
            Rule(event_type="aggregated", target="complete"),
        ])

        region["collect"] = CollectState()
        region["transform"] = TransformState()
        region["aggregate"] = AggregateState()
        region["complete"] = DataComplete()

        chart = StateChart(name="data_pipeline", regions=[region], auto_finish=True)
        await chart.start()

        completed = await wait_for_chart(chart, timeout=1.0)

        assert completed is True
        assert region.current_state_name == "complete"

        scope = chart._scope
        ctx = scope.ctx(0)
        # raw_data: [1,2,3,4,5]
        # transformed: [2,4,6,8,10]
        # total: 30
        assert ctx.get("total") == 30
