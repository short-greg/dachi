"""Unit tests for StateChart state classes.

Tests cover BaseState, State, StreamState, and FinalState following the
framework testing conventions.
"""

import asyncio
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dachi.act._chart._state import BaseState, AtomState, State, StreamState, FinalState, PseudoState, ReadyState, BoundState, BoundStreamState
from dachi.act._chart._base import ChartStatus, InvalidTransition
from dachi.act._chart._event import EventQueue, EventPost
from dachi.core import Scope


class ConcreteState(State):
    """Concrete State implementation for testing."""

    async def execute(self, post, **inputs):
        return {"output": "result"}


class ConcreteStateWithInputs(State):
    """State with explicit inputs declaration."""

    class inputs:
        param1: str
        param2: int = 42

    async def execute(self, post, param1, param2):
        return {"processed": f"{param1}_{param2}"}


class ConcreteStreamState(StreamState):
    """Concrete StreamState implementation for testing."""

    async def execute(self, post, **inputs):
        yield {"step": 1}
        yield {"step": 2}
        yield {"step": 3}


class ConcreteStreamStateWithInputs(StreamState):
    """StreamState with explicit inputs declaration."""

    class inputs:
        count: int = 3
        prefix: str = "item"

    async def execute(self, post, count, prefix):
        for i in range(count):
            yield {f"{prefix}_{i}": i}


class ConcreteLeafState(AtomState):
    """Concrete LeafState for testing base functionality."""

    async def execute(self, post, **inputs):
        return {"leaf": True}

    async def run(self, post, ctx):
        pass


class LeafStateWithOutputs(AtomState):
    """LeafState with outputs declaration."""

    class outputs:
        result: str
        value: int

    async def execute(self, post, **inputs):
        return {"result": "test", "value": 42}

    async def run(self, post, ctx):
        pass


class LeafStateWithEmit(AtomState):
    """LeafState with emit declaration."""

    class emit:
        TestEvent: str
        DataEvent: dict

    async def execute(self, post, **inputs):
        return {"data": "test"}

    async def run(self, post, ctx):
        pass


class SlowState(State):
    """State with configurable delay for concurrent execution tests."""

    async def execute(self, post, **inputs):
        await asyncio.sleep(0.1)
        return {"slow": True}


class SlowStreamState(StreamState):
    """StreamState with delays between yields."""

    async def execute(self, post, **inputs):
        for i in range(3):
            await asyncio.sleep(0.05)
            yield {"step": i}


class CancelledState(State):
    """State that raises asyncio.CancelledError."""

    async def execute(self, post, **inputs):
        raise asyncio.CancelledError()


class CancelledStreamState(StreamState):
    """StreamState that raises asyncio.CancelledError."""

    async def execute(self, post, **inputs):
        yield {"start": True}
        raise asyncio.CancelledError()


class FailingState(State):
    """State that raises ValueError."""

    async def execute(self, post, **inputs):
        raise ValueError("Test error")


class FailingStreamState(StreamState):
    """StreamState that raises after first yield."""

    async def execute(self, post, **inputs):
        yield {"start": True}
        raise ValueError("Test error")


class EmptyOutputState(State):
    """State that returns None."""

    async def execute(self, post, **inputs):
        return None


class EmptyStreamState(StreamState):
    """StreamState that yields nothing."""

    async def execute(self, post, **inputs):
        return
        yield


class StateWithAllPorts(State):
    """State with inputs, outputs, and emit declarations."""

    class inputs:
        param1: str
        param2: int = 42

    class outputs:
        result: str

    class emit:
        StartEvent: str
        EndEvent: str

    async def execute(self, post, param1, param2):
        return {"result": f"{param1}_{param2}"}


class TrackingState(State):
    """State that tracks lifecycle method calls."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def enter(self, post, ctx):
        self.calls.append('enter')
        super().enter(post, ctx)

    async def execute(self, post, **inputs):
        self.calls.append('execute')
        return {"executed": True}

    def exit(self, post, ctx):
        self.calls.append('exit')
        super().exit(post, ctx)

    def reset(self):
        self.calls.append('reset')
        super().reset()


class TestBaseState:

    def test_post_init_initializes_status_to_waiting(self):
        state = ConcreteState()
        assert state._status.get() == ChartStatus.WAITING

    def test_post_init_initializes_termination_requested_to_false(self):
        state = ConcreteState()
        assert state._termination_requested.get() is False

    def test_post_init_initializes_run_completed_to_false(self):
        state = ConcreteState()
        assert state._run_completed.get() is False

    def test_post_init_initializes_is_executing_to_false(self):
        state = ConcreteState()
        assert state._executing.get() is False

    def test_can_enter_returns_true_when_waiting(self):
        state = ConcreteState()
        assert state.can_enter() is True

    def test_can_enter_returns_false_when_already_entered(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        assert state.can_enter() is False

    def test_can_run_returns_true_when_running_and_not_executing(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        assert state.can_run() is True

    def test_can_run_returns_false_when_waiting(self):
        state = ConcreteState()
        assert state.can_run() is False

    def test_can_run_returns_false_when_is_executing(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._executing.set(True)
        assert state.can_run() is False

    def test_can_run_returns_false_when_run_completed(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._run_completed.set(True)
        assert state.can_run() is False

    def test_can_exit_returns_true_when_executing(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._executing.set(True)
        assert state.can_exit() is True

    def test_can_exit_returns_true_when_run_completed(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._run_completed.set(True)
        assert state.can_exit() is True

    def test_can_exit_returns_false_when_already_exiting(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._executing.set(True)
        state._exiting.set(True)
        assert state.can_exit() is False

    def test_can_exit_returns_false_when_waiting(self):
        state = ConcreteState()
        assert state.can_exit() is False

    def test_enter_sets_status_to_running_when_waiting(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        assert state._status.get() == ChartStatus.RUNNING

    def test_enter_resets_termination_requested_flag(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state._termination_requested.set(True)
        state.enter(post, ctx)
        assert state._termination_requested.get() is False

    def test_enter_resets_run_completed_flag(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state._run_completed.set(True)
        state.enter(post, ctx)
        assert state._run_completed.get() is False

    def test_enter_resets_is_executing_flag(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state._executing.set(True)
        state.enter(post, ctx)
        assert state._executing.get() is False

    def test_enter_raises_exception_when_already_entered(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        with pytest.raises(InvalidTransition):
            state.enter(post, ctx)

    @pytest.mark.asyncio
    async def test_exit_sets_status_to_preempting_when_not_run_completed(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._executing.set(True)
        state.exit(post, ctx)
        assert state._status.get() == ChartStatus.PREEMPTING

    @pytest.mark.asyncio
    async def test_exit_sets_termination_requested_when_not_run_completed(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._executing.set(True)
        state.exit(post, ctx)
        assert state._termination_requested.get() is True

    @pytest.mark.asyncio
    async def test_exit_sets_status_to_success_when_run_completed(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._run_completed.set(True)
        state.exit(post, ctx)
        assert state._status.get() == ChartStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_exit_raises_exception_when_waiting(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        with pytest.raises(InvalidTransition):
            state.exit(post, ctx)

    @pytest.mark.asyncio
    async def test_exit_sets_exiting_flag(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._executing.set(True)
        state.exit(post, ctx)
        assert state._exiting.get() is True

    @pytest.mark.asyncio
    async def test_exit_does_not_change_status_when_failure(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._run_completed.set(True)
        state._status.set(ChartStatus.FAILURE)
        state.exit(post, ctx)
        assert state._status.get() == ChartStatus.FAILURE

    @pytest.mark.asyncio
    async def test_exit_calls_finish_when_run_completed_and_failure(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        called = False

        def callback():
            nonlocal called
            called = True

        state.register_finish_callback(callback)
        state.enter(post, ctx)
        state._run_completed.set(True)
        state._status.set(ChartStatus.FAILURE)
        state.exit(post, ctx)
        await asyncio.sleep(0)  # Yield to allow finish() task to run
        assert called is True

    @pytest.mark.asyncio
    async def test_exit_raises_exception_when_already_exiting(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._executing.set(True)
        state._exiting.set(True)
        with pytest.raises(InvalidTransition):
            state.exit(post, ctx)

    @pytest.mark.asyncio
    async def test_exit_raises_exception_when_not_entered(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        with pytest.raises(InvalidTransition):
            state.exit(post, ctx)

    def test_reset_sets_status_to_waiting_when_success(self):
        state = ConcreteState()
        state._status.set(ChartStatus.SUCCESS)
        state.reset()
        assert state._status.get() == ChartStatus.WAITING

    def test_reset_resets_termination_requested_flag(self):
        state = ConcreteState()
        state._status.set(ChartStatus.SUCCESS)
        state._termination_requested.set(True)
        state.reset()
        assert state._termination_requested.get() is False

    def test_reset_resets_run_completed_flag(self):
        state = ConcreteState()
        state._status.set(ChartStatus.SUCCESS)
        state._run_completed.set(True)
        state.reset()
        assert state._run_completed.get() is False

    def test_reset_resets_is_executing_flag(self):
        state = ConcreteState()
        state._status.set(ChartStatus.SUCCESS)
        state._executing.set(True)
        state.reset()
        assert state._executing.get() is False

    def test_reset_raises_exception_when_running(self):
        state = ConcreteState()
        state._status.set(ChartStatus.RUNNING)
        with pytest.raises(InvalidTransition):
            state.reset()

    def test_reset_works_when_failure(self):
        state = ConcreteState()
        state._status.set(ChartStatus.FAILURE)
        state.reset()
        assert state._status.get() == ChartStatus.WAITING

    def test_reset_works_when_canceled(self):
        state = ConcreteState()
        state._status.set(ChartStatus.CANCELED)
        state.reset()
        assert state._status.get() == ChartStatus.WAITING

    def test_build_inputs_resolves_from_context_with_inputs_class(self):
        state = ConcreteStateWithInputs()
        scope = Scope()
        ctx = scope.ctx()
        ctx["param1"] = "test"
        ctx["param2"] = 100
        inputs = state.build_inputs(ctx)
        assert inputs["param1"] == "test"
        assert inputs["param2"] == 100

    def test_build_inputs_uses_defaults_when_not_in_context(self):
        state = ConcreteStateWithInputs()
        scope = Scope()
        ctx = scope.ctx()
        ctx["param1"] = "test"
        inputs = state.build_inputs(ctx)
        assert inputs["param2"] == 42

    def test_build_inputs_returns_empty_dict_without_inputs_class(self):
        state = ConcreteState()
        scope = Scope()
        ctx = scope.ctx()
        inputs = state.build_inputs(ctx)
        assert inputs == {}

    def test_is_final_returns_false_for_base_state(self):
        state = ConcreteState()
        assert state.is_final() is False

    def test_get_status_returns_current_status(self):
        state = ConcreteState()
        assert state.get_status() == ChartStatus.WAITING

    def test_get_status_returns_updated_status(self):
        state = ConcreteState()
        state._status.set(ChartStatus.RUNNING)
        assert state.get_status() == ChartStatus.RUNNING

    def test_request_termination_sets_flag_to_true(self):
        state = ConcreteState()
        state.request_termination()
        assert state._termination_requested.get() is True

    def test_request_termination_from_false_to_true(self):
        state = ConcreteState()
        assert state._termination_requested.get() is False
        state.request_termination()
        assert state._termination_requested.get() is True


class TestLeafState:

    def test_init_subclass_processes_inputs_class(self):
        class TestLeafState(AtomState):
            class inputs:
                param1: str
                param2: int = 42
            async def execute(self, post, **inputs):
                return {}
            async def run(self, post, ctx):
                pass
        assert hasattr(TestLeafState, 'sc_params')
        assert 'param1' in TestLeafState.sc_params['inputs']

    def test_init_subclass_processes_outputs_class(self):
        assert hasattr(LeafStateWithOutputs, 'sc_params')
        assert 'result' in LeafStateWithOutputs.sc_params['outputs']

    def test_init_subclass_processes_emit_class(self):
        assert hasattr(LeafStateWithEmit, 'sc_params')
        assert 'TestEvent' in LeafStateWithEmit.sc_params['emit']

    def test_init_subclass_with_no_declarations_creates_empty_params(self):
        state = ConcreteLeafState()
        assert state.sc_params['inputs'] == {}
        assert state.sc_params['outputs'] == {}
        assert state.sc_params['emit'] == {}

    @pytest.mark.asyncio
    async def test_execute_can_return_dict(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        state = ConcreteLeafState()
        result = await state.execute(post)
        assert result == {"leaf": True}

    @pytest.mark.asyncio
    async def test_execute_can_return_none(self):
        class NoneLeafState(AtomState):
            async def execute(self, post, **inputs):
                return None
            async def run(self, post, ctx):
                pass
        queue = EventQueue()
        post = EventPost(queue=queue)
        state = NoneLeafState()
        result = await state.execute(post)
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_receives_post_parameter(self):
        class PostCheckingLeafState(AtomState):
            async def execute(self, post, **inputs):
                return {"has_post": post is not None}
            async def run(self, post, ctx):
                pass
        queue = EventQueue()
        post = EventPost(queue=queue)
        state = PostCheckingLeafState()
        result = await state.execute(post)
        assert result["has_post"] is True

    @pytest.mark.asyncio
    async def test_execute_receives_inputs_from_kwargs(self):
        class InputCheckingLeafState(AtomState):
            class inputs:
                test_param: str
            async def execute(self, post, test_param):
                return {"received": test_param}
            async def run(self, post, ctx):
                pass
        queue = EventQueue()
        post = EventPost(queue=queue)
        state = InputCheckingLeafState()
        result = await state.execute(post, test_param="value")
        assert result["received"] == "value"

    def test_outputs_class_processed_into_sc_params(self):
        outputs_params = LeafStateWithOutputs.sc_params['outputs']
        assert 'result' in outputs_params
        assert 'value' in outputs_params

    def test_emit_class_processed_into_sc_params(self):
        emit_params = LeafStateWithEmit.sc_params['emit']
        assert 'TestEvent' in emit_params
        assert 'DataEvent' in emit_params

    def test_run_is_abstract_method(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteLeafState()
        asyncio.run(state.run(post, ctx))


class TestState:

    @pytest.mark.asyncio
    async def test_run_sets_is_executing_flag_during_execution(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = SlowState()
        state.enter(post, ctx)
        task = asyncio.create_task(state.run(post, ctx))
        await asyncio.sleep(0.01)
        assert state._executing.get() is True
        await task

    @pytest.mark.asyncio
    async def test_run_clears_is_executing_flag_after_execution(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._executing.get() is False

    @pytest.mark.asyncio
    async def test_run_sets_run_completed_flag_on_success(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._run_completed.get() is True

    @pytest.mark.asyncio
    async def test_run_keeps_status_running_when_not_exiting(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._status.get() == ChartStatus.RUNNING

    @pytest.mark.asyncio
    async def test_run_updates_context_with_execute_result(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert ctx["output"] == "result"

    @pytest.mark.asyncio
    async def test_run_builds_inputs_from_context(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        ctx["param1"] = "hello"
        state = ConcreteStateWithInputs()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert ctx["processed"] == "hello_42"

    @pytest.mark.asyncio
    async def test_run_does_not_update_context_when_result_is_none(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = EmptyOutputState()
        state.enter(post, ctx)
        ctx["existing"] = "value"
        await state.run(post, ctx)
        assert ctx["existing"] == "value"
        assert "output" not in ctx

    @pytest.mark.asyncio
    async def test_run_raises_exception_when_not_can_run(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        with pytest.raises(InvalidTransition):
            await state.run(post, ctx)

    @pytest.mark.asyncio
    async def test_run_sets_status_to_failure_on_exception(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = FailingState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._status.get() == ChartStatus.FAILURE
        assert "__exception__" in ctx
        assert ctx["__exception__"]["type"] == "ValueError"
        assert ctx["__exception__"]["message"] == "Test error"

    @pytest.mark.asyncio
    async def test_run_sets_run_completed_on_exception(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = FailingState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._run_completed.get() is True

    @pytest.mark.asyncio
    async def test_run_stores_exception_in_context(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = FailingState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert "__exception__" in ctx
        assert "Test error" in ctx["__exception__"]["message"]

    @pytest.mark.asyncio
    async def test_run_sets_status_to_canceled_on_cancelled_error(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = CancelledState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._status.get() == ChartStatus.CANCELED

    @pytest.mark.asyncio
    async def test_run_sets_run_completed_on_cancelled_error(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = CancelledState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._run_completed.get() is True

    @pytest.mark.asyncio
    async def test_run_clears_is_executing_flag_on_exception(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = FailingState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._executing.get() is False

    @pytest.mark.asyncio
    async def test_run_clears_is_executing_flag_on_cancelled_error(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = CancelledState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._executing.get() is False

    @pytest.mark.asyncio
    async def test_run_prevents_concurrent_execution(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = SlowState()
        state.enter(post, ctx)
        task1 = asyncio.create_task(state.run(post, ctx))
        await asyncio.sleep(0.01)
        with pytest.raises(InvalidTransition):
            await state.run(post, ctx)
        await task1


class TestStreamState:

    @pytest.mark.asyncio
    async def test_run_sets_is_executing_flag_during_execution(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = SlowStreamState()
        state.enter(post, ctx)
        task = asyncio.create_task(state.run(post, ctx))
        await asyncio.sleep(0.01)
        assert state._executing.get() is True
        await task

    @pytest.mark.asyncio
    async def test_run_clears_is_executing_flag_after_execution(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._executing.get() is False

    @pytest.mark.asyncio
    async def test_run_sets_run_completed_flag_when_not_terminated(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._run_completed.get() is True

    @pytest.mark.asyncio
    async def test_run_sets_run_completed_when_terminated(self):
        class TerminatingStreamState(StreamState):
            async def execute(self, post, **inputs):
                for i in range(10):
                    yield {"count": i}
                    await asyncio.sleep(0.01)
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = TerminatingStreamState()
        state.enter(post, ctx)
        task = asyncio.create_task(state.run(post, ctx))
        await asyncio.sleep(0.02)
        state.request_termination()
        await task
        assert state._run_completed.get() is True

    @pytest.mark.asyncio
    async def test_run_keeps_status_running_when_completed_naturally(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._status.get() == ChartStatus.RUNNING

    @pytest.mark.asyncio
    async def test_run_sets_status_to_canceled_when_terminated(self):
        class TerminatingStreamState(StreamState):
            async def execute(self, post, **inputs):
                for i in range(10):
                    yield {"count": i}
                    await asyncio.sleep(0.01)
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = TerminatingStreamState()
        state.enter(post, ctx)
        task = asyncio.create_task(state.run(post, ctx))
        await asyncio.sleep(0.02)
        state.request_termination()
        await task
        assert state._status.get() == ChartStatus.CANCELED

    @pytest.mark.asyncio
    async def test_run_updates_context_with_each_yield(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert ctx["step"] == 3

    @pytest.mark.asyncio
    async def test_run_builds_inputs_from_context(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        ctx["count"] = 2
        ctx["prefix"] = "test"
        state = ConcreteStreamStateWithInputs()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert ctx["test_0"] == 0
        assert ctx["test_1"] == 1

    @pytest.mark.asyncio
    async def test_run_does_not_update_context_when_yield_is_none(self):
        class NoneYieldStreamState(StreamState):
            async def execute(self, post, **inputs):
                yield None
                yield {"step": 2}
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = NoneYieldStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert ctx["step"] == 2

    @pytest.mark.asyncio
    async def test_run_raises_exception_when_not_can_run(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteStreamState()
        with pytest.raises(InvalidTransition):
            await state.run(post, ctx)

    @pytest.mark.asyncio
    async def test_run_checks_termination_after_each_yield(self):
        class TerminatingStreamState(StreamState):
            async def execute(self, post, **inputs):
                for i in range(10):
                    yield {"count": i}
                    await asyncio.sleep(0.01)
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = TerminatingStreamState()
        state.enter(post, ctx)
        task = asyncio.create_task(state.run(post, ctx))
        await asyncio.sleep(0.02)
        state.request_termination()
        await task
        assert ctx.get("count", -1) < 9

    @pytest.mark.asyncio
    async def test_run_sets_status_to_failure_on_exception(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = FailingStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._status.get() == ChartStatus.FAILURE
        assert "__exception__" in ctx
        assert ctx["__exception__"]["type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_run_sets_run_completed_on_exception(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = FailingStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._run_completed.get() is True

    @pytest.mark.asyncio
    async def test_run_processes_yields_before_exception(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = FailingStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert ctx["start"] is True

    @pytest.mark.asyncio
    async def test_run_stores_exception_with_yield_count(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = FailingStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert "__exception__" in ctx
        assert "yielded_count" in ctx["__exception__"]

    @pytest.mark.asyncio
    async def test_run_sets_status_to_canceled_on_cancelled_error(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = CancelledStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._status.get() == ChartStatus.CANCELED

    @pytest.mark.asyncio
    async def test_run_sets_run_completed_on_cancelled_error(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = CancelledStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._run_completed.get() is True

    @pytest.mark.asyncio
    async def test_run_clears_is_executing_flag_on_exception(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = FailingStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._executing.get() is False

    @pytest.mark.asyncio
    async def test_run_clears_is_executing_flag_on_cancelled_error(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = CancelledStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._executing.get() is False

    @pytest.mark.asyncio
    async def test_run_prevents_concurrent_execution(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = SlowStreamState()
        state.enter(post, ctx)
        task1 = asyncio.create_task(state.run(post, ctx))
        await asyncio.sleep(0.01)
        with pytest.raises(InvalidTransition):
            await state.run(post, ctx)
        await task1

    @pytest.mark.asyncio
    async def test_run_handles_empty_stream(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = EmptyStreamState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._status.get() == ChartStatus.RUNNING


class TestPseudoState:
    """Test PseudoState base class."""

    def test_pseudostate_has_name_attribute(self):
        # PseudoState is abstract, so test via FinalState
        state = FinalState()
        assert hasattr(state, 'name')
        assert state.name == "FINAL"

    def test_pseudostate_is_base_module(self):
        from dachi.core import BaseModule
        state = FinalState()
        assert isinstance(state, BaseModule)


class TestFinalState:
    """Test FinalState - a PseudoState that marks region completion."""

    def test_finalstate_is_pseudostate(self):
        state = FinalState()
        assert isinstance(state, PseudoState)

    def test_finalstate_has_name_attribute(self):
        state = FinalState()
        assert state.name == "FINAL"

    def test_finalstate_has_status_attribute(self):
        state = FinalState()
        assert hasattr(state, 'status')
        # status is an Attr, get its value
        assert state.status.get() == ChartStatus.SUCCESS

    def test_finalstate_does_not_have_run_method(self):
        state = FinalState()
        # FinalState is a PseudoState, not a State, so it doesn't have run()
        assert not hasattr(state, 'run') or not callable(getattr(state, 'run', None))

    def test_finalstate_does_not_have_execute_method(self):
        state = FinalState()
        # FinalState is a PseudoState, not a State, so it doesn't have execute()
        assert not hasattr(state, 'execute') or not callable(getattr(state, 'execute', None))


class TestReadyState:
    """Test ReadyState - the built-in initial PseudoState."""

    def test_readystate_is_pseudostate(self):
        state = ReadyState()
        assert isinstance(state, PseudoState)

    def test_readystate_has_name_attribute(self):
        state = ReadyState()
        assert state.name == "READY"

    def test_readystate_status_is_waiting(self):
        state = ReadyState()
        # ReadyState.status is a property that returns WAITING
        assert state.status == ChartStatus.WAITING

    def test_readystate_status_is_property(self):
        state = ReadyState()
        # Verify it's a property, not an Attr
        assert isinstance(type(state).status, property)


class TestStateLifecycle:

    @pytest.mark.asyncio
    async def test_enter_run_exit_completes_successfully(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        state.exit(post, ctx)
        assert state._status.get() == ChartStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_enter_exit_without_run_sets_preempting(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        state._executing.set(True)
        state.exit(post, ctx)
        assert state._status.get() == ChartStatus.PREEMPTING

    @pytest.mark.asyncio
    async def test_reset_after_success_allows_reentry(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        state._status.set(ChartStatus.SUCCESS)
        state.reset()
        state.enter(post, ctx)
        assert state._status.get() == ChartStatus.RUNNING

    @pytest.mark.asyncio
    async def test_reset_after_failure_allows_reentry(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = FailingState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._status.get() == ChartStatus.FAILURE
        state.reset()
        assert state._status.get() == ChartStatus.WAITING

    @pytest.mark.asyncio
    async def test_tracking_state_records_lifecycle_calls(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = TrackingState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        state.exit(post, ctx)
        state.reset()
        assert 'enter' in state.calls
        assert 'execute' in state.calls
        assert 'exit' in state.calls
        assert 'reset' in state.calls

    @pytest.mark.asyncio
    async def test_cannot_enter_after_enter_without_reset(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        with pytest.raises(InvalidTransition):
            state.enter(post, ctx)

    @pytest.mark.asyncio
    async def test_cannot_run_twice_without_reset(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        with pytest.raises(InvalidTransition):
            await state.run(post, ctx)

    @pytest.mark.asyncio
    async def test_state_reusable_after_reset(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        state.enter(post, ctx)
        await state.run(post, ctx)
        state._status.set(ChartStatus.SUCCESS)
        state.reset()
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert state._status.get() == ChartStatus.RUNNING


class TestStateFinishCallbacks:
    
    @pytest.mark.asyncio
    async def test_exit_calls_finish_when_run_completed(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = ConcreteState()
        called = False

        def callback():
            nonlocal called
            called = True

        state.register_finish_callback(callback)
        state.enter(post, ctx)
        await state.run(post, ctx)
        state.exit(post, ctx)
        await asyncio.sleep(0)  # Yield to allow finish() task to run
        assert called is True

    @pytest.mark.asyncio
    async def test_exit_does_not_call_finish_when_preempting(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = SlowState()
        called = False

        def callback():
            nonlocal called
            called = True

        state.register_finish_callback(callback)
        state.enter(post, ctx)
        # Start run but don't await - it's executing slowly
        task = asyncio.create_task(state.run(post, ctx))
        await asyncio.sleep(0.01)  # Let it start executing
        # Now exit before run completes
        state.exit(post, ctx)
        assert called is False  # finish() not called yet because run not completed
        await task  # Wait for run to finish
    
    @pytest.mark.asyncio
    async def test_state_run_calls_finish_on_cancellation(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = CancelledState()
        called = False
        
        def callback():
            nonlocal called
            called = True
        
        state.register_finish_callback(callback)
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert called is True
    
    @pytest.mark.asyncio
    async def test_streamstate_run_calls_finish_on_termination(self):
        class TerminatingStreamState(StreamState):
            async def execute(self, post, **inputs):
                for i in range(10):
                    yield {"count": i}
                    await asyncio.sleep(0.01)
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = TerminatingStreamState()
        called = False
        
        def callback():
            nonlocal called
            called = True
        
        state.register_finish_callback(callback)
        state.enter(post, ctx)
        task = asyncio.create_task(state.run(post, ctx))
        await asyncio.sleep(0.02)
        state.request_termination()
        await task
        assert called is True
    
    @pytest.mark.asyncio
    async def test_streamstate_run_calls_finish_on_cancellation(self):
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        state = CancelledStreamState()
        called = False

        def callback():
            nonlocal called
            called = True

        state.register_finish_callback(callback)
        state.enter(post, ctx)
        await state.run(post, ctx)
        assert called is True


# ============================================================================
# BoundState & BoundStreamState Tests
# ============================================================================

# Test helper states

class SimpleState(State):
    """State with simple inputs and outputs."""
    class inputs:
        x: int
        y: int = 10

    async def execute(self, post, x, y=10):
        return {"result": x + y, "doubled": x * 2}


class MultiInputState(State):
    """State with multiple input types."""
    class inputs:
        data: int
        config: dict
        flag: bool = True

    async def execute(self, post, data, config, flag=True):
        return {"processed": data, "mode": config.get("mode"), "enabled": flag}


# class FailingState(State):
#     """State that raises an exception."""
#     async def execute(self, post):
#         raise ValueError("Intentional failure")


class SimpleStreamState(StreamState):
    """Streaming state with multiple yields."""
    class inputs:
        count: int

    async def execute(self, post, count):
        for i in range(count):
            yield {"iteration": i, "value": i * 10}


class LongRunningStreamState(StreamState):
    """Streaming state for testing preemption."""
    async def execute(self, post):
        for i in range(5):
            yield {"count": i}
            await asyncio.sleep(0.01)


# ============================================================================
# TestBoundStateInputResolution
# ============================================================================

@pytest.mark.asyncio
class TestBoundStateInputResolution:
    """Verify bindings map context data to state inputs correctly."""

    async def test_bound_state_resolves_simple_binding(self):
        """Test that input 'x' bound to context key 'data' resolves correctly."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["data"] = 42
        ctx["y"] = 100

        bound = BoundState(
            state=SimpleState(),
            bindings={"x": "data"}  # x -> data
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)
        await bound.run(post, ctx)

        # Assert
        assert ctx["result"] == 142  # data(42) + y(100)
        assert ctx["doubled"] == 84  # data(42) * 2

    async def test_bound_state_resolves_multiple_bindings(self):
        """Test that multiple inputs bound to different context keys all resolve."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["sensor_data"] = 999
        ctx["global_config"] = {"mode": "auto"}
        ctx["enabled"] = False

        bound = BoundState(
            state=MultiInputState(),
            bindings={"data": "sensor_data", "config": "global_config", "flag": "enabled"}
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)
        await bound.run(post, ctx)

        # Assert
        assert ctx["processed"] == 999
        assert ctx["mode"] == "auto"
        assert ctx["enabled"] == False

    async def test_bound_state_uses_unbound_name_when_not_in_bindings(self):
        """Test that input 'y' not in bindings resolves directly from context['y']."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["x"] = 50  # Not bound, resolves directly
        ctx["y"] = 25  # Not bound, resolves directly

        bound = BoundState(
            state=SimpleState(),
            bindings={}  # Empty bindings
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)
        await bound.run(post, ctx)

        # Assert
        assert ctx["result"] == 75  # 50 + 25

    async def test_bound_state_raises_error_when_bound_key_missing(self):
        """Test that input bound to missing key raises KeyError."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["y"] = 10  # 'x' is missing

        bound = BoundState(
            state=SimpleState(),
            bindings={"x": "missing_key"}
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act & Assert
        bound.enter(post, ctx)
        await bound.run(post, ctx)

        # State should fail and store exception
        assert bound._status.get() == ChartStatus.FAILURE
        assert "__exception__" in ctx


# ============================================================================
# TestBoundStateOutputPropagation
# ============================================================================

@pytest.mark.asyncio
class TestBoundStateOutputPropagation:
    """Verify outputs write to original context, not bound paths."""

    async def test_bound_state_writes_output_to_original_context(self):
        """Test that state output writes to ctx['result'], not bound path."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["data"] = 10
        ctx["y"] = 5

        bound = BoundState(
            state=SimpleState(),
            bindings={"x": "data"}
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)
        await bound.run(post, ctx)

        # Assert - outputs go to original context keys
        assert ctx["result"] == 15
        assert ctx["doubled"] == 20
        # Verify bound input key still exists
        assert ctx["data"] == 10

    async def test_bound_state_writes_multiple_outputs_to_original_context(self):
        """Test that multiple outputs all propagate to original context."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["sensor"] = 123
        ctx["cfg"] = {"mode": "manual"}
        ctx["flag"] = True

        bound = BoundState(
            state=MultiInputState(),
            bindings={"data": "sensor", "config": "cfg"}
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)
        await bound.run(post, ctx)

        # Assert - all outputs in original context
        assert ctx["processed"] == 123
        assert ctx["mode"] == "manual"
        assert ctx["enabled"] == True


# ============================================================================
# TestBoundStateLifecycle
# ============================================================================

@pytest.mark.asyncio
class TestBoundStateLifecycle:
    """Verify state lifecycle (enter/run/exit/reset) works correctly."""

    async def test_bound_state_can_enter_when_waiting(self):
        """Test that fresh BoundState can enter successfully."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["x"] = 1
        ctx["y"] = 1

        bound = BoundState(state=SimpleState(), bindings={})
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act & Assert
        assert bound.can_enter() == True
        bound.enter(post, ctx)
        assert bound._status.get() == ChartStatus.RUNNING

    async def test_bound_state_cannot_enter_when_already_entered(self):
        """Test that state cannot be entered twice."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["x"] = 1
        ctx["y"] = 1

        bound = BoundState(state=SimpleState(), bindings={})
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)

        # Assert
        assert bound.can_enter() == False
        with pytest.raises(InvalidTransition):
            bound.enter(post, ctx)

    async def test_bound_state_completes_successfully_after_run(self):
        """Test that after run() completes, status is SUCCESS."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["x"] = 1
        ctx["y"] = 1

        bound = BoundState(state=SimpleState(), bindings={})
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act - proper lifecycle: enter → run → exit
        bound.enter(post, ctx)
        await bound.run(post, ctx)
        bound.exit(post, ctx)

        # Assert
        assert bound._status.get() == ChartStatus.SUCCESS

    async def test_bound_state_can_be_reset_after_completion(self):
        """Test that after SUCCESS, state can be reset to WAITING."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["x"] = 1
        ctx["y"] = 1

        bound = BoundState(state=SimpleState(), bindings={})
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act - proper lifecycle: enter → run → exit
        bound.enter(post, ctx)
        await bound.run(post, ctx)
        bound.exit(post, ctx)

        # Assert state can be reset
        assert bound.can_reset() == True
        bound.reset()
        assert bound._status.get() == ChartStatus.WAITING

    async def test_bound_state_cannot_be_reset_while_running(self):
        """Test that reset during execution is prevented."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["x"] = 1
        ctx["y"] = 1

        bound = BoundState(state=SimpleState(), bindings={})
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)

        # Assert - cannot reset while running
        assert bound.can_reset() == False
        with pytest.raises(InvalidTransition):
            bound.reset()


# ============================================================================
# TestBoundStateExceptionHandling
# ============================================================================

@pytest.mark.asyncio
class TestBoundStateExceptionHandling:
    """Verify error handling behavior."""

    async def test_bound_state_sets_failure_status_when_execute_raises(self):
        """Test that exception in execute() sets status to FAILURE."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()

        bound = BoundState(state=FailingState(), bindings={})
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)
        await bound.run(post, ctx)

        # Assert
        assert bound._status.get() == ChartStatus.FAILURE

    async def test_bound_state_stores_exception_info_in_context(self):
        """Test that exception details are stored in ctx['__exception__']."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()

        bound = BoundState(state=FailingState(), bindings={})
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)
        await bound.run(post, ctx)

        # Assert
        assert "__exception__" in ctx
        assert ctx["__exception__"]["message"] == "Test error"
        assert ctx["__exception__"]["type"] == "ValueError"


# ============================================================================
# TestBoundStreamStateInputResolution
# ============================================================================

@pytest.mark.asyncio
class TestBoundStreamStateInputResolution:
    """Verify bindings work for StreamState."""

    async def test_bound_stream_state_resolves_bindings_before_streaming(self):
        """Test that bound input resolves correctly before streaming begins."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["num_iterations"] = 3

        bound = BoundStreamState(
            state=SimpleStreamState(),
            bindings={"count": "num_iterations"}
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act - proper lifecycle: enter → run → exit
        bound.enter(post, ctx)
        await bound.run(post, ctx)
        bound.exit(post, ctx)

        # Assert - should have yielded 3 times (0, 1, 2)
        assert ctx["iteration"] == 2  # Last iteration
        assert ctx["value"] == 20  # 2 * 10


# ============================================================================
# TestBoundStreamStateOutputAccumulation
# ============================================================================

@pytest.mark.asyncio
class TestBoundStreamStateOutputAccumulation:
    """Verify streaming outputs accumulate in context."""

    async def test_bound_stream_state_accumulates_yielded_outputs(self):
        """Test that multiple yields update context with latest value."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["count"] = 3

        bound = BoundStreamState(
            state=SimpleStreamState(),
            bindings={}
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act - proper lifecycle: enter → run → exit
        bound.enter(post, ctx)
        await bound.run(post, ctx)
        bound.exit(post, ctx)

        # Assert - last yield wins
        assert ctx["iteration"] == 2
        assert ctx["value"] == 20

    async def test_bound_stream_state_writes_to_original_context_not_bound_path(self):
        """Test that yields write to original context, not bound paths."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["num"] = 2

        bound = BoundStreamState(
            state=SimpleStreamState(),
            bindings={"count": "num"}
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act - proper lifecycle: enter → run → exit
        bound.enter(post, ctx)
        await bound.run(post, ctx)
        bound.exit(post, ctx)

        # Assert - outputs go to original context keys
        assert ctx["iteration"] == 1
        assert ctx["value"] == 10
        # Bound input key still exists
        assert ctx["num"] == 2


# ============================================================================
# TestBoundStreamStatePreemption
# ============================================================================

@pytest.mark.asyncio
class TestBoundStreamStatePreemption:
    """Verify preemption behavior for streaming states."""

    async def test_bound_stream_state_returns_canceled_status_when_preempted(self):
        """Test that request_termination() during run() results in CANCELED status."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()

        bound = BoundStreamState(
            state=LongRunningStreamState(),
            bindings={}
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)

        # Start run in background
        run_task = asyncio.create_task(bound.run(post, ctx))

        # Request termination after first yield
        await asyncio.sleep(0.02)
        bound.request_termination()

        # Wait for run to complete
        await run_task

        # Assert
        assert bound._status.get() == ChartStatus.CANCELED

    async def test_bound_stream_state_stops_at_next_yield_when_terminated(self):
        """Test that termination stops execution at next checkpoint."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()

        bound = BoundStreamState(
            state=LongRunningStreamState(),
            bindings={}
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)

        # Start run in background
        run_task = asyncio.create_task(bound.run(post, ctx))

        # Terminate after 1-2 yields
        await asyncio.sleep(0.02)
        bound.request_termination()

        await run_task

        # Assert - should not have completed all 5 yields
        assert ctx["count"] < 4  # Stopped early


# ============================================================================
# TestBoundStateEdgeCases
# ============================================================================

@pytest.mark.asyncio
class TestBoundStateEdgeCases:
    """Unusual but valid scenarios."""

    async def test_bound_state_with_empty_bindings_dict_works(self):
        """Test that bindings={} resolves inputs directly from context."""
        # Arrange
        scope = Scope()
        ctx = scope.ctx()
        ctx["x"] = 11
        ctx["y"] = 22

        bound = BoundState(
            state=SimpleState(),
            bindings={}
        )
        queue = EventQueue()
        post = EventPost(queue=queue)

        # Act
        bound.enter(post, ctx)
        await bound.run(post, ctx)

        # Assert
        assert ctx["result"] == 33

    async def test_bound_state_can_have_explicit_name(self):
        """Test that BoundState can be given an explicit name."""
        # Arrange
        state = SimpleState()

        bound = BoundState(state=state, bindings={}, name="CustomBound")

        # Assert
        assert bound.name == "CustomBound"
