"""Unit tests for StateChart state classes.

Tests cover BaseState, State, StreamState, and FinalState following the
framework testing conventions.
"""

import asyncio
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dachi.act._chart._state import BaseState, State, StreamState, FinalState, StateStatus
from dachi.act._chart._event import EventQueue, Post
from dachi.core import Scope


class TestStateStatus:
    
    def test_state_status_enum_values(self):
        assert StateStatus.WAITING.value == "waiting"
        assert StateStatus.RUNNING.value == "running"
        assert StateStatus.COMPLETED.value == "completed"
        assert StateStatus.PREEMPTED.value == "preempted"


class ConcreteState(State):
    """Concrete State implementation for testing."""
    
    async def execute(self, post, **inputs):
        await post("TestEvent", {"data": "test"})
        return {"output": "result"}


class ConcreteStateWithInputs(State):
    """State with explicit inputs declaration."""
    
    class inputs:
        param1: str
        param2: int = 42
        
    async def execute(self, post, param1, param2):
        await post("ProcessedEvent", {"param1": param1, "param2": param2})
        return {"processed": f"{param1}_{param2}"}


class ConcreteStreamState(StreamState):
    """Concrete StreamState implementation for testing."""
    
    async def astream(self, post, **inputs):
        await post("StartEvent")
        yield {"step": 1}
        yield {"step": 2} 
        await post("EndEvent")
        yield {"step": 3}


class ConcreteStreamStateWithInputs(StreamState):
    """StreamState with explicit inputs declaration."""
    
    class inputs:
        count: int = 3
        prefix: str = "item"
        
    async def astream(self, post, count, prefix):
        for i in range(count):
            yield {f"{prefix}_{i}": i}


class TestBaseState:
    
    def test_init_subclass_processes_inputs_outputs_emit(self):
        class TestState(BaseState):
            class inputs:
                param1: str
                param2: int = 42
                
            class outputs:
                result: str
                
            class emit:
                StartEvent: str
                EndEvent: str
        
        assert hasattr(TestState, 'sc_params')
        assert 'inputs' in TestState.sc_params
        assert 'outputs' in TestState.sc_params
        assert 'emit' in TestState.sc_params
        
        # Check inputs processing
        inputs_info = TestState.sc_params['inputs']
        assert 'param1' in inputs_info
        assert 'param2' in inputs_info
        assert inputs_info['param2']['default'] == 42
        
        # Check outputs processing
        outputs_info = TestState.sc_params['outputs']
        assert 'result' in outputs_info
        
        # Check emit processing
        emit_info = TestState.sc_params['emit']
        assert 'StartEvent' in emit_info
        assert 'EndEvent' in emit_info
    
    def test_init_subclass_with_no_declarations(self):
        class SimpleState(BaseState):
            pass
        
        assert hasattr(SimpleState, 'sc_params')
        assert SimpleState.sc_params['inputs'] == {}
        assert SimpleState.sc_params['outputs'] == {}
        assert SimpleState.sc_params['emit'] == {}
    
    def test_initial_state(self):
        state = ConcreteState()
        assert state.get_status() == StateStatus.WAITING
        assert not state._termination_requested.get()
        assert not state._run_completed.get()
        assert not state.is_final()
    
    def test_enter_sets_running_status(self):
        state = ConcreteState()
        state.enter()
        
        assert state.get_status() == StateStatus.RUNNING
        assert not state._termination_requested.get()
        assert not state._run_completed.get()
    
    def test_exit_when_run_completed_sets_completed(self):
        state = ConcreteState()
        state.enter()
        state._run_completed.set(True)
        state.exit()
        
        assert state.get_status() == StateStatus.COMPLETED
    
    def test_exit_when_run_not_completed_sets_preempted(self):
        state = ConcreteState()
        state.enter()
        state.exit()
        
        assert state.get_status() == StateStatus.PREEMPTED
        assert state._termination_requested.get()
    
    def test_request_termination_sets_flag(self):
        state = ConcreteState()
        state.request_termination()
        
        assert state._termination_requested.get()
    
    def test_build_inputs_with_inputs_class_uses_resolve_fields(self):
        state = ConcreteStateWithInputs()
        scope = Scope()
        ctx = scope.ctx()
        ctx["param1"] = "test"
        ctx["param2"] = 100
        
        inputs = state.build_inputs(ctx)
        
        assert inputs["param1"] == "test"
        assert inputs["param2"] == 100
    
    def test_build_inputs_with_inputs_class_uses_defaults(self):
        state = ConcreteStateWithInputs()
        scope = Scope()
        ctx = scope.ctx()
        ctx["param1"] = "test"  # param2 should use default
        
        inputs = state.build_inputs(ctx)
        
        assert inputs["param1"] == "test"
        assert inputs["param2"] == 42  # default value
    
    def test_build_inputs_without_inputs_class_uses_signature(self):
        state = ConcreteState()
        scope = Scope()
        ctx = scope.ctx()
        ctx["some_param"] = "value"
        
        inputs = state.build_inputs(ctx)
        
        # ConcreteState.execute takes **inputs, so should return empty dict
        assert inputs == {}


class TestState:
    
    @pytest.mark.asyncio
    async def test_run_executes_aforward_and_finishes(self):
        queue = EventQueue()
        post = Post(queue=queue)
        state = ConcreteState()
        scope = Scope()
        ctx = scope.ctx()
        
        state.enter()
        await state.run(post, ctx)
        
        assert state.get_status() == StateStatus.RUNNING  # exit() not called yet
        assert state._run_completed.get()
        
        # Check that post.finish() was called (should have posted Finished event)
        assert queue.size() == 2  # TestEvent + Finished
        events = [queue.pop_nowait(), queue.pop_nowait()]
        assert events[0]["type"] == "TestEvent"
        assert events[1]["type"] == "Finished"
    
    @pytest.mark.asyncio
    async def test_run_updates_context_with_result(self):
        queue = EventQueue()
        post = Post(queue=queue)
        state = ConcreteState()
        scope = Scope()
        ctx = scope.ctx()
        
        state.enter()
        await state.run(post, ctx)
        
        # Check context was updated
        assert ctx["output"] == "result"
    
    @pytest.mark.asyncio
    async def test_run_with_inputs_resolves_parameters(self):
        queue = EventQueue()
        post = Post(queue=queue)
        state = ConcreteStateWithInputs()
        scope = Scope()
        ctx = scope.ctx()
        ctx["param1"] = "hello"  # param2 will use default
        
        state.enter()
        await state.run(post, ctx)
        
        # Check event was posted with resolved inputs
        assert queue.size() == 2  # ProcessedEvent + Finished
        event = queue.pop_nowait()
        assert event["type"] == "ProcessedEvent"
        assert event["payload"]["param1"] == "hello"
        assert event["payload"]["param2"] == 42
        
        # Check context was updated
        assert ctx["processed"] == "hello_42"
    
    @pytest.mark.asyncio
    async def test_run_with_exception_does_not_set_completed(self):
        class FailingState(State):
            async def execute(self, post, **inputs):
                raise ValueError("Test error")
        
        queue = EventQueue()
        post = Post(queue=queue)
        state = FailingState()
        scope = Scope()
        ctx = scope.ctx()
        
        state.enter()
        
        with pytest.raises(ValueError):
            await state.run(post, ctx)
        
        assert not state._run_completed.get()
        
        # post.finish() should still be called
        assert queue.size() == 1
        event = queue.pop_nowait()
        assert event["type"] == "Finished"


class TestStreamState:
    
    @pytest.mark.asyncio
    async def test_run_executes_astream_and_finishes(self):
        queue = EventQueue()
        post = Post(queue=queue)
        state = ConcreteStreamState()
        scope = Scope()
        ctx = scope.ctx()
        
        state.enter()
        await state.run(post, ctx)
        
        assert state.get_status() == StateStatus.RUNNING  # exit() not called yet
        assert state._run_completed.get()
        
        # Check events were posted
        assert queue.size() == 3  # StartEvent + EndEvent + Finished
        events = [queue.pop_nowait() for _ in range(3)]
        assert events[0]["type"] == "StartEvent"
        assert events[1]["type"] == "EndEvent"
        assert events[2]["type"] == "Finished"
    
    @pytest.mark.asyncio
    async def test_run_updates_context_with_streamed_outputs(self):
        queue = EventQueue()
        post = Post(queue=queue)
        state = ConcreteStreamState()
        scope = Scope()
        ctx = scope.ctx()
        
        state.enter()
        await state.run(post, ctx)
        
        # Check context was updated with all yields
        assert ctx["step"] == 3  # Last yield wins
    
    @pytest.mark.asyncio
    async def test_run_with_inputs_resolves_parameters(self):
        queue = EventQueue()
        post = Post(queue=queue)
        state = ConcreteStreamStateWithInputs()
        scope = Scope()
        ctx = scope.ctx()
        ctx["count"] = 2
        ctx["prefix"] = "test"
        
        state.enter()
        await state.run(post, ctx)
        
        # Check context was updated with streamed outputs
        assert ctx["test_0"] == 0
        assert ctx["test_1"] == 1
        assert "test_2" not in ctx  # Only 2 items
    
    @pytest.mark.asyncio
    async def test_run_respects_termination_request(self):
        class LongStreamState(StreamState):
            async def astream(self, post, **inputs):
                for i in range(10):
                    yield {"count": i}
                    # Use a small delay to allow termination processing
                    await asyncio.sleep(0.001)
        
        queue = EventQueue()
        post = Post(queue=queue)
        state = LongStreamState()
        scope = Scope()
        ctx = scope.ctx()
        
        state.enter()
        
        # Start the state in a task
        task = asyncio.create_task(state.run(post, ctx))
        
        # Let it yield a few times with more time
        await asyncio.sleep(0.005)
        
        # Request termination
        state.request_termination()
        
        # Wait for completion
        await task
        
        # Should not have completed all iterations
        assert not state._run_completed.get()
        assert ctx.get("count", -1) < 9  # Didn't reach the end
        
        # post.finish() should still be called
        assert queue.size() == 1
        event = queue.pop_nowait()
        assert event["type"] == "Finished"
    
    @pytest.mark.asyncio
    async def test_run_with_exception_does_not_set_completed(self):
        class FailingStreamState(StreamState):
            async def astream(self, post, **inputs):
                yield {"start": True}
                raise ValueError("Test error")
        
        queue = EventQueue()
        post = Post(queue=queue)
        state = FailingStreamState()
        scope = Scope()
        ctx = scope.ctx()
        
        state.enter()
        
        with pytest.raises(ValueError):
            await state.run(post, ctx)
        
        assert not state._run_completed.get()
        
        # Should have processed first yield and called finish
        assert ctx["start"] is True
        assert queue.size() == 1
        event = queue.pop_nowait()
        assert event["type"] == "Finished"


class TestFinalState:
    
    def test_is_final_returns_true(self):
        state = FinalState()
        assert state.is_final() is True
    
    @pytest.mark.asyncio
    async def test_execute_returns_none(self):
        queue = EventQueue()
        post = Post(queue=queue)
        state = FinalState()
        
        result = await state.execute(post)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_run_completes_successfully(self):
        queue = EventQueue()
        post = Post(queue=queue)
        state = FinalState()
        scope = Scope()
        ctx = scope.ctx()
        
        state.enter()
        await state.run(post, ctx)
        
        assert state._run_completed.get()
        
        # Should call finish
        assert queue.size() == 1
        event = queue.pop_nowait()
        assert event["type"] == "Finished"


class TestRunResult:
    
    def test_completed_exists(self):
        from dachi.act._chart._state import RunResult
        assert RunResult.COMPLETED is not None
        
    def test_preempted_exists(self):
        from dachi.act._chart._state import RunResult
        assert RunResult.PREEMPTED is not None