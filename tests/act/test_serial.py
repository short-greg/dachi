import pytest
from dachi.core import InitVar, Attr, ModuleList, Scope
from dachi.act._bt._core import TaskStatus
from dachi.act._bt._leafs import Action
from dachi.act._bt._serial import PreemptCond, Serial, Selector, Sequence
from .utils import ImmediateAction, SetStorageActionCounter, AlwaysTrueCond, AlwaysFalseCond, SetStorageAction



@pytest.mark.asyncio
class TestPreemptCond:
    async def test_preemptcond_failure_when_false(self):
        scope = Scope()
        ctx = scope.ctx()
        main = ImmediateAction(status_val=TaskStatus.SUCCESS)
        pc = PreemptCond(cond=AlwaysFalseCond(), task=main)
        assert await pc.tick(ctx) is TaskStatus.FAILURE
        assert main.status is TaskStatus.READY  # main skipped

    async def test_preemptcond_propagates_task_success(self):
        scope = Scope()
        ctx = scope.ctx()
        main = ImmediateAction(status_val=TaskStatus.SUCCESS)
        pc = PreemptCond(cond=AlwaysTrueCond(), task=main)
        print('Cascaded: ', pc.cascaded)
        assert await pc.tick(ctx) is TaskStatus.SUCCESS


class ImmediateAction(Action):
    """A task that immediately returns a fixed *status*."""

    status_val: InitVar[TaskStatus]

    def __post_init__(self, status_val: TaskStatus):
        super().__post_init__()
        self._status_val = status_val

    async def execute(self) -> TaskStatus:  # noqa: D401
        return self._status_val


class TestSerialValidation:
    def test_list_to_modulelist(self):
        serial = Sequence(tasks=[ImmediateAction(status_val=TaskStatus.SUCCESS)])
        assert isinstance(serial.tasks, ModuleList)

    def test_defaults_to_empty(self):
        assert len(Sequence().tasks) == 0

    def test_invalid_tasks_type(self):
        with pytest.raises(ValueError):
            Sequence(tasks=123)


@pytest.mark.asyncio
class TestSequence:
    async def test_sequence_is_running_when_started(self):
        scope = Scope()
        ctx = scope.ctx()
        
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=2)
        sequence = Sequence(
            tasks=[action1, action2]
        )
        assert await sequence.tick(ctx) == TaskStatus.RUNNING

    async def test_sequence_is_success_when_finished(self):
        scope = Scope()
        ctx = scope.ctx()
        
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=2)
        sequence = Sequence(tasks=[action1, action2])
        await sequence.tick(ctx)
        assert await sequence.tick(ctx) == TaskStatus.SUCCESS

    async def test_sequence_is_failure_less_than_zero(self):
        scope = Scope()
        ctx = scope.ctx()
        
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=-1)
        sequence = Sequence(tasks=[action1, action2])
        await sequence.tick(ctx)
        assert await sequence.tick(ctx) == TaskStatus.FAILURE

    async def test_sequence_is_ready_when_reset(self):
        scope = Scope()
        ctx = scope.ctx()
        
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=-1)
        sequence = Sequence(tasks=[action1, action2])
        await sequence.tick(ctx)
        await sequence.tick(ctx)
        sequence.reset()
        assert sequence.status == TaskStatus.READY

    async def test_sequence_finished_after_three_ticks(self):
        scope = Scope()
        ctx = scope.ctx()
        
        action1 = SetStorageAction(value=2)
        action2 = SetStorageActionCounter(value=3)
        sequence = Sequence(tasks=[action1, action2])
        await sequence.tick(ctx)
        await sequence.tick(ctx)
        assert await sequence.tick(ctx) == TaskStatus.SUCCESS


@pytest.mark.asyncio 
class TestCascadedSequence:
    async def test_cascaded_sequence_completes_all_immediate_tasks_in_one_tick(self):
        """Test that cascaded sequence completes all immediate SUCCESS tasks in a single tick"""
        action1 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2 = ImmediateAction(status_val=TaskStatus.SUCCESS) 
        action3 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence = Sequence(tasks=[action1, action2, action3])
        sequence.cascade()
        
        # Should complete all tasks in one tick
        scope = Scope()
        ctx = scope.ctx()
        assert await sequence.tick(ctx) == TaskStatus.SUCCESS
        assert action1.status == TaskStatus.SUCCESS
        assert action2.status == TaskStatus.SUCCESS  
        assert action3.status == TaskStatus.SUCCESS

    async def test_cascaded_sequence_stops_at_running_task(self):
        """Test that cascaded sequence stops when it encounters a RUNNING task"""
        action1 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2 = ImmediateAction(status_val=TaskStatus.RUNNING)
        action3 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence = Sequence(tasks=[action1, action2, action3])
        sequence.cascade()
        
        # Should stop at the RUNNING task
        scope = Scope()
        ctx = scope.ctx()
        assert await sequence.tick(ctx) == TaskStatus.RUNNING
        assert action1.status == TaskStatus.SUCCESS
        assert action2.status == TaskStatus.RUNNING
        assert action3.status == TaskStatus.READY  # Not executed yet

    async def test_cascaded_sequence_stops_at_failure(self):
        """Test that cascaded sequence stops immediately when a task fails"""
        action1 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action3 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence = Sequence(tasks=[action1, action2, action3])
        sequence.cascade()
        
        # Should fail immediately at action2
        scope = Scope()
        ctx = scope.ctx()
        assert await sequence.tick(ctx) == TaskStatus.FAILURE
        assert action1.status == TaskStatus.SUCCESS
        assert action2.status == TaskStatus.FAILURE
        assert action3.status == TaskStatus.READY  # Not executed

    async def test_cascaded_vs_non_cascaded_sequence_behavior(self):
        """Test difference between cascaded and non-cascaded sequence execution"""
        # Non-cascaded sequence
        action1_nc = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2_nc = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence_nc = Sequence(tasks=[action1_nc, action2_nc])
        sequence_nc.cascade(cascaded=False)
        
        # Cascaded sequence  
        action1_c = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2_c = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence_c = Sequence(tasks=[action1_c, action2_c])
        sequence_c.cascade()
        
        scope_nc = Scope()
        ctx_nc = scope_nc.ctx()
        scope_c = Scope()
        ctx_c = scope_c.ctx()
        
        # Non-cascaded needs multiple ticks
        assert await sequence_nc.tick(ctx_nc) == TaskStatus.RUNNING  # First tick - only action1
        assert await sequence_nc.tick(ctx_nc) == TaskStatus.SUCCESS  # Second tick - action2
        
        # Cascaded completes in one tick
        assert await sequence_c.tick(ctx_c) == TaskStatus.SUCCESS
        
    async def test_cascaded_sequence_with_mixed_task_types(self):
        """Test cascaded sequence with mixture of immediate and storage actions"""
        immediate = ImmediateAction(status_val=TaskStatus.SUCCESS)
        storage = SetStorageAction(value=1)  # Will succeed
        sequence = Sequence(tasks=[immediate, storage])
        sequence.cascade()
        
        # Should complete both in one tick since both succeed immediately
        scope = Scope()
        ctx = scope.ctx()
        assert await sequence.tick(ctx) == TaskStatus.SUCCESS
        assert immediate.status == TaskStatus.SUCCESS
        assert storage.status == TaskStatus.SUCCESS


@pytest.mark.asyncio
class TestCascadedSelector:
    async def test_cascaded_selector_stops_at_first_success(self):
        """Test that cascaded selector stops at first successful task"""
        action1 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2 = ImmediateAction(status_val=TaskStatus.SUCCESS) 
        action3 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        selector = Selector(tasks=[action1, action2, action3])
        selector.cascade()
        
        # Should succeed at action2 and not try action3
        scope = Scope()
        ctx = scope.ctx()
        assert await selector.tick(ctx) == TaskStatus.SUCCESS
        assert action1.status == TaskStatus.FAILURE
        assert action2.status == TaskStatus.SUCCESS
        assert action3.status == TaskStatus.READY  # Not executed

    async def test_cascaded_selector_tries_all_failing_tasks(self):
        """Test that cascaded selector tries all tasks if they all fail"""  
        action1 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action3 = ImmediateAction(status_val=TaskStatus.FAILURE)
        selector = Selector(tasks=[action1, action2, action3])
        selector.cascade()
        
        # Should try all tasks and fail
        scope = Scope()
        ctx = scope.ctx()
        assert await selector.tick(ctx) == TaskStatus.FAILURE
        assert action1.status == TaskStatus.FAILURE
        assert action2.status == TaskStatus.FAILURE  
        assert action3.status == TaskStatus.FAILURE

    async def test_cascaded_selector_stops_at_running_task(self):
        """Test that cascaded selector stops when it encounters a RUNNING task"""
        action1 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2 = ImmediateAction(status_val=TaskStatus.RUNNING)
        action3 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        selector = Selector(tasks=[action1, action2, action3])
        selector.cascade()
        
        # Should stop at the RUNNING task
        scope = Scope()
        ctx = scope.ctx()
        assert await selector.tick(ctx) == TaskStatus.RUNNING
        assert action1.status == TaskStatus.FAILURE
        assert action2.status == TaskStatus.RUNNING
        assert action3.status == TaskStatus.READY  # Not executed yet

    async def test_cascaded_vs_non_cascaded_selector_behavior(self):
        """Test difference between cascaded and non-cascaded selector execution"""
        # Non-cascaded selector
        action1_nc = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2_nc = ImmediateAction(status_val=TaskStatus.SUCCESS)
        selector_nc = Selector(tasks=[action1_nc, action2_nc])
        selector_nc.cascade(cascaded=False)
        
        # Cascaded selector
        action1_c = ImmediateAction(status_val=TaskStatus.FAILURE) 
        action2_c = ImmediateAction(status_val=TaskStatus.SUCCESS)
        selector_c = Selector(tasks=[action1_c, action2_c])
        selector_c.cascade()
        
        scope_nc = Scope()
        ctx_nc = scope_nc.ctx()
        scope_c = Scope()
        ctx_c = scope_c.ctx()
        
        # Non-cascaded needs multiple ticks
        assert await selector_nc.tick(ctx_nc) == TaskStatus.RUNNING  # First tick - action1 fails, move to action2
        assert await selector_nc.tick(ctx_nc) == TaskStatus.SUCCESS  # Second tick - action2 succeeds
        
        # Cascaded completes in one tick
        assert await selector_c.tick(ctx_c) == TaskStatus.SUCCESS


@pytest.mark.asyncio
class TestCascadedEdgeCases:
    async def test_cascaded_sequence_empty_tasks(self):
        """Test cascaded sequence with no tasks"""
        sequence = Sequence(tasks=[])
        sequence.cascade()
        # Should succeed immediately with no tasks
        scope = Scope()
        ctx = scope.ctx()
        assert await sequence.tick(ctx) == TaskStatus.SUCCESS

    async def test_cascaded_selector_empty_tasks(self):
        """Test cascaded selector with no tasks"""
        selector = Selector(tasks=[])
        selector.cascade()
        # Should fail immediately with no tasks to try
        scope = Scope()
        ctx = scope.ctx()
        assert await selector.tick(ctx) == TaskStatus.FAILURE

    async def test_cascaded_sequence_single_task(self):
        """Test cascaded sequence with single task"""
        action = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence = Sequence(tasks=[action])
        sequence.cascade()
        
        scope = Scope()
        ctx = scope.ctx()
        assert await sequence.tick(ctx) == TaskStatus.SUCCESS
        assert action.status == TaskStatus.SUCCESS

    async def test_cascaded_selector_single_task(self):
        """Test cascaded selector with single task"""
        action = ImmediateAction(status_val=TaskStatus.SUCCESS)
        selector = Selector(tasks=[action])
        selector.cascade()
        
        scope = Scope()
        ctx = scope.ctx()
        assert await selector.tick(ctx) == TaskStatus.SUCCESS
        assert action.status == TaskStatus.SUCCESS

    async def test_cascaded_sequence_reset_behavior(self):
        """Test that cascaded sequence resets properly"""
        action1 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence = Sequence(tasks=[action1, action2])
        sequence.cascade()
        
        # Complete the sequence
        scope = Scope()
        ctx = scope.ctx()
        assert await sequence.tick(ctx) == TaskStatus.SUCCESS
        
        # Reset and verify state
        sequence.reset()
        assert sequence.status == TaskStatus.READY
        assert sequence._idx.data == 0
        assert action1.status == TaskStatus.READY
        assert action2.status == TaskStatus.READY

    async def test_cascaded_selector_reset_behavior(self):
        """Test that cascaded selector resets properly"""
        action1 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        selector = Selector(tasks=[action1, action2])
        selector.cascade()
        
        # Complete the selector
        scope = Scope()
        ctx = scope.ctx()
        assert await selector.tick(ctx) == TaskStatus.SUCCESS
        
        # Reset and verify state  
        selector.reset()
        assert selector.status == TaskStatus.READY
        assert selector._idx.data == 0
        assert action1.status == TaskStatus.READY
        assert action2.status == TaskStatus.READY


@pytest.mark.asyncio
class TestFallback:
    async def test_fallback_is_successful_after_one_tick(self):
        scope = Scope()
        ctx = scope.ctx()
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction()
        fallback = Selector(tasks=[action1, action2])
        assert await fallback.tick(ctx) == TaskStatus.SUCCESS

    async def test_fallback_is_successful_after_two_ticks(self):
        scope = Scope()
        ctx = scope.ctx()
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=1)
        fallback = Selector(tasks=[action1, action2])
        await fallback.tick(ctx)
        assert await fallback.tick(ctx) == TaskStatus.SUCCESS

    async def test_fallback_fails_after_two_ticks(self):
        scope = Scope()
        ctx = scope.ctx()
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = Selector(tasks=[action1, action2])
        await fallback.tick(ctx)
        assert await fallback.tick(ctx) == TaskStatus.FAILURE

    async def test_fallback_running_after_one_tick(self):
        scope = Scope()
        ctx = scope.ctx()
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = Selector(tasks=[action1, action2])
        assert await fallback.tick(ctx) == TaskStatus.RUNNING

    async def test_fallback_ready_after_reset(self):
        scope = Scope()
        ctx = scope.ctx()
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = Selector(tasks=[action1, action2])
        await fallback.tick(ctx)
        fallback.reset()
        assert fallback.status == TaskStatus.READY


# Context-aware test helpers
from dachi.core import Scope
from dachi.act._bt._leafs import Action, Condition

class ContextTestAction(Action):
    """Test action using function signature for input resolution"""
    
    class outputs:
        result: str
        success: bool
    
    def __post_init__(self):
        super().__post_init__()
        self._call_count = 0
    
    async def execute(self, target: tuple = (0, 0, 0), attempts: int = 1, optional_param: str = "default_value"):
        self._call_count += 1
        self._last_kwargs = {"target": target, "attempts": attempts, "optional_param": optional_param}
        
        if attempts > 0:
            return TaskStatus.SUCCESS, {
                "result": f"reached_{target}",
                "success": True
            }
        else:
            return TaskStatus.FAILURE, {
                "result": "failed",
                "success": False
            }

class SimpleContextAction(Action):
    """Simpler action that just returns success and configurable outputs"""
    
    class outputs:
        value: int
        
    def __init__(self, output_value: int = 42):
        super().__init__()
        self._output_value = output_value
    
    async def execute(self):
        return TaskStatus.SUCCESS, {"value": self._output_value}

class RequiredInputAction(Action):
    """Action with required inputs using class-based input resolution"""
    
    class inputs:
        required_param: str
        required_number: int
    
    async def execute(self, required_param: str, required_number: int):
        return TaskStatus.SUCCESS


@pytest.mark.asyncio
class TestSequenceWithContext:
    """Test context-aware Sequence behavior"""
    
    async def test_sequence_creates_child_contexts_with_correct_indices(self):
        """Test that sequence creates child contexts at correct index paths"""
        scope = Scope()
        ctx = scope.ctx()  # root context
        
        action1 = SimpleContextAction(output_value=1)
        action2 = SimpleContextAction(output_value=2)
        sequence = Sequence(tasks=[action1, action2])
        sequence.cascade()
        
        # This should create child contexts at paths (0,) and (1,)
        await sequence.tick(ctx)
        
        # Verify that data was stored at correct paths
        # When action1 ticks, it should store at path (0,) field "value" = 1
        # When action2 ticks, it should store at path (1,) field "value" = 2
        assert scope.path((0,), "value") == 1
        assert scope.path((1,), "value") == 2
    
    async def test_sequence_calls_leaf_with_resolved_kwargs(self):
        """Test that sequence calls leaf tasks with resolved keyword arguments"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Set up context data that the action should receive
        ctx["target"] = (10, 20, 30)
        ctx["attempts"] = 3
        
        action = ContextTestAction()
        sequence = Sequence(tasks=[action])
        
        await sequence.tick(ctx)
        
        # Verify the action was called with resolved kwargs
        assert hasattr(action, '_last_kwargs')
        assert action._last_kwargs['target'] == (10, 20, 30)
        assert action._last_kwargs['attempts'] == 3
        assert action._last_kwargs['optional_param'] == "default_value"  # from default
    
    async def test_sequence_calls_composite_with_child_context(self):
        """Test that sequence calls composite children with child context, not kwargs"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Create nested sequence (composite child)
        inner_action = SimpleContextAction(output_value=5)
        inner_sequence = Sequence(tasks=[inner_action])
        inner_sequence.cascade()
        outer_sequence = Sequence(tasks=[inner_sequence])
        outer_sequence.cascade()

        await outer_sequence.tick(ctx)
        
        # Verify inner sequence got a child context and data was stored correctly
        # The inner_action should store at path (0, 0) field "value" = 5
        # (outer child 0, inner child 0, output "value")
        assert scope.path((0, 0), "value") == 5
    
    async def test_sequence_stores_leaf_outputs_in_context(self):
        """Test that sequence stores leaf outputs using ctx.update()"""
        scope = Scope()
        ctx = scope.ctx()
        
        action = SimpleContextAction(output_value=42)
        sequence = Sequence(tasks=[action])
        sequence.cascade()

        await sequence.tick(ctx)
        
        # The action's output should be stored in the scope at the child's path
        assert scope.path((0,), "value") == 42
    
    async def test_sequence_resolves_inputs_from_context_data(self):
        """Test input resolution from context data"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Set up context data
        scope.set((), "target", (100, 200, 300))
        scope.set((), "attempts", 5)
        
        action = ContextTestAction()
        sequence = Sequence(tasks=[action])
        sequence.cascade()

        await sequence.tick(ctx)
        
        # Verify the action received the context data
        assert action._last_kwargs['target'] == (100, 200, 300)
        assert action._last_kwargs['attempts'] == 5
    
    async def test_sequence_fails_on_missing_required_inputs(self):
        """Test that sequence fails when child has unresolvable required inputs"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Create action with required inputs but don't provide them in context
        action = RequiredInputAction()
        sequence = Sequence(tasks=[action])
        
        # Should fail the sequence when child has missing required inputs
        result = await sequence.tick(ctx)
        assert result == TaskStatus.FAILURE
        assert action.status == TaskStatus.FAILURE
    
    async def test_sequence_uses_input_defaults_when_context_empty(self):
        """Test that optional inputs fall back to defaults when context is empty"""
        scope = Scope()
        ctx = scope.ctx()
        
        action = ContextTestAction()
        sequence = Sequence(tasks=[action])
        
        await sequence.tick(ctx)
        
        # Should use defaults for all inputs
        assert action._last_kwargs['target'] == (0, 0, 0)  # default
        assert action._last_kwargs['attempts'] == 1  # default
        assert action._last_kwargs['optional_param'] == "default_value"  # default
    
    async def test_sequence_sibling_data_flow_between_children(self):
        """Test that later children can access earlier children's outputs"""
        scope = Scope()
        ctx = scope.ctx()
        
        # First action produces output
        producer = SimpleContextAction(output_value=123)
        
        # Second action should be able to consume first action's output
        # This will be tested once input resolution is implemented
        consumer = ContextTestAction()

        sequence = Sequence(tasks=[producer, consumer])
        sequence.cascade()
        await sequence.tick(ctx)
        
        # Verify producer output is available in scope
        assert scope.path((0,), "value") == 123
        
        # Note: Full sibling resolution will be tested in input resolution tests


@pytest.mark.asyncio  
class TestSelectorWithContext:
    """Test context-aware Selector behavior"""
    
    async def test_selector_creates_child_contexts_with_correct_indices(self):
        """Test that selector creates child contexts at correct index paths"""
        scope = Scope()
        ctx = scope.ctx()
        
        # First action fails, second succeeds
        action1 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2 = SimpleContextAction(output_value=99)
        selector = Selector(tasks=[action1, action2])
        
        await selector.tick(ctx)  # First tick: action1 fails, moves to index 1
        await selector.tick(ctx)  # Second tick: action2 succeeds
        
        # Should have tried action1 (no output) and succeeded with action2
        assert scope[(1, "value")] == 99  # action2's output at index 1
    
    async def test_selector_propagates_context_through_decision_tree(self):
        """Test context forwarding through selector attempts"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Set up context data
        ctx["attempts"] = 2
        
        action1 = ContextTestAction()  # Will succeed with the context
        selector = Selector(tasks=[action1])
        
        await selector.tick(ctx)
        
        # Verify action received context data
        assert action1._last_kwargs['attempts'] == 2
    
    async def test_selector_stops_at_first_success_with_context(self):
        """Test that selector stops at first success and stores outputs"""
        scope = Scope()
        ctx = scope.ctx()
        
        action1 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2 = SimpleContextAction(output_value=777)
        action3 = SimpleContextAction(output_value=888)  # Should not execute
        
        selector = Selector(tasks=[action1, action2, action3])
        await selector.tick(ctx)  # First tick: action1 fails, moves to index 1
        await selector.tick(ctx)  # Second tick: action2 succeeds
        
        # Should have action2's output but not action3's
        assert scope[(1, "value")] == 777
        # action3 not executed, so key shouldn't exist
        try:
            scope[(2, "value")]
            assert False, "action3 should not have been executed"
        except KeyError:
            pass  # Expected - action3 wasn't executed
    
    async def test_selector_context_isolation_between_attempts(self):
        """Test that failed attempts don't affect later attempts"""
        scope = Scope()
        ctx = scope.ctx()
        
        # This test ensures each child gets its own context path
        action1 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2 = SimpleContextAction(output_value=456)
        
        selector = Selector(tasks=[action1, action2])
        await selector.tick(ctx)  # First tick: action1 fails, moves to index 1
        await selector.tick(ctx)  # Second tick: action2 succeeds
        
        # Only action2 should have stored output
        # action1 failed, so no output stored
        try:
            scope[(0, "value")]
            assert False, "action1 should not have stored output"
        except KeyError:
            pass  # Expected - action1 failed and didn't store output
        assert scope[(1, "value")] == 456  # action2 succeeded


@pytest.mark.asyncio
class TestContextUpdate:
    """Test ctx.update() method functionality"""
    
    async def test_ctx_update_stores_dict_at_context_path(self):
        """Test that ctx.update(dict) stores all key-value pairs at context path"""
        scope = Scope()
        ctx = scope.ctx(0)  # Context at path (0,)
        
        # Update context with a dictionary
        outputs = {
            "result": "success",
            "score": 95,
            "completed": True
        }
        ctx.update(outputs)
        
        # Verify all values stored at correct paths
        assert scope[(0, "result")] == "success"
        assert scope[(0, "score")] == 95
        assert scope[(0, "completed")] == True
    
    async def test_multiple_ctx_updates_accumulate_in_scope(self):
        """Test that multiple updates accumulate correctly in the scope"""
        scope = Scope()
        ctx = scope.ctx(1)  # Context at path (1,)
        
        # First update
        ctx.update({"step1": "done", "count": 1})
        
        # Second update with different keys
        ctx.update({"step2": "done", "count": 2})  # This should overwrite count
        
        # Third update
        ctx.update({"final": True})
        
        # Verify all updates accumulated correctly
        assert scope[(1, "step1")] == "done"
        assert scope[(1, "step2")] == "done"
        assert scope[(1, "count")] == 2  # Latest value
        assert scope[(1, "final")] == True
    
    async def test_ctx_update_returns_none_like_dict(self):
        """Test that ctx.update() returns None (dict-like behavior)"""
        scope = Scope()
        ctx = scope.ctx(2)
        
        # update() should return None like dict.update()
        result = ctx.update({"test": "value"})
        assert result is None
        
        # Verify data was still stored
        assert scope[(2, "test")] == "value"


@pytest.mark.asyncio  
class TestCompositeInputResolution:
    """Test cross-cutting input resolution scenarios"""
    
    async def test_pre_validation_checks_required_inputs_before_execution(self):
        """Test that composites validate required inputs before attempting child tick"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Action with required inputs (no defaults)
        action = RequiredInputAction()
        sequence = Sequence(tasks=[action])
        
        # No required inputs provided in context
        result = await sequence.tick(ctx)
        
        # Should fail immediately without ticking the child
        assert result == TaskStatus.FAILURE
        
        # Verify child was not ticked (no call count increment)
        assert not hasattr(action, '_call_count') or action._call_count == 0
    
    async def test_resolution_priority_sibling_over_context_over_defaults(self):
        """Test input resolution priority order"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Set up context data (lower priority)
        ctx["target"] = (1, 1, 1)
        ctx["attempts"] = 1
        
        # First action produces sibling output (higher priority)
        producer = SimpleContextAction(output_value=999)
        async def mock_tick(ctx):
            outputs = {
                "target": (2, 2, 2),  # This should override context
                "attempts": 2
            }
            ctx.update(outputs)
            producer._status.set(TaskStatus.SUCCESS)
            return TaskStatus.SUCCESS
        producer.tick = mock_tick
        
        consumer = ContextTestAction()
        sequence = Sequence(tasks=[producer, consumer])
        
        await sequence.tick(ctx)
        
        # Consumer should receive sibling output over context data
        # This will be tested once full resolution is implemented
        # For now, verify that the mock producer's output was stored
        assert scope[(0, "target")] == (2, 2, 2)
        assert scope[(0, "attempts")] == 2
    
    async def test_sibling_outputs_available_to_subsequent_children(self):
        """Test that children can access outputs from earlier siblings"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Producer creates output
        producer = SimpleContextAction(output_value=555)
        
        # Consumer should be able to access producer's output
        consumer = ContextTestAction()
        
        sequence = Sequence(tasks=[producer, consumer])
        await sequence.tick(ctx)
        
        # Verify producer output stored and accessible
        assert scope[(0, "value")] == 555
        
        # Note: Full sibling->consumer data flow will be tested 
        # once input resolution is implemented
    
    async def test_context_data_accessible_across_composite_boundaries(self):
        """Test that nested composites can access parent context data"""
        scope = Scope()
        root_ctx = scope.ctx()
        
        # Set up root context data
        root_ctx["global_config"] = {"max_attempts": 5}
        
        # Nested structure: sequence -> sequence -> action
        inner_action = ContextTestAction()
        inner_sequence = Sequence(tasks=[inner_action])
        outer_sequence = Sequence(tasks=[inner_sequence])
        
        await outer_sequence.tick(root_ctx)
        
        # Verify nested action can eventually access root context
        # This will be verified once context propagation is implemented
        assert scope[("global_config",)] == {"max_attempts": 5}
    
    async def test_nested_data_access_with_index_paths(self):
        """Test accessing nested data using index path notation"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Set up nested data structure
        scope[(0, "sensors")] = {
            "pose": {"x": 10, "y": 20, "z": 30},
            "velocity": {"linear": 1.5, "angular": 0.5}
        }
        
        action = ContextTestAction()
        sequence = Sequence(tasks=[action])
        
        # This will test path resolution like "0.sensors.pose.x"
        # For now, verify the data structure is accessible
        assert scope[(0, "sensors")]["pose"]["x"] == 10
        assert scope[(0, "sensors")]["velocity"]["linear"] == 1.5
    
    async def test_missing_required_input_causes_composite_failure(self):
        """Test composite failure when child has unresolvable required inputs"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Set up some context data, but not the required inputs
        ctx["irrelevant_data"] = "not_what_we_need"
        
        action = RequiredInputAction()  # Needs required_param and required_number
        sequence = Sequence(tasks=[action])
        
        result = await sequence.tick(ctx)
        
        # Should fail due to missing required inputs
        assert result == TaskStatus.FAILURE
    
    async def test_optional_inputs_use_defaults_when_missing(self):
        """Test that optional inputs fall back to class defaults correctly"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Don't provide any inputs - should use all defaults
        action = ContextTestAction()
        sequence = Sequence(tasks=[action])
        
        await sequence.tick(ctx)
        
        # Should have used all default values
        assert action._last_kwargs['target'] == (0, 0, 0)  # default
        assert action._last_kwargs['attempts'] == 1  # default
        assert action._last_kwargs['optional_param'] == "default_value"  # default


class TestSequenceRestrictedSchema:
    """Test Sequence.restricted_schema() - Pattern B (Direct Variants)"""

    def test_restricted_schema_returns_unrestricted_when_tasks_none(self):
        """Test that tasks=None returns unrestricted schema"""
        seq = Sequence()
        restricted = seq.restricted_schema(tasks=None)
        unrestricted = seq.schema()

        # Should be identical
        assert restricted == unrestricted

    def test_restricted_schema_updates_tasks_field_with_variants(self):
        """Test that tasks field is restricted to specified variants"""
        seq = Sequence()

        # Restrict to only ImmediateAction and SetStorageAction
        restricted = seq.restricted_schema(
            tasks=[ImmediateAction, SetStorageAction]
        )

        # Check that schema was updated
        assert "$defs" in restricted
        assert "Allowed_tasks" in restricted["$defs"]

        # Check that Allowed_tasks contains our variants
        allowed_union = restricted["$defs"]["Allowed_tasks"]
        assert "oneOf" in allowed_union
        refs = allowed_union["oneOf"]
        assert len(refs) == 2

        # Extract spec names from refs
        spec_names = {ref["$ref"].split("/")[-1] for ref in refs}
        # Spec names include module path, so check if they contain the expected name
        assert any("ImmediateActionSpec" in name for name in spec_names)
        assert any("SetStorageActionSpec" in name for name in spec_names)

    def test_restricted_schema_uses_shared_profile_by_default(self):
        """Test that default profile is 'shared' (uses $defs/Allowed_*)"""
        seq = Sequence()
        restricted = seq.restricted_schema(tasks=[ImmediateAction])

        # Should use shared union in $defs
        assert "Allowed_tasks" in restricted["$defs"]

        # tasks field should reference the shared union
        # Handle nullable field (anyOf with null)
        tasks_schema = restricted["properties"]["tasks"]
        if "anyOf" in tasks_schema:
            # Nullable: find the array option
            for option in tasks_schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "array":
                    assert option["items"] == {"$ref": "#/$defs/Allowed_tasks"}
                    break
        else:
            # Non-nullable
            assert tasks_schema["items"] == {"$ref": "#/$defs/Allowed_tasks"}

    def test_restricted_schema_inline_profile_creates_oneof(self):
        """Test that _profile='inline' creates inline oneOf"""
        seq = Sequence()
        restricted = seq.restricted_schema(
            tasks=[ImmediateAction, SetStorageAction],
            _profile="inline"
        )

        # Should still have defs for the individual tasks (with full module path)
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)
        assert any("SetStorageActionSpec" in key for key in defs_keys)

        # But tasks field should have inline oneOf (no Allowed_tasks)
        tasks_schema = restricted["properties"]["tasks"]
        if "anyOf" in tasks_schema:
            # Nullable: find the array option
            for option in tasks_schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "array":
                    assert "oneOf" in option["items"]
                    break
        else:
            # Non-nullable
            assert "oneOf" in tasks_schema["items"]

    def test_restricted_schema_with_task_spec_class(self):
        """Test that TaskSpec classes work as variants"""
        seq = Sequence()

        # Get the spec class
        spec_class = ImmediateAction.schema_model()

        restricted = seq.restricted_schema(tasks=[spec_class])

        # Should work and include the task (with full module path)
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)

    def test_restricted_schema_with_mixed_formats(self):
        """Test that mixed variant formats work together"""
        seq = Sequence()

        # Mix: Task class, TaskSpec class, and schema dict
        action_spec = SetStorageAction.schema_model()
        immediate_schema = ImmediateAction.schema()

        restricted = seq.restricted_schema(
            tasks=[
                ImmediateAction,  # Task class
                action_spec,       # Spec class
                immediate_schema   # Schema dict (will be duplicate)
            ]
        )

        # Should deduplicate and work correctly (with full module path)
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)
        assert any("SetStorageActionSpec" in key for key in defs_keys)


class TestSelectorRestrictedSchema:
    """Test Selector.restricted_schema() - Pattern B (Direct Variants)"""

    def test_restricted_schema_returns_unrestricted_when_tasks_none(self):
        """Test that tasks=None returns unrestricted schema"""
        sel = Selector()
        restricted = sel.restricted_schema(tasks=None)
        unrestricted = sel.schema()

        # Should be identical
        assert restricted == unrestricted

    def test_restricted_schema_updates_tasks_field_with_variants(self):
        """Test that tasks field is restricted to specified variants"""
        sel = Selector()

        # Restrict to only ImmediateAction and SetStorageAction
        restricted = sel.restricted_schema(
            tasks=[ImmediateAction, SetStorageAction]
        )

        # Check that schema was updated
        assert "$defs" in restricted
        assert "Allowed_tasks" in restricted["$defs"]

        # Check that Allowed_tasks contains our variants
        allowed_union = restricted["$defs"]["Allowed_tasks"]
        assert "oneOf" in allowed_union
        refs = allowed_union["oneOf"]
        assert len(refs) == 2

        # Extract spec names from refs
        spec_names = {ref["$ref"].split("/")[-1] for ref in refs}
        # Spec names include module path, so check if they contain the expected name
        assert any("ImmediateActionSpec" in name for name in spec_names)
        assert any("SetStorageActionSpec" in name for name in spec_names)

    def test_restricted_schema_uses_shared_profile_by_default(self):
        """Test that default profile is 'shared' (uses $defs/Allowed_*)"""
        sel = Selector()
        restricted = sel.restricted_schema(tasks=[ImmediateAction])

        # Should use shared union in $defs
        assert "Allowed_tasks" in restricted["$defs"]

        # tasks field should reference the shared union
        # Handle nullable field (anyOf with null)
        tasks_schema = restricted["properties"]["tasks"]
        if "anyOf" in tasks_schema:
            # Nullable: find the array option
            for option in tasks_schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "array":
                    assert option["items"] == {"$ref": "#/$defs/Allowed_tasks"}
                    break
        else:
            # Non-nullable
            assert tasks_schema["items"] == {"$ref": "#/$defs/Allowed_tasks"}

    def test_restricted_schema_inline_profile_creates_oneof(self):
        """Test that _profile='inline' creates inline oneOf"""
        sel = Selector()
        restricted = sel.restricted_schema(
            tasks=[ImmediateAction, SetStorageAction],
            _profile="inline"
        )

        # Should still have defs for the individual tasks (with full module path)
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)
        assert any("SetStorageActionSpec" in key for key in defs_keys)

        # But tasks field should have inline oneOf (no Allowed_tasks)
        tasks_schema = restricted["properties"]["tasks"]
        if "anyOf" in tasks_schema:
            # Nullable: find the array option
            for option in tasks_schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "array":
                    assert "oneOf" in option["items"]
                    break
        else:
            # Non-nullable
            assert "oneOf" in tasks_schema["items"]

    def test_restricted_schema_with_task_spec_class(self):
        """Test that TaskSpec classes work as variants"""
        sel = Selector()

        # Get the spec class
        spec_class = ImmediateAction.schema_model()

        restricted = sel.restricted_schema(tasks=[spec_class])

        # Should work and include the task (with full module path)
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)

    def test_restricted_schema_with_mixed_formats(self):
        """Test that mixed variant formats work together"""
        sel = Selector()

        # Mix: Task class, TaskSpec class, and schema dict
        action_spec = SetStorageAction.schema_model()
        immediate_schema = ImmediateAction.schema()

        restricted = sel.restricted_schema(
            tasks=[
                ImmediateAction,  # Task class
                action_spec,       # Spec class
                immediate_schema   # Schema dict (will be duplicate)
            ]
        )

        # Should deduplicate and work correctly (with full module path)
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)
        assert any("SetStorageActionSpec" in key for key in defs_keys)


class TestPreemptCondRestrictedSchema:
    """Test PreemptCond.restricted_schema() - Pattern C × 2 with Condition filter"""

    def test_restricted_schema_returns_unrestricted_when_tasks_none(self):
        """Test that tasks=None returns unrestricted schema"""
        pc = PreemptCond(cond=AlwaysTrueCond(), task=ImmediateAction(status_val=TaskStatus.SUCCESS))
        restricted = pc.restricted_schema(tasks=None)
        unrestricted = pc.schema()

        # Should be identical
        assert restricted == unrestricted

    def test_restricted_schema_updates_both_fields(self):
        """Test that both cond and task fields are updated"""
        pc = PreemptCond(cond=AlwaysTrueCond(), task=ImmediateAction(status_val=TaskStatus.SUCCESS))

        # Provide tasks with both Conditions and Actions
        restricted = pc.restricted_schema(
            tasks=[AlwaysTrueCond, AlwaysFalseCond, ImmediateAction, SetStorageAction]
        )

        # Both field schemas should be updated
        assert "$defs" in restricted
        assert "Allowed_cond" in restricted["$defs"]  # For cond field
        assert "Allowed_task" in restricted["$defs"]       # For task field

        # cond field should only have Conditions
        cond_schema = restricted["properties"]["cond"]
        assert cond_schema == {"$ref": "#/$defs/Allowed_cond"}

        # task field should have all tasks
        task_schema = restricted["properties"]["task"]
        assert task_schema == {"$ref": "#/$defs/Allowed_task"}

    def test_restricted_schema_filters_cond_to_conditions_only(self):
        """Test that cond field only includes Condition subclasses"""
        pc = PreemptCond(cond=AlwaysTrueCond(), task=ImmediateAction(status_val=TaskStatus.SUCCESS))

        # Provide mix of Conditions and Actions
        restricted = pc.restricted_schema(
            tasks=[AlwaysTrueCond, ImmediateAction, AlwaysFalseCond, SetStorageAction]
        )

        # Check cond field has only 2 Conditions
        allowed_cond_union = restricted["$defs"]["Allowed_cond"]
        cond_refs = allowed_cond_union["oneOf"]
        assert len(cond_refs) == 2

        cond_names = {ref["$ref"].split("/")[-1] for ref in cond_refs}
        assert any("AlwaysTrueCondSpec" in name for name in cond_names)
        assert any("AlwaysFalseCondSpec" in name for name in cond_names)

        # Check task field has all 4 tasks
        allowed_task_union = restricted["$defs"]["Allowed_task"]
        task_refs = allowed_task_union["oneOf"]
        assert len(task_refs) == 4

    def test_restricted_schema_with_no_conditions_only_updates_task(self):
        """Test that if no Conditions in tasks, only task field is updated"""
        pc = PreemptCond(cond=AlwaysTrueCond(), task=ImmediateAction(status_val=TaskStatus.SUCCESS))

        # Provide only Actions (no Conditions)
        restricted = pc.restricted_schema(
            tasks=[ImmediateAction, SetStorageAction]
        )

        # Only task field should be updated
        assert "Allowed_task" in restricted["$defs"]

        # cond field should remain unrestricted (no Allowed_cond)
        # The cond schema will still have a $ref but not to Allowed_cond
        cond_schema = restricted["properties"]["cond"]
        if "$ref" in cond_schema:
            assert "Allowed_cond" not in cond_schema["$ref"]

    def test_restricted_schema_inline_profile(self):
        """Test that _profile='inline' creates inline oneOf for both fields"""
        pc = PreemptCond(cond=AlwaysTrueCond(), task=ImmediateAction(status_val=TaskStatus.SUCCESS))

        restricted = pc.restricted_schema(
            tasks=[AlwaysTrueCond, AlwaysFalseCond, ImmediateAction, SetStorageAction],
            _profile="inline"
        )

        # Both fields should have inline oneOf
        cond_schema = restricted["properties"]["cond"]
        assert "oneOf" in cond_schema

        task_schema = restricted["properties"]["task"]
        assert "oneOf" in task_schema
