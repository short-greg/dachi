import pytest
from dachi.core import InitVar, Attr, ModuleList
from dachi.act._core import TaskStatus
from dachi.act._leafs import Action
from dachi.act._serial import PreemptCond, Serial, Selector, Sequence
from .utils import ImmediateAction, SetStorageActionCounter, AlwaysTrueCond, AlwaysFalseCond, SetStorageAction



@pytest.mark.asyncio
class TestPreemptCond:
    async def test_preemptcond_failure_when_false(self):
        main = ImmediateAction(status_val=TaskStatus.SUCCESS)
        pc = PreemptCond(cond=[AlwaysFalseCond()], task=main)
        assert await pc.tick() is TaskStatus.FAILURE
        assert main.status is TaskStatus.READY  # main skipped

    async def test_preemptcond_propagates_task_success(self):
        main = ImmediateAction(status_val=TaskStatus.SUCCESS)
        pc = PreemptCond(cond=[AlwaysTrueCond()], task=main)
        assert await pc.tick() is TaskStatus.SUCCESS


class ImmediateAction(Action):
    """A task that immediately returns a fixed *status*."""

    status_val: InitVar[TaskStatus]

    def __post_init__(self, status_val: TaskStatus):
        super().__post_init__()
        self._status_val = status_val

    async def act(self) -> TaskStatus:  # noqa: D401
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
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=2)
        sequence = Sequence(
            tasks=[action1, action2]
        )
        assert await sequence.tick() == TaskStatus.RUNNING

    async def test_sequence_is_success_when_finished(self):
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=2)
        sequence = Sequence(tasks=[action1, action2])
        await sequence.tick()
        assert await sequence.tick() == TaskStatus.SUCCESS

    async def test_sequence_is_failure_less_than_zero(self):
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=-1)
        sequence = Sequence(tasks=[action1, action2])
        await sequence.tick()
        assert await sequence.tick() == TaskStatus.FAILURE

    async def test_sequence_is_ready_when_reset(self):
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=-1)
        sequence = Sequence(tasks=[action1, action2])
        await sequence.tick()
        await sequence.tick()
        sequence.reset()
        assert sequence.status == TaskStatus.READY

    async def test_sequence_finished_after_three_ticks(self):
        action1 = SetStorageAction(value=2)
        action2 = SetStorageActionCounter(value=3)
        sequence = Sequence(tasks=[action1, action2])
        await sequence.tick()
        await sequence.tick()
        assert await sequence.tick() == TaskStatus.SUCCESS


@pytest.mark.asyncio 
class TestCascadedSequence:
    async def test_cascaded_sequence_completes_all_immediate_tasks_in_one_tick(self):
        """Test that cascaded sequence completes all immediate SUCCESS tasks in a single tick"""
        action1 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2 = ImmediateAction(status_val=TaskStatus.SUCCESS) 
        action3 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence = Sequence(tasks=[action1, action2, action3], cascaded=True)
        
        # Should complete all tasks in one tick
        assert await sequence.tick() == TaskStatus.SUCCESS
        assert action1.status == TaskStatus.SUCCESS
        assert action2.status == TaskStatus.SUCCESS  
        assert action3.status == TaskStatus.SUCCESS

    async def test_cascaded_sequence_stops_at_running_task(self):
        """Test that cascaded sequence stops when it encounters a RUNNING task"""
        action1 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2 = ImmediateAction(status_val=TaskStatus.RUNNING)
        action3 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence = Sequence(tasks=[action1, action2, action3], cascaded=True)
        
        # Should stop at the RUNNING task
        assert await sequence.tick() == TaskStatus.RUNNING
        assert action1.status == TaskStatus.SUCCESS
        assert action2.status == TaskStatus.RUNNING
        assert action3.status == TaskStatus.READY  # Not executed yet

    async def test_cascaded_sequence_stops_at_failure(self):
        """Test that cascaded sequence stops immediately when a task fails"""
        action1 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action3 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence = Sequence(tasks=[action1, action2, action3], cascaded=True)
        
        # Should fail immediately at action2
        assert await sequence.tick() == TaskStatus.FAILURE
        assert action1.status == TaskStatus.SUCCESS
        assert action2.status == TaskStatus.FAILURE
        assert action3.status == TaskStatus.READY  # Not executed

    async def test_cascaded_vs_non_cascaded_sequence_behavior(self):
        """Test difference between cascaded and non-cascaded sequence execution"""
        # Non-cascaded sequence
        action1_nc = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2_nc = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence_nc = Sequence(tasks=[action1_nc, action2_nc], cascaded=False)
        
        # Cascaded sequence  
        action1_c = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2_c = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence_c = Sequence(tasks=[action1_c, action2_c], cascaded=True)
        
        # Non-cascaded needs multiple ticks
        assert await sequence_nc.tick() == TaskStatus.RUNNING  # First tick - only action1
        assert await sequence_nc.tick() == TaskStatus.SUCCESS  # Second tick - action2
        
        # Cascaded completes in one tick
        assert await sequence_c.tick() == TaskStatus.SUCCESS
        
    async def test_cascaded_sequence_with_mixed_task_types(self):
        """Test cascaded sequence with mixture of immediate and storage actions"""
        immediate = ImmediateAction(status_val=TaskStatus.SUCCESS)
        storage = SetStorageAction(value=1)  # Will succeed
        sequence = Sequence(tasks=[immediate, storage], cascaded=True)
        
        # Should complete both in one tick since both succeed immediately
        assert await sequence.tick() == TaskStatus.SUCCESS
        assert immediate.status == TaskStatus.SUCCESS
        assert storage.status == TaskStatus.SUCCESS


@pytest.mark.asyncio
class TestCascadedSelector:
    async def test_cascaded_selector_stops_at_first_success(self):
        """Test that cascaded selector stops at first successful task"""
        action1 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2 = ImmediateAction(status_val=TaskStatus.SUCCESS) 
        action3 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        selector = Selector(tasks=[action1, action2, action3], cascaded=True)
        
        # Should succeed at action2 and not try action3
        assert await selector.tick() == TaskStatus.SUCCESS
        assert action1.status == TaskStatus.FAILURE
        assert action2.status == TaskStatus.SUCCESS
        assert action3.status == TaskStatus.READY  # Not executed

    async def test_cascaded_selector_tries_all_failing_tasks(self):
        """Test that cascaded selector tries all tasks if they all fail"""  
        action1 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action3 = ImmediateAction(status_val=TaskStatus.FAILURE)
        selector = Selector(tasks=[action1, action2, action3], cascaded=True)
        
        # Should try all tasks and fail
        assert await selector.tick() == TaskStatus.FAILURE
        assert action1.status == TaskStatus.FAILURE
        assert action2.status == TaskStatus.FAILURE  
        assert action3.status == TaskStatus.FAILURE

    async def test_cascaded_selector_stops_at_running_task(self):
        """Test that cascaded selector stops when it encounters a RUNNING task"""
        action1 = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2 = ImmediateAction(status_val=TaskStatus.RUNNING)
        action3 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        selector = Selector(tasks=[action1, action2, action3], cascaded=True)
        
        # Should stop at the RUNNING task
        assert await selector.tick() == TaskStatus.RUNNING
        assert action1.status == TaskStatus.FAILURE
        assert action2.status == TaskStatus.RUNNING
        assert action3.status == TaskStatus.READY  # Not executed yet

    async def test_cascaded_vs_non_cascaded_selector_behavior(self):
        """Test difference between cascaded and non-cascaded selector execution"""
        # Non-cascaded selector
        action1_nc = ImmediateAction(status_val=TaskStatus.FAILURE)
        action2_nc = ImmediateAction(status_val=TaskStatus.SUCCESS)
        selector_nc = Selector(tasks=[action1_nc, action2_nc], cascaded=False)
        
        # Cascaded selector
        action1_c = ImmediateAction(status_val=TaskStatus.FAILURE) 
        action2_c = ImmediateAction(status_val=TaskStatus.SUCCESS)
        selector_c = Selector(tasks=[action1_c, action2_c], cascaded=True)
        
        # Non-cascaded needs multiple ticks
        assert await selector_nc.tick() == TaskStatus.RUNNING  # First tick - action1 fails, move to action2
        assert await selector_nc.tick() == TaskStatus.SUCCESS  # Second tick - action2 succeeds
        
        # Cascaded completes in one tick
        assert await selector_c.tick() == TaskStatus.SUCCESS


@pytest.mark.asyncio
class TestCascadedEdgeCases:
    async def test_cascaded_sequence_empty_tasks(self):
        """Test cascaded sequence with no tasks"""
        sequence = Sequence(tasks=[], cascaded=True)
        # Should succeed immediately with no tasks
        assert await sequence.tick() == TaskStatus.SUCCESS

    async def test_cascaded_selector_empty_tasks(self):
        """Test cascaded selector with no tasks"""
        selector = Selector(tasks=[], cascaded=True) 
        # Should fail immediately with no tasks to try
        assert await selector.tick() == TaskStatus.FAILURE

    async def test_cascaded_sequence_single_task(self):
        """Test cascaded sequence with single task"""
        action = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence = Sequence(tasks=[action], cascaded=True)
        
        assert await sequence.tick() == TaskStatus.SUCCESS
        assert action.status == TaskStatus.SUCCESS

    async def test_cascaded_selector_single_task(self):
        """Test cascaded selector with single task"""
        action = ImmediateAction(status_val=TaskStatus.SUCCESS)
        selector = Selector(tasks=[action], cascaded=True)
        
        assert await selector.tick() == TaskStatus.SUCCESS
        assert action.status == TaskStatus.SUCCESS

    async def test_cascaded_sequence_reset_behavior(self):
        """Test that cascaded sequence resets properly"""
        action1 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        action2 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        sequence = Sequence(tasks=[action1, action2], cascaded=True)
        
        # Complete the sequence
        assert await sequence.tick() == TaskStatus.SUCCESS
        
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
        selector = Selector(tasks=[action1, action2], cascaded=True)
        
        # Complete the selector
        assert await selector.tick() == TaskStatus.SUCCESS
        
        # Reset and verify state  
        selector.reset()
        assert selector.status == TaskStatus.READY
        assert selector._idx.data == 0
        assert action1.status == TaskStatus.READY
        assert action2.status == TaskStatus.READY


@pytest.mark.asyncio
class TestFallback:
    async def test_fallback_is_successful_after_one_tick(self):
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction()
        fallback = Selector(tasks=[action1, action2])
        assert await fallback.tick() == TaskStatus.SUCCESS

    async def test_fallback_is_successful_after_two_ticks(self):
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=1)
        fallback = Selector(tasks=[action1, action2])
        await fallback.tick()
        assert await fallback.tick() == TaskStatus.SUCCESS

    async def test_fallback_fails_after_two_ticks(self):
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = Selector(tasks=[action1, action2])
        await fallback.tick()
        assert await fallback.tick() == TaskStatus.FAILURE

    async def test_fallback_running_after_one_tick(self):
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = Selector(tasks=[action1, action2])
        assert await fallback.tick() == TaskStatus.RUNNING

    async def test_fallback_ready_after_reset(self):
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = Selector(tasks=[action1, action2])
        await fallback.tick()
        fallback.reset()
        assert fallback.status == TaskStatus.READY
